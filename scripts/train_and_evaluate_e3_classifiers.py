"""
E3 Experiment: Classifier Training, Calibration, Evaluation, and Error Analysis
Compares 5 models:
  M0: Existing frozen probe features (cosine_similarity, p_entail, p_neutral)
  M1: Existing features + claim_cosine
  M2: Existing features + reasoning_cosine
  M3: Existing features + warrant_cosine
  M4: Existing features + claim_cosine + reasoning_cosine + warrant_cosine

Protocol constraints:
  - Train strictly on Train partition (32 claims, 104 pairs: 51 C2, 53 C1)
  - Calibrate threshold T* strictly on Calibration partition (16 claims, 100 pairs)
  - Zero access to Held-Out Test partition during training or threshold tuning
  - Evaluate cascaded performance on Held-Out Test partition (17 claims, 88 pairs)
  - 2,000 resample Claim-Clustered Bootstrap for confidence intervals & paired tests
  - Detailed qualitative and quantitative error analysis
"""

import os
import json
import math
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

def get_wilson_ci(k, n, z=1.95996):
    """Computes exact 95% Wilson score interval for binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p_hat = k / n
    denom = 1.0 + (z ** 2) / n
    center = (p_hat + (z ** 2) / (2.0 * n)) / denom
    spread = (z * math.sqrt((p_hat * (1.0 - p_hat) + (z ** 2) / (4.0 * n)) / n)) / denom
    return max(0.0, center - spread), min(1.0, center + spread)

def fast_claim_clustered_ci(sub_df, prob_col, t_probe, n_boot=2000, seed=42):
    rng = np.random.RandomState(seed)
    
    # Precompute per-claim counts for speed
    claims = []
    c2_totals = []
    c1_totals = []
    c0a_totals = []
    c0b_totals = []
    
    tps = []
    fps_1 = []
    fps_0a = []
    fps_0b = []
    
    # Stage 1 + Stage 2a filters
    is_contra = (
        ((sub_df['stance_gold'] == 'support') & (sub_df['stance_generated'] == 'undermine')) |
        ((sub_df['stance_gold'] == 'undermine') & (sub_df['stance_generated'] == 'support'))
    )
    s1_pass = sub_df['cosine_similarity'] >= 0.55
    s2a_pass = s1_pass & (~is_contra)
    s2b_pass = s2a_pass & (sub_df[prob_col] >= t_probe)
    
    for cid, grp_idx in sub_df.groupby('claim_id').groups.items():
        claims.append(cid)
        grp_labels = sub_df.loc[grp_idx, 'human_ground_truth_label'].values
        grp_pass = s2b_pass.loc[grp_idx].values
        
        c2_mask = (grp_labels == 'class_2')
        c1_mask = (grp_labels == 'class_1')
        c0a_mask = (grp_labels == 'class_0a')
        c0b_mask = (grp_labels == 'class_0b')
        
        c2_totals.append(c2_mask.sum())
        c1_totals.append(c1_mask.sum())
        c0a_totals.append(c0a_mask.sum())
        c0b_totals.append(c0b_mask.sum())
        
        tps.append((grp_pass & c2_mask).sum())
        fps_1.append((grp_pass & c1_mask).sum())
        fps_0a.append((grp_pass & c0a_mask).sum())
        fps_0b.append((grp_pass & c0b_mask).sum())
        
    claims = np.array(claims)
    c2_totals = np.array(c2_totals)
    c1_totals = np.array(c1_totals)
    c0a_totals = np.array(c0a_totals)
    c0b_totals = np.array(c0b_totals)
    
    tps = np.array(tps)
    fps_1 = np.array(fps_1)
    fps_0a = np.array(fps_0a)
    fps_0b = np.array(fps_0b)
    
    n_claims = len(claims)
    
    boot_rec, boot_prec, boot_f1 = [], [], []
    boot_rej_c1, boot_rej_c0a, boot_rej_c0b = [], [], []
    
    for _ in range(n_boot):
        idx = rng.choice(n_claims, size=n_claims, replace=True)
        tot_c2 = c2_totals[idx].sum()
        if tot_c2 == 0:
            continue
        tot_tp = tps[idx].sum()
        tot_fp = fps_1[idx].sum() + fps_0a[idx].sum() + fps_0b[idx].sum()
        tot_c1 = c1_totals[idx].sum()
        tot_c0a = c0a_totals[idx].sum()
        tot_c0b = c0b_totals[idx].sum()
        
        rec = tot_tp / tot_c2
        prec = tot_tp / max(tot_tp + tot_fp, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        
        rej_c1 = (tot_c1 - fps_1[idx].sum()) / max(tot_c1, 1)
        rej_c0a = (tot_c0a - fps_0a[idx].sum()) / max(tot_c0a, 1)
        rej_c0b = (tot_c0b - fps_0b[idx].sum()) / max(tot_c0b, 1)
        
        boot_rec.append(rec)
        boot_prec.append(prec)
        boot_f1.append(f1)
        boot_rej_c1.append(rej_c1)
        boot_rej_c0a.append(rej_c0a)
        boot_rej_c0b.append(rej_c0b)
        
    return {
        "recall_ci": (float(np.percentile(boot_rec, 2.5)), float(np.percentile(boot_rec, 97.5))),
        "precision_ci": (float(np.percentile(boot_prec, 2.5)), float(np.percentile(boot_prec, 97.5))),
        "f1_ci": (float(np.percentile(boot_f1, 2.5)), float(np.percentile(boot_f1, 97.5))),
        "rej_c1_ci": (float(np.percentile(boot_rej_c1, 2.5)), float(np.percentile(boot_rej_c1, 97.5))),
        "rej_c0a_ci": (float(np.percentile(boot_rej_c0a, 2.5)), float(np.percentile(boot_rej_c0a, 97.5))),
        "rej_c0b_ci": (float(np.percentile(boot_rej_c0b, 2.5)), float(np.percentile(boot_rej_c0b, 97.5))),
        "boot_f1_dist": boot_f1,
        "boot_rec_dist": boot_rec,
        "boot_prec_dist": boot_prec,
        "boot_c1_rej_dist": boot_rej_c1
    }

def main():
    print("=" * 90)
    print("PHASE 4 & 5: TRAIN EXPERIMENTAL CLASSIFIERS & FULL BENCHMARK EVALUATION")
    print("=" * 90)
    
    # 1. Load pairwise features
    features_csv = "results/calibration/e3_pairwise_features_341.csv"
    if not os.path.exists(features_csv):
        raise FileNotFoundError(f"Missing {features_csv}. Run run_e3_feature_extraction.py first.")
    df = pd.read_csv(features_csv)
    
    # Partitions
    train_c21 = df[(df['partition'] == 'train') & (df['human_ground_truth_label'].isin(['class_2', 'class_1']))].copy()
    cal_c21 = df[(df['partition'] == 'calibration') & (df['human_ground_truth_label'].isin(['class_2', 'class_1']))].copy()
    test_c21 = df[(df['partition'] == 'test') & (df['human_ground_truth_label'].isin(['class_2', 'class_1']))].copy()
    
    cal_all = df[df['partition'] == 'calibration'].copy()
    test_all = df[df['partition'] == 'test'].copy()
    
    y_train = (train_c21['human_ground_truth_label'] == 'class_2').astype(int).values
    y_cal = (cal_c21['human_ground_truth_label'] == 'class_2').astype(int).values
    y_test = (test_c21['human_ground_truth_label'] == 'class_2').astype(int).values
    
    print(f"Dataset split counts:")
    print(f"  Train:       {len(train_c21)} pairs (C2={sum(y_train==1)}, C1={sum(y_train==0)}) across {train_c21['claim_id'].nunique()} claims")
    print(f"  Calibration: {len(cal_all)} total pairs ({len(cal_c21)} C2/C1) across {cal_all['claim_id'].nunique()} claims")
    print(f"  Held-Out:    {len(test_all)} total pairs ({len(test_c21)} C2/C1) across {test_all['claim_id'].nunique()} claims")
    
    # Model feature definitions
    base_feats = ['cosine_similarity', 'p_entail', 'p_neutral']
    models_config = {
        "M0_baseline": {
            "name": "M0: Frozen Probe Baseline",
            "features": base_feats,
            "description": "Existing features (BGE cosine, p_entail, p_neutral)"
        },
        "M1_claim": {
            "name": "M1: + Claim Similarity",
            "features": base_feats + ['claim_cosine'],
            "description": "Existing features + extracted central claim cosine"
        },
        "M2_reasoning": {
            "name": "M2: + Reasoning Similarity",
            "features": base_feats + ['reasoning_cosine'],
            "description": "Existing features + extracted core reasoning cosine"
        },
        "M3_warrant": {
            "name": "M3: + Warrant Similarity",
            "features": base_feats + ['warrant_cosine'],
            "description": "Existing features + extracted underlying warrant cosine"
        },
        "M4_all_three": {
            "name": "M4: + All Three Similarities",
            "features": base_feats + ['claim_cosine', 'reasoning_cosine', 'warrant_cosine'],
            "description": "Existing features + claim + reasoning + warrant cosines"
        }
    }
    
    trained_models = {}
    scalers = {}
    calibrated_thresholds = {}
    model_checkpoints = {}
    
    os.makedirs("models/e3_classifiers", exist_ok=True)
    
    print("\n" + "=" * 90)
    print("TRAINING CLASSIFIERS & CALIBRATING THRESHOLDS STRICTLY ON CALIBRATION SPLIT")
    print("=" * 90)
    
    for m_key, m_info in models_config.items():
        feats = m_info["features"]
        
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(train_c21[feats].values)
        X_ca = scaler.transform(cal_all[feats].values)
        X_te = scaler.transform(test_all[feats].values)
        
        clf = LogisticRegression(C=1.0, random_state=42)
        clf.fit(X_tr, y_train)
        
        trained_models[m_key] = clf
        scalers[m_key] = scaler
        
        # Probabilities
        prob_col = f"prob_{m_key}"
        df[prob_col] = clf.predict_proba(scaler.transform(df[feats].values))[:, 1]
        cal_all[prob_col] = clf.predict_proba(X_ca)[:, 1]
        test_all[prob_col] = clf.predict_proba(X_te)[:, 1]
        
        # Calibration Sweep: find T* strictly on Calibration partition
        # For M0, the frozen production threshold is T* = 0.20
        # For experimental models, sweep T in [0.10, 0.85] on calibration data
        best_f1 = -1.0
        best_t = 0.20
        cal_sweep_records = []
        
        # Stance concordance for calibration
        is_contra_cal = (
            ((cal_all['stance_gold'] == 'support') & (cal_all['stance_generated'] == 'undermine')) |
            ((cal_all['stance_gold'] == 'undermine') & (cal_all['stance_generated'] == 'support'))
        )
        s1_pass_cal = cal_all['cosine_similarity'] >= 0.55
        s2a_pass_cal = s1_pass_cal & (~is_contra_cal)
        
        for t_val in np.arange(0.10, 0.851, 0.01):
            t_val = round(float(t_val), 2)
            passed = s2a_pass_cal & (cal_all[prob_col] >= t_val)
            
            c2_mask = (cal_all['human_ground_truth_label'] == 'class_2')
            c1_mask = (cal_all['human_ground_truth_label'] == 'class_1')
            
            tp = (passed & c2_mask).sum()
            fp = (passed & (~c2_mask)).sum()
            fn = ((~passed) & c2_mask).sum()
            tn_c1 = ((~passed) & c1_mask).sum()
            
            rec = tp / max(tp + fn, 1)
            prec = tp / max(tp + fp, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-8)
            rej_c1 = tn_c1 / max(c1_mask.sum(), 1)
            
            cal_sweep_records.append({
                "t": t_val, "f1": f1, "prec": prec, "rec": rec, "rej_c1": rej_c1
            })
            
            if f1 > best_f1:
                best_f1 = f1
                best_t = t_val
                
        # If baseline M0, ensure comparison at official frozen T*=0.20 as well
        if m_key == "M0_baseline":
            t_star = 0.20
        else:
            t_star = best_t
            
        calibrated_thresholds[m_key] = t_star
        
        # Feature coefficients dict
        coef_dict = {f: float(c) for f, c in zip(feats, clf.coef_[0])}
        
        checkpoint = {
            "model_key": m_key,
            "model_name": m_info["name"],
            "features": feats,
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "coefficients": coef_dict,
            "intercept": float(clf.intercept_[0]),
            "calibrated_threshold": t_star,
            "calibration_f1_optimal_threshold": best_t,
            "calibration_optimal_f1": round(best_f1, 4),
            "training_samples": len(train_c21),
            "training_claims": int(train_c21['claim_id'].nunique())
        }
        model_checkpoints[m_key] = checkpoint
        
        ckpt_path = f"models/e3_classifiers/{m_key}.json"
        with open(ckpt_path, "w") as f:
            json.dump(checkpoint, f, indent=2)
            
        print(f"[{m_key}] T* = {t_star:.2f} (Cal F1 = {best_f1:.4f}) | Coefs: {coef_dict} | Intercept: {clf.intercept_[0]:.4f}")
        
    # Save all pairwise predictions with model probabilities
    df.to_csv("results/calibration/e3_pairwise_features_and_predictions_341.csv", index=False)
    
    # 2. Evaluate Full Cascaded Pipeline on Untouched Held-Out Test Split (17 claims, 88 pairs)
    print("\n" + "=" * 90)
    print("EVALUATION ON HELD-OUT TEST CLAIMS (17 CLAIMS, 88 PAIRS)")
    print("=" * 90)
    
    test_eval_rows = []
    test_bootstrap_results = {}
    
    c2_test = (test_all['human_ground_truth_label'] == 'class_2')
    c1_test = (test_all['human_ground_truth_label'] == 'class_1')
    c0a_test = (test_all['human_ground_truth_label'] == 'class_0a')
    c0b_test = (test_all['human_ground_truth_label'] == 'class_0b')
    non_c2_test = (~c2_test)
    
    n_c2 = c2_test.sum()   # 24
    n_c1 = c1_test.sum()   # 29
    n_c0a = c0a_test.sum() # 26
    n_c0b = c0b_test.sum() # 9
    n_tot = len(test_all)  # 88
    
    # Baseline filters
    is_contra_te = (
        ((test_all['stance_gold'] == 'support') & (test_all['stance_generated'] == 'undermine')) |
        ((test_all['stance_gold'] == 'undermine') & (test_all['stance_generated'] == 'support'))
    )
    s1_pass_te = test_all['cosine_similarity'] >= 0.55
    s2a_pass_te = s1_pass_te & (~is_contra_te)
    
    for m_key, m_info in models_config.items():
        t_star = calibrated_thresholds[m_key]
        prob_col = f"prob_{m_key}"
        
        s2b_pass = s2a_pass_te & (test_all[prob_col] >= t_star)
        
        tp = int((s2b_pass & c2_test).sum())
        fn = int(((~s2b_pass) & c2_test).sum())
        fp_c1 = int((s2b_pass & c1_test).sum())
        fp_c0a = int((s2b_pass & c0a_test).sum())
        fp_c0b = int((s2b_pass & c0b_test).sum())
        fp_tot = fp_c1 + fp_c0a + fp_c0b
        tn_tot = int(((~s2b_pass) & non_c2_test).sum())
        
        rec = tp / n_c2
        prec = tp / max(tp + fp_tot, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        
        rej_c1 = (n_c1 - fp_c1) / n_c1
        rej_c0a = (n_c0a - fp_c0a) / n_c0a
        rej_c0b = (n_c0b - fp_c0b) / n_c0b
        
        # Wilson CIs
        rec_w = get_wilson_ci(tp, n_c2)
        prec_w = get_wilson_ci(tp, max(tp + fp_tot, 1))
        
        # Clustered Bootstrap CI
        boot_res = fast_claim_clustered_ci(test_all, prob_col, t_star, n_boot=2000, seed=42)
        test_bootstrap_results[m_key] = boot_res
        
        row_dict = {
            "Model": m_key,
            "Description": m_info["description"],
            "Threshold": t_star,
            "TP": tp,
            "FP_Total": fp_tot,
            "FP_C1": fp_c1,
            "FP_C0a": fp_c0a,
            "FP_C0b": fp_c0b,
            "FN": fn,
            "Recall_C2": round(rec, 4),
            "Recall_Wilson_CI": f"[{rec_w[0]:.4f}, {rec_w[1]:.4f}]",
            "Recall_Clustered_CI": f"[{boot_res['recall_ci'][0]:.4f}, {boot_res['recall_ci'][1]:.4f}]",
            "Precision_C2": round(prec, 4),
            "Precision_Wilson_CI": f"[{prec_w[0]:.4f}, {prec_w[1]:.4f}]",
            "Precision_Clustered_CI": f"[{boot_res['precision_ci'][0]:.4f}, {boot_res['precision_ci'][1]:.4f}]",
            "F1_C2": round(f1, 4),
            "F1_Clustered_CI": f"[{boot_res['f1_ci'][0]:.4f}, {boot_res['f1_ci'][1]:.4f}]",
            "Rejection_C1": round(rej_c1, 4),
            "Rej_C1_Clustered_CI": f"[{boot_res['rej_c1_ci'][0]:.4f}, {boot_res['rej_c1_ci'][1]:.4f}]",
            "Rejection_C0a": round(rej_c0a, 4),
            "Rejection_C0b": round(rej_c0b, 4),
        }
        test_eval_rows.append(row_dict)
        
    summary_df = pd.DataFrame(test_eval_rows)
    print("\nHeld-Out Test Evaluation Summary Table:")
    cols_to_print = ["Model", "Threshold", "Recall_C2", "Precision_C2", "F1_C2", "Rejection_C1", "Rejection_C0a", "Rejection_C0b"]
    print(summary_df[cols_to_print].to_string(index=False))
    
    # Save CSV
    cascade_csv_path = "results/calibration/e3_full_cascade_table.csv"
    summary_df.to_csv(cascade_csv_path, index=False)
    print(f"\nSaved full cascade table to: {cascade_csv_path}")
    
    # 3. Paired Claim-Clustered Bootstrap Tests vs M0 Baseline
    print("\n" + "=" * 90)
    print("PAIRED CLAIM-CLUSTERED BOOTSTRAP HYPOTHESIS TESTS (DELTA VS M0 BASELINE)")
    print("=" * 90)
    
    m0_boot = test_bootstrap_results["M0_baseline"]
    paired_diff_rows = []
    
    for m_key in models_config.keys():
        if m_key == "M0_baseline":
            continue
        m_boot = test_bootstrap_results[m_key]
        
        # Paired delta distributions
        delta_f1 = np.array(m_boot["boot_f1_dist"]) - np.array(m0_boot["boot_f1_dist"])
        delta_rec = np.array(m_boot["boot_rec_dist"]) - np.array(m0_boot["boot_rec_dist"])
        delta_prec = np.array(m_boot["boot_prec_dist"]) - np.array(m0_boot["boot_prec_dist"])
        delta_c1_rej = np.array(m_boot["boot_c1_rej_dist"]) - np.array(m0_boot["boot_c1_rej_dist"])
        
        # Two-tailed bootstrap p-value
        p_val_f1 = 2.0 * min(np.mean(delta_f1 <= 0), np.mean(delta_f1 >= 0))
        p_val_c1 = 2.0 * min(np.mean(delta_c1_rej <= 0), np.mean(delta_c1_rej >= 0))
        
        diff_dict = {
            "Comparison": f"{m_key} vs M0",
            "Delta_F1_Mean": round(float(np.mean(delta_f1)), 4),
            "Delta_F1_95_CI": f"[{np.percentile(delta_f1, 2.5):+.4f}, {np.percentile(delta_f1, 97.5):+.4f}]",
            "p_value_F1": round(float(p_val_f1), 4),
            "Delta_Recall_Mean": round(float(np.mean(delta_rec)), 4),
            "Delta_Recall_95_CI": f"[{np.percentile(delta_rec, 2.5):+.4f}, {np.percentile(delta_rec, 97.5):+.4f}]",
            "Delta_Precision_Mean": round(float(np.mean(delta_prec)), 4),
            "Delta_Precision_95_CI": f"[{np.percentile(delta_prec, 2.5):+.4f}, {np.percentile(delta_prec, 97.5):+.4f}]",
            "Delta_C1_Rej_Mean": round(float(np.mean(delta_c1_rej)), 4),
            "Delta_C1_Rej_95_CI": f"[{np.percentile(delta_c1_rej, 2.5):+.4f}, {np.percentile(delta_c1_rej, 97.5):+.4f}]",
            "p_value_C1_Rej": round(float(p_val_c1), 4),
        }
        paired_diff_rows.append(diff_dict)
        print(f"{diff_dict['Comparison']:<20} | Delta F1: {diff_dict['Delta_F1_Mean']:+.4f} (95% CI {diff_dict['Delta_F1_95_CI']}, p={diff_dict['p_value_F1']:.4f}) | Delta C1 Rej: {diff_dict['Delta_C1_Rej_Mean']:+.4f} (p={diff_dict['p_value_C1_Rej']:.4f})")
        
    diff_df = pd.DataFrame(paired_diff_rows)
    diff_csv_path = "results/calibration/e3_paired_bootstrap_tests.csv"
    diff_df.to_csv(diff_csv_path, index=False)
    
    # 4. Per-Claim Performance Breakdown across all 17 Held-Out Claims
    print("\n" + "=" * 90)
    print("PER-CLAIM PERFORMANCE BREAKDOWN ON HELD-OUT TEST CLAIMS")
    print("=" * 90)
    
    per_claim_rows = []
    test_claims = sorted(test_all['claim_id'].unique())
    
    for cid in test_claims:
        c_sub = test_all[test_all['claim_id'] == cid]
        c_topic = c_sub['claim_text'].iloc[0]
        n_pairs = len(c_sub)
        n_c2 = (c_sub['human_ground_truth_label'] == 'class_2').sum()
        n_c1 = (c_sub['human_ground_truth_label'] == 'class_1').sum()
        
        row_c = {
            "claim_id": cid,
            "claim_text": c_topic,
            "total_pairs": n_pairs,
            "class_2_count": n_c2,
            "class_1_count": n_c1
        }
        
        for m_key in models_config.keys():
            t_star = calibrated_thresholds[m_key]
            prob_col = f"prob_{m_key}"
            
            # Cascade pass
            is_contra = (
                ((c_sub['stance_gold'] == 'support') & (c_sub['stance_generated'] == 'undermine')) |
                ((c_sub['stance_gold'] == 'undermine') & (c_sub['stance_generated'] == 'support'))
            )
            passed = (c_sub['cosine_similarity'] >= 0.55) & (~is_contra) & (c_sub[prob_col] >= t_star)
            
            tp = (passed & (c_sub['human_ground_truth_label'] == 'class_2')).sum()
            fp_c1 = (passed & (c_sub['human_ground_truth_label'] == 'class_1')).sum()
            
            row_c[f"{m_key}_TP"] = tp
            row_c[f"{m_key}_FP_C1"] = fp_c1
            row_c[f"{m_key}_F1"] = round(2 * tp / max(2 * tp + (n_c2 - tp) + (passed & (c_sub['human_ground_truth_label'] != 'class_2')).sum(), 1e-8), 4)
            
        per_claim_rows.append(row_c)
        
    per_claim_df = pd.DataFrame(per_claim_rows)
    per_claim_csv_path = "results/calibration/e3_per_claim_performance.csv"
    per_claim_df.to_csv(per_claim_csv_path, index=False)
    print(f"Saved per-claim breakdown to: {per_claim_csv_path}")
    
    # 5. Error Analysis: Recovered Class 2 pairs and Newly Rejected Class 1 pairs
    print("\n" + "=" * 90)
    print("DETAILED ERROR ANALYSIS (M4 vs M0)")
    print("=" * 90)
    
    # Classify each pair under M0 and M4
    is_contra_all = (
        ((test_all['stance_gold'] == 'support') & (test_all['stance_generated'] == 'undermine')) |
        ((test_all['stance_gold'] == 'undermine') & (test_all['stance_generated'] == 'support'))
    )
    s1_all = test_all['cosine_similarity'] >= 0.55
    s2a_all = s1_all & (~is_contra_all)
    
    test_all['accept_M0'] = s2a_all & (test_all['prob_M0_baseline'] >= calibrated_thresholds['M0_baseline'])
    test_all['accept_M4'] = s2a_all & (test_all['prob_M4_all_three'] >= calibrated_thresholds['M4_all_three'])
    
    # Recovered Class 2: M0 rejected (FN), but M4 accepted (TP)
    rec_c2 = test_all[(test_all['human_ground_truth_label'] == 'class_2') & (~test_all['accept_M0']) & (test_all['accept_M4'])]
    # Lost Class 2: M0 accepted (TP), but M4 rejected (FN)
    lost_c2 = test_all[(test_all['human_ground_truth_label'] == 'class_2') & (test_all['accept_M0']) & (~test_all['accept_M4'])]
    
    # Newly Rejected Class 1: M0 accepted (FP), but M4 rejected (TN)
    new_rej_c1 = test_all[(test_all['human_ground_truth_label'] == 'class_1') & (test_all['accept_M0']) & (~test_all['accept_M4'])]
    # Newly Accepted Class 1: M0 rejected (TN), but M4 accepted (FP)
    new_acc_c1 = test_all[(test_all['human_ground_truth_label'] == 'class_1') & (~test_all['accept_M0']) & (test_all['accept_M4'])]
    
    print(f"Error Analysis Summary:")
    print(f"  Class 2 Pairs Recovered by M4:       {len(rec_c2)}")
    print(f"  Class 2 Pairs Regressed (Lost):      {len(lost_c2)}")
    print(f"  Class 1 Pairs Newly Rejected by M4:  {len(new_rej_c1)}")
    print(f"  Class 1 Pairs Newly Accepted by M4:  {len(new_acc_c1)}")
    
    error_analysis_data = {
        "recovered_class_2": [],
        "lost_class_2": [],
        "newly_rejected_class_1": [],
        "newly_accepted_class_1": []
    }
    
    for idx, r in rec_c2.iterrows():
        item = {
            "pair_id": int(r["pair_id"]),
            "claim_id": int(r["claim_id"]),
            "claim_text": r["claim_text"],
            "gold_perspective_text": r["gold_perspective_text"],
            "generated_viewpoint_text": r["generated_viewpoint_text"],
            "m0_prob": float(r["prob_M0_baseline"]),
            "m4_prob": float(r["prob_M4_all_three"]),
            "cosine_similarity": float(r["cosine_similarity"]),
            "claim_cosine": float(r["claim_cosine"]),
            "reasoning_cosine": float(r["reasoning_cosine"]),
            "warrant_cosine": float(r["warrant_cosine"]),
            "p_entail": float(r["p_entail"])
        }
        error_analysis_data["recovered_class_2"].append(item)
        print(f"\n[RECOVERED C2] PID {r['pair_id']} (Claim {r['claim_id']}): M0_prob={r['prob_M0_baseline']:.4f} -> M4_prob={r['prob_M4_all_three']:.4f}")
        print(f"  Gold: {r['gold_perspective_text']}")
        print(f"  Gen : {r['generated_viewpoint_text']}")
        print(f"  Features: Cos={r['cosine_similarity']:.3f}, Claim={r['claim_cosine']:.3f}, Reason={r['reasoning_cosine']:.3f}, Warrant={r['warrant_cosine']:.3f}")
        
    for idx, r in new_rej_c1.iterrows():
        item = {
            "pair_id": int(r["pair_id"]),
            "claim_id": int(r["claim_id"]),
            "claim_text": r["claim_text"],
            "gold_perspective_text": r["gold_perspective_text"],
            "generated_viewpoint_text": r["generated_viewpoint_text"],
            "m0_prob": float(r["prob_M0_baseline"]),
            "m4_prob": float(r["prob_M4_all_three"]),
            "cosine_similarity": float(r["cosine_similarity"]),
            "claim_cosine": float(r["claim_cosine"]),
            "reasoning_cosine": float(r["reasoning_cosine"]),
            "warrant_cosine": float(r["warrant_cosine"]),
            "p_entail": float(r["p_entail"])
        }
        error_analysis_data["newly_rejected_class_1"].append(item)
        print(f"\n[NEWLY REJECTED C1] PID {r['pair_id']} (Claim {r['claim_id']}): M0_prob={r['prob_M0_baseline']:.4f} -> M4_prob={r['prob_M4_all_three']:.4f}")
        print(f"  Gold: {r['gold_perspective_text']}")
        print(f"  Gen : {r['generated_viewpoint_text']}")
        print(f"  Features: Cos={r['cosine_similarity']:.3f}, Claim={r['claim_cosine']:.3f}, Reason={r['reasoning_cosine']:.3f}, Warrant={r['warrant_cosine']:.3f}")
        
    for idx, r in lost_c2.iterrows():
        item = {
            "pair_id": int(r["pair_id"]),
            "claim_id": int(r["claim_id"]),
            "claim_text": r["claim_text"],
            "gold_perspective_text": r["gold_perspective_text"],
            "generated_viewpoint_text": r["generated_viewpoint_text"],
            "m0_prob": float(r["prob_M0_baseline"]),
            "m4_prob": float(r["prob_M4_all_three"]),
            "cosine_similarity": float(r["cosine_similarity"]),
            "claim_cosine": float(r["claim_cosine"]),
            "reasoning_cosine": float(r["reasoning_cosine"]),
            "warrant_cosine": float(r["warrant_cosine"])
        }
        error_analysis_data["lost_class_2"].append(item)

    for idx, r in new_acc_c1.iterrows():
        item = {
            "pair_id": int(r["pair_id"]),
            "claim_id": int(r["claim_id"]),
            "claim_text": r["claim_text"],
            "gold_perspective_text": r["gold_perspective_text"],
            "generated_viewpoint_text": r["generated_viewpoint_text"],
            "m0_prob": float(r["prob_M0_baseline"]),
            "m4_prob": float(r["prob_M4_all_three"]),
            "cosine_similarity": float(r["cosine_similarity"]),
            "claim_cosine": float(r["claim_cosine"]),
            "reasoning_cosine": float(r["reasoning_cosine"]),
            "warrant_cosine": float(r["warrant_cosine"])
        }
        error_analysis_data["newly_accepted_class_1"].append(item)

    error_analysis_json = "results/calibration/e3_recovered_and_rejected_pairs_analysis.json"
    with open(error_analysis_json, "w") as f:
        json.dump(error_analysis_data, f, indent=2)
    print(f"\nSaved detailed error analysis to: {error_analysis_json}")
    
    # 6. Comprehensive JSON Benchmark Comparison Report
    full_report = {
        "experiment": "E3_Reasoning_Aware_Feature_Extraction",
        "partitions": {
            "train": {"num_claims": int(train_c21['claim_id'].nunique()), "num_pairs": len(train_c21)},
            "calibration": {"num_claims": int(cal_all['claim_id'].nunique()), "num_pairs": len(cal_all)},
            "test": {"num_claims": int(test_all['claim_id'].nunique()), "num_pairs": len(test_all)}
        },
        "models": models_config,
        "calibrated_thresholds": calibrated_thresholds,
        "held_out_test_summary": test_eval_rows,
        "paired_bootstrap_tests_vs_m0": paired_diff_rows,
        "error_analysis_counts": {
            "recovered_c2": len(rec_c2),
            "lost_c2": len(lost_c2),
            "newly_rejected_c1": len(new_rej_c1),
            "newly_accepted_c1": len(new_acc_c1)
        }
    }
    
    report_path = "results/calibration/e3_benchmark_comparison_report.json"
    with open(report_path, "w") as f:
        json.dump(full_report, f, indent=2)
    print(f"Saved complete benchmark comparison report to: {report_path}")
    print("\nTraining, evaluation, bootstrap testing, and error analysis complete!")

if __name__ == "__main__":
    main()
