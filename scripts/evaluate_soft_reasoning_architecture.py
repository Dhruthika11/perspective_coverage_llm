"""
Evaluate a Softer Reasoning-Aware Architecture
Comprehensive calibration-only feature selection and threshold tuning.
Compares:
  - M0: Frozen baseline probe
  - M2: Unconstrained reasoning-aware probe (v1.1)
  - Candidate 1: Logit-Space Convex Ensemble (M_logit_blend)
  - Candidate 2: Probability-Space Weighted Blend (M_prob_blend)
  - Candidate 3: Constrained-Ratio Feature Model (M_ratio_LR)
  - Candidate 4: One-Sided Gated Reasoning Penalty (M_gated_penalty)

Strict Calibration-Only Rule:
  All hyper-parameters (alpha, gamma, lambda, tau_div) and decision thresholds (T*)
  are selected STRICTLY on the Calibration partition (16 claims, 100 pairs)
  under a prospective recall floor (Recall >= 95.83% or >= 91.67%).
  ZERO access to held-out test labels during tuning!
"""

import os
import json
import math
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def wilson_ci(k, n, conf=0.95):
    if n == 0:
        return (0.0, 0.0)
    z = stats.norm.ppf(1 - (1 - conf) / 2)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, center - half), min(1.0, center + half))

def fast_paired_bootstrap(sub_df, prob_a_col, t_a, prob_b_col, t_b, n_boot=10000, seed=42):
    """Claim-clustered paired bootstrap test comparing Model A and Model B cascade."""
    rng = np.random.RandomState(seed)
    
    # Cascade passes
    is_contra = (
        ((sub_df['stance_gold'] == 'support') & (sub_df['stance_generated'] == 'undermine')) |
        ((sub_df['stance_gold'] == 'undermine') & (sub_df['stance_generated'] == 'support'))
    )
    s1_pass = sub_df['cosine_similarity'] >= 0.55
    s2a_pass = s1_pass & (~is_contra)
    
    pass_a = s2a_pass & (sub_df[prob_a_col] >= t_a)
    pass_b = s2a_pass & (sub_df[prob_b_col] >= t_b)
    
    claims = []
    tps_a, fps_a = [], []
    tps_b, fps_b = [], []
    c2_tots, c1_tots = [], []
    c1_rejs_a, c1_rejs_b = [], []
    
    for cid, grp in sub_df.groupby('claim_id'):
        claims.append(cid)
        c2 = (grp['human_ground_truth_label'] == 'class_2').values
        c1 = (grp['human_ground_truth_label'] == 'class_1').values
        c_neg = ~c2
        
        pa = pass_a.loc[grp.index].values
        pb = pass_b.loc[grp.index].values
        
        c2_tots.append(c2.sum())
        c1_tots.append(c1.sum())
        
        tps_a.append((pa & c2).sum())
        fps_a.append((pa & c_neg).sum())
        c1_rejs_a.append(((~pa) & c1).sum())
        
        tps_b.append((pb & c2).sum())
        fps_b.append((pb & c_neg).sum())
        c1_rejs_b.append(((~pb) & c1).sum())
        
    c2_tots = np.array(c2_tots)
    c1_tots = np.array(c1_tots)
    tps_a = np.array(tps_a)
    fps_a = np.array(fps_a)
    c1_rejs_a = np.array(c1_rejs_a)
    tps_b = np.array(tps_b)
    fps_b = np.array(fps_b)
    c1_rejs_b = np.array(c1_rejs_b)
    
    n_claims = len(claims)
    delta_f1 = []
    delta_c1_rej = []
    
    for _ in range(n_boot):
        idx = rng.choice(n_claims, size=n_claims, replace=True)
        tot_c2 = c2_tots[idx].sum()
        tot_c1 = c1_tots[idx].sum()
        if tot_c2 == 0:
            continue
            
        tp_a = tps_a[idx].sum()
        fp_a = fps_a[idx].sum()
        rec_a = tp_a / tot_c2
        prec_a = tp_a / max(tp_a + fp_a, 1)
        f1_a = 2 * prec_a * rec_a / max(prec_a + rec_a, 1e-8)
        rej_c1_a = c1_rejs_a[idx].sum() / max(tot_c1, 1)
        
        tp_b = tps_b[idx].sum()
        fp_b = fps_b[idx].sum()
        rec_b = tp_b / tot_c2
        prec_b = tp_b / max(tp_b + fp_b, 1)
        f1_b = 2 * prec_b * rec_b / max(prec_b + rec_b, 1e-8)
        rej_c1_b = c1_rejs_b[idx].sum() / max(tot_c1, 1)
        
        delta_f1.append(f1_b - f1_a)
        delta_c1_rej.append(rej_c1_b - rej_c1_a)
        
    delta_f1 = np.array(delta_f1)
    delta_c1_rej = np.array(delta_c1_rej)
    
    p_f1 = (np.sum(delta_f1 <= 0) / len(delta_f1)) if np.mean(delta_f1) > 0 else (np.sum(delta_f1 >= 0) / len(delta_f1))
    p_c1 = (np.sum(delta_c1_rej <= 0) / len(delta_c1_rej)) if np.mean(delta_c1_rej) > 0 else (np.sum(delta_c1_rej >= 0) / len(delta_c1_rej))
    
    return {
        "mean_delta_f1": float(np.mean(delta_f1)),
        "ci_delta_f1": (float(np.percentile(delta_f1, 2.5)), float(np.percentile(delta_f1, 97.5))),
        "p_val_f1": float(min(1.0, p_f1 * 2)),
        "mean_delta_c1_rej": float(np.mean(delta_c1_rej)),
        "ci_delta_c1_rej": (float(np.percentile(delta_c1_rej, 2.5)), float(np.percentile(delta_c1_rej, 97.5))),
        "p_val_c1_rej": float(min(1.0, p_c1 * 2))
    }

def main():
    print("=" * 105)
    print("EVALUATING A SOFTER REASONING-AWARE ARCHITECTURE")
    print("=" * 105)
    
    # 1. Load 341 benchmark features
    df = pd.read_csv("results/calibration/e3_1_pairwise_features_341.csv")
    
    train_c21 = df[(df['partition'] == 'train') & (df['human_ground_truth_label'].isin(['class_2', 'class_1']))].copy()
    cal_all = df[df['partition'] == 'calibration'].copy()
    test_all = df[df['partition'] == 'test'].copy()
    
    y_train = (train_c21['human_ground_truth_label'] == 'class_2').astype(int).values
    
    # Fit Base M0 (global cosine, p_entail, p_neutral)
    feats_m0 = ['cosine_similarity', 'p_entail', 'p_neutral']
    sc_m0 = StandardScaler()
    X_tr_m0 = sc_m0.fit_transform(train_c21[feats_m0].values)
    clf_m0 = LogisticRegression(C=1.0, random_state=42).fit(X_tr_m0, y_train)
    
    # Fit Base M2 (unconstrained reasoning-aware probe)
    feats_m2 = ['cosine_similarity', 'p_entail', 'p_neutral', 'reasoning_cosine_v1_1']
    sc_m2 = StandardScaler()
    X_tr_m2 = sc_m2.fit_transform(train_c21[feats_m2].values)
    clf_m2 = LogisticRegression(C=1.0, random_state=42).fit(X_tr_m2, y_train)
    
    print("Base Models Trained:")
    print("  M0 Coefficients:", dict(zip(feats_m0, clf_m0.coef_[0].round(4))), f"Intercept: {clf_m0.intercept_[0]:.4f}")
    print("  M2 Coefficients:", dict(zip(feats_m2, clf_m2.coef_[0].round(4))), f"Intercept: {clf_m2.intercept_[0]:.4f}")
    
    # Compute base probabilities across all splits
    for d in [cal_all, test_all]:
        d['prob_m0'] = clf_m0.predict_proba(sc_m0.transform(d[feats_m0].values))[:, 1]
        d['prob_m2'] = clf_m2.predict_proba(sc_m2.transform(d[feats_m2].values))[:, 1]
        
        # Logit representations with numerical clipping
        p0_clipped = np.clip(d['prob_m0'].values, 1e-6, 1.0 - 1e-6)
        p2_clipped = np.clip(d['prob_m2'].values, 1e-6, 1.0 - 1e-6)
        d['logit_m0'] = logit(p0_clipped)
        d['logit_m2'] = logit(p2_clipped)
        
        # Cascade pre-conditions
        is_contra = (
            ((d['stance_gold'] == 'support') & (d['stance_generated'] == 'undermine')) |
            ((d['stance_gold'] == 'undermine') & (d['stance_generated'] == 'support'))
        )
        s1_pass = d['cosine_similarity'] >= 0.55
        d['s2a_pass'] = s1_pass & (~is_contra)
        
    c2_cal = (cal_all['human_ground_truth_label'] == 'class_2')
    c1_cal = (cal_all['human_ground_truth_label'] == 'class_1')
    tot_c2_cal = c2_cal.sum()
    tot_c1_cal = c1_cal.sum()
    
    print(f"\nCalibration Partition: {len(cal_all)} total pairs ({tot_c2_cal} C2, {tot_c1_cal} C1) across {cal_all['claim_id'].nunique()} claims.")
    
    # -------------------------------------------------------------------------
    # EXPLORATION ON CALIBRATION: Systematic Candidate Search
    # -------------------------------------------------------------------------
    calibration_sweep_results = []
    
    # Helper to evaluate score column on Calibration across thresholds
    def sweep_on_calibration(score_arr, model_family, param_name, param_val):
        best_f1_floor = -1.0
        best_t_floor = 0.20
        best_rec_floor = 0.0
        best_prec_floor = 0.0
        best_c1_rej_floor = 0.0
        best_fp_floor = 999
        
        # Also best unconstrained F1
        best_f1_unconstrained = -1.0
        best_t_unconstrained = 0.20
        
        for t_val in np.arange(0.10, 0.85, 0.01):
            t_val = round(float(t_val), 2)
            passed = cal_all['s2a_pass'] & (score_arr >= t_val)
            
            tp = (passed & c2_cal).sum()
            fp = (passed & (~c2_cal)).sum()
            fn = ((~passed) & c2_cal).sum()
            
            rec = tp / max(tot_c2_cal, 1)
            prec = tp / max(tp + fp, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-8)
            rej_c1 = ((~passed) & c1_cal).sum() / max(tot_c1_cal, 1)
            
            if f1 > best_f1_unconstrained:
                best_f1_unconstrained = f1
                best_t_unconstrained = t_val
                
            # Floor constraint: Must retain M0's calibration recall (>= 95.83%, i.e. 23/24)
            if rec >= 0.9583:
                if f1 > best_f1_floor or (f1 == best_f1_floor and prec > best_prec_floor):
                    best_f1_floor = f1
                    best_t_floor = t_val
                    best_rec_floor = rec
                    best_prec_floor = prec
                    best_c1_rej_floor = rej_c1
                    best_fp_floor = fp
                    
        return {
            "family": model_family,
            "param_name": param_name,
            "param_val": param_val,
            "best_t_floor": best_t_floor,
            "cal_recall_floor": round(best_rec_floor * 100, 2),
            "cal_precision_floor": round(best_prec_floor * 100, 2),
            "cal_f1_floor": round(best_f1_floor, 4),
            "cal_c1_rej_floor": round(best_c1_rej_floor * 100, 2),
            "cal_total_fp_floor": int(best_fp_floor),
            "best_t_unconstrained": best_t_unconstrained,
            "cal_f1_unconstrained": round(best_f1_unconstrained, 4)
        }
        
    # Baseline M0 on Calibration
    m0_cal_res = sweep_on_calibration(cal_all['prob_m0'].values, "M0_Baseline", "None", 0)
    calibration_sweep_results.append(m0_cal_res)
    
    # Baseline M2 on Calibration
    m2_cal_res = sweep_on_calibration(cal_all['prob_m2'].values, "M2_Unconstrained", "None", 0)
    calibration_sweep_results.append(m2_cal_res)
    
    # Family 1: Logit-Space Convex Ensemble
    # logit(P_soft) = (1 - alpha) * logit(P_M0) + alpha * logit(P_M2)
    for alpha in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70]:
        score_cal = expit((1.0 - alpha) * cal_all['logit_m0'].values + alpha * cal_all['logit_m2'].values)
        res = sweep_on_calibration(score_cal, "Logit_Convex_Blend", "alpha", alpha)
        calibration_sweep_results.append(res)
        
    # Family 2: Probability-Space Weighted Blend
    # P_soft = (1 - alpha) * P_M0 + alpha * P_M2
    for alpha in [0.10, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60]:
        score_cal = (1.0 - alpha) * cal_all['prob_m0'].values + alpha * cal_all['prob_m2'].values
        res = sweep_on_calibration(score_cal, "Prob_Weighted_Blend", "alpha", alpha)
        calibration_sweep_results.append(res)
        
    # Family 3: Constrained Feature Ratio Logistic Regression
    # Scale reasoning_cosine before training or fit with ratio gamma
    # X_tr = [cos_sim, gamma * reasoning_cosine, p_entail, p_neutral]
    trained_ratio_models = {}
    for gamma in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.75]:
        tr_df = train_c21.copy()
        ca_df = cal_all.copy()
        tr_df['scaled_reason'] = tr_df['reasoning_cosine_v1_1'] * gamma
        ca_df['scaled_reason'] = ca_df['reasoning_cosine_v1_1'] * gamma
        
        feats_sub = ['cosine_similarity', 'p_entail', 'p_neutral', 'scaled_reason']
        sc_r = StandardScaler()
        X_tr_r = sc_r.fit_transform(tr_df[feats_sub].values)
        clf_r = LogisticRegression(C=1.0, random_state=42).fit(X_tr_r, y_train)
        
        score_cal = clf_r.predict_proba(sc_r.transform(ca_df[feats_sub].values))[:, 1]
        trained_ratio_models[gamma] = (clf_r, sc_r, feats_sub)
        res = sweep_on_calibration(score_cal, "Constrained_Ratio_LR", "gamma", gamma)
        calibration_sweep_results.append(res)
        
    # Family 4: Gated Residual Reasoning Penalty
    # P_soft = P_M0 - lambda * max(0, tau_div - reasoning_cosine)
    gated_configs = []
    for tau_div in [0.50, 0.55, 0.60, 0.65]:
        for lam in [0.10, 0.15, 0.20, 0.30, 0.40, 0.50]:
            diff = np.maximum(0.0, tau_div - cal_all['reasoning_cosine_v1_1'].values)
            score_cal = np.clip(cal_all['prob_m0'].values - lam * diff, 0.0, 1.0)
            res = sweep_on_calibration(score_cal, "Gated_Reasoning_Penalty", f"tau_{tau_div}_lam", lam)
            res['tau_div'] = tau_div
            calibration_sweep_results.append(res)
            
    cal_df = pd.DataFrame(calibration_sweep_results)
    
    print("\n" + "=" * 110)
    print("CALIBRATION SWEEP: CANDIDATE SOFT ARCHITECTURES (UNDER RECALL FLOOR >= 95.83%)")
    print("=" * 110)
    top_candidates = cal_df.sort_values(by=['cal_f1_floor', 'cal_precision_floor', 'cal_c1_rej_floor'], ascending=False)
    print(top_candidates[['family', 'param_name', 'param_val', 'best_t_floor', 'cal_recall_floor', 'cal_precision_floor', 'cal_f1_floor', 'cal_c1_rej_floor', 'cal_total_fp_floor']].head(15).to_string(index=False))
    
    # Save full calibration sweep
    cal_df.to_csv("results/calibration/soft_reasoning_calibration_sweep.csv", index=False)
    
    # -------------------------------------------------------------------------
    # SELECT WINNING SOFT ARCHITECTURES ON CALIBRATION ONLY
    # -------------------------------------------------------------------------
    # Winner 1: Best Logit Ensemble with Recall Floor >= 95.83%
    sub_logit = cal_df[(cal_df['family'] == 'Logit_Convex_Blend') & (cal_df['cal_recall_floor'] >= 95.83)].sort_values(by='cal_f1_floor', ascending=False).iloc[0]
    win_alpha_logit = sub_logit['param_val']
    win_t_logit = sub_logit['best_t_floor']
    
    # Winner 2: Best Probability Blend with Recall Floor >= 95.83%
    sub_prob = cal_df[(cal_df['family'] == 'Prob_Weighted_Blend') & (cal_df['cal_recall_floor'] >= 95.83)].sort_values(by='cal_f1_floor', ascending=False).iloc[0]
    win_alpha_prob = sub_prob['param_val']
    win_t_prob = sub_prob['best_t_floor']
    
    # Winner 3: Best Gated Penalty with Recall Floor >= 95.83%
    sub_gate = cal_df[(cal_df['family'] == 'Gated_Reasoning_Penalty') & (cal_df['cal_recall_floor'] >= 95.83)].sort_values(by=['cal_f1_floor', 'cal_c1_rej_floor'], ascending=False).iloc[0]
    win_lam_gate = sub_gate['param_val']
    win_tau_gate = sub_gate.get('tau_div', 0.60)
    win_t_gate = sub_gate['best_t_floor']
    
    # Winner 4: Best Constrained Ratio LR
    sub_ratio = cal_df[(cal_df['family'] == 'Constrained_Ratio_LR') & (cal_df['cal_recall_floor'] >= 95.83)].sort_values(by='cal_f1_floor', ascending=False).iloc[0]
    win_gamma = sub_ratio['param_val']
    win_t_ratio = sub_ratio['best_t_floor']
    
    print("\n" + "=" * 90)
    print("PROSPECTIVELY SELECTED SOFT ARCHITECTURES (FREEZING ALL PARAMETERS):")
    print(f"  1. Logit Blend:   alpha = {win_alpha_logit}, Threshold T* = {win_t_logit:.2f} (Cal F1 = {sub_logit['cal_f1_floor']})")
    print(f"  2. Prob Blend:    alpha = {win_alpha_prob}, Threshold T* = {win_t_prob:.2f} (Cal F1 = {sub_prob['cal_f1_floor']})")
    print(f"  3. Gated Penalty: tau = {win_tau_gate}, lambda = {win_lam_gate}, Threshold T* = {win_t_gate:.2f} (Cal F1 = {sub_gate['cal_f1_floor']})")
    print(f"  4. Constrained LR:gamma = {win_gamma}, Threshold T* = {win_t_ratio:.2f} (Cal F1 = {sub_ratio['cal_f1_floor']})")
    print("=" * 90)
    
    # -------------------------------------------------------------------------
    # TEST EVALUATION: UNTOUCHED HELD-OUT TEST SPLIT (17 CLAIMS, 88 PAIRS)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 105)
    print("EVALUATING ON UNTOUCHED HELD-OUT TEST PARTITION (17 CLAIMS, 88 PAIRS)")
    print("=" * 105)
    
    # Compute test scores for selected models
    test_all['score_soft_logit'] = expit((1.0 - win_alpha_logit) * test_all['logit_m0'].values + win_alpha_logit * test_all['logit_m2'].values)
    test_all['score_soft_prob'] = (1.0 - win_alpha_prob) * test_all['prob_m0'].values + win_alpha_prob * test_all['prob_m2'].values
    
    diff_te = np.maximum(0.0, win_tau_gate - test_all['reasoning_cosine_v1_1'].values)
    test_all['score_soft_gated'] = np.clip(test_all['prob_m0'].values - win_lam_gate * diff_te, 0.0, 1.0)
    
    clf_r, sc_r, feats_sub = trained_ratio_models[win_gamma]
    te_ratio_df = test_all.copy()
    te_ratio_df['scaled_reason'] = te_ratio_df['reasoning_cosine_v1_1'] * win_gamma
    test_all['score_soft_ratio'] = clf_r.predict_proba(sc_r.transform(te_ratio_df[feats_sub].values))[:, 1]
    
    c2_te = (test_all['human_ground_truth_label'] == 'class_2')
    c1_te = (test_all['human_ground_truth_label'] == 'class_1')
    c0a_te = (test_all['human_ground_truth_label'] == 'class_0a')
    c0b_te = (test_all['human_ground_truth_label'] == 'class_0b')
    tot_c2_te = c2_te.sum()
    tot_c1_te = c1_te.sum()
    
    eval_configs = [
        ("M0 Frozen Baseline", "prob_m0", 0.20),
        ("M2 Unconstrained (T*=0.18)", "prob_m2", 0.18),
        ("M2 Unconstrained (T*=0.31)", "prob_m2", 0.31),
        (f"M_soft: Logit Blend (alpha={win_alpha_logit})", "score_soft_logit", win_t_logit),
        (f"M_soft: Prob Blend (alpha={win_alpha_prob})", "score_soft_prob", win_t_prob),
        (f"M_soft: Gated Penalty (tau={win_tau_gate}, lam={win_lam_gate})", "score_soft_gated", win_t_gate),
        (f"M_soft: Constrained LR (gamma={win_gamma})", "score_soft_ratio", win_t_ratio),
    ]
    
    test_report_records = []
    
    for label, col_score, t_val in eval_configs:
        passed = test_all['s2a_pass'] & (test_all[col_score] >= t_val)
        
        tp = (passed & c2_te).sum()
        fp = (passed & (~c2_te)).sum()
        fn = ((~passed) & c2_te).sum()
        tn = ((~passed) & (~c2_te)).sum()
        
        fp1 = (passed & c1_te).sum()
        fp0a = (passed & c0a_te).sum()
        fp0b = (passed & c0b_te).sum()
        
        rec = tp / tot_c2_te
        prec = tp / max(tp + fp, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        rej_c1 = (tot_c1_te - fp1) / max(tot_c1_te, 1)
        rej_c0a = ((~passed) & c0a_te).sum() / c0a_te.sum()
        rej_c0b = ((~passed) & c0b_te).sum() / c0b_te.sum()
        
        prec_ci = wilson_ci(tp, tp + fp)
        rec_ci = wilson_ci(tp, tot_c2_te)
        
        print(f"{label:42s} | T*={t_val:.2f} | Rec: {rec*100:5.2f}% ({tp}/{tot_c2_te}) | Prec: {prec*100:5.2f}% [95% CI: {prec_ci[0]*100:4.1f}-{prec_ci[1]*100:4.1f}%] | F1: {f1:.4f} | Rej C1: {rej_c1*100:5.2f}% ({tot_c1_te-fp1}/{tot_c1_te}) | Rej C0a: {rej_c0a*100:5.2f}% | Tot FP: {fp} (C1={fp1}, C0a={fp0a}, C0b={fp0b})")
        
        test_report_records.append({
            "model": label,
            "threshold": t_val,
            "recall": round(rec * 100, 2),
            "precision": round(prec * 100, 2),
            "precision_ci_lower": round(prec_ci[0] * 100, 2),
            "precision_ci_upper": round(prec_ci[1] * 100, 2),
            "f1": round(f1, 4),
            "rej_c1": round(rej_c1 * 100, 2),
            "rej_c0a": round(rej_c0a * 100, 2),
            "rej_c0b": round(rej_c0b * 100, 2),
            "total_fp": int(fp),
            "fp_c1": int(fp1),
            "fp_c0a": int(fp0a),
            "fp_c0b": int(fp0b),
            "tp": int(tp),
            "fn": int(fn)
        })
        
    # Paired Bootstrap Tests against M0 on Test Split
    print("\n--- PAIRED CLAIM-CLUSTERED BOOTSTRAP TESTS VS M0 ON TEST SPLIT (10,000 Resamples) ---")
    bootstrap_records = []
    for label, col_score, t_val in eval_configs[1:]:
        boot_res = fast_paired_bootstrap(test_all, "prob_m0", 0.20, col_score, t_val, n_boot=10000, seed=42)
        print(f"{label} vs M0:")
        print(f"  Delta F1:     {boot_res['mean_delta_f1']:+.4f} (95% CI: [{boot_res['ci_delta_f1'][0]:+.4f}, {boot_res['ci_delta_f1'][1]:+.4f}], p = {boot_res['p_val_f1']:.4f})")
        print(f"  Delta C1 Rej: {boot_res['mean_delta_c1_rej']*100:+.2f}% (95% CI: [{boot_res['ci_delta_c1_rej'][0]*100:+.2f}%, {boot_res['ci_delta_c1_rej'][1]*100:+.2f}%], p = {boot_res['p_val_c1_rej']:.4f})")
        bootstrap_records.append({"comparison": f"{label} vs M0", **boot_res})
        
    # -------------------------------------------------------------------------
    # INDEPENDENT EVALUATION: NON-TEST AUDIT COHORT (76 PAIRS, 34 INDEPENDENT CLAIMS)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 105)
    print("INDEPENDENT CLAIM-DISJOINT EVALUATION ON NON-TEST PRODUCTION CLAIMS (N = 76)")
    print("=" * 105)
    
    u100 = pd.read_csv("data/audit/unified_production_audit_benchmark_100_enriched_e3.csv")
    with open("data/calibration/stage2_claim_disjoint_partitions_human_341.json", "r") as f:
        p_json = json.load(f)
    test_claims = set(p_json['partitions']['test']['claim_ids'])
    u_non_test = u100[~u100['claim_id'].isin(test_claims)].copy()
    
    # Compute soft model scores for independent audit set
    p0_u = np.clip(u_non_test['prob_m0'].values, 1e-6, 1.0 - 1e-6)
    p2_u = np.clip(u_non_test['prob_m2'].values, 1e-6, 1.0 - 1e-6)
    logit_p0_u = logit(p0_u)
    logit_p2_u = logit(p2_u)
    
    u_non_test['score_soft_logit'] = expit((1.0 - win_alpha_logit) * logit_p0_u + win_alpha_logit * logit_p2_u)
    u_non_test['score_soft_prob'] = (1.0 - win_alpha_prob) * u_non_test['prob_m0'].values + win_alpha_prob * u_non_test['prob_m2'].values
    
    diff_u = np.maximum(0.0, win_tau_gate - u_non_test['reasoning_cosine_v1_1'].values)
    u_non_test['score_soft_gated'] = np.clip(u_non_test['prob_m0'].values - win_lam_gate * diff_u, 0.0, 1.0)
    
    u_ratio_df = u_non_test.copy()
    u_ratio_df['scaled_reason'] = u_ratio_df['reasoning_cosine_v1_1'] * win_gamma
    u_non_test['score_soft_ratio'] = 0.0
    s2a_idx_u = u_non_test[u_non_test['pass_s2a']].index
    if len(s2a_idx_u) > 0:
        u_non_test.loc[s2a_idx_u, 'score_soft_ratio'] = clf_r.predict_proba(sc_r.transform(u_ratio_df.loc[s2a_idx_u, feats_sub].values))[:, 1]
    
    c2_u = (u_non_test['human_audit_decision'] == 'class_2')
    c1_u = (u_non_test['human_audit_decision'] == 'class_1')
    c0a_u = (u_non_test['human_audit_decision'] == 'class_0a')
    c0b_u = (u_non_test['human_audit_decision'] == 'class_0b')
    tot_c2_u = c2_u.sum()
    tot_c1_u = c1_u.sum()
    
    ind_eval_records = []
    for label, col_score, t_val in eval_configs:
        passed = u_non_test['pass_s2a'] & (u_non_test[col_score] >= t_val)
        
        tp = (passed & c2_u).sum()
        fp = (passed & (~c2_u)).sum()
        fn = ((~passed) & c2_u).sum()
        
        fp1 = (passed & c1_u).sum()
        fp0a = (passed & c0a_u).sum()
        fp0b = (passed & c0b_u).sum()
        
        rec = tp / tot_c2_u
        prec = tp / max(tp + fp, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        rej_c1 = (tot_c1_u - fp1) / max(tot_c1_u, 1)
        rej_c0a = ((~passed) & c0a_u).sum() / c0a_u.sum()
        
        prec_ci = wilson_ci(tp, tp + fp)
        
        print(f"{label:42s} | T*={t_val:.2f} | Rec: {rec*100:5.1f}% ({tp}/{tot_c2_u}) | Prec: {prec*100:5.1f}% [95% CI: {prec_ci[0]*100:4.1f}-{prec_ci[1]*100:4.1f}%] | F1: {f1:.4f} | Rej C1: {rej_c1*100:5.1f}% ({tot_c1_u-fp1}/{tot_c1_u}) | Rej C0a: {rej_c0a*100:5.1f}% | Tot FP: {fp} (C1={fp1}, C0a={fp0a}, C0b={fp0b})")
        
        ind_eval_records.append({
            "model": label,
            "threshold": t_val,
            "recall": round(rec * 100, 2),
            "precision": round(prec * 100, 2),
            "f1": round(f1, 4),
            "rej_c1": round(rej_c1 * 100, 2),
            "rej_c0a": round(rej_c0a * 100, 2),
            "total_fp": int(fp),
            "fp_c1": int(fp1),
            "fp_c0a": int(fp0a),
            "fp_c0b": int(fp0b),
            "tp": int(tp),
            "fn": int(fn)
        })
        
    # -------------------------------------------------------------------------
    # PRODUCTION WORKLOAD (N = 250) HORVITZ-THOMPSON ESTIMATION
    # -------------------------------------------------------------------------
    print("\n" + "=" * 105)
    print("PRODUCTION WORKLOAD (N = 250) EXACT 7-SUBSTRATA HORVITZ-THOMPSON ESTIMATES")
    print("=" * 105)
    
    df_pop = pd.read_csv('results/production_verifier_predictions_250.csv')
    df_pop['is_unknown'] = df_pop['stance_generated'].astype(str).str.lower().str.strip() == 'unknown'
    u100['is_unknown'] = u100['stance_generated'].astype(str).str.lower().str.strip() == 'unknown'
    
    sub_strata = [
        ('ACCEPT_CLASS2_EQUIVALENT', False),
        ('ACCEPT_CLASS2_EQUIVALENT', True),
        ('REJECT_STAGE2B_DIFFERENT_WARRANT', False),
        ('REJECT_STAGE2B_DIFFERENT_WARRANT', True),
        ('REJECT_STAGE2A_STANCE', False),
        ('REJECT_STAGE1', False),
        ('REJECT_STAGE1', True),
    ]
    sub_weights = {}
    for dec, unk in sub_strata:
        N_h = len(df_pop[(df_pop['final_decision'] == dec) & (df_pop['is_unknown'] == unk)])
        n_h = len(u100[(u100['final_decision'] == dec) & (u100['is_unknown'] == unk)])
        sub_weights[(dec, unk)] = N_h / max(n_h, 1)
        
    u100['sub_weight'] = [sub_weights[(r['final_decision'], r['is_unknown'])] for _, r in u100.iterrows()]
    
    # Compute soft model scores for all 100 pairs
    p0_all = np.clip(u100['prob_m0'].values, 1e-6, 1.0 - 1e-6)
    p2_all = np.clip(u100['prob_m2'].values, 1e-6, 1.0 - 1e-6)
    u100['score_soft_logit'] = expit((1.0 - win_alpha_logit) * logit(p0_all) + win_alpha_logit * logit(p2_all))
    u100['score_soft_prob'] = (1.0 - win_alpha_prob) * u100['prob_m0'].values + win_alpha_prob * u100['prob_m2'].values
    
    diff_all = np.maximum(0.0, win_tau_gate - u100['reasoning_cosine_v1_1'].values)
    u100['score_soft_gated'] = np.clip(u100['prob_m0'].values - win_lam_gate * diff_all, 0.0, 1.0)
    
    u_all_ratio = u100.copy()
    u_all_ratio['scaled_reason'] = u_all_ratio['reasoning_cosine_v1_1'] * win_gamma
    u100['score_soft_ratio'] = 0.0
    s2a_idx_all = u100[u100['pass_s2a']].index
    if len(s2a_idx_all) > 0:
        u100.loc[s2a_idx_all, 'score_soft_ratio'] = clf_r.predict_proba(sc_r.transform(u_all_ratio.loc[s2a_idx_all, feats_sub].values))[:, 1]
    
    ht_records = []
    for label, col_score, t_val in eval_configs:
        passed = u100['pass_s2a'] & (u100[col_score] >= t_val)
        
        pop_tp = 0.0
        pop_fp = 0.0
        pop_fn = 0.0
        pop_tn = 0.0
        pop_fp_c1 = 0.0
        pop_fp_c0a = 0.0
        pop_c2 = 0.0
        pop_c1 = 0.0
        pop_c0a = 0.0
        
        for idx, r in u100.iterrows():
            w = r['sub_weight']
            lbl = r['human_audit_decision']
            pred = passed.loc[idx]
            
            if lbl == 'class_2':
                pop_c2 += w
                if pred:
                    pop_tp += w
                else:
                    pop_fn += w
            else:
                if lbl == 'class_1':
                    pop_c1 += w
                    if pred:
                        pop_fp_c1 += w
                elif lbl == 'class_0a':
                    pop_c0a += w
                    if pred:
                        pop_fp_c0a += w
                if pred:
                    pop_fp += w
                else:
                    pop_tn += w
                    
        wt_rec = pop_tp / max(pop_c2, 1e-8)
        wt_prec = pop_tp / max(pop_tp + pop_fp, 1e-8)
        wt_f1 = 2 * wt_prec * wt_rec / max(wt_prec + wt_rec, 1e-8)
        wt_c1_rej = (pop_c1 - pop_fp_c1) / max(pop_c1, 1e-8)
        
        print(f"{label:42s} | Rec: {wt_rec*100:5.2f}% | Prec: {wt_prec*100:5.2f}% | F1: {wt_f1:.4f} | C1 Rej: {wt_c1_rej*100:5.2f}% | Est Pop FP: {pop_fp:4.1f} (C1: {pop_fp_c1:4.1f}, C0a: {pop_fp_c0a:4.1f})")
        
        ht_records.append({
            "model": label,
            "threshold": t_val,
            "weighted_recall": round(wt_rec * 100, 2),
            "weighted_precision": round(wt_prec * 100, 2),
            "weighted_f1": round(wt_f1, 4),
            "weighted_c1_rejection": round(wt_c1_rej * 100, 2),
            "est_pop_fp": round(pop_fp, 1),
            "est_fp_c1": round(pop_fp_c1, 1),
            "est_fp_c0a": round(pop_fp_c0a, 1),
            "est_pop_tp": round(pop_tp, 1),
            "est_pop_fn": round(pop_fn, 1)
        })
        
    # Save full reports
    out_dict = {
        "calibrated_parameters": {
            "logit_blend": {"alpha": win_alpha_logit, "threshold": win_t_logit},
            "prob_blend": {"alpha": win_alpha_prob, "threshold": win_t_prob},
            "gated_penalty": {"tau_div": win_tau_gate, "lambda": win_lam_gate, "threshold": win_t_gate},
            "constrained_lr": {"gamma": win_gamma, "threshold": win_t_ratio}
        },
        "held_out_test_results": test_report_records,
        "independent_audit_results": ind_eval_records,
        "production_ht_results": ht_records,
        "bootstrap_tests_vs_m0": bootstrap_records
    }
    
    with open("results/calibration/soft_reasoning_architecture_report.json", "w") as f:
        json.dump(out_dict, f, indent=2)
    pd.DataFrame(test_report_records).to_csv("results/calibration/soft_reasoning_test_comparison.csv", index=False)
    pd.DataFrame(ht_records).to_csv("results/calibration/soft_reasoning_production_comparison.csv", index=False)
    print("\nSaved soft reasoning architecture report to results/calibration/")

if __name__ == '__main__':
    main()
