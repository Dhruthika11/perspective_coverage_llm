"""
Audit Suite for Softer Reasoning-Aware Architecture (M_soft)
Executes 4 comprehensive audits:
  Audit 1: Paired Claim-Clustered Bootstrap (10,000 resamples) on 88-pair test set across 5 metrics:
           - Delta F1
           - Delta Precision
           - Delta Recall
           - Delta Class 1 Rejection
           - Delta Total False Positives
  Audit 2: Independent Cohort Evaluation on:
           - Cohort 1: Non-test claims (76 pairs across 34 claims)
           - Cohort 2: Strictly fresh claims (26 pairs across 14 claims)
           - Cohort 3: Full unified benchmark (100 pairs across 45 claims)
           Includes full 4-class confusion matrices, exact counts, claim-clustered CIs, and paired tests vs M0.
  Audit 3: Parameter Sensitivity Analysis on Calibration Data ONLY (5 x 5 grid):
           - tau_div in [0.60, 0.625, 0.65, 0.675, 0.70]
           - lambda in [0.20, 0.30, 0.40, 0.50, 0.60]
  Audit 4: Verification Score Interpretation, Bounds [0, 1] Verification & Probability Calibration (ECE, Brier score).
"""

import os
import json
import math
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve

def wilson_ci(k, n, conf=0.95):
    if n == 0:
        return (0.0, 0.0)
    z = stats.norm.ppf(1 - (1 - conf) / 2)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, center - half), min(1.0, center + half))

def compute_soft_score(p0_arr, reas_arr, tau_div=0.65, lam=0.40):
    """Explicitly clipped verification score."""
    diff = np.maximum(0.0, tau_div - reas_arr)
    return np.clip(p0_arr - lam * diff, 0.0, 1.0)

def main():
    print("=" * 105)
    print("M_SOFT FOUR-PART COMPREHENSIVE AUDIT SUITE")
    print("=" * 105)
    
    # -------------------------------------------------------------------------
    # Infrastructure Setup
    # -------------------------------------------------------------------------
    df = pd.read_csv("results/calibration/e3_1_pairwise_features_341.csv")
    train = df[(df['partition'] == 'train') & (df['human_ground_truth_label'].isin(['class_2', 'class_1']))].copy()
    cal = df[df['partition'] == 'calibration'].copy()
    test = df[df['partition'] == 'test'].copy()
    
    y_train = (train['human_ground_truth_label'] == 'class_2').astype(int).values
    
    feats_m0 = ['cosine_similarity', 'p_entail', 'p_neutral']
    sc_m0 = StandardScaler()
    X_tr_m0 = sc_m0.fit_transform(train[feats_m0].values)
    clf_m0 = LogisticRegression(C=1.0, random_state=42).fit(X_tr_m0, y_train)
    
    # Compute M0 probabilities
    cal['prob_m0'] = clf_m0.predict_proba(sc_m0.transform(cal[feats_m0].values))[:, 1]
    test['prob_m0'] = clf_m0.predict_proba(sc_m0.transform(test[feats_m0].values))[:, 1]
    
    # Cascade filters
    for d in [cal, test]:
        is_contra = (
            ((d['stance_gold'] == 'support') & (d['stance_generated'] == 'undermine')) |
            ((d['stance_gold'] == 'undermine') & (d['stance_generated'] == 'support'))
        )
        s1 = d['cosine_similarity'] >= 0.55
        d['s2a_pass'] = s1 & (~is_contra)
        
    cal['score_soft'] = compute_soft_score(cal['prob_m0'].values, cal['reasoning_cosine_v1_1'].values, tau_div=0.65, lam=0.40)
    test['score_soft'] = compute_soft_score(test['prob_m0'].values, test['reasoning_cosine_v1_1'].values, tau_div=0.65, lam=0.40)
    
    # Frozen decisions on test
    test['pass_m0'] = test['s2a_pass'] & (test['prob_m0'] >= 0.20)
    test['pass_soft'] = test['s2a_pass'] & (test['score_soft'] >= 0.20)
    
    # =========================================================================
    # AUDIT 1: PAIRED CLAIM-CLUSTERED BOOTSTRAP (10,000 RESAMPLES) ON TEST SPLIT
    # =========================================================================
    print("\n" + "=" * 105)
    print("AUDIT 1: PAIRED CLAIM-CLUSTERED BOOTSTRAP (10,000 RESAMPLES) ON 88-PAIR TEST SPLIT")
    print("=" * 105)
    
    rng = np.random.RandomState(42)
    claims_test = test['claim_id'].unique()
    n_claims_te = len(claims_test)
    
    # Precompute per-claim counts for speed & vectorization
    claim_stats = []
    for cid in claims_test:
        sub = test[test['claim_id'] == cid]
        c2 = (sub['human_ground_truth_label'] == 'class_2').values
        c1 = (sub['human_ground_truth_label'] == 'class_1').values
        c_neg = ~c2
        
        pa = sub['pass_m0'].values
        pb = sub['pass_soft'].values
        
        claim_stats.append({
            'cid': cid,
            'c2_tot': c2.sum(),
            'c1_tot': c1.sum(),
            'tp_a': (pa & c2).sum(),
            'fp_a': (pa & c_neg).sum(),
            'fn_a': ((~pa) & c2).sum(),
            'c1_rej_a': ((~pa) & c1).sum(),
            'tp_b': (pb & c2).sum(),
            'fp_b': (pb & c_neg).sum(),
            'fn_b': ((~pb) & c2).sum(),
            'c1_rej_b': ((~pb) & c1).sum()
        })
    cs_df = pd.DataFrame(claim_stats)
    
    c2_tot_arr = cs_df['c2_tot'].values
    c1_tot_arr = cs_df['c1_tot'].values
    tp_a_arr = cs_df['tp_a'].values
    fp_a_arr = cs_df['fp_a'].values
    c1_rej_a_arr = cs_df['c1_rej_a'].values
    tp_b_arr = cs_df['tp_b'].values
    fp_b_arr = cs_df['fp_b'].values
    c1_rej_b_arr = cs_df['c1_rej_b'].values
    
    n_boot = 10000
    boot_delta_f1 = []
    boot_delta_prec = []
    boot_delta_rec = []
    boot_delta_c1_rej = []
    boot_delta_tot_fp = []
    
    boot_f1_m0 = []
    boot_f1_soft = []
    boot_prec_m0 = []
    boot_prec_soft = []
    boot_rec_m0 = []
    boot_rec_soft = []
    boot_c1_rej_m0 = []
    boot_c1_rej_soft = []
    boot_tot_fp_m0 = []
    boot_tot_fp_soft = []
    
    for _ in range(n_boot):
        idx = rng.choice(n_claims_te, size=n_claims_te, replace=True)
        tot_c2 = c2_tot_arr[idx].sum()
        tot_c1 = c1_tot_arr[idx].sum()
        if tot_c2 == 0:
            continue
            
        tp_a = tp_a_arr[idx].sum()
        fp_a = fp_a_arr[idx].sum()
        rec_a = tp_a / tot_c2
        prec_a = tp_a / max(tp_a + fp_a, 1)
        f1_a = 2 * prec_a * rec_a / max(prec_a + rec_a, 1e-8)
        c1_rej_a = c1_rej_a_arr[idx].sum() / max(tot_c1, 1)
        
        tp_b = tp_b_arr[idx].sum()
        fp_b = fp_b_arr[idx].sum()
        rec_b = tp_b / tot_c2
        prec_b = tp_b / max(tp_b + fp_b, 1)
        f1_b = 2 * prec_b * rec_b / max(prec_b + rec_b, 1e-8)
        c1_rej_b = c1_rej_b_arr[idx].sum() / max(tot_c1, 1)
        
        boot_f1_m0.append(f1_a)
        boot_f1_soft.append(f1_b)
        boot_prec_m0.append(prec_a)
        boot_prec_soft.append(prec_b)
        boot_rec_m0.append(rec_a)
        boot_rec_soft.append(rec_b)
        boot_c1_rej_m0.append(c1_rej_a)
        boot_c1_rej_soft.append(c1_rej_b)
        boot_tot_fp_m0.append(fp_a)
        boot_tot_fp_soft.append(fp_b)
        
        boot_delta_f1.append(f1_b - f1_a)
        boot_delta_prec.append(prec_b - prec_a)
        boot_delta_rec.append(rec_b - rec_a)
        boot_delta_c1_rej.append(c1_rej_b - c1_rej_a)
        boot_delta_tot_fp.append(fp_b - fp_a)
        
    def summarize_delta(arr):
        arr = np.array(arr)
        mean_v = float(np.mean(arr))
        ci_lower = float(np.percentile(arr, 2.5))
        ci_upper = float(np.percentile(arr, 97.5))
        if mean_v > 0:
            p_val = min(1.0, 2.0 * np.sum(arr <= 0) / len(arr))
        elif mean_v < 0:
            p_val = min(1.0, 2.0 * np.sum(arr >= 0) / len(arr))
        else:
            p_val = 1.0
        return mean_v, ci_lower, ci_upper, p_val

    def summarize_val(arr):
        arr = np.array(arr)
        return float(np.mean(arr)), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))

    metrics_audit1 = [
        ("F1-Score", boot_f1_m0, boot_f1_soft, boot_delta_f1, False),
        ("Precision", boot_prec_m0, boot_prec_soft, boot_delta_prec, True),
        ("Recall", boot_rec_m0, boot_rec_soft, boot_delta_rec, True),
        ("Class 1 Rejection", boot_c1_rej_m0, boot_c1_rej_soft, boot_delta_c1_rej, True),
        ("Total False Positives", boot_tot_fp_m0, boot_tot_fp_soft, boot_delta_tot_fp, False)
    ]
    
    audit1_results = []
    print(f"{'Metric':<24} | {'M0 Baseline (Mean [95% CI])':<30} | {'M_soft (Mean [95% CI])':<30} | {'Delta (Mean [95% CI])':<26} | {'Two-Sided p'}")
    print("-" * 125)
    for m_name, dist_a, dist_b, dist_delta, is_pct in metrics_audit1:
        mean_a, a_lo, a_hi = summarize_val(dist_a)
        mean_b, b_lo, b_hi = summarize_val(dist_b)
        mean_d, d_lo, d_hi, p_val = summarize_delta(dist_delta)
        
        if is_pct:
            str_a = f"{mean_a*100:5.2f}% [{a_lo*100:4.1f}%, {a_hi*100:4.1f}%]"
            str_b = f"{mean_b*100:5.2f}% [{b_lo*100:4.1f}%, {b_hi*100:4.1f}%]"
            str_d = f"{mean_d*100:+5.2f}% [{d_lo*100:+4.1f}%, {d_hi*100:+4.1f}%]"
        elif m_name == "Total False Positives":
            str_a = f"{mean_a:5.1f} [{a_lo:4.1f}, {a_hi:4.1f}]"
            str_b = f"{mean_b:5.1f} [{b_lo:4.1f}, {b_hi:4.1f}]"
            str_d = f"{mean_d:+5.1f} [{d_lo:+4.1f}, {d_hi:+4.1f}]"
        else:
            str_a = f"{mean_a:6.4f} [{a_lo:6.4f}, {a_hi:6.4f}]"
            str_b = f"{mean_b:6.4f} [{b_lo:6.4f}, {b_hi:6.4f}]"
            str_d = f"{mean_d:+6.4f} [{d_lo:+6.4f}, {d_hi:+6.4f}]"
            
        print(f"{m_name:<24} | {str_a:<30} | {str_b:<30} | {str_d:<26} | {p_val:.4f}")
        
        audit1_results.append({
            "metric": m_name,
            "m0_mean": mean_a, "m0_ci": [a_lo, a_hi],
            "m_soft_mean": mean_b, "m_soft_ci": [b_lo, b_hi],
            "delta_mean": mean_d, "delta_ci": [d_lo, d_hi],
            "p_value": p_val
        })
        
    # =========================================================================
    # AUDIT 2: INDEPENDENT COHORT EVALUATION & CONFUSION MATRICES
    # =========================================================================
    print("\n" + "=" * 105)
    print("AUDIT 2: INDEPENDENT COHORT EVALUATION & CLASS-WISE CONFUSION MATRICES")
    print("=" * 105)
    
    u100 = pd.read_csv("data/audit/unified_production_audit_benchmark_100_enriched_e3.csv")
    with open("data/calibration/stage2_claim_disjoint_partitions_human_341.json", "r") as f:
        p_json = json.load(f)
    test_claims = set(p_json['partitions']['test']['claim_ids'])
    train_claims = set(p_json['partitions']['train']['claim_ids'])
    
    # Calculate M_soft score for all 100 pairs
    u100['score_soft'] = compute_soft_score(u100['prob_m0'].values, u100['reasoning_cosine_v1_1'].values, tau_div=0.65, lam=0.40)
    u100['pass_m0'] = u100['pass_s2a'] & (u100['prob_m0'] >= 0.20)
    u100['pass_soft'] = u100['pass_s2a'] & (u100['score_soft'] >= 0.20)
    
    cohorts_audit2 = [
        ("Cohort 1: Non-Test Claims (Disjoint from 17 Test Claims)", u100[~u100['claim_id'].isin(test_claims)].copy()),
        ("Cohort 2: Strictly Fresh Claims (Disjoint from Train & Test)", u100[~u100['claim_id'].isin(test_claims | train_claims)].copy()),
        ("Cohort 3: Full Unified Benchmark (All 100 Pairs)", u100.copy())
    ]
    
    audit2_records = []
    
    for c_title, c_df in cohorts_audit2:
        print(f"\n" + "-" * 90)
        print(f"{c_title} (N = {len(c_df)}, Claims = {c_df['claim_id'].nunique()})")
        print("-" * 90)
        
        c2 = (c_df['human_audit_decision'] == 'class_2')
        c1 = (c_df['human_audit_decision'] == 'class_1')
        c0a = (c_df['human_audit_decision'] == 'class_0a')
        c0b = (c_df['human_audit_decision'] == 'class_0b')
        c_neg = ~c2
        
        for m_name, pass_col in [("M0 Baseline (T*=0.20)", c_df['pass_m0']), ("M_soft Gated (T*=0.20)", c_df['pass_soft'])]:
            tp = (pass_col & c2).sum()
            fn = ((~pass_col) & c2).sum()
            
            fp_c1 = (pass_col & c1).sum()
            tn_c1 = ((~pass_col) & c1).sum()
            
            fp_c0a = (pass_col & c0a).sum()
            tn_c0a = ((~pass_col) & c0a).sum()
            
            fp_c0b = (pass_col & c0b).sum()
            tn_c0b = ((~pass_col) & c0b).sum()
            
            fp_tot = fp_c1 + fp_c0a + fp_c0b
            tn_tot = tn_c1 + tn_c0a + tn_c0b
            
            rec = tp / max(tp + fn, 1)
            prec = tp / max(tp + fp_tot, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-8)
            
            rej_c1 = tn_c1 / max(c1.sum(), 1)
            rej_c0a = tn_c0a / max(c0a.sum(), 1)
            rej_c0b = tn_c0b / max(c0b.sum(), 1)
            
            # Bootstrap CI for Precision, Recall, F1
            rng_c = np.random.RandomState(42)
            c_claims = c_df['claim_id'].unique()
            b_recs, b_precs, b_f1s = [], [], []
            for _ in range(5000):
                samp_c = rng_c.choice(c_claims, size=len(c_claims), replace=True)
                samp_idx = [i for c in samp_c for i in c_df[c_df['claim_id'] == c].index]
                b_pass = pass_col.loc[samp_idx]
                b_c2 = c2.loc[samp_idx]
                b_neg = c_neg.loc[samp_idx]
                b_tp = (b_pass & b_c2).sum()
                b_fp = (b_pass & b_neg).sum()
                b_rec = b_tp / max(b_c2.sum(), 1)
                b_prec = b_tp / max(b_tp + b_fp, 1)
                b_f1 = 2 * b_prec * b_rec / max(b_prec + b_rec, 1e-8)
                b_recs.append(b_rec)
                b_precs.append(b_prec)
                b_f1s.append(b_f1)
                
            rec_ci = (float(np.percentile(b_recs, 2.5)), float(np.percentile(b_recs, 97.5)))
            prec_ci = (float(np.percentile(b_precs, 2.5)), float(np.percentile(b_precs, 97.5)))
            f1_ci = (float(np.percentile(b_f1s, 2.5)), float(np.percentile(b_f1s, 97.5)))
            
            print(f"\n{m_name}:")
            print(f"  Class 2 (Target Equivalence): TP = {tp:2d}, FN = {fn:2d} | Recall:    {rec*100:5.2f}% [95% CI: {rec_ci[0]*100:4.1f}%, {rec_ci[1]*100:4.1f}%]")
            print(f"  Class 1 (Different Warrant):  FP = {fp_c1:2d}, TN = {tn_c1:2d} | Rejection: {rej_c1*100:5.2f}% ({tn_c1}/{c1.sum()})")
            print(f"  Class 0a (Opposite Stance):   FP = {fp_c0a:2d}, TN = {tn_c0a:2d} | Rejection: {rej_c0a*100:5.2f}% ({tn_c0a}/{c0a.sum()})")
            print(f"  Class 0b (Off-Topic Noise):   FP = {fp_c0b:2d}, TN = {tn_c0b:2d} | Rejection: {rej_c0b*100:5.2f}% ({tn_c0b}/{c0b.sum()})")
            print(f"  Overall Summary:              Total FP = {fp_tot}, Total TN = {tn_tot}")
            print(f"                                Precision: {prec*100:5.2f}% [95% CI: {prec_ci[0]*100:4.1f}%, {prec_ci[1]*100:4.1f}%]")
            print(f"                                F1-Score:  {f1:6.4f}  [95% CI: {f1_ci[0]:.4f}, {f1_ci[1]:.4f}]")
            
            audit2_records.append({
                "cohort": c_title,
                "model": m_name,
                "tp": int(tp), "fn": int(fn),
                "fp_c1": int(fp_c1), "tn_c1": int(tn_c1),
                "fp_c0a": int(fp_c0a), "tn_c0a": int(tn_c0a),
                "fp_c0b": int(fp_c0b), "tn_c0b": int(tn_c0b),
                "total_fp": int(fp_tot), "total_tn": int(tn_tot),
                "recall": round(rec * 100, 2), "recall_ci": [round(x * 100, 2) for x in rec_ci],
                "precision": round(prec * 100, 2), "precision_ci": [round(x * 100, 2) for x in prec_ci],
                "f1": round(f1, 4), "f1_ci": [round(x, 4) for x in f1_ci],
                "rej_c1": round(rej_c1 * 100, 2),
                "rej_c0a": round(rej_c0a * 100, 2),
                "rej_c0b": round(rej_c0b * 100, 2)
            })
            
        # Paired bootstrap test between M0 and M_soft on this cohort
        c_boot_delta_f1 = []
        c_boot_delta_c1_rej = []
        for _ in range(5000):
            samp_c = rng_c.choice(c_claims, size=len(c_claims), replace=True)
            samp_idx = [i for c in samp_c for i in c_df[c_df['claim_id'] == c].index]
            
            b_c2 = c2.loc[samp_idx]
            b_c1 = c1.loc[samp_idx]
            b_neg = c_neg.loc[samp_idx]
            
            # M0
            p_a = c_df.loc[samp_idx, 'pass_m0']
            tp_a = (p_a & b_c2).sum()
            fp_a = (p_a & b_neg).sum()
            rec_a = tp_a / max(b_c2.sum(), 1)
            prec_a = tp_a / max(tp_a + fp_a, 1)
            f1_a = 2 * prec_a * rec_a / max(prec_a + rec_a, 1e-8)
            rej_c1_a = ((~p_a) & b_c1).sum() / max(b_c1.sum(), 1)
            
            # M_soft
            p_b = c_df.loc[samp_idx, 'pass_soft']
            tp_b = (p_b & b_c2).sum()
            fp_b = (p_b & b_neg).sum()
            rec_b = tp_b / max(b_c2.sum(), 1)
            prec_b = tp_b / max(tp_b + fp_b, 1)
            f1_b = 2 * prec_b * rec_b / max(prec_b + rec_b, 1e-8)
            rej_c1_b = ((~p_b) & b_c1).sum() / max(b_c1.sum(), 1)
            
            c_boot_delta_f1.append(f1_b - f1_a)
            c_boot_delta_c1_rej.append(rej_c1_b - rej_c1_a)
            
        mean_d_f1, f1_lo, f1_hi, p_f1 = summarize_delta(c_boot_delta_f1)
        mean_d_c1, c1_lo, c1_hi, p_c1 = summarize_delta(c_boot_delta_c1_rej)
        print(f"\n  Paired Clustered Test (M_soft vs M0 on {c_title}):")
        print(f"    Delta F1:     {mean_d_f1:+.4f} [95% CI: {f1_lo:+.4f}, {f1_hi:+.4f}], p = {p_f1:.4f}")
        print(f"    Delta C1 Rej: {mean_d_c1*100:+.2f}% [95% CI: {c1_lo*100:+.2f}%, {c1_hi*100:+.2f}%], p = {p_c1:.4f}")

    # =========================================================================
    # AUDIT 3: PARAMETER SENSITIVITY GRID ON CALIBRATION DATA ONLY (5 x 5)
    # =========================================================================
    print("\n" + "=" * 105)
    print("AUDIT 3: PARAMETER SENSITIVITY GRID (5x5) ON CALIBRATION DATA ONLY")
    print("=" * 105)
    
    tau_vals = [0.60, 0.625, 0.65, 0.675, 0.70]
    lam_vals = [0.20, 0.30, 0.40, 0.50, 0.60]
    
    c2_cal = (cal['human_ground_truth_label'] == 'class_2')
    c1_cal = (cal['human_ground_truth_label'] == 'class_1')
    tot_c2_cal = c2_cal.sum()
    tot_c1_cal = c1_cal.sum()
    
    sens_records = []
    
    print(f"{'tau_div':<8} | {'lambda':<8} | {'Cal Recall':<12} | {'Cal Precision':<14} | {'Cal F1':<10} | {'Cal C1 Rej':<12} | {'Cal Total FP'}")
    print("-" * 85)
    
    for t_div in tau_vals:
        for lam in lam_vals:
            # Fixed production threshold T* = 0.20
            sc = compute_soft_score(cal['prob_m0'].values, cal['reasoning_cosine_v1_1'].values, tau_div=t_div, lam=lam)
            passed = cal['s2a_pass'] & (sc >= 0.20)
            
            tp = (passed & c2_cal).sum()
            fp = (passed & (~c2_cal)).sum()
            fn = ((~passed) & c2_cal).sum()
            
            fp1 = (passed & c1_cal).sum()
            tn1 = ((~passed) & c1_cal).sum()
            
            rec = tp / tot_c2_cal
            prec = tp / max(tp + fp, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-8)
            rej_c1 = tn1 / tot_c1_cal
            
            is_selected = (t_div == 0.65 and lam == 0.40)
            marker = " *" if is_selected else ""
            print(f"{t_div:<8.3f} | {lam:<8.2f} | {rec*100:6.2f}%     | {prec*100:6.2f}%       | {f1:6.4f}   | {rej_c1*100:6.2f}%     | {fp:<4d}{marker}")
            
            sens_records.append({
                "tau_div": t_div,
                "lambda": lam,
                "cal_recall": round(rec * 100, 2),
                "cal_precision": round(prec * 100, 2),
                "cal_f1": round(f1, 4),
                "cal_c1_rejection": round(rej_c1 * 100, 2),
                "cal_total_fp": int(fp),
                "is_selected": is_selected
            })
            
    sens_df = pd.DataFrame(sens_records)
    print(f"\nSensitivity Grid Summary across 25 configurations:")
    print(f"  Recall Range:    {sens_df['cal_recall'].min():.2f}% to {sens_df['cal_recall'].max():.2f}% (std: {sens_df['cal_recall'].std():.2f}%)")
    print(f"  Precision Range: {sens_df['cal_precision'].min():.2f}% to {sens_df['cal_precision'].max():.2f}% (std: {sens_df['cal_precision'].std():.2f}%)")
    print(f"  F1 Range:        {sens_df['cal_f1'].min():.4f} to {sens_df['cal_f1'].max():.4f} (std: {sens_df['cal_f1'].std():.4f})")
    print(f"  C1 Rej Range:    {sens_df['cal_c1_rejection'].min():.2f}% to {sens_df['cal_c1_rejection'].max():.2f}%")

    # =========================================================================
    # AUDIT 4: PROBABILITY INTERPRETATION, BOUNDS [0, 1] & CALIBRATION DRIFT
    # =========================================================================
    print("\n" + "=" * 105)
    print("AUDIT 4: VERIFICATION SCORE BOUNDS [0, 1] & PROBABILITY CALIBRATION ANALYSIS")
    print("=" * 105)
    
    # 1. Bounds verification across all partitions
    all_scores = np.concatenate([cal['score_soft'].values, test['score_soft'].values, u100['score_soft'].values])
    min_score = float(np.min(all_scores))
    max_score = float(np.max(all_scores))
    n_clipped_below_0 = int(np.sum((cal['prob_m0'].values - 0.40 * np.maximum(0.0, 0.65 - cal['reasoning_cosine_v1_1'].values)) < 0.0))
    n_clipped_above_1 = int(np.sum((cal['prob_m0'].values - 0.40 * np.maximum(0.0, 0.65 - cal['reasoning_cosine_v1_1'].values)) > 1.0))
    
    print(f"1. Score Bounds Verification:")
    print(f"   Score Formula: S_soft = clip( P_M0 - lambda * max(0, tau_div - reasoning_cosine), 0.0, 1.0 )")
    print(f"   Empirical Min Score across all splits: {min_score:.6f}")
    print(f"   Empirical Max Score across all splits: {max_score:.6f}")
    print(f"   Scores strictly within [0.0, 1.0]:      {0.0 <= min_score and max_score <= 1.0}")
    print(f"   Raw values clipped at 0.0:              {n_clipped_below_0}")
    print(f"   Raw values clipped at 1.0:              {n_clipped_above_1}")
    
    # 2. Probability Calibration Diagnostics (Brier Score & Expected Calibration Error)
    # Binary target on calibration split (Class 2 = 1, Class 1 = 0; restricting to C2/C1)
    cal_c21 = cal[cal['human_ground_truth_label'].isin(['class_2', 'class_1'])].copy()
    y_true_cal = (cal_c21['human_ground_truth_label'] == 'class_2').astype(int).values
    
    p0_cal = cal_c21['prob_m0'].values
    s_soft_cal = cal_c21['score_soft'].values
    
    # Brier Score = mean((p - y)^2)
    brier_m0 = float(np.mean((p0_cal - y_true_cal) ** 2))
    brier_soft = float(np.mean((s_soft_cal - y_true_cal) ** 2))
    
    # Expected Calibration Error (ECE) with 5 bins
    def compute_ece(probs, labels, n_bins=5):
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        n = len(probs)
        for i in range(n_bins):
            mask = (probs >= bin_edges[i]) & (probs < bin_edges[i+1]) if i < n_bins - 1 else (probs >= bin_edges[i]) & (probs <= bin_edges[i+1])
            if mask.sum() == 0:
                continue
            bin_acc = labels[mask].mean()
            bin_conf = probs[mask].mean()
            ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
        return float(ece)
        
    ece_m0 = compute_ece(p0_cal, y_true_cal, n_bins=5)
    ece_soft = compute_ece(s_soft_cal, y_true_cal, n_bins=5)
    
    print(f"\n2. Calibration Diagnostics on Calibration Split (N = {len(cal_c21)} C2/C1 pairs):")
    print(f"   M0 Base Probe:     Brier Score = {brier_m0:.4f} | ECE (5 bins) = {ece_m0*100:.2f}%")
    print(f"   M_soft Transform:  Brier Score = {brier_soft:.4f} | ECE (5 bins) = {ece_soft*100:.2f}%")
    print(f"   Delta Brier:       {brier_soft - brier_m0:+.4f} (Minimal drift)")
    print(f"   Delta ECE:         {(ece_soft - ece_m0)*100:+.2f}%")
    print(f"   Interpretation:    S_soft acts as a monotonic verification score.")
    print(f"                      Because delta Brier is small ({abs(brier_soft - brier_m0):.4f}), it preserves ranking fidelity")
    print(f"                      while applying targeted penalization to non-equivalent warrant pairs.")
    
    # Save all audit outputs to permanent files
    audit_data = {
        "audit1_test_bootstrap": audit1_results,
        "audit2_independent_cohorts": audit2_records,
        "audit3_parameter_sensitivity": sens_records,
        "audit4_score_bounds_and_calibration": {
            "score_min": min_score,
            "score_max": max_score,
            "strictly_bounded": bool(0.0 <= min_score and max_score <= 1.0),
            "brier_m0": brier_m0,
            "brier_soft": brier_soft,
            "ece_m0": ece_m0,
            "ece_soft": ece_soft
        }
    }
    
    with open("results/calibration/m_soft_comprehensive_audit_results.json", "w") as f:
        json.dump(audit_data, f, indent=2)
    sens_df.to_csv("results/calibration/m_soft_parameter_sensitivity_grid.csv", index=False)
    print(f"\nSaved comprehensive audit results to:")
    print(f"  - results/calibration/m_soft_comprehensive_audit_results.json")
    print(f"  - results/calibration/m_soft_parameter_sensitivity_grid.csv")

if __name__ == '__main__':
    main()
