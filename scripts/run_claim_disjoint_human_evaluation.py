"""
Experiment 3: New Claim-Disjoint Human Evaluation
Compares 4 frozen models:
  1. M0 Baseline (Frozen T* = 0.20)
  2. M2 Reasoning (Frozen T* = 0.18, No Guard)
  3. M2 Reasoning (Frozen T* = 0.31, No Guard)
  4. M2 + Calibrated Length Guard (Rule B, L* = 14, Frozen T* = 0.18)
Evaluated across:
  - Cohort 1: Non-Test Claims (76 pairs across 34 independent claims)
  - Cohort 2: Strictly Fresh Claims (26 pairs across 14 claims outside train & test)
  - Cohort 3: Full Unified Human Audit Benchmark (100 pairs across 45 claims)
"""

import json
import numpy as np
import pandas as pd
from scipy import stats

def wilson_ci(k, n, conf=0.95):
    if n == 0:
        return (0.0, 0.0)
    z = stats.norm.ppf(1 - (1 - conf) / 2)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, center - half), min(1.0, center + half))

def fast_paired_bootstrap_test(df_cohort, col_a, col_b, n_boot=10000, seed=42):
    """Fast precomputed claim-clustered paired bootstrap test."""
    rng = np.random.RandomState(seed)
    
    claims = []
    tps_a, fps_a, fns_a = [], [], []
    tps_b, fps_b, fns_b = [], [], []
    c2_tots, c1_tots = [], []
    c1_rejs_a, c1_rejs_b = [], []
    
    for cid, grp in df_cohort.groupby('claim_id'):
        claims.append(cid)
        c2 = (grp['human_audit_decision'] == 'class_2').values
        c1 = (grp['human_audit_decision'] == 'class_1').values
        c_neg = ~c2
        
        pass_a = grp[col_a].values
        pass_b = grp[col_b].values
        
        c2_tots.append(c2.sum())
        c1_tots.append(c1.sum())
        
        tps_a.append((pass_a & c2).sum())
        fps_a.append((pass_a & c_neg).sum())
        fns_a.append(((~pass_a) & c2).sum())
        c1_rejs_a.append(((~pass_a) & c1).sum())
        
        tps_b.append((pass_b & c2).sum())
        fps_b.append((pass_b & c_neg).sum())
        fns_b.append(((~pass_b) & c2).sum())
        c1_rejs_b.append(((~pass_b) & c1).sum())
        
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
    print("EXPERIMENT 3: NEW CLAIM-DISJOINT HUMAN EVALUATION (FOUR FROZEN CONFIGURATIONS)")
    print("=" * 105)
    
    df = pd.read_csv("data/audit/unified_production_audit_benchmark_100_enriched_e3.csv")
    df['gold_word_count'] = df['gold_perspective_text'].apply(lambda x: len(str(x).strip().split()))
    
    # Assign frozen model decisions
    df['score_m2_guard'] = df.apply(lambda r: r['prob_m0'] if r['gold_word_count'] <= 14 else r['prob_m2'], axis=1)
    df['pass_m2_guard_018'] = df['pass_s2a'] & (df['score_m2_guard'] >= 0.18)
    
    models = [
        ("M0 Baseline (T*=0.20)", "pass_m0_020"),
        ("M2 Reasoning (T*=0.18, No Guard)", "pass_m2_018"),
        ("M2 Reasoning (T*=0.31, No Guard)", "pass_m2_031"),
        ("M2 + Length Guard (L*=14, T*=0.18)", "pass_m2_guard_018")
    ]
    
    with open("data/calibration/stage2_claim_disjoint_partitions_human_341.json", "r") as f:
        p_json = json.load(f)
    test_claims = set(p_json['partitions']['test']['claim_ids'])
    train_claims = set(p_json['partitions']['train']['claim_ids'])
    
    cohorts = [
        ("Cohort 1: Non-Test Claims (Disjoint from 17 Test Claims)", df[~df['claim_id'].isin(test_claims)].copy()),
        ("Cohort 2: Strictly Fresh Claims (Disjoint from Train & Test Claims)", df[~df['claim_id'].isin(test_claims | train_claims)].copy()),
        ("Cohort 3: Full Unified Human Audit Benchmark (All 100 Pairs)", df.copy())
    ]
    
    all_eval_records = []
    
    for c_name, c_df in cohorts:
        print(f"\n=========================================================================================")
        print(f"{c_name}")
        print(f"Total Pairs: {len(c_df)} across {c_df['claim_id'].nunique()} claims")
        print(f"Ground Truth Distribution: {c_df['human_audit_decision'].value_counts().to_dict()}")
        print("-" * 105)
        
        c2 = (c_df['human_audit_decision'] == 'class_2')
        c1 = (c_df['human_audit_decision'] == 'class_1')
        c0a = (c_df['human_audit_decision'] == 'class_0a')
        c0b = (c_df['human_audit_decision'] == 'class_0b')
        c_neg = ~c2
        tot_c2 = c2.sum()
        tot_c1 = c1.sum()
        tot_c0a = c0a.sum()
        tot_c0b = c0b.sum()
        
        for m_label, col in models:
            passed = c_df[col]
            tp = (passed & c2).sum()
            fp = (passed & c_neg).sum()
            fn = ((~passed) & c2).sum()
            tn = ((~passed) & c_neg).sum()
            
            fp1 = (passed & c1).sum()
            fp0a = (passed & c0a).sum()
            fp0b = (passed & c0b).sum()
            
            rec = tp / max(tot_c2, 1)
            prec = tp / max(tp + fp, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-8)
            rej_c1 = (tot_c1 - fp1) / max(tot_c1, 1)
            rej_c0a = (tot_c0a - fp0a) / max(tot_c0a, 1)
            rej_c0b = (tot_c0b - fp0b) / max(tot_c0b, 1)
            
            prec_ci = wilson_ci(tp, tp + fp)
            rec_ci = wilson_ci(tp, tot_c2)
            
            print(f"{m_label:36s} | Rec: {rec*100:5.1f}% ({tp}/{tot_c2}) | Prec: {prec*100:5.1f}% [95% CI: {prec_ci[0]*100:4.1f}-{prec_ci[1]*100:4.1f}%] | F1: {f1:.4f} | Rej C1: {rej_c1*100:5.1f}% ({tot_c1-fp1}/{tot_c1}) | Rej C0a: {rej_c0a*100:5.1f}% | Tot FP: {fp} (C1={fp1}, C0a={fp0a}, C0b={fp0b})")
            
            all_eval_records.append({
                "cohort": c_name,
                "model": m_label,
                "n_pairs": len(c_df),
                "n_claims": c_df['claim_id'].nunique(),
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
            
        print("\n--- Paired Claim-Clustered Bootstrap Tests vs M0 Baseline (10,000 Resamples) ---")
        for m_label, col in models[1:]:
            boot_res = fast_paired_bootstrap_test(c_df, "pass_m0_020", col, n_boot=10000, seed=42)
            print(f"{m_label} vs M0:")
            print(f"  Delta F1:     {boot_res['mean_delta_f1']:+.4f} (95% CI: [{boot_res['ci_delta_f1'][0]:+.4f}, {boot_res['ci_delta_f1'][1]:+.4f}], p = {boot_res['p_val_f1']:.4f})")
            print(f"  Delta C1 Rej: {boot_res['mean_delta_c1_rej']*100:+.2f}% (95% CI: [{boot_res['ci_delta_c1_rej'][0]*100:+.2f}%, {boot_res['ci_delta_c1_rej'][1]*100:+.2f}%], p = {boot_res['p_val_c1_rej']:.4f})")
            
    out_csv = "results/calibration/new_claim_disjoint_human_evaluation_results.csv"
    pd.DataFrame(all_eval_records).to_csv(out_csv, index=False)
    print(f"\nSaved claim-disjoint human evaluation results to: {out_csv}")

if __name__ == '__main__':
    main()
