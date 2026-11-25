"""
Reconcile Production Estimates:
Reproduces Baseline M0 and Experimental M2 projections from the same frozen 250-pair population,
using both:
  1. Exact 7-Substratum Horvitz-Thompson Estimator (accounting for UNKNOWN stance oversampling).
  2. Coarse 4-Stratum Horvitz-Thompson Estimator.
  3. Batch 1 Pure Stratified Random Sample (n = 50).
"""

import json
import numpy as np
import pandas as pd

def reconcile():
    print("=" * 90)
    print("EXPERIMENT 1: RECONCILE PRODUCTION ESTIMATES ON FROZEN 250 POPULATION")
    print("=" * 90)
    
    # 1. Load frozen population predictions (N = 250)
    df_pop = pd.read_csv('results/production_verifier_predictions_250.csv')
    df_pop['is_unknown'] = df_pop['stance_generated'].astype(str).str.lower().str.strip() == 'unknown'
    N_pop = len(df_pop)
    print(f"Loaded frozen production population: N = {N_pop}")
    
    # 2. Load enriched 100-pair human audit dataset
    df_u100 = pd.read_csv('data/audit/unified_production_audit_benchmark_100_enriched_e3.csv')
    df_u100['is_unknown'] = df_u100['stance_generated'].astype(str).str.lower().str.strip() == 'unknown'
    print(f"Loaded unified human audit sample: n = {len(df_u100)}")
    
    # 3. Define 7 Sub-Strata (Decision x UNKNOWN)
    sub_strata = [
        ('ACCEPT_CLASS2_EQUIVALENT', False, 'ACCEPT_KNOWN'),
        ('ACCEPT_CLASS2_EQUIVALENT', True,  'ACCEPT_UNKNOWN'),
        ('REJECT_STAGE2B_DIFFERENT_WARRANT', False, 'REJ_S2B_KNOWN'),
        ('REJECT_STAGE2B_DIFFERENT_WARRANT', True,  'REJ_S2B_UNKNOWN'),
        ('REJECT_STAGE2A_STANCE', False, 'REJ_S2A_KNOWN'),
        ('REJECT_STAGE1', False, 'REJ_S1_KNOWN'),
        ('REJECT_STAGE1', True,  'REJ_S1_UNKNOWN'),
    ]
    
    sub_weights = {}
    print("\n--- EXACT 7-SUBSTRATA INCLUSION PROBABILITIES & HORVITZ-THOMPSON WEIGHTS ---")
    print(f"{'Sub-Stratum':<20} | {'Pop N_h':<8} | {'Sample n_h':<10} | {'Incl Prob pi_h':<14} | {'Design Weight W_h'}")
    print("-" * 75)
    for dec, unk, name in sub_strata:
        N_h = len(df_pop[(df_pop['final_decision'] == dec) & (df_pop['is_unknown'] == unk)])
        n_h = len(df_u100[(df_u100['final_decision'] == dec) & (df_u100['is_unknown'] == unk)])
        pi_h = n_h / max(N_h, 1)
        w_h = N_h / max(n_h, 1)
        sub_weights[(dec, unk)] = w_h
        print(f"{name:<20} | {N_h:<8d} | {n_h:<10d} | {pi_h:<14.4f} | {w_h:.4f}")
        
    df_u100['weight_7sub'] = [sub_weights[(r['final_decision'], r['is_unknown'])] for _, r in df_u100.iterrows()]
    
    # 4. Define Coarse 4-Strata Weights
    coarse_strata = [
        'ACCEPT_CLASS2_EQUIVALENT',
        'REJECT_STAGE2B_DIFFERENT_WARRANT',
        'REJECT_STAGE2A_STANCE',
        'REJECT_STAGE1'
    ]
    coarse_weights = {}
    print("\n--- COARSE 4-STRATA INCLUSION PROBABILITIES & HORVITZ-THOMPSON WEIGHTS ---")
    print(f"{'Stratum':<35} | {'Pop N_h':<8} | {'Sample n_h':<10} | {'Incl Prob pi_h':<14} | {'Design Weight W_h'}")
    print("-" * 85)
    for dec in coarse_strata:
        N_h = len(df_pop[df_pop['final_decision'] == dec])
        n_h = len(df_u100[df_u100['final_decision'] == dec])
        pi_h = n_h / max(N_h, 1)
        w_h = N_h / max(n_h, 1)
        coarse_weights[dec] = w_h
        print(f"{dec:<35} | {N_h:<8d} | {n_h:<10d} | {pi_h:<14.4f} | {w_h:.4f}")
        
    df_u100['weight_4coarse'] = [coarse_weights[r['final_decision']] for _, r in df_u100.iterrows()]
    
    # 5. Helper function to compute Horvitz-Thompson population metrics
    def compute_ht_metrics(sample_df, weight_col, pred_col, model_name, est_type):
        pop_tp = 0.0
        pop_fp = 0.0
        pop_fn = 0.0
        pop_tn = 0.0
        
        pop_fp_c1 = 0.0
        pop_fp_c0a = 0.0
        pop_fp_c0b = 0.0
        
        pop_c2 = 0.0
        pop_c1 = 0.0
        pop_c0a = 0.0
        pop_c0b = 0.0
        
        for _, r in sample_df.iterrows():
            w = r[weight_col]
            lbl = r['human_audit_decision']
            pred = bool(r[pred_col])
            
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
                elif lbl == 'class_0b':
                    pop_c0b += w
                    if pred:
                        pop_fp_c0b += w
                if pred:
                    pop_fp += w
                else:
                    pop_tn += w
                    
        rec = pop_tp / max(pop_c2, 1e-8)
        prec = pop_tp / max(pop_tp + pop_fp, 1e-8)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        c1_rej = (pop_c1 - pop_fp_c1) / max(pop_c1, 1e-8)
        c0a_rej = (pop_c0a - pop_fp_c0a) / max(pop_c0a, 1e-8)
        
        return {
            "model": model_name,
            "estimator": est_type,
            "est_population_total_predicted": round(pop_tp + pop_fp, 1),
            "est_pop_tp": round(pop_tp, 1),
            "est_pop_fp": round(pop_fp, 1),
            "est_pop_fn": round(pop_fn, 1),
            "est_pop_tn": round(pop_tn, 1),
            "est_pop_c2": round(pop_c2, 1),
            "est_pop_c1": round(pop_c1, 1),
            "est_pop_c0a": round(pop_c0a, 1),
            "est_pop_c0b": round(pop_c0b, 1),
            "est_fp_c1": round(pop_fp_c1, 1),
            "est_fp_c0a": round(pop_fp_c0a, 1),
            "est_fp_c0b": round(pop_fp_c0b, 1),
            "weighted_recall": round(rec * 100, 2),
            "weighted_precision": round(prec * 100, 2),
            "weighted_f1": round(f1, 4),
            "weighted_c1_rejection": round(c1_rej * 100, 2),
            "weighted_c0a_rejection": round(c0a_rej * 100, 2)
        }
        
    # Models to evaluate
    models_to_test = [
        ("M0 Frozen Baseline (T*=0.20)", "pass_m0_020"),
        ("M2 Reasoning (T*=0.18)", "pass_m2_018"),
        ("M2 Reasoning (T*=0.31)", "pass_m2_031")
    ]
    
    results = []
    
    # 1. Exact 7-Substrata Horvitz-Thompson
    for m_name, col in models_to_test:
        res = compute_ht_metrics(df_u100, "weight_7sub", col, m_name, "7-Substrata Exact HT")
        results.append(res)
        
    # 2. Coarse 4-Strata Horvitz-Thompson
    for m_name, col in models_to_test:
        res = compute_ht_metrics(df_u100, "weight_4coarse", col, m_name, "4-Strata Coarse HT")
        results.append(res)
        
    # 3. Batch 1 Pure Stratified Sample (n = 50)
    b1_df = df_u100[df_u100['batch'] == 1].copy()
    b1_weights = {
        'ACCEPT_CLASS2_EQUIVALENT': 74 / 20,
        'REJECT_STAGE2B_DIFFERENT_WARRANT': 70 / 15,
        'REJECT_STAGE2A_STANCE': 56 / 10,
        'REJECT_STAGE1': 50 / 5
    }
    b1_df['weight_b1'] = [b1_weights[r['final_decision']] for _, r in b1_df.iterrows()]
    for m_name, col in models_to_test:
        res = compute_ht_metrics(b1_df, "weight_b1", col, m_name, "Batch 1 Pure Stratified (n=50)")
        results.append(res)
        
    res_df = pd.DataFrame(results)
    print("\n" + "=" * 110)
    print("RECONCILED PRODUCTION POPULATION ESTIMATES (N = 250)")
    print("=" * 110)
    cols_show = ['estimator', 'model', 'weighted_precision', 'weighted_recall', 'weighted_f1', 'weighted_c1_rejection', 'est_pop_fp', 'est_fp_c1', 'est_fp_c0a']
    print(res_df[cols_show].to_string(index=False))
    
    # Export reconciled results
    out_path = 'results/calibration/reconciled_production_estimates_comparison.csv'
    res_df.to_csv(out_path, index=False)
    print(f"\nSaved reconciled comparison table to: {out_path}")

if __name__ == '__main__':
    reconcile()
