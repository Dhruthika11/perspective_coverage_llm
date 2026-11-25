import json
import numpy as np
import pandas as pd
from scipy import stats

print("=" * 85)
print("UNIFIED 100-PAIR INDEPENDENT PRODUCTION AUDIT EVALUATION")
print("=" * 85)

# Load Batch 1 and Batch 2
df1 = pd.read_csv('data/audit/production_verifier_audit_sample_50.csv')
df2 = pd.read_csv('data/audit/expanded_production_audit_queue_batch2_50.csv')

df1['batch'] = 1
df2['batch'] = 2

# Verify columns match
cols = ['pair_id', 'claim_id', 'claim_text', 'gold_perspective_text', 'generated_viewpoint_text',
        'stance_gold', 'stance_generated', 'cosine_similarity', 'p_entail', 'p_neutral',
        'probe_prob', 'final_decision', 'human_audit_decision', 'human_audit_notes', 'batch']

combined = pd.concat([df1[cols], df2[cols]], ignore_index=True)
print(f"Total unified audited pairs: {len(combined)}")

# Check for any duplicates
dups = combined[combined.duplicated(subset=['pair_id'], keep=False)]
print(f"Duplicate pair_ids across batches: {len(dups)}")

out_path = 'data/audit/unified_production_audit_benchmark_100.csv'
combined.to_csv(out_path, index=False)
print(f"Saved unified 100-pair audit dataset to: {out_path}")

# Pipeline Decision mapping:
# ACCEPT_CLASS2_EQUIVALENT -> Predicted Positive (Class 2)
# All others -> Predicted Negative (Class 1 / Class 0a / Class 0b)
combined['predicted_positive'] = combined['final_decision'] == 'ACCEPT_CLASS2_EQUIVALENT'
combined['actual_positive'] = combined['human_audit_decision'] == 'class_2'

# 1. Unweighted Sample Metrics (N = 100)
tp = (combined['predicted_positive'] & combined['actual_positive']).sum()
fp = (combined['predicted_positive'] & ~combined['actual_positive']).sum()
fn = (~combined['predicted_positive'] & combined['actual_positive']).sum()
tn = (~combined['predicted_positive'] & ~combined['actual_positive']).sum()

prec = tp / max(tp + fp, 1)
rec = tp / max(tp + fn, 1)
f1 = 2 * prec * rec / max(prec + rec, 1e-8)
spec = tn / max(tn + fp, 1)

def wilson_ci(k, n, conf=0.95):
    if n == 0:
        return (0.0, 0.0)
    z = stats.norm.ppf(1 - (1 - conf) / 2)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, center - half), min(1.0, center + half))

prec_ci = wilson_ci(tp, tp + fp)
rec_ci = wilson_ci(tp, tp + fn)
spec_ci = wilson_ci(tn, tn + fp)

print(f"\n--- UNWEIGHTED SAMPLE METRICS (N = 100) ---")
print(f"Confusion Matrix: TP={tp}, FP={fp}, FN={fn}, TN={tn}")
print(f"  Class 2 Acceptance Precision: {prec*100:5.2f}% ({tp}/{tp+fp}) [95% Wilson CI: {prec_ci[0]*100:.2f}%, {prec_ci[1]*100:.2f}%]")
print(f"  Class 2 Recall:               {rec*100:5.2f}% ({tp}/{tp+fn}) [95% Wilson CI: {rec_ci[0]*100:.2f}%, {rec_ci[1]*100:.2f}%]")
print(f"  Class 2 F1-Score:             {f1:6.4f}")
print(f"  Negative Specificity:         {spec*100:5.2f}% ({tn}/{tn+fp}) [95% Wilson CI: {spec_ci[0]*100:.2f}%, {spec_ci[1]*100:.2f}%]")

# Error Breakdown of False Positives
fps = combined[combined['predicted_positive'] & ~combined['actual_positive']]
print(f"\nFalse Positive Composition (Total {len(fps)}):")
print(fps['human_audit_decision'].value_counts())

# Error Breakdown of False Negatives
fns = combined[~combined['predicted_positive'] & combined['actual_positive']]
print(f"\nFalse Negative Composition (Total {len(fns)}):")
print(fns['final_decision'].value_counts())

# 2. Population Sampling-Weighted Metrics (Horvitz-Thompson Estimator)
# Total production workload N = 250
# Decision strata counts in production:
# Stratum 1 (ACCEPT_CLASS2_EQUIVALENT): 74
# Stratum 2 (REJECT_STAGE2B_DIFFERENT_WARRANT): 70
# Stratum 3 (REJECT_STAGE2A_STANCE): 56
# Stratum 4 (REJECT_STAGE1): 50
prod_strata_sizes = {
    'ACCEPT_CLASS2_EQUIVALENT': 74,
    'REJECT_STAGE2B_DIFFERENT_WARRANT': 70,
    'REJECT_STAGE2A_STANCE': 56,
    'REJECT_STAGE1': 50
}

# Sample counts in combined 100
sample_strata_counts = combined['final_decision'].value_counts().to_dict()
print("\n--- STRATIFIED SAMPLING DESIGN (N_pop = 250, n_sample = 100) ---")
print(f"{'Decision Stratum':<35} | {'Pop Size (N_h)':<14} | {'Sample (n_h)':<14} | {'Sampling Weight (W_h)'}")
print("-" * 80)
strata_weights = {}
for dec, N_h in prod_strata_sizes.items():
    n_h = sample_strata_counts.get(dec, 0)
    w_h = N_h / max(n_h, 1)
    strata_weights[dec] = w_h
    print(f"{dec:<35} | {N_h:<14} | {n_h:<14} | {w_h:.4f}")

combined['stratum_weight'] = combined['final_decision'].map(strata_weights)

# Horvitz-Thompson estimates of population totals:
# Class counts
pop_class2_est = 0.0
pop_class1_est = 0.0
pop_class0a_est = 0.0
pop_class0b_est = 0.0

pop_tp_est = 0.0
pop_fp_est = 0.0
pop_fn_est = 0.0
pop_tn_est = 0.0

for idx, r in combined.iterrows():
    w = r['stratum_weight']
    lbl = r['human_audit_decision']
    pred = r['predicted_positive']
    
    if lbl == 'class_2':
        pop_class2_est += w
        if pred:
            pop_tp_est += w
        else:
            pop_fn_est += w
    else:
        if pred:
            pop_fp_est += w
        else:
            pop_tn_est += w
            
    if lbl == 'class_1':
        pop_class1_est += w
    elif lbl == 'class_0a':
        pop_class0a_est += w
    elif lbl == 'class_0b':
        pop_class0b_est += w

pop_prec_est = pop_tp_est / max(pop_tp_est + pop_fp_est, 1)
pop_rec_est = pop_tp_est / max(pop_tp_est + pop_fn_est, 1)
pop_f1_est = 2 * pop_prec_est * pop_rec_est / max(pop_prec_est + pop_rec_est, 1e-8)
pop_spec_est = pop_tn_est / max(pop_tn_est + pop_fp_est, 1)

print(f"\n--- POPULATION-WEIGHTED HORVITZ-THOMPSON ESTIMATES (N = 250) ---")
print(f"Estimated Population Class Totals:")
print(f"  Class 2 (Equivalent Reasoning):         {pop_class2_est:6.2f} ({pop_class2_est/250*100:.2f}%)")
print(f"  Class 1 (Different Reasoning, Same St): {pop_class1_est:6.2f} ({pop_class1_est/250*100:.2f}%)")
print(f"  Class 0a (Opposite Stance):             {pop_class0a_est:6.2f} ({pop_class0a_est/250*100:.2f}%)")
print(f"  Class 0b (Unrelated / Off-topic):       {pop_class0b_est:6.2f} ({pop_class0b_est/250*100:.2f}%)")
print(f"Estimated Population Confusion Totals:")
print(f"  TP: {pop_tp_est:6.2f}, FP: {pop_fp_est:6.2f}, FN: {pop_fn_est:6.2f}, TN: {pop_tn_est:6.2f}")
print(f"Estimated Production Verifier Performance:")
print(f"  Population Precision:  {pop_prec_est*100:5.2f}%")
print(f"  Population Recall:     {pop_rec_est*100:5.2f}%")
print(f"  Population F1-Score:   {pop_f1_est:6.4f}")
print(f"  Population Specificity:{pop_spec_est*100:5.2f}%")

# Save results to json
summary = {
    "total_audited_pairs": len(combined),
    "batch1_count": len(df1),
    "batch2_count": len(df2),
    "sample_metrics": {
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "precision": float(prec),
        "precision_ci_wilson_95": [float(prec_ci[0]), float(prec_ci[1])],
        "recall": float(rec),
        "recall_ci_wilson_95": [float(rec_ci[0]), float(rec_ci[1])],
        "f1": float(f1),
        "specificity": float(spec),
        "specificity_ci_wilson_95": [float(spec_ci[0]), float(spec_ci[1])]
    },
    "population_horvitz_thompson_estimates_n250": {
        "pop_tp": float(pop_tp_est),
        "pop_fp": float(pop_fp_est),
        "pop_fn": float(pop_fn_est),
        "pop_tn": float(pop_tn_est),
        "precision": float(pop_prec_est),
        "recall": float(pop_rec_est),
        "f1": float(pop_f1_est),
        "specificity": float(pop_spec_est),
        "class_totals": {
            "class_2": float(pop_class2_est),
            "class_1": float(pop_class1_est),
            "class_0a": float(pop_class0a_est),
            "class_0b": float(pop_class0b_est)
        }
    }
}

with open('results/calibration/unified_production_audit_100_report.json', 'w') as f:
    json.dump(summary, f, indent=2)
print("\nSaved unified report to: results/calibration/unified_production_audit_100_report.json")
