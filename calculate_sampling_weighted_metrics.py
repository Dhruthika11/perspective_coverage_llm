import json
import numpy as np
import pandas as pd

print("=" * 85)
print("PRIORITY 4: SAMPLING-WEIGHTED PRODUCTION VERIFIER METRICS")
print("=" * 85)

# 1. Total Population Counts from results/production_verifier_predictions_250.csv
df_pop = pd.read_csv('results/production_verifier_predictions_250.csv')
pop_counts = df_pop['final_decision'].value_counts()

N_total = len(df_pop)
N_h = {
    'ACCEPT_CLASS2_EQUIVALENT': int(pop_counts.get('ACCEPT_CLASS2_EQUIVALENT', 0)),
    'REJECT_STAGE2B_DIFFERENT_WARRANT': int(pop_counts.get('REJECT_STAGE2B_DIFFERENT_WARRANT', 0)),
    'REJECT_STAGE2A_STANCE': int(pop_counts.get('REJECT_STAGE2A_STANCE', 0)),
    'REJECT_STAGE1': int(pop_counts.get('REJECT_STAGE1', 0))
}

# 2. Audit Sample Counts from data/audit/production_verifier_audit_sample_50.csv
df_sample = pd.read_csv('data/audit/production_verifier_audit_sample_50.csv')
sample_counts = df_sample['final_decision'].value_counts()

n_h = {k: int(sample_counts.get(k, 0)) for k in N_h}

# 3. Survey Sampling Weights W_h = N_h / n_h
W_h = {k: N_h[k] / n_h[k] for k in N_h}

print(f"Total Production Candidates: N = {N_total}")
print(f"Audit Sample Size:          n = {len(df_sample)}")
print("\nStrata Breakdown and Sampling Weights:")
print(f"{'Stratum':<35} | {'Pop N_h':<8} | {'Sample n_h':<10} | {'Sample Frac f_h':<15} | {'Weight W_h':<10}")
print("-" * 85)
for k in N_h:
    f_h = n_h[k] / N_h[k]
    print(f"{k:<35} | {N_h[k]:<8d} | {n_h[k]:<10d} | {f_h*100:<14.2f}% | {W_h[k]:<10.4f}")

# 4. Human Audit Distribution by Stratum
crosstab = pd.crosstab(df_sample['final_decision'], df_sample['human_audit_decision'])
classes = ['class_2', 'class_1', 'class_0a', 'class_0b']

# 5. Horvitz-Thompson Population Totals: \hat{N}_c = \sum_h W_h * n_{h, c}
pop_class_estimates = {}
for c in classes:
    tot_est = 0.0
    for strat in N_h:
        sub_cnt = int(((df_sample['final_decision'] == strat) & (df_sample['human_audit_decision'] == c)).sum())
        tot_est += W_h[strat] * sub_cnt
    pop_class_estimates[c] = tot_est

print("\n" + "=" * 85)
print("HORVITZ-THOMPSON ESTIMATES OF PRODUCTION POPULATION DISTRIBUTION (N = 250)")
print("=" * 85)
for c in classes:
    est = pop_class_estimates[c]
    print(f"Estimated Total {c:10s} in Population: {est:6.2f} / {N_total} ({est/N_total*100:.2f}%)")

# 6. Sampling-Weighted Precision, Recall, and Rejection Metrics
# True Positives in Population = W_1 * n_{1, class_2}
pop_tp = W_h['ACCEPT_CLASS2_EQUIVALENT'] * ((df_sample['final_decision'] == 'ACCEPT_CLASS2_EQUIVALENT') & (df_sample['human_audit_decision'] == 'class_2')).sum()
pop_accepted = N_h['ACCEPT_CLASS2_EQUIVALENT']

weighted_prec = pop_tp / pop_accepted
weighted_rec = pop_tp / pop_class_estimates['class_2']
weighted_f1 = 2 * weighted_prec * weighted_rec / (weighted_prec + weighted_rec)

# Negative Rejections
c1_pop_rej = W_h['REJECT_STAGE2B_DIFFERENT_WARRANT'] * ((df_sample['final_decision'] == 'REJECT_STAGE2B_DIFFERENT_WARRANT') & (df_sample['human_audit_decision'] == 'class_1')).sum()
weighted_c1_rej = c1_pop_rej / pop_class_estimates['class_1']

c0a_pop_rej = (
    W_h['REJECT_STAGE2B_DIFFERENT_WARRANT'] * ((df_sample['final_decision'] == 'REJECT_STAGE2B_DIFFERENT_WARRANT') & (df_sample['human_audit_decision'] == 'class_0a')).sum() +
    W_h['REJECT_STAGE2A_STANCE'] * ((df_sample['final_decision'] == 'REJECT_STAGE2A_STANCE') & (df_sample['human_audit_decision'] == 'class_0a')).sum()
)
weighted_c0a_rej = c0a_pop_rej / pop_class_estimates['class_0a']

c0b_pop_rej = (
    W_h['REJECT_STAGE2B_DIFFERENT_WARRANT'] * ((df_sample['final_decision'] == 'REJECT_STAGE2B_DIFFERENT_WARRANT') & (df_sample['human_audit_decision'] == 'class_0b')).sum() +
    W_h['REJECT_STAGE1'] * ((df_sample['final_decision'] == 'REJECT_STAGE1') & (df_sample['human_audit_decision'] == 'class_0b')).sum()
)
weighted_c0b_rej = c0b_pop_rej / pop_class_estimates['class_0b']

# Overall Specificity
neg_pop_total = pop_class_estimates['class_1'] + pop_class_estimates['class_0a'] + pop_class_estimates['class_0b']
neg_pop_rej = c1_pop_rej + c0a_pop_rej + c0b_pop_rej
weighted_specificity = neg_pop_rej / neg_pop_total

print("\n" + "=" * 85)
print("SAMPLING-WEIGHTED PRODUCTION METRICS (POPULATION ESTIMATES)")
print("=" * 85)
print(f"Class 2 Precision (Weighted): {weighted_prec*100:.2f}% (Unweighted Sample: 85.00%)")
print(f"Class 2 Recall (Weighted):    {weighted_rec*100:.2f}% (Unweighted Sample: 80.95%)")
print(f"Class 2 F1-Score (Weighted):  {weighted_f1:.4f}  (Unweighted Sample: 0.8293)")
print(f"Class 1 Rejection (Weighted): {weighted_c1_rej*100:.2f}%")
print(f"Class 0a Rejection (Weighted):{weighted_c0a_rej*100:.2f}%")
print(f"Class 0b Rejection (Weighted):{weighted_c0b_rej*100:.2f}%")
print(f"Negative Specificity (Weight):{weighted_specificity*100:.2f}% (Unweighted Sample: 89.66%)")

# Export to JSON
weighted_report = {
    "population_size": N_total,
    "sample_size": len(df_sample),
    "strata_weights": W_h,
    "estimated_population_class_distribution": pop_class_estimates,
    "weighted_metrics": {
        "class_2_precision": round(float(weighted_prec), 4),
        "class_2_recall": round(float(weighted_rec), 4),
        "class_2_f1": round(float(weighted_f1), 4),
        "class_1_rejection_e2e": round(float(weighted_c1_rej), 4),
        "class_0a_rejection_e2e": round(float(weighted_c0a_rej), 4),
        "class_0b_rejection_e2e": round(float(weighted_c0b_rej), 4),
        "negative_class_specificity": round(float(weighted_specificity), 4)
    }
}

with open('results/calibration/sampling_weighted_production_metrics.json', 'w') as f:
    json.dump(weighted_report, f, indent=2)
print("\nSaved sampling-weighted production metrics to results/calibration/sampling_weighted_production_metrics.json")
