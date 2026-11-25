import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.proportion import proportion_confint

# 1. Load Data
bench = pd.read_csv('data/calibration/human_llm_benchmark_1700.csv')
df = pd.read_csv('results/calibration/stage2_cross_encoder_predictions_341.csv')

# Merge Stances
merged = pd.merge(df, bench[['pair_id', 'stance_gold', 'stance_generated']], on='pair_id', how='left')

# 2. Train Probe strictly on Train partition (C2 vs C1, 32 claims, 104 pairs)
train_c21 = merged[(merged['partition'] == 'train') & (merged['human_ground_truth_label'].isin(['class_2', 'class_1']))].copy()
feats = ['cosine_similarity', 'p_entail', 'p_neutral']
scaler = StandardScaler()
X_tr = scaler.fit_transform(train_c21[feats].values)
y_tr = (train_c21['human_ground_truth_label'] == 'class_2').astype(int).values

clf = LogisticRegression(C=1.0, random_state=42)
clf.fit(X_tr, y_tr)

# 3. Apply Probe across the entire dataset
X_all = scaler.transform(merged[feats].values)
merged['probe_prob'] = clf.predict_proba(X_all)[:, 1]

# 4. Filters Definition
# Stance concordance with uncertainty preservation:
# Reject ONLY when explicit opposite stance
is_contra = (
    ((merged['stance_gold'] == 'support') & (merged['stance_generated'] == 'undermine')) |
    ((merged['stance_gold'] == 'undermine') & (merged['stance_generated'] == 'support'))
)
merged['pass_stance'] = ~is_contra
merged['pass_s1'] = merged['cosine_similarity'] >= 0.55

# Isolate Test Partition (17 claims, 88 pairs)
test_df = merged[merged['partition'] == 'test'].copy()

print("=" * 100)
print("EXPERIMENT C1: FULL CASCADED PIPELINE EVALUATION ON HELD-OUT TEST CLAIMS (N = 88)")
print("=" * 100)

total_c2 = (test_df['human_ground_truth_label'] == 'class_2').sum() # 24
total_c1 = (test_df['human_ground_truth_label'] == 'class_1').sum() # 29
total_c0a = (test_df['human_ground_truth_label'] == 'class_0a').sum() # 26
total_c0b = (test_df['human_ground_truth_label'] == 'class_0b').sum() # 9

# Stage 1
s1_surv = test_df[test_df['pass_s1']]
# Stage 2a
s2a_surv = s1_surv[s1_surv['pass_stance']]

def eval_cascade(s2b_condition, name):
    s2b_surv = s2a_surv[s2b_condition(s2a_surv)]
    
    tp = (s2b_surv['human_ground_truth_label'] == 'class_2').sum()
    fp_1 = (s2b_surv['human_ground_truth_label'] == 'class_1').sum()
    fp_0a = (s2b_surv['human_ground_truth_label'] == 'class_0a').sum()
    fp_0b = (s2b_surv['human_ground_truth_label'] == 'class_0b').sum()
    total_acc = len(s2b_surv)
    
    prec = tp / total_acc if total_acc > 0 else 0.0
    rec = tp / total_c2
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    
    # Rejections
    # Stage 2 Conditional Rejection R_2,c = (s1_surv_c - final_c) / s1_surv_c
    r2_c2 = ((s1_surv['human_ground_truth_label'] == 'class_2').sum() - tp) / (s1_surv['human_ground_truth_label'] == 'class_2').sum()
    r2_c1 = ((s1_surv['human_ground_truth_label'] == 'class_1').sum() - fp_1) / (s1_surv['human_ground_truth_label'] == 'class_1').sum()
    r2_c0a = ((s1_surv['human_ground_truth_label'] == 'class_0a').sum() - fp_0a) / (s1_surv['human_ground_truth_label'] == 'class_0a').sum()
    r2_c0b = ((s1_surv['human_ground_truth_label'] == 'class_0b').sum() - fp_0b) / (s1_surv['human_ground_truth_label'] == 'class_0b').sum()
    
    # E2E Rejection R_e2e,c = (total_c - final_c) / total_c
    e2e_c2 = (total_c2 - tp) / total_c2
    e2e_c1 = (total_c1 - fp_1) / total_c1
    e2e_c0a = (total_c0a - fp_0a) / total_c0a
    e2e_c0b = (total_c0b - fp_0b) / total_c0b
    
    w_prec_l, w_prec_h = proportion_confint(tp, total_acc, alpha=0.05, method='wilson') if total_acc > 0 else (0,0)
    w_rec_l, w_rec_h = proportion_confint(tp, total_c2, alpha=0.05, method='wilson')
    
    return {
        'name': name,
        'acc': total_acc,
        'tp': tp,
        'fp_c1': fp_1,
        'fp_c0a': fp_0a,
        'fp_c0b': fp_0b,
        'prec': prec,
        'prec_ci': (w_prec_l, w_prec_h),
        'rec': rec,
        'rec_ci': (w_rec_l, w_rec_h),
        'f1': f1,
        'r2_cond': {'c2': r2_c2, 'c1': r2_c1, 'c0a': r2_c0a, 'c0b': r2_c0b},
        'r_e2e': {'c2': e2e_c2, 'c1': e2e_c1, 'c0a': e2e_c0a, 'c0b': e2e_c0b}
    }

results = []
# 1. Baseline NLI without stance, theta=0.80
b0_surv = s1_surv[s1_surv['p_entail'] >= 0.80]
tp_b0 = (b0_surv['human_ground_truth_label'] == 'class_2').sum()
fp1_b0 = (b0_surv['human_ground_truth_label'] == 'class_1').sum()
fp0a_b0 = (b0_surv['human_ground_truth_label'] == 'class_0a').sum()
fp0b_b0 = (b0_surv['human_ground_truth_label'] == 'class_0b').sum()
tot_b0 = len(b0_surv)
prec_b0 = tp_b0 / tot_b0
rec_b0 = tp_b0 / total_c2
f1_b0 = 2 * prec_b0 * rec_b0 / (prec_b0 + rec_b0)

# Baseline NLI WITH stance, theta=0.80
results.append(eval_cascade(lambda d: d['p_entail'] >= 0.80, "S1 + S2a + NLI Baseline (theta=0.80)"))
# Probe calibrated at T* = 0.20 (F1-optimal on Calibration)
results.append(eval_cascade(lambda d: d['probe_prob'] >= 0.20, "S1 + S2a + Probe T*=0.20 (Cal F1-Opt)"))
# Probe at T* = 0.35 (Calibrated Balanced / Precision point)
results.append(eval_cascade(lambda d: d['probe_prob'] >= 0.35, "S1 + S2a + Probe T*=0.35 (High Precision)"))

print(f"\n--- 1. Baseline Cascade: Stage 1 (tau=0.55) + NLI Verifier (theta=0.80, NO Stance Filter) ---")
print(f"Accepted: {tot_b0}/88 ({tot_b0/88:.1%}) | TP (C2): {tp_b0}/24 | FP (C1): {fp1_b0}/29 | FP (C0a): {fp0a_b0}/26 | FP (C0b): {fp0b_b0}/9")
print(f"Class 2 Precision: {prec_b0:.2%} | Class 2 Recall: {rec_b0:.2%} | Class 2 F1: {f1_b0:.4f}")

print("\n--- 2. Cascaded Architectures (Stage 1 + Stage 2a Stance + Stage 2b Verifier) ---")
for r in results:
    print(f"\nConfiguration: {r['name']}")
    print(f"  Accepted Candidates: {r['acc']}/88 ({r['acc']/88:.1%})")
    print(f"  Class 2 (TP):        {r['tp']}/24 (Recall: {r['rec']:.2%}, 95% Wilson: [{r['rec_ci'][0]:.1%}, {r['rec_ci'][1]:.1%}])")
    print(f"  Class 2 Precision:   {r['prec']:.2%} (95% Wilson: [{r['prec_ci'][0]:.1%}, {r['prec_ci'][1]:.1%}])")
    print(f"  Class 2 F1-Score:    {r['f1']:.4f}")
    print(f"  Stage 2 Conditional Rejection R_2,c:")
    print(f"    Class 2:   {r['r2_cond']['c2']:.2%}")
    print(f"    Class 1:   {r['r2_cond']['c1']:.2%}")
    print(f"    Class 0a:  {r['r2_cond']['c0a']:.2%}")
    print(f"    Class 0b:  {r['r2_cond']['c0b']:.2%}")
    print(f"  End-to-End Rejection R_e2e,c:")
    print(f"    Class 2:   {r['r_e2e']['c2']:.2%} (Retention: {1 - r['r_e2e']['c2']:.2%})")
    print(f"    Class 1:   {r['r_e2e']['c1']:.2%}")
    print(f"    Class 0a:  {r['r_e2e']['c0a']:.2%}")
    print(f"    Class 0b:  {r['r_e2e']['c0b']:.2%}")

# Export full report JSON
with open('results/calibration/experiment_c1_cascade_report.json', 'w') as f:
    json.dump({
        'baseline_no_stance': {
            'accepted': int(tot_b0), 'tp': int(tp_b0), 'fp_c1': int(fp1_b0),
            'fp_c0a': int(fp0a_b0), 'fp_c0b': int(fp0b_b0),
            'prec': float(prec_b0), 'rec': float(rec_b0), 'f1': float(f1_b0)
        },
        'cascades': [
            {
                'name': r['name'], 'accepted': int(r['acc']), 'tp': int(r['tp']),
                'fp_c1': int(r['fp_c1']), 'fp_c0a': int(r['fp_c0a']), 'fp_c0b': int(r['fp_c0b']),
                'prec': float(r['prec']), 'prec_ci': [float(r['prec_ci'][0]), float(r['prec_ci'][1])],
                'rec': float(r['rec']), 'rec_ci': [float(r['rec_ci'][0]), float(r['rec_ci'][1])],
                'f1': float(r['f1']),
                'r2_cond': r['r2_cond'],
                'r_e2e': r['r_e2e']
            }
            for r in results
        ]
    }, f, indent=2)

print("\nSaved full cascade metrics to results/calibration/experiment_c1_cascade_report.json")
