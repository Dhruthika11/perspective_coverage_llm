import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# 1. Load data
bench = pd.read_csv('data/calibration/human_llm_benchmark_1700.csv')
df = pd.read_csv('results/calibration/stage2_cross_encoder_predictions_341.csv')
merged = pd.merge(df, bench[['pair_id', 'stance_gold', 'stance_generated']], on='pair_id', how='left')

# 2. Train Probe on train partition
train_c21 = merged[(merged['partition'] == 'train') & (merged['human_ground_truth_label'].isin(['class_2', 'class_1']))].copy()
feats = ['cosine_similarity', 'p_entail', 'p_neutral']
scaler = StandardScaler()
X_tr = scaler.fit_transform(train_c21[feats].values)
y_tr = (train_c21['human_ground_truth_label'] == 'class_2').astype(int).values

clf = LogisticRegression(C=1.0, random_state=42)
clf.fit(X_tr, y_tr)

# 3. Apply to test set (17 claims, 88 pairs)
test_df = merged[merged['partition'] == 'test'].copy()
X_te = scaler.transform(test_df[feats].values)
test_df['probe_prob'] = clf.predict_proba(X_te)[:, 1]

# Stance filter
is_contra = (
    ((test_df['stance_gold'] == 'support') & (test_df['stance_generated'] == 'undermine')) |
    ((test_df['stance_gold'] == 'undermine') & (test_df['stance_generated'] == 'support'))
)
test_df['pass_stance'] = ~is_contra
test_df['pass_s1'] = test_df['cosine_similarity'] >= 0.55

# Claim-clustered bootstrap function
def run_cascade_on_data(d, t_probe):
    # S1 filter
    s1 = d[d['pass_s1']]
    # S2a filter
    s2a = s1[s1['pass_stance']]
    # S2b filter
    s2b = s2a[s2a['probe_prob'] >= t_probe]
    
    tot_c2 = (d['human_ground_truth_label'] == 'class_2').sum()
    tot_c1 = (d['human_ground_truth_label'] == 'class_1').sum()
    tot_c0a = (d['human_ground_truth_label'] == 'class_0a').sum()
    tot_c0b = (d['human_ground_truth_label'] == 'class_0b').sum()
    
    tp = (s2b['human_ground_truth_label'] == 'class_2').sum()
    fp_1 = (s2b['human_ground_truth_label'] == 'class_1').sum()
    fp_0a = (s2b['human_ground_truth_label'] == 'class_0a').sum()
    fp_0b = (s2b['human_ground_truth_label'] == 'class_0b').sum()
    tot_acc = len(s2b)
    
    rec = tp / tot_c2 if tot_c2 > 0 else 0.0
    prec = tp / tot_acc if tot_acc > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    
    rej_c1_e2e = (tot_c1 - fp_1) / tot_c1 if tot_c1 > 0 else 0.0
    rej_c0a_e2e = (tot_c0a - fp_0a) / tot_c0a if tot_c0a > 0 else 0.0
    rej_c0b_e2e = (tot_c0b - fp_0b) / tot_c0b if tot_c0b > 0 else 0.0
    
    return {
        'rec': rec, 'prec': prec, 'f1': f1,
        'rej_c1': rej_c1_e2e, 'rej_c0a': rej_c0a_e2e, 'rej_c0b': rej_c0b_e2e
    }

# 2,000 Claim-Clustered Bootstrap
claims = test_df['claim_id'].unique()
np.random.seed(42)
n_resamples = 2000

print(f"Running {n_resamples} Claim-Clustered Bootstrap Resamples across {len(claims)} Test Claims...")

for t_probe, name in [(0.20, "T* = 0.20 (F1-Optimal)"), (0.35, "T* = 0.35 (High Precision)")]:
    point_est = run_cascade_on_data(test_df, t_probe)
    
    boot_rec, boot_prec, boot_f1 = [], [], []
    boot_c1_rej, boot_c0a_rej, boot_c0b_rej = [], [], []
    
    for _ in range(n_resamples):
        sampled_claims = np.random.choice(claims, size=len(claims), replace=True)
        # assemble sampled dataframe
        sampled_dfs = [test_df[test_df['claim_id'] == c] for c in sampled_claims]
        boot_df = pd.concat(sampled_dfs, ignore_index=True)
        
        m = run_cascade_on_data(boot_df, t_probe)
        boot_rec.append(m['rec'])
        boot_prec.append(m['prec'])
        boot_f1.append(m['f1'])
        boot_c1_rej.append(m['rej_c1'])
        boot_c0a_rej.append(m['rej_c0a'])
        boot_c0b_rej.append(m['rej_c0b'])
        
    print("\n" + "=" * 80)
    print(f"CLAIM-CLUSTERED BOOTSTRAP RESULTS FOR CASCADE AT {name}")
    print("=" * 80)
    print(f"Class 2 Recall:    Point = {point_est['rec']*100:.2f}% | Clustered 95% CI: [{np.percentile(boot_rec, 2.5)*100:.2f}%, {np.percentile(boot_rec, 97.5)*100:.2f}%]")
    print(f"Class 2 Precision: Point = {point_est['prec']*100:.2f}% | Clustered 95% CI: [{np.percentile(boot_prec, 2.5)*100:.2f}%, {np.percentile(boot_prec, 97.5)*100:.2f}%]")
    print(f"Class 2 F1-Score:  Point = {point_est['f1']:.4f}   | Clustered 95% CI: [{np.percentile(boot_f1, 2.5):.4f}, {np.percentile(boot_f1, 97.5):.4f}]")
    print(f"Class 1 Rej (E2E): Point = {point_est['rej_c1']*100:.2f}% | Clustered 95% CI: [{np.percentile(boot_c1_rej, 2.5)*100:.2f}%, {np.percentile(boot_c1_rej, 97.5)*100:.2f}%]")
    print(f"Class 0a Rej (E2E):Point = {point_est['rej_c0a']*100:.2f}% | Clustered 95% CI: [{np.percentile(boot_c0a_rej, 2.5)*100:.2f}%, {np.percentile(boot_c0a_rej, 97.5)*100:.2f}%]")
    print(f"Class 0b Rej (E2E):Point = {point_est['rej_c0b']*100:.2f}% | Clustered 95% CI: [{np.percentile(boot_c0b_rej, 2.5)*100:.2f}%, {np.percentile(boot_c0b_rej, 97.5)*100:.2f}%]")
