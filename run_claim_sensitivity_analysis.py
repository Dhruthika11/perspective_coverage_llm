import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# 1. Load data
bench = pd.read_csv('data/calibration/human_llm_benchmark_1700.csv')
df = pd.read_csv('results/calibration/stage2_cross_encoder_predictions_341.csv')
merged = pd.merge(df, bench[['pair_id', 'stance_gold', 'stance_generated']], on='pair_id', how='left')

# Train probe on Train partition (C2 vs C1)
train_c21 = merged[(merged['partition'] == 'train') & (merged['human_ground_truth_label'].isin(['class_2', 'class_1']))].copy()
feats = ['cosine_similarity', 'p_entail', 'p_neutral']
scaler = StandardScaler()
X_tr = scaler.fit_transform(train_c21[feats].values)
y_tr = (train_c21['human_ground_truth_label'] == 'class_2').astype(int).values

clf = LogisticRegression(C=1.0, random_state=42)
clf.fit(X_tr, y_tr)

# Predict probe on all
X_all = scaler.transform(merged[feats].values)
merged['probe_prob'] = clf.predict_proba(X_all)[:, 1]

# Stance filter
is_contra = (
    ((merged['stance_gold'] == 'support') & (merged['stance_generated'] == 'undermine')) |
    ((merged['stance_gold'] == 'undermine') & (merged['stance_generated'] == 'support'))
)
merged['pass_stance'] = ~is_contra
merged['pass_s1'] = merged['cosine_similarity'] >= 0.55

# 1. Calibration Partition Claim-Level LOOCV (Leave-One-Claim-Out)
cal_df = merged[merged['partition'] == 'calibration'].copy()
cal_c21 = cal_df[cal_df['human_ground_truth_label'].isin(['class_2', 'class_1'])].copy()
cal_claims = cal_c21['claim_id'].unique()

print("=" * 85)
print(f"PART 1: LEAVE-ONE-CLAIM-OUT CROSS-VALIDATION ON CALIBRATION (16 CLAIMS)")
print("=" * 85)

loocv_thresholds = []
for out_c in cal_claims:
    in_folds = cal_c21[cal_c21['claim_id'] != out_c]
    
    best_f1 = -1
    best_th = 0.20
    for th in np.arange(0.15, 0.50, 0.05):
        pred = (in_folds['probe_prob'] >= th).astype(int)
        y_true = (in_folds['human_ground_truth_label'] == 'class_2').astype(int)
        tp = np.sum((pred == 1) & (y_true == 1))
        fp = np.sum((pred == 1) & (y_true == 0))
        fn = np.sum((pred == 0) & (y_true == 1))
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f1 = 2 * p * r / max(p + r, 1e-8)
        if f1 > best_f1:
            best_f1 = f1
            best_th = round(float(th), 2)
    loocv_thresholds.append(best_th)

loocv_thresholds = np.array(loocv_thresholds)
print(f"Optimal Thresholds selected across 16 LOOCV folds: {loocv_thresholds.tolist()}")
print(f"Mean Selected Threshold: {np.mean(loocv_thresholds):.3f} +/- {np.std(loocv_thresholds):.3f}")
print(f"Median Selected Threshold: {np.median(loocv_thresholds):.2f}")
print(f"Threshold Stability: {np.sum(loocv_thresholds == 0.20)} / 16 folds selected exactly T* = 0.20")

# 2. Test Set Claim-Level Breakdown (17 Claims)
test_df = merged[merged['partition'] == 'test'].copy()
test_claims = test_df['claim_id'].unique()

print("\n" + "=" * 85)
print(f"PART 2: CLAIM-LEVEL SENSITIVITY BREAKDOWN ON TEST PARTITION (17 CLAIMS)")
print("=" * 85)

claim_stats = []
for cid in test_claims:
    c_df = test_df[test_df['claim_id'] == cid]
    c_text = c_df['claim_text'].iloc[0]
    
    tot_c2 = (c_df['human_ground_truth_label'] == 'class_2').sum()
    tot_c1 = (c_df['human_ground_truth_label'] == 'class_1').sum()
    tot_c0a = (c_df['human_ground_truth_label'] == 'class_0a').sum()
    tot_c0b = (c_df['human_ground_truth_label'] == 'class_0b').sum()
    
    # Cascade at T = 0.20
    s1_pass = c_df[c_df['pass_s1']]
    s2a_pass = s1_pass[s1_pass['pass_stance']]
    s2b_pass_20 = s2a_pass[s2a_pass['probe_prob'] >= 0.20]
    s2b_pass_35 = s2a_pass[s2a_pass['probe_prob'] >= 0.35]
    
    c2_acc_20 = (s2b_pass_20['human_ground_truth_label'] == 'class_2').sum()
    c2_acc_35 = (s2b_pass_35['human_ground_truth_label'] == 'class_2').sum()
    
    c1_acc_20 = (s2b_pass_20['human_ground_truth_label'] == 'class_1').sum()
    c1_acc_35 = (s2b_pass_35['human_ground_truth_label'] == 'class_1').sum()
    
    claim_stats.append({
        'claim_id': cid,
        'claim_text': c_text[:40] + '...',
        'n_pairs': len(c_df),
        'c2_total': tot_c2,
        'c2_rec_20': f"{c2_acc_20}/{tot_c2}" if tot_c2 > 0 else "N/A",
        'c2_rec_35': f"{c2_acc_35}/{tot_c2}" if tot_c2 > 0 else "N/A",
        'c1_rej_20': f"{tot_c1 - c1_acc_20}/{tot_c1}" if tot_c1 > 0 else "N/A",
        'c1_rej_35': f"{tot_c1 - c1_acc_35}/{tot_c1}" if tot_c1 > 0 else "N/A"
    })

claim_df = pd.DataFrame(claim_stats)
print(claim_df.to_string(index=False))

# Sensitivity across thresholds on test partition:
print("\n" + "=" * 85)
print(f"PART 3: THRESHOLD SENSITIVITY CURVE ON TEST PARTITION (T in [0.15, 0.45])")
print("=" * 85)
print(f"{'Threshold':<10} | {'C2 Rec (Pair)':<15} | {'Claims w/ 100% C2 Rec':<25} | {'C2 Prec':<10} | {'C2 F1':<10} | {'C1 Rej':<10}")
print("-" * 85)

for th in np.arange(0.15, 0.50, 0.05):
    th = round(float(th), 2)
    s1 = test_df[test_df['pass_s1']]
    s2a = s1[s1['pass_stance']]
    s2b = s2a[s2a['probe_prob'] >= th]
    
    tp = (s2b['human_ground_truth_label'] == 'class_2').sum()
    fp1 = (s2b['human_ground_truth_label'] == 'class_1').sum()
    tot_acc = len(s2b)
    
    rec = tp / 24
    prec = tp / tot_acc if tot_acc > 0 else 0
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    rej1 = (29 - fp1) / 29
    
    # count claims with 100% C2 recall
    perfect_c2_claims = 0
    c2_claims_count = 0
    for cid in test_claims:
        c_sub = test_df[test_df['claim_id'] == cid]
        n_c2 = (c_sub['human_ground_truth_label'] == 'class_2').sum()
        if n_c2 > 0:
            c2_claims_count += 1
            s2b_c = s2b[s2b['claim_id'] == cid]
            tp_c = (s2b_c['human_ground_truth_label'] == 'class_2').sum()
            if tp_c == n_c2:
                perfect_c2_claims += 1
                
    print(f"{th:<10.2f} | {tp}/24 ({rec*100:.1f}%)   | {perfect_c2_claims}/{c2_claims_count} claims ({perfect_c2_claims/c2_claims_count*100:.1f}%)     | {prec*100:.2f}%    | {f1:.4f}    | {rej1*100:.1f}%")
