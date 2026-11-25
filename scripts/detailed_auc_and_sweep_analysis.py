import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

df = pd.read_csv("results/calibration/e3_pairwise_features_and_predictions_341.csv")

test_df = df[df['partition'] == 'test'].copy()
test_c21 = test_df[test_df['human_ground_truth_label'].isin(['class_2', 'class_1'])].copy()

y_c21 = (test_c21['human_ground_truth_label'] == 'class_2').astype(int).values

models = ["M0_baseline", "M1_claim", "M2_reasoning", "M3_warrant", "M4_all_three"]

print("=" * 80)
print("DISCRIMINATIVE POWER ON HELD-OUT TEST CLAIMS (CLASS 2 VS CLASS 1, N=53)")
print("=" * 80)

auc_records = []
for m in models:
    probs = test_c21[f"prob_{m}"].values
    auc = roc_auc_score(y_c21, probs)
    ap = average_precision_score(y_c21, probs)
    auc_records.append({
        "Model": m,
        "ROC_AUC (C2 vs C1)": round(auc, 4),
        "PR_AUC (Average Prec)": round(ap, 4)
    })

print(pd.DataFrame(auc_records).to_string(index=False))

print("\n" + "=" * 80)
print("THRESHOLD SWEEP ON FULL CASCADE (STAGE 1 + STAGE 2a + PROBE) ON TEST SPLIT (N=88)")
print("=" * 80)

# Stance filter
is_contra = (
    ((test_df['stance_gold'] == 'support') & (test_df['stance_generated'] == 'undermine')) |
    ((test_df['stance_gold'] == 'undermine') & (test_df['stance_generated'] == 'support'))
)
s1_pass = test_df['cosine_similarity'] >= 0.55
s2a_pass = s1_pass & (~is_contra)

c2_mask = (test_df['human_ground_truth_label'] == 'class_2')
c1_mask = (test_df['human_ground_truth_label'] == 'class_1')

for m in models:
    print(f"\n--- {m} ---")
    prob_col = f"prob_{m}"
    
    # Check operating point where Recall == 23/24 (95.83%)
    rec_c23_rows = []
    for t in np.arange(0.05, 0.85, 0.01):
        t = round(float(t), 2)
        passed = s2a_pass & (test_df[prob_col] >= t)
        tp = (passed & c2_mask).sum()
        fp = (passed & (~c2_mask)).sum()
        fp_c1 = (passed & c1_mask).sum()
        
        rec = tp / 24.0
        prec = tp / max(tp + fp, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        rej_c1 = (29 - fp_c1) / 29.0
        
        if tp == 23: # exactly matching M0's recall of 23/24
            rec_c23_rows.append((t, prec, f1, rej_c1, fp_c1))
            
    if rec_c23_rows:
        best_match = max(rec_c23_rows, key=lambda x: x[1])
        print(f"  At Matched Recall = 23/24 (95.83%): Max Threshold T={best_match[0]:.2f} -> Prec={best_match[1]:.2%}, F1={best_match[2]:.4f}, C1 Rejection={best_match[3]:.2%} (C1 FP={best_match[4]})")
    else:
        print("  No operating point with exactly 23/24 recall found.")
