import pandas as pd
import numpy as np

df = pd.read_csv('results/calibration/e3_pairwise_features_and_predictions_341.csv')
test_df = df[df['partition'] == 'test'].copy()

# Pipeline filters
is_contra = (
    ((test_df['stance_gold'] == 'support') & (test_df['stance_generated'] == 'undermine')) |
    ((test_df['stance_gold'] == 'undermine') & (test_df['stance_generated'] == 'support'))
)
s1_pass = test_df['cosine_similarity'] >= 0.55
s2a_pass = s1_pass & (~is_contra)

c2 = (test_df['human_ground_truth_label'] == 'class_2')
c1 = (test_df['human_ground_truth_label'] == 'class_1')
c0a = (test_df['human_ground_truth_label'] == 'class_0a')
c0b = (test_df['human_ground_truth_label'] == 'class_0b')
non_c2 = (~c2)

print("=" * 90)
print("AUDIT 5: FULL FOUR-CLASS VERIFIER EVALUATION ON HELD-OUT TEST (N=88)")
print("Composition: Class 2 = 24, Class 1 = 29, Class 0a = 26, Class 0b = 9")
print("=" * 90)

models_to_test = [
    ("M0 (Frozen Baseline)", 'prob_M0_baseline', 0.20),
    ("M2 (Cal F1-Opt)", 'prob_M2_reasoning', 0.31),
    ("M2 (Cal Recall>=90%)", 'prob_M2_reasoning', 0.18),
    ("M4 (Cal F1-Opt)", 'prob_M4_all_three', 0.11),
]

for name, prob_col, t_val in models_to_test:
    passed = s2a_pass & (test_df[prob_col] >= t_val)
    
    tp = (passed & c2).sum()
    fn = ((~passed) & c2).sum()
    
    fp_c1 = (passed & c1).sum()
    fp_c0a = (passed & c0a).sum()
    fp_c0b = (passed & c0b).sum()
    fp_tot = fp_c1 + fp_c0a + fp_c0b
    
    tn_c1 = ((~passed) & c1).sum()
    tn_c0a = ((~passed) & c0a).sum()
    tn_c0b = ((~passed) & c0b).sum()
    tn_tot = ((~passed) & non_c2).sum()
    
    rec_c2 = tp / 24.0
    prec_c2 = tp / max(tp + fp_tot, 1)
    f1_c2 = 2 * prec_c2 * rec_c2 / max(prec_c2 + rec_c2, 1e-8)
    
    # Specificity across all noise
    spec_all = tn_tot / 64.0
    # Overall 4-class binary accuracy (Class 2 vs Non-Class 2)
    acc_binary = (tp + tn_tot) / 88.0
    
    print(f"\n--- {name} (Threshold T* = {t_val:.2f}) ---")
    print(f"Confusion Matrix: TP = {tp:2d}, FN = {fn:2d} | FP = {fp_tot:2d} (C1={fp_c1:2d}, C0a={fp_c0a:2d}, C0b={fp_c0b:2d}) | TN = {tn_tot:2d}")
    print(f"Key Metrics:")
    print(f"  Class 2 Recall:    {rec_c2*100:5.2f}% ({tp}/24)")
    print(f"  Class 2 Precision: {prec_c2*100:5.2f}% ({tp}/{tp+fp_tot})")
    print(f"  Class 2 F1-Score:  {f1_c2:6.4f}")
    print(f"  All-Noise Specificity: {spec_all*100:5.2f}% ({tn_tot}/64)")
    print(f"  Overall Binary Acc:    {acc_binary*100:5.2f}% ({tp+tn_tot}/88)")
    print(f"Class-Specific Rejection Rates:")
    print(f"  Class 1 Rejection (Nuance):   {tn_c1/29*100:5.2f}% ({tn_c1}/29)")
    print(f"  Class 0a Rejection (Stance):  {tn_c0a/26*100:5.2f}% ({tn_c0a}/26)")
    print(f"  Class 0b Rejection (Topic):   {tn_c0b/9*100:5.2f}% ({tn_c0b}/9)")
