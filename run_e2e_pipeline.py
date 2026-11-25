import pandas as pd
import numpy as np

df = pd.read_csv('results/calibration/stage2_cross_encoder_predictions_341.csv')
test_df = df[df['partition'] == 'test'].copy()

print("=" * 85)
print("END-TO-END PIPELINE AUDIT ON HELD-OUT TEST CLAIMS (17 CLAIMS, 88 PAIRS)")
print("=" * 85)

# Stage 1: tau* = 0.55
tau_star = 0.55
s1_passed = test_df['cosine_similarity'] >= tau_star

tot_c2 = (test_df['human_ground_truth_label'] == 'class_2').sum()
tot_c1 = (test_df['human_ground_truth_label'] == 'class_1').sum()
tot_c0a = (test_df['human_ground_truth_label'] == 'class_0a').sum()
tot_c0b = (test_df['human_ground_truth_label'] == 'class_0b').sum()

print(f"Ground Truth in Test Split (N = {len(test_df)}):")
print(f"  Class 2 (Target Equivalence): {tot_c2}")
print(f"  Class 1 (Distinct Nuance):    {tot_c1}")
print(f"  Class 0a (Counter-Argument):  {tot_c0a}")
print(f"  Class 0b (Off-topic Noise):   {tot_c0b}")

print(f"\n--- 1. STAGE 1 ALONE (tau* = {tau_star}) ---")
s1_c2 = np.sum(s1_passed & (test_df['human_ground_truth_label'] == 'class_2'))
s1_c1 = np.sum(s1_passed & (test_df['human_ground_truth_label'] == 'class_1'))
s1_c0a = np.sum(s1_passed & (test_df['human_ground_truth_label'] == 'class_0a'))
s1_c0b = np.sum(s1_passed & (test_df['human_ground_truth_label'] == 'class_0b'))

print(f"Candidate Recall (Class 2): {s1_c2}/{tot_c2} = {s1_c2/tot_c2*100:.2f}%")
print(f"Class 0b Noise Rejection:   {(tot_c0b - s1_c0b)}/{tot_c0b} = {(tot_c0b - s1_c0b)/tot_c0b*100:.2f}%")
print(f"Class 0a Stance Rejection:  {(tot_c0a - s1_c0a)}/{tot_c0a} = {(tot_c0a - s1_c0a)/tot_c0a*100:.2f}%")
print(f"Class 1 Nuance Rejection:   {(tot_c1 - s1_c1)}/{tot_c1} = {(tot_c1 - s1_c1)/tot_c1*100:.2f}%")
print(f"Stage 1 Candidate Pass Rate:{s1_passed.sum()}/{len(test_df)} = {s1_passed.mean()*100:.2f}%")

print(f"\n--- 2. STAGE 2 & CASCADED END-TO-END PIPELINE ---")
for theta in [0.70, 0.75, 0.80, 0.85, 0.90]:
    s2_passed = (test_df['p_entail'] >= theta)
    e2e_passed = s1_passed & s2_passed
    
    e2e_c2 = np.sum(e2e_passed & (test_df['human_ground_truth_label'] == 'class_2'))
    e2e_c1 = np.sum(e2e_passed & (test_df['human_ground_truth_label'] == 'class_1'))
    e2e_c0a = np.sum(e2e_passed & (test_df['human_ground_truth_label'] == 'class_0a'))
    e2e_c0b = np.sum(e2e_passed & (test_df['human_ground_truth_label'] == 'class_0b'))
    
    total_accepted = e2e_passed.sum()
    prec_c2 = e2e_c2 / max(total_accepted, 1)
    rec_c2 = e2e_c2 / tot_c2
    f1_c2 = 2 * prec_c2 * rec_c2 / max(prec_c2 + rec_c2, 1e-8)
    
    rej_c0b = (tot_c0b - e2e_c0b) / tot_c0b
    rej_c0a = (tot_c0a - e2e_c0a) / tot_c0a
    rej_c1 = (tot_c1 - e2e_c1) / tot_c1
    
    print(f"\nTheta = {theta:.2f}:")
    print(f"  Stage 2 on S1 Survivors: Accepted {e2e_passed.sum()}/{s1_passed.sum()} ({e2e_passed.sum()/s1_passed.sum()*100:.2f}%)")
    print(f"  End-to-End Class 2 Recall:    {e2e_c2}/{tot_c2} = {rec_c2*100:.2f}%")
    print(f"  End-to-End Class 2 Precision: {e2e_c2}/{total_accepted} = {prec_c2*100:.2f}%")
    print(f"  End-to-End Class 2 F1-Score:  {f1_c2:.4f}")
    print(f"  Rejection of Noise: Class 0b={rej_c0b*100:.2f}%, Class 0a={rej_c0a*100:.2f}%, Class 1={rej_c1*100:.2f}%")
    print(f"  Final End-to-End Acceptance:  {total_accepted}/{len(test_df)} = {total_accepted/len(test_df)*100:.2f}%")
