import json
import pandas as pd

with open("results/calibration/e3_recovered_and_rejected_pairs_analysis.json") as f:
    err_data = json.load(f)

print("=" * 80)
print("ERROR ANALYSIS: RECOVERED CLASS 2 PAIRS (M4 vs M0)")
print("=" * 80)
for r in err_data["recovered_class_2"]:
    print(f"Pair ID: {r['pair_id']} (Claim {r['claim_id']}: {r['claim_text']})")
    print(f"  Gold: {r['gold_perspective_text']}")
    print(f"  Gen : {r['generated_viewpoint_text']}")
    print(f"  M0 Prob: {r['m0_prob']:.4f} (Rejected by M0 at T=0.20)")
    print(f"  M4 Prob: {r['m4_prob']:.4f} (Accepted by M4 at T=0.11)")
    print(f"  Signals: Cos={r['cosine_similarity']:.3f}, Claim={r['claim_cosine']:.3f}, Reason={r['reasoning_cosine']:.3f}, Warrant={r['warrant_cosine']:.3f}, p_ent={r['p_entail']:.3f}")

print("\n" + "=" * 80)
print("ERROR ANALYSIS: NEWLY REJECTED CLASS 1 PAIRS (M2 vs M0)")
print("=" * 80)
df = pd.read_csv("results/calibration/e3_pairwise_features_and_predictions_341.csv")
test_df = df[df['partition'] == 'test'].copy()

is_contra = (
    ((test_df['stance_gold'] == 'support') & (test_df['stance_generated'] == 'undermine')) |
    ((test_df['stance_gold'] == 'undermine') & (test_df['stance_generated'] == 'support'))
)
s1_pass = test_df['cosine_similarity'] >= 0.55
s2a_pass = s1_pass & (~is_contra)

test_df['m0_accept'] = s2a_pass & (test_df['prob_M0_baseline'] >= 0.20)
test_df['m2_accept'] = s2a_pass & (test_df['prob_M2_reasoning'] >= 0.31)

m2_new_rej_c1 = test_df[(test_df['human_ground_truth_label'] == 'class_1') & (test_df['m0_accept']) & (~test_df['m2_accept'])]
print(f"Total Class 1 False Positives eliminated by M2: {len(m2_new_rej_c1)} pairs!")
for idx, r in m2_new_rej_c1.iterrows():
    print(f"\nPair ID {r['pair_id']} (Claim {r['claim_id']}: {r['claim_text']})")
    print(f"  Gold: {r['gold_perspective_text']}")
    print(f"  Gen : {r['generated_viewpoint_text']}")
    print(f"  M0 Prob: {r['prob_M0_baseline']:.4f} -> M2 Prob: {r['prob_M2_reasoning']:.4f}")
    print(f"  Signals: Cos={r['cosine_similarity']:.3f}, Reason_Cos={r['reasoning_cosine']:.3f}, p_ent={r['p_entail']:.3f}")
