import pandas as pd

df = pd.read_csv('results/calibration/e3_pairwise_features_and_predictions_341.csv')
test_df = df[df['partition'] == 'test'].copy()

is_contra = (
    ((test_df['stance_gold'] == 'support') & (test_df['stance_generated'] == 'undermine')) |
    ((test_df['stance_gold'] == 'undermine') & (test_df['stance_generated'] == 'support'))
)
s1_pass = test_df['cosine_similarity'] >= 0.55
s2a_pass = s1_pass & (~is_contra)

test_df['m0_accept'] = s2a_pass & (test_df['prob_M0_baseline'] >= 0.20)
test_df['m2_accept'] = s2a_pass & (test_df['prob_M2_reasoning'] >= 0.31)

m2_new_rej_c1 = test_df[(test_df['human_ground_truth_label'] == 'class_1') & (test_df['m0_accept']) & (~test_df['m2_accept'])].copy()

print(f"Auditing {len(m2_new_rej_c1)} Class 1 pairs newly rejected by M2:")
for idx, r in m2_new_rej_c1.iterrows():
    print("=" * 90)
    print(f"Pair ID: {r['pair_id']} | Claim ID: {r['claim_id']}")
    print(f"Debate Claim: {r['claim_text']}")
    print(f"Gold Perspective: {r['gold_perspective_text']}")
    print(f"Generated Viewpoint: {r['generated_viewpoint_text']}")
    print(f"Human Label: {r['human_ground_truth_label']}")
    print(f"Human Rationale: {r.get('human_adjudication_rationale', 'N/A')}")
    print(f"Jury Label: {r.get('jury_consensus_label', 'N/A')} ({r.get('jury_consensus_type', 'N/A')})")
    print(f"M0 Prob: {r['prob_M0_baseline']:.4f} -> M2 Prob: {r['prob_M2_reasoning']:.4f}")
    print(f"Global Cos: {r['cosine_similarity']:.4f} | Reason Cos: {r['reasoning_cosine']:.4f} | p_ent: {r['p_entail']:.4f}")
