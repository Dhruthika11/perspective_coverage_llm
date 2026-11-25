import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('results/calibration/stage2_cross_encoder_predictions_341.csv')
c21 = df[df['human_ground_truth_label'].isin(['class_2', 'class_1'])].copy()

train_df = c21[c21['partition'] == 'train'].copy()
test_df = c21[c21['partition'] == 'test'].copy()

feats = ['cosine_similarity', 'p_entail', 'p_neutral']
scaler = StandardScaler()
X_tr = scaler.fit_transform(train_df[feats].values)
X_te = scaler.transform(test_df[feats].values)

clf = LogisticRegression(C=1.0, random_state=42)
clf.fit(X_tr, (train_df['human_ground_truth_label'] == 'class_2').astype(int).values)

test_df['probe_prob'] = clf.predict_proba(X_te)[:, 1]
test_df['probe_pred'] = (test_df['probe_prob'] >= 0.20).astype(int)

c1_fp = test_df[(test_df['human_ground_truth_label'] == 'class_1') & (test_df['probe_pred'] == 1)].sort_values(by='probe_prob', ascending=False)

print(f"TOTAL CLASS 1 FALSE POSITIVES AT T* = 0.20: {len(c1_fp)}")
for i, (idx, r) in enumerate(c1_fp.iterrows(), 1):
    print("=" * 80)
    print(f"[{i}/17] PID {r['pair_id']} | Claim {r['claim_id']}: \"{r['claim_text']}\"")
    print(f"Probe: {r['probe_prob']:.4f} | Cosine: {r['cosine_similarity']:.4f} | p_ent: {r['p_entail']:.4f} | p_neu: {r['p_neutral']:.4f}")
    print(f"Gold: \"{r['gold_perspective_text']}\"")
    print(f"Gen:  \"{r['generated_viewpoint_text']}\"")
    print(f"Human Rationale: {r['human_adjudication_rationale']}")
