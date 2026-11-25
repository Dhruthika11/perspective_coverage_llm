import json
import pandas as pd
import numpy as np

df = pd.read_csv('results/calibration/stage2_cross_encoder_predictions_341.csv')
cal_df = df[df['partition'] == 'calibration'].copy()

# Load probe config
with open('results/calibration/frozen_warrant_probe_model.json', 'r') as f:
    config = json.load(f)

scaler_mean = np.array(config['scaler_mean'])
scaler_scale = np.array(config['scaler_scale'])
w_cos = config['coefficients']['cosine_similarity']
w_ent = config['coefficients']['p_entail']
w_neu = config['coefficients']['p_neutral']
intercept = config['intercept']

z_cos = (cal_df['cosine_similarity'] - scaler_mean[0]) / scaler_scale[0]
z_ent = (cal_df['p_entail'] - scaler_mean[1]) / scaler_scale[1]
z_neu = (cal_df['p_neutral'] - scaler_mean[2]) / scaler_scale[2]
cal_df['probe_prob'] = 1.0 / (1.0 + np.exp(-(z_cos * w_cos + z_ent * w_ent + z_neu * w_neu + intercept)))

# Filter Class 1 errors accepted at 0.20
c1_acc = cal_df[(cal_df['human_ground_truth_label'] == 'class_1') & (cal_df['probe_prob'] >= 0.20)].copy()
print(f"Total Class 1 false positives on Calibration partition: {len(c1_acc)}")

# Group by reason:
# 1. High NLI Entailment (P_entail >= 0.70): Shared conclusion/topic mistaken for warrant entailment
# 2. Moderate NLI Entailment (0.20 <= P_entail < 0.70)
# 3. Dense Cosine Dominance (P_entail < 0.20, High Cosine)

high_ent = c1_acc[c1_acc['p_entail'] >= 0.70]
mod_ent = c1_acc[(c1_acc['p_entail'] >= 0.20) & (c1_acc['p_entail'] < 0.70)]
cos_dom = c1_acc[c1_acc['p_entail'] < 0.20]

print(f"\n1. High NLI Entailment (P_entail >= 0.70): {len(high_ent)} pairs")
print(f"2. Moderate NLI Entailment (0.20 <= P_entail < 0.70): {len(mod_ent)} pairs")
print(f"3. Dense Cosine Dominance (P_entail < 0.20, High Cosine): {len(cos_dom)} pairs")

print("\nSample High NLI Entailment Class 1 Errors (Shared conclusion mistaken for warrant equivalence):")
for idx, r in high_ent.head(5).iterrows():
    print(f"\nPID {r.pair_id} | Claim: \"{r.claim_text}\" | Cos: {r.cosine_similarity:.4f} | Ent: {r.p_entail:.4f} | Probe: {r.probe_prob:.4f}")
    print(f"  Gold Warrant: \"{r.gold_perspective_text}\"")
    print(f"  Gen Warrant:  \"{r.generated_viewpoint_text}\"")

print("\nSample Dense Cosine Dominance Class 1 Errors:")
for idx, r in cos_dom.head(5).iterrows():
    print(f"\nPID {r.pair_id} | Claim: \"{r.claim_text}\" | Cos: {r.cosine_similarity:.4f} | Ent: {r.p_entail:.4f} | Probe: {r.probe_prob:.4f}")
    print(f"  Gold Warrant: \"{r.gold_perspective_text}\"")
    print(f"  Gen Warrant:  \"{r.generated_viewpoint_text}\"")
