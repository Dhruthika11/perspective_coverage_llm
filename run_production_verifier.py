import json
import os
import time
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

print("=" * 85)
print("RUNNING FROZEN VERIFIER ON PRODUCTION PERSPECTIVE-MATCHING WORKLOAD")
print("=" * 85)

# 1. Load Frozen Verifier Configuration
with open('results/calibration/frozen_warrant_probe_model.json', 'r') as f:
    config = json.load(f)

tau_s1 = 0.55
t_probe = config['calibrated_threshold']  # 0.20
feats = config['input_features']
scaler_mean = np.array(config['scaler_mean'])
scaler_scale = np.array(config['scaler_scale'])
w_cos = config['coefficients']['cosine_similarity']
w_ent = config['coefficients']['p_entail']
w_neu = config['coefficients']['p_neutral']
intercept = config['intercept']

print(f"Loaded Frozen Config:")
print(f"  Stage 1 Threshold:   tau* = {tau_s1:.2f}")
print(f"  Stage 2b Threshold:  T* = {t_probe:.2f}")
print(f"  Weights: cos={w_cos:.4f}, entail={w_ent:.4f}, neutral={w_neu:.4f}, b={intercept:.4f}")

# 2. Load Production Candidate Workload
prod_csv = "data/calibration/dataset_b_stage_b_sampled_250.csv"
df_prod = pd.read_csv(prod_csv)
print(f"\nLoaded {len(df_prod)} production candidates from: {prod_csv}")

# 3. Stage 1: Dense Retrieval Filter
df_prod['stage1_pass'] = df_prod['cosine_similarity'] >= tau_s1
s1_count = df_prod['stage1_pass'].sum()
print(f"\nStage 1 Retrieval (tau* = {tau_s1:.2f}):")
print(f"  Passed: {s1_count} / {len(df_prod)} ({s1_count/len(df_prod)*100:.2f}%)")
print(f"  Pruned: {len(df_prod) - s1_count} / {len(df_prod)} ({(len(df_prod)-s1_count)/len(df_prod)*100:.2f}%)")

# 4. Stage 2a: Stance Concordance Filter (Preserves UNKNOWN)
# Reject only if explicit contradiction
g_st = df_prod['stance_gold'].astype(str).str.lower().str.strip()
h_st = df_prod['stance_generated'].astype(str).str.lower().str.strip()
is_contra = ((g_st == 'support') & (h_st == 'undermine')) | ((g_st == 'undermine') & (h_st == 'support'))
df_prod['stage2a_pass'] = df_prod['stage1_pass'] & (~is_contra)
s2a_count = df_prod['stage2a_pass'].sum()
print(f"\nStage 2a Stance Filter (Uncertainty-Preserving):")
print(f"  Passed Stage 2a: {s2a_count} / {s1_count} ({s2a_count/s1_count*100:.2f}% of S1 survivors)")
print(f"  Pruned by Stance: {s1_count - s2a_count} counter-arguments")

# 5. Stage 2b: Cross-Encoder Inference & Warrant Probe on Surviving Candidates
s2a_survivors = df_prod[df_prod['stage2a_pass']].copy()
print(f"\nStage 2b Warrant Discrimination:")
print(f"  Running Cross-Encoder on {len(s2a_survivors)} surviving candidates...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "models/argument_nli_deberta_v3_v3"
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
model.eval()
print(f"  Cross-Encoder loaded in {time.time() - t0:.2f}s on {device}")

batch_size = 16
premises = s2a_survivors['generated_viewpoint_text'].tolist()
hypotheses = s2a_survivors['gold_perspective_text'].tolist()

p_ent_list = []
p_neu_list = []
p_contra_list = []

t1 = time.time()
for i in range(0, len(premises), batch_size):
    p_b = premises[i : i + batch_size]
    h_b = hypotheses[i : i + batch_size]
    inputs = tokenizer(p_b, h_b, truncation=True, max_length=256, padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    p_contra_list.extend(probs[:, 0])
    p_ent_list.extend(probs[:, 1])
    p_neu_list.extend(probs[:, 2])

print(f"  Cross-Encoder inference completed in {time.time() - t1:.2f}s")

# Populate signals in dataframe
df_prod['p_contra'] = np.nan
df_prod['p_entail'] = np.nan
df_prod['p_neutral'] = np.nan
df_prod['probe_prob'] = np.nan
df_prod['stage2b_pass'] = False
df_prod['final_decision'] = "REJECT_STAGE1"

# Assign Stage 1 rejections
df_prod.loc[~df_prod['stage1_pass'], 'final_decision'] = "REJECT_STAGE1"
# Assign Stage 2a rejections
df_prod.loc[df_prod['stage1_pass'] & ~df_prod['stage2a_pass'], 'final_decision'] = "REJECT_STAGE2A_STANCE"

# Compute Probe for S2a survivors
surv_idx = s2a_survivors.index
df_prod.loc[surv_idx, 'p_contra'] = [round(float(x), 4) for x in p_contra_list]
df_prod.loc[surv_idx, 'p_entail'] = [round(float(x), 4) for x in p_ent_list]
df_prod.loc[surv_idx, 'p_neutral'] = [round(float(x), 4) for x in p_neu_list]

# Scale features: (x - mean) / scale
cos_vals = df_prod.loc[surv_idx, 'cosine_similarity'].values
ent_vals = df_prod.loc[surv_idx, 'p_entail'].values
neu_vals = df_prod.loc[surv_idx, 'p_neutral'].values

z_cos = (cos_vals - scaler_mean[0]) / scaler_scale[0]
z_ent = (ent_vals - scaler_mean[1]) / scaler_scale[1]
z_neu = (neu_vals - scaler_mean[2]) / scaler_scale[2]

linear_score = z_cos * w_cos + z_ent * w_ent + z_neu * w_neu + intercept
probe_prob = 1.0 / (1.0 + np.exp(-linear_score))

df_prod.loc[surv_idx, 'probe_prob'] = [round(float(x), 4) for x in probe_prob]
df_prod.loc[surv_idx, 'stage2b_pass'] = df_prod.loc[surv_idx, 'probe_prob'] >= t_probe

# Final Decision
df_prod.loc[surv_idx, 'final_decision'] = np.where(
    df_prod.loc[surv_idx, 'stage2b_pass'],
    "ACCEPT_CLASS2_EQUIVALENT",
    "REJECT_STAGE2B_DIFFERENT_WARRANT"
)

# Summary of Production Decisions
print("\n" + "=" * 85)
print("FINAL PRODUCTION VERIFIER DECISION BREAKDOWN (N = 250)")
print("=" * 85)
dec_counts = df_prod['final_decision'].value_counts()
for dec, cnt in dec_counts.items():
    print(f"  {dec:35s}: {cnt:4d} ({cnt/len(df_prod)*100:.2f}%)")

# Export Full Predictions
out_pred_path = "results/production_verifier_predictions_250.csv"
df_prod.to_csv(out_pred_path, index=False)
print(f"\nSaved production verifier predictions to: {out_pred_path}")

# 6. Sample 50 Stratified Pairs for Independent Manual Audit
# We stratify across the four decision buckets:
# 1. ACCEPT_CLASS2_EQUIVALENT (20 pairs)
# 2. REJECT_STAGE2B_DIFFERENT_WARRANT (15 pairs)
# 3. REJECT_STAGE2A_STANCE (10 pairs)
# 4. REJECT_STAGE1 (5 pairs)
np.random.seed(42)

sample_accepted = df_prod[df_prod['final_decision'] == 'ACCEPT_CLASS2_EQUIVALENT'].sample(min(20, (df_prod['final_decision'] == 'ACCEPT_CLASS2_EQUIVALENT').sum()), random_state=42)
sample_warrant_rej = df_prod[df_prod['final_decision'] == 'REJECT_STAGE2B_DIFFERENT_WARRANT'].sample(min(15, (df_prod['final_decision'] == 'REJECT_STAGE2B_DIFFERENT_WARRANT').sum()), random_state=42)
sample_stance_rej = df_prod[df_prod['final_decision'] == 'REJECT_STAGE2A_STANCE'].sample(min(10, (df_prod['final_decision'] == 'REJECT_STAGE2A_STANCE').sum()), random_state=42)
sample_stage1_rej = df_prod[df_prod['final_decision'] == 'REJECT_STAGE1'].sample(min(5, (df_prod['final_decision'] == 'REJECT_STAGE1').sum()), random_state=42)

audit_sample = pd.concat([sample_accepted, sample_warrant_rej, sample_stance_rej, sample_stage1_rej]).reset_index(drop=True)

# Select and order audit columns cleanly
audit_cols = [
    'pair_id', 'claim_id', 'claim_text', 'gold_perspective_text', 'generated_viewpoint_text',
    'stance_gold', 'stance_generated', 'cosine_similarity', 'p_entail', 'p_neutral', 'probe_prob',
    'final_decision'
]
audit_export = audit_sample[audit_cols].copy()
audit_export['human_audit_decision'] = ""
audit_export['human_audit_notes'] = ""

audit_path = "data/audit/production_verifier_audit_sample_50.csv"
os.makedirs("data/audit", exist_ok=True)
audit_export.to_csv(audit_path, index=False)
print(f"Saved stratified 50-pair independent audit fixture to: {audit_path}")
print(f"Audit Breakdown: {len(sample_accepted)} Accepted, {len(sample_warrant_rej)} Warrant Rejections, {len(sample_stance_rej)} Stance Rejections, {len(sample_stage1_rej)} S1 Rejections")
