import pandas as pd
import numpy as np

# Load full 250 predictions and first 50 audited sample
df_all = pd.read_csv('results/production_verifier_predictions_250.csv')
df_audited = pd.read_csv('data/audit/production_verifier_audit_sample_50.csv')

audited_pids = set(df_audited['pair_id'])
df_remaining = df_all[~df_all['pair_id'].isin(audited_pids)].copy()

print(f"Total remaining unaudited production pairs: {len(df_remaining)}")
print("Remaining pairs by pipeline decision:")
print(df_remaining['final_decision'].value_counts())

# Check UNKNOWN stance in remaining pairs
df_remaining['is_unknown_stance'] = df_remaining['stance_generated'].astype(str).str.lower().str.strip() == 'unknown'
print(f"Remaining pairs with UNKNOWN generated stance: {df_remaining['is_unknown_stance'].sum()}")

# Stratified Priority Sampling for Batch 2 (50 pairs):
# 1. Accepted pairs (25 pairs)
# 2. Remaining UNKNOWN stance candidates across rejection strata (12 pairs)
# 3. Stage 2b Warrant Rejections (known stance) (7 pairs)
# 4. Stage 2a Stance Rejections (4 pairs)
# 5. Stage 1 Retrieval Rejections (2 pairs)

np.random.seed(42)

# Subsets
rem_accepted = df_remaining[df_remaining['final_decision'] == 'ACCEPT_CLASS2_EQUIVALENT']
rem_unknown_rej = df_remaining[(df_remaining['final_decision'] != 'ACCEPT_CLASS2_EQUIVALENT') & df_remaining['is_unknown_stance']]
rem_warrant_rej = df_remaining[(df_remaining['final_decision'] == 'REJECT_STAGE2B_DIFFERENT_WARRANT') & ~df_remaining['is_unknown_stance']]
rem_stance_rej = df_remaining[df_remaining['final_decision'] == 'REJECT_STAGE2A_STANCE']
rem_stage1_rej = df_remaining[(df_remaining['final_decision'] == 'REJECT_STAGE1') & ~df_remaining['is_unknown_stance']]

sample_b2_acc = rem_accepted.sample(min(25, len(rem_accepted)), random_state=42)
sample_b2_unk = rem_unknown_rej.sample(min(12, len(rem_unknown_rej)), random_state=42)
sample_b2_war = rem_warrant_rej.sample(min(7, len(rem_warrant_rej)), random_state=42)
sample_b2_sta = rem_stance_rej.sample(min(4, len(rem_stance_rej)), random_state=42)
sample_b2_s1  = rem_stage1_rej.sample(min(2, len(rem_stage1_rej)), random_state=42)

batch2 = pd.concat([sample_b2_acc, sample_b2_unk, sample_b2_war, sample_b2_sta, sample_b2_s1]).reset_index(drop=True)

audit_cols = [
    'pair_id', 'claim_id', 'claim_text', 'gold_perspective_text', 'generated_viewpoint_text',
    'stance_gold', 'stance_generated', 'cosine_similarity', 'p_entail', 'p_neutral', 'probe_prob',
    'final_decision'
]
batch2_export = batch2[audit_cols].copy()
batch2_export['human_audit_decision'] = ""
batch2_export['human_audit_notes'] = ""

out_path = 'data/audit/expanded_production_audit_queue_batch2_50.csv'
batch2_export.to_csv(out_path, index=False)
print(f"\nSaved Batch 2 Expanded Audit Queue ({len(batch2_export)} pairs) to: {out_path}")
print("Batch 2 Composition:")
print(f"  * Pipeline Accepted Candidates: {len(sample_b2_acc)}")
print(f"  * Enriched UNKNOWN-Stance Rejections: {len(sample_b2_unk)}")
print(f"  * Stage 2b Warrant Rejections: {len(sample_b2_war)}")
print(f"  * Stage 2a Stance Rejections: {len(sample_b2_sta)}")
print(f"  * Stage 1 Retrieval Rejections: {len(sample_b2_s1)}")
