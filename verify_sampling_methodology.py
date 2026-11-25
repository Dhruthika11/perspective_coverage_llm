import json
import numpy as np
import pandas as pd

# Load 250 population predictions
df_pop = pd.read_csv('results/production_verifier_predictions_250.csv')
df_b1 = pd.read_csv('data/audit/production_verifier_audit_sample_50.csv')
df_b2 = pd.read_csv('data/audit/expanded_production_audit_queue_batch2_50.csv')

print("=" * 85)
print("METHODOLOGICAL VERIFICATION OF SAMPLING DESIGN & POPULATION COUNTS")
print("=" * 85)

# 1. Population Stratum Counts in Frozen Pipeline (results/production_verifier_predictions_250.csv)
print("\n1. POPULATION STRATUM COUNTS IN FROZEN PIPELINE (N = 250):")
print(df_pop['final_decision'].value_counts())

# Check UNKNOWN stance breakdown in population
df_pop['is_unknown'] = df_pop['stance_generated'].astype(str).str.lower().str.strip() == 'unknown'
print("\nPopulation breakdown by Decision and UNKNOWN stance:")
print(pd.crosstab(df_pop['final_decision'], df_pop['is_unknown'], margins=True))

# 2. Batch 1 Sampling Verification (n = 50)
print("\n2. BATCH 1 SAMPLING VERIFICATION:")
print("Batch 1 sample sizes per decision stratum:")
print(df_b1['final_decision'].value_counts())
# Check UNKNOWN distribution in Batch 1
df_b1['is_unknown'] = df_b1['stance_generated'].astype(str).str.lower().str.strip() == 'unknown'
print("Batch 1 UNKNOWN stance distribution:")
print(pd.crosstab(df_b1['final_decision'], df_b1['is_unknown'], margins=True))

# 3. Batch 2 Sampling Verification (n = 50)
print("\n3. BATCH 2 SAMPLING VERIFICATION:")
print("Batch 2 sample sizes per decision stratum:")
print(df_b2['final_decision'].value_counts())
df_b2['is_unknown'] = df_b2['stance_generated'].astype(str).str.lower().str.strip() == 'unknown'
print("Batch 2 UNKNOWN stance distribution:")
print(pd.crosstab(df_b2['final_decision'], df_b2['is_unknown'], margins=True))

# 4. Joint (Pooled) Sampling Inclusion Probabilities
# Check whether sampling was simple random within strata across the combined 100:
# For each decision stratum h, check inclusion rate of UNKNOWN vs known stance
df_all = df_pop.copy()
b1_pids = set(df_b1['pair_id'])
b2_pids = set(df_b2['pair_id'])
df_all['in_b1'] = df_all['pair_id'].isin(b1_pids)
df_all['in_b2'] = df_all['pair_id'].isin(b2_pids)
df_all['in_sample'] = df_all['in_b1'] | df_all['in_b2']

print("\n4. INCLUSION RATES IN POOLED SAMPLE (N_pop = 250, n_sample = 100):")
for dec, grp in df_all.groupby('final_decision'):
    total = len(grp)
    sampled = grp['in_sample'].sum()
    rate = sampled / total
    
    unk_total = grp['is_unknown'].sum()
    unk_sampled = (grp['is_unknown'] & grp['in_sample']).sum()
    unk_rate = unk_sampled / max(unk_total, 1)
    
    known_total = total - unk_total
    known_sampled = sampled - unk_sampled
    known_rate = known_sampled / max(known_total, 1)
    
    print(f"Stratum: {dec}")
    print(f"  Overall:  {sampled}/{total} ({rate*100:.2f}%)")
    print(f"  UNKNOWN:  {unk_sampled}/{unk_total} ({unk_rate*100:.2f}%)")
    print(f"  Known St: {known_sampled}/{known_total} ({known_rate*100:.2f}%)")
    print(f"  Equal inclusion? {abs(unk_rate - known_rate) < 0.05}")
