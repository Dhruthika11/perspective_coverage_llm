import json
import numpy as np
import pandas as pd

df_pop = pd.read_csv('results/production_verifier_predictions_250.csv')
df_u100 = pd.read_csv('data/audit/unified_production_audit_benchmark_100.csv')
df_b1 = pd.read_csv('data/audit/production_verifier_audit_sample_50.csv')

df_pop['is_unknown'] = df_pop['stance_generated'].astype(str).str.lower().str.strip() == 'unknown'
df_u100['is_unknown'] = df_u100['stance_generated'].astype(str).str.lower().str.strip() == 'unknown'

# Define the exact 7 sub-strata:
# 1. ACCEPT + Known Stance
# 2. ACCEPT + UNKNOWN
# 3. REJECT_STAGE2B + Known Stance
# 4. REJECT_STAGE2B + UNKNOWN
# 5. REJECT_STAGE2A + Known Stance (UNKNOWN is 0 in pop)
# 6. REJECT_STAGE1 + Known Stance
# 7. REJECT_STAGE1 + UNKNOWN

sub_strata = [
    ('ACCEPT_CLASS2_EQUIVALENT', False, 'ACCEPT_KNOWN'),
    ('ACCEPT_CLASS2_EQUIVALENT', True,  'ACCEPT_UNKNOWN'),
    ('REJECT_STAGE2B_DIFFERENT_WARRANT', False, 'REJ_S2B_KNOWN'),
    ('REJECT_STAGE2B_DIFFERENT_WARRANT', True,  'REJ_S2B_UNKNOWN'),
    ('REJECT_STAGE2A_STANCE', False, 'REJ_S2A_KNOWN'),
    ('REJECT_STAGE1', False, 'REJ_S1_KNOWN'),
    ('REJECT_STAGE1', True,  'REJ_S1_UNKNOWN'),
]

print(f"{'Sub-Stratum':<20} | {'Pop N_h':<8} | {'Sample n_h':<10} | {'Incl Prob pi_h':<14} | {'Design Weight W_h'}")
print("-" * 75)

weights_dict = {}
for dec, unk, name in sub_strata:
    N_h = len(df_pop[(df_pop['final_decision'] == dec) & (df_pop['is_unknown'] == unk)])
    n_h = len(df_u100[(df_u100['final_decision'] == dec) & (df_u100['is_unknown'] == unk)])
    pi_h = n_h / max(N_h, 1)
    W_h = N_h / max(n_h, 1)
    weights_dict[(dec, unk)] = W_h
    print(f"{name:<20} | {N_h:<8} | {n_h:<10} | {pi_h:<14.4f} | {W_h:.4f}")

# Compute Sub-Stratum Horvitz-Thompson Estimator
df_u100['sub_weight'] = [weights_dict[(r['final_decision'], r['is_unknown'])] for idx, r in df_u100.iterrows()]

pop_tp = 0.0
pop_fp = 0.0
pop_fn = 0.0
pop_tn = 0.0

pop_c2 = 0.0
pop_c1 = 0.0
pop_c0a = 0.0
pop_c0b = 0.0

for idx, r in df_u100.iterrows():
    w = r['sub_weight']
    lbl = r['human_audit_decision']
    pred = r['final_decision'] == 'ACCEPT_CLASS2_EQUIVALENT'
    
    if lbl == 'class_2':
        pop_c2 += w
        if pred:
            pop_tp += w
        else:
            pop_fn += w
    else:
        if pred:
            pop_fp += w
        else:
            pop_tn += w
            
    if lbl == 'class_1':
        pop_c1 += w
    elif lbl == 'class_0a':
        pop_c0a += w
    elif lbl == 'class_0b':
        pop_c0b += w

prec_sub = pop_tp / max(pop_tp + pop_fp, 1)
rec_sub = pop_tp / max(pop_tp + pop_fn, 1)
f1_sub = 2 * prec_sub * rec_sub / max(prec_sub + rec_sub, 1e-8)
spec_sub = pop_tn / max(pop_tn + pop_fp, 1)

print("\n--- SUB-STRATUM DESIGN-UNBIASED ESTIMATES (N = 250) ---")
print(f"Population Totals: Class 2 = {pop_c2:.2f}, Class 1 = {pop_c1:.2f}, Class 0a = {pop_c0a:.2f}, Class 0b = {pop_c0b:.2f}")
print(f"Confusion Matrix:  TP = {pop_tp:.2f}, FP = {pop_fp:.2f}, FN = {pop_fn:.2f}, TN = {pop_tn:.2f}")
print(f"Performance:")
print(f"  Precision:   {prec_sub*100:.2f}%")
print(f"  Recall:      {rec_sub*100:.2f}%")
print(f"  F1-Score:    {f1_sub:.4f}")
print(f"  Specificity: {spec_sub*100:.2f}%")

# Compare with Batch 1 Alone (Pure Simple Random Sampling within 4 strata)
w1_b1 = 74 / 20
w2_b1 = 70 / 15
w3_b1 = 56 / 10
w4_b1 = 50 / 5
df_b1['w'] = df_b1['final_decision'].map({
    'ACCEPT_CLASS2_EQUIVALENT': w1_b1,
    'REJECT_STAGE2B_DIFFERENT_WARRANT': w2_b1,
    'REJECT_STAGE2A_STANCE': w3_b1,
    'REJECT_STAGE1': w4_b1
})

tp_b1 = (df_b1[df_b1['human_audit_decision'] == 'class_2']['final_decision'] == 'ACCEPT_CLASS2_EQUIVALENT') * df_b1['w']
fn_b1 = (df_b1[df_b1['human_audit_decision'] == 'class_2']['final_decision'] != 'ACCEPT_CLASS2_EQUIVALENT') * df_b1['w']
fp_b1 = (df_b1[df_b1['human_audit_decision'] != 'class_2']['final_decision'] == 'ACCEPT_CLASS2_EQUIVALENT') * df_b1['w']
tn_b1 = (df_b1[df_b1['human_audit_decision'] != 'class_2']['final_decision'] != 'ACCEPT_CLASS2_EQUIVALENT') * df_b1['w']

prec_b1 = tp_b1.sum() / max(tp_b1.sum() + fp_b1.sum(), 1)
rec_b1 = tp_b1.sum() / max(tp_b1.sum() + fn_b1.sum(), 1)
f1_b1 = 2 * prec_b1 * rec_b1 / max(prec_b1 + rec_b1, 1e-8)
spec_b1 = tn_b1.sum() / max(tn_b1.sum() + fp_b1.sum(), 1)

print("\n--- BATCH 1 PURE STRATIFIED RANDOM SAMPLE ESTIMATES (n = 50) ---")
print(f"  Precision:   {prec_b1*100:.2f}%")
print(f"  Recall:      {rec_b1*100:.2f}%")
print(f"  F1-Score:    {f1_b1:.4f}")
print(f"  Specificity: {spec_b1*100:.2f}%")
