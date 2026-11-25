import json
import numpy as np
import pandas as pd

# Load predictions & benchmark
df = pd.read_csv('results/calibration/stage2_cross_encoder_predictions_341.csv')
df1700 = pd.read_csv('data/calibration/human_llm_benchmark_1700.csv')
df = df.merge(df1700[['pair_id', 'stance_gold', 'stance_generated']], on='pair_id', how='left')

# Load Frozen Warrant Probe
with open('results/calibration/frozen_warrant_probe_model.json', 'r') as f:
    config = json.load(f)

scaler_mean = np.array(config['scaler_mean'])
scaler_scale = np.array(config['scaler_scale'])
w_cos = config['coefficients']['cosine_similarity']
w_ent = config['coefficients']['p_entail']
w_neu = config['coefficients']['p_neutral']
intercept = config['intercept']
tau_s1 = 0.55
t_probe = 0.20

# Stage 1 and Stage 2a
df['stage1_pass'] = df['cosine_similarity'] >= tau_s1
g_st = df['stance_gold'].astype(str).str.lower().str.strip()
h_st = df['stance_generated'].astype(str).str.lower().str.strip()
is_contra = ((g_st == 'support') & (h_st == 'undermine')) | ((g_st == 'undermine') & (h_st == 'support'))
df['stage2a_pass'] = df['stage1_pass'] & (~is_contra)

# Probe score
z_cos = (df['cosine_similarity'] - scaler_mean[0]) / scaler_scale[0]
z_ent = (df['p_entail'] - scaler_mean[1]) / scaler_scale[1]
z_neu = (df['p_neutral'] - scaler_mean[2]) / scaler_scale[2]
df['probe_prob'] = 1.0 / (1.0 + np.exp(-(z_cos * w_cos + z_ent * w_ent + z_neu * w_neu + intercept)))
df['final_accept'] = df['stage2a_pass'] & (df['probe_prob'] >= t_probe)

print("=" * 90)
print("EXPERIMENT E1: THRESHOLD SENSITIVITY AND ERROR AUDIT")
print("=" * 90)

# 1. Inspect the 5 Class 2 Misses across the full 341 benchmark
c2_misses = df[(df['human_ground_truth_label'] == 'class_2') & (~df['final_accept'])].copy()
print(f"\n1. THE FIVE CLASS 2 MISSES ACROSS ALL 341 PAIRS (Total {len(c2_misses)}):")
for idx, r in c2_misses.iterrows():
    print(f"\n[Miss #{len(c2_misses) - idx if False else idx}] Pair ID {r.pair_id} | Partition: {r.partition} | Claim {r.claim_id}: \"{r.claim_text}\"")
    print(f"  Stance: Gold={r.stance_gold}, Gen={r.stance_generated} | Stage1 Pass: {r.stage1_pass} | Stage2a Pass: {r.stage2a_pass}")
    print(f"  Scores: Cosine={r.cosine_similarity:.4f}, P(Entail)={r.p_entail:.4f}, P(Neutral)={r.p_neutral:.4f} -> Probe Score = {r.probe_prob:.4f} (Threshold T*={t_probe})")
    print(f"  Gold: \"{r.gold_perspective_text}\"")
    print(f"  Gen:  \"{r.generated_viewpoint_text}\"")

# 2. Inspect the accepted Class 1 errors on Calibration partition (and overall)
cal_df = df[df['partition'] == 'calibration'].copy()
c1_cal_accepted = cal_df[(cal_df['human_ground_truth_label'] == 'class_1') & cal_df['final_accept']].copy()
print(f"\n" + "=" * 90)
print(f"2. ACCEPTED CLASS 1 ERRORS ON CALIBRATION DATA (Total {len(c1_cal_accepted)} out of 45 Class 1 pairs):")
print("=" * 90)
for idx, r in c1_cal_accepted.iterrows():
    print(f"\nPair ID {r.pair_id} | Claim {r.claim_id}: \"{r.claim_text}\"")
    print(f"  Scores: Cosine={r.cosine_similarity:.4f}, P(Entail)={r.p_entail:.4f}, P(Neutral)={r.p_neutral:.4f} -> Probe Score = {r.probe_prob:.4f}")
    print(f"  Gold: \"{r.gold_perspective_text}\"")
    print(f"  Gen:  \"{r.generated_viewpoint_text}\"")

# 3. Fine-Grained Threshold Sweep around 0.20 on Training and Calibration data
print("\n" + "=" * 90)
print("3. THRESHOLD SENSITIVITY SWEEP AROUND 0.20 (TRAINING & CALIBRATION DATA)")
print("=" * 90)

thresholds_fine = [0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30, 0.35, 0.40, 0.50]

for name, d_part in [("Calibration Partition (16 Claims, 100 Pairs)", cal_df),
                     ("Training Partition (32 Claims, 153 Pairs)", df[df['partition'] == 'train'].copy()),
                     ("Train + Calibration Pooled (48 Claims, 253 Pairs)", df[df['partition'].isin(['train', 'calibration'])].copy())]:
    print(f"\n--- {name} ---")
    print(f"{'Threshold':<10} | {'C2 Recall':<15} | {'C2 Precision':<15} | {'F1-Score':<10} | {'C1 Rejection':<15} | {'TP':<4} | {'FP (C1/0a/0b)':<14} | {'FN':<4}")
    print("-" * 95)
    
    c2_p = d_part[d_part['human_ground_truth_label'] == 'class_2']
    c1_p = d_part[d_part['human_ground_truth_label'] == 'class_1']
    c0a_p = d_part[d_part['human_ground_truth_label'] == 'class_0a']
    c0b_p = d_part[d_part['human_ground_truth_label'] == 'class_0b']
    non_c2_p = d_part[d_part['human_ground_truth_label'] != 'class_2']
    
    for th in thresholds_fine:
        pass_mask = d_part['stage2a_pass'] & (d_part['probe_prob'] >= th)
        tp = pass_mask.loc[c2_p.index].sum()
        fp = pass_mask.loc[non_c2_p.index].sum()
        fn = len(c2_p) - tp
        tn = len(non_c2_p) - fp
        
        fp_c1 = pass_mask.loc[c1_p.index].sum()
        fp_0a = pass_mask.loc[c0a_p.index].sum()
        fp_0b = pass_mask.loc[c0b_p.index].sum()
        
        rec = tp / len(c2_p) * 100
        prec = tp / max(tp + fp, 1) * 100
        f1 = 2 * (prec * rec) / max(prec + rec, 1e-8)
        c1_rej = (~pass_mask.loc[c1_p.index]).mean() * 100
        
        print(f"{th:<10.2f} | {rec:5.2f}% ({tp}/{len(c2_p)})  | {prec:5.2f}% ({tp}/{tp+fp})  | {f1/100:6.4f}   | {c1_rej:5.2f}% ({(~pass_mask.loc[c1_p.index]).sum()}/{len(c1_p)}) | {tp:<4} | {fp} ({fp_c1}/{fp_0a}/{fp_0b}) | {fn:<4}")
