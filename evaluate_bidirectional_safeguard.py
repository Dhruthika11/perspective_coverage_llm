import json
import numpy as np
import pandas as pd

df = pd.read_csv("results/calibration/stage2_bidirectional_predictions_341.csv")
print("Total rows:", len(df))
print("Partitions:")
print(df['partition'].value_counts())

thresholds = [0.01, 0.025, 0.05, 0.10, 0.20]

# Check existing probe predictions
# In the frozen pipeline:
# Stage 1: tau* = 0.55
# Stage 2a: Stance filter (reject explicit contradiction: support vs undermine)
# Stage 2b: Probe prob >= 0.20

# Let's inspect columns to see stance info
df1700 = pd.read_csv('data/calibration/human_llm_benchmark_1700.csv')
df = df.merge(df1700[['pair_id', 'stance_gold', 'stance_generated']], on='pair_id', how='left')

g_st = df['stance_gold'].astype(str).str.lower().str.strip()
h_st = df['stance_generated'].astype(str).str.lower().str.strip()
is_contra = ((g_st == 'support') & (h_st == 'undermine')) | ((g_st == 'undermine') & (h_st == 'support'))
df['stage2a_pass'] = ~is_contra

# Probe probability from frozen model
with open('results/calibration/frozen_warrant_probe_model.json', 'r') as f:
    config = json.load(f)

scaler_mean = np.array(config['scaler_mean'])
scaler_scale = np.array(config['scaler_scale'])
w_cos = config['coefficients']['cosine_similarity']
w_ent = config['coefficients']['p_entail']
w_neu = config['coefficients']['p_neutral']
intercept = config['intercept']
t_probe = config['calibrated_threshold'] # 0.20

z_cos = (df['cosine_similarity'] - scaler_mean[0]) / scaler_scale[0]
z_ent = (df['p_entail_fwd'] - scaler_mean[1]) / scaler_scale[1]
z_neu = (df['p_neutral'] - scaler_mean[2]) / scaler_scale[2]
probe_prob = 1.0 / (1.0 + np.exp(-(z_cos * w_cos + z_ent * w_ent + z_neu * w_neu + intercept)))
df['probe_prob'] = probe_prob
df['probe_pass'] = (df['probe_prob'] >= t_probe) & df['stage2a_pass']

print(f"\nFrozen Probe Alone (T* = {t_probe}):")
for part in ['train', 'calibration', 'test']:
    sub = df[df['partition'] == part]
    c2 = sub[sub['human_ground_truth_label'] == 'class_2']
    c1 = sub[sub['human_ground_truth_label'] == 'class_1']
    c0a = sub[sub['human_ground_truth_label'] == 'class_0a']
    c0b = sub[sub['human_ground_truth_label'] == 'class_0b']
    
    c2_rec = c2['probe_pass'].mean() * 100
    c1_rej = (~c1['probe_pass']).mean() * 100
    c0a_rej = (~c0a['probe_pass']).mean() * 100
    c0b_rej = (~c0b['probe_pass']).mean() * 100
    
    tp = c2['probe_pass'].sum()
    fp = sub[sub['human_ground_truth_label'] != 'class_2']['probe_pass'].sum()
    prec = tp / max(tp + fp, 1) * 100
    f1 = 2 * (prec * c2_rec) / max(prec + c2_rec, 1e-8)
    
    print(f"  {part:12s} (N={len(sub):2d}): C2 Rec: {c2_rec:5.2f}% ({c2['probe_pass'].sum()}/{len(c2)}), "
          f"C1 Rej: {c1_rej:5.2f}% ({(~c1['probe_pass']).sum()}/{len(c1)}), "
          f"C0a Rej: {c0a_rej:5.2f}% ({(~c0a['probe_pass']).sum()}/{len(c0a)}), "
          f"C0b Rej: {c0b_rej:5.2f}% ({(~c0b['probe_pass']).sum()}/{len(c0b)}) | "
          f"Prec: {prec:5.2f}%, F1: {f1:5.2f}%")

print("\n" + "=" * 90)
print("EVALUATING BIDIRECTIONAL ENTAILMENT AS A SECOND-STAGE SAFEGUARD")
print("Rule: Accept iff (Probe Pass) AND (min(P_fwd, P_bwd) >= tau_bidi)")
print("=" * 90)

# Evaluate across partitions
for part in ['calibration', 'test']:
    sub = df[df['partition'] == part]
    c2 = sub[sub['human_ground_truth_label'] == 'class_2']
    c1 = sub[sub['human_ground_truth_label'] == 'class_1']
    c0a = sub[sub['human_ground_truth_label'] == 'class_0a']
    c0b = sub[sub['human_ground_truth_label'] == 'class_0b']
    
    print(f"\n--- PARTITION: {part.upper()} (N={len(sub)}: C2={len(c2)}, C1={len(c1)}, C0a={len(c0a)}, C0b={len(c0b)}) ---")
    print(f"{'Threshold':<10} | {'C2 Recall':<15} | {'C1 Rejection':<15} | {'C0a Rejection':<15} | {'C0b Rejection':<15} | {'Precision':<12} | {'F1-Score':<10}")
    print("-" * 90)
    
    # Baseline (no bidi safeguard, tau=0.0)
    for tau in [0.0] + thresholds:
        pass_mask = df['probe_pass'] & (df['bidirectional_min_entail'] >= tau)
        sub_pass = pass_mask.loc[sub.index]
        
        c2_pass = sub_pass.loc[c2.index]
        c1_pass = sub_pass.loc[c1.index]
        c0a_pass = sub_pass.loc[c0a.index]
        c0b_pass = sub_pass.loc[c0b.index]
        
        c2_rec = c2_pass.mean() * 100
        c1_rej = (~c1_pass).mean() * 100
        c0a_rej = (~c0a_pass).mean() * 100
        c0b_rej = (~c0b_pass).mean() * 100
        
        tp = c2_pass.sum()
        fp = (~sub_pass.loc[c2.index]).sum() # wait, fp is non-C2 that passed
        fp = sub_pass.loc[sub[sub['human_ground_truth_label'] != 'class_2'].index].sum()
        prec = tp / max(tp + fp, 1) * 100
        f1 = 2 * (prec * c2_rec) / max(prec + c2_rec, 1e-8)
        
        lbl = "None (0.0)" if tau == 0.0 else f"{tau:.3f}"
        print(f"{lbl:<10} | {c2_rec:5.2f}% ({c2_pass.sum()}/{len(c2)})   | "
              f"{c1_rej:5.2f}% ({(~c1_pass).sum()}/{len(c1)})   | "
              f"{c0a_rej:5.2f}% ({(~c0a_pass).sum()}/{len(c0a)})   | "
              f"{c0b_rej:5.2f}% ({(~c0b_pass).sum()}/{len(c0b)})   | "
              f"{prec:5.2f}% ({tp}/{tp+fp}) | "
              f"{f1:5.2f}%")

# Also evaluate Standalone Bidirectional Entailment (without the probe)
print("\n" + "=" * 90)
print("COMPARISON: STANDALONE BIDIRECTIONAL ENTAILMENT (WITHOUT PROBE)")
print("Rule: Accept iff (min(P_fwd, P_bwd) >= tau_bidi)")
print("=" * 90)
for part in ['calibration', 'test']:
    sub = df[df['partition'] == part]
    c2 = sub[sub['human_ground_truth_label'] == 'class_2']
    c1 = sub[sub['human_ground_truth_label'] == 'class_1']
    c0a = sub[sub['human_ground_truth_label'] == 'class_0a']
    c0b = sub[sub['human_ground_truth_label'] == 'class_0b']
    
    print(f"\n--- STANDALONE PARTITION: {part.upper()} ---")
    print(f"{'Threshold':<10} | {'C2 Recall':<15} | {'C1 Rejection':<15} | {'C0a Rejection':<15} | {'C0b Rejection':<15} | {'Precision':<12} | {'F1-Score':<10}")
    print("-" * 90)
    for tau in thresholds:
        pass_mask = df['bidirectional_min_entail'] >= tau
        sub_pass = pass_mask.loc[sub.index]
        
        c2_pass = sub_pass.loc[c2.index]
        c1_pass = sub_pass.loc[c1.index]
        c0a_pass = sub_pass.loc[c0a.index]
        c0b_pass = sub_pass.loc[c0b.index]
        
        c2_rec = c2_pass.mean() * 100
        c1_rej = (~c1_pass).mean() * 100
        c0a_rej = (~c0a_pass).mean() * 100
        c0b_rej = (~c0b_pass).mean() * 100
        
        tp = c2_pass.sum()
        fp = sub_pass.loc[sub[sub['human_ground_truth_label'] != 'class_2'].index].sum()
        prec = tp / max(tp + fp, 1) * 100
        f1 = 2 * (prec * c2_rec) / max(prec + c2_rec, 1e-8)
        
        print(f"{tau:<10.3f} | {c2_rec:5.2f}% ({c2_pass.sum()}/{len(c2)})   | "
              f"{c1_rej:5.2f}% ({(~c1_pass).sum()}/{len(c1)})   | "
              f"{c0a_rej:5.2f}% ({(~c0a_pass).sum()}/{len(c0a)})   | "
              f"{c0b_rej:5.2f}% ({(~c0b_pass).sum()}/{len(c0b)})   | "
              f"{prec:5.2f}% ({tp}/{tp+fp}) | "
              f"{f1:5.2f}%")
