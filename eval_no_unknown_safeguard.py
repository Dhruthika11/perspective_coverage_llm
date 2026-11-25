import json
import numpy as np
import pandas as pd

print("=" * 85)
print("EVALUATION OF THE FROZEN VERIFIER PIPELINE WITHOUT ANY UNKNOWN SAFEGUARD")
print("=" * 85)

# 1. EVALUATION ON 341 HUMAN GROUND TRUTH BENCHMARK PAIRS
# Model: Frozen Warrant Probe (tau_S1 = 0.55, Stance Filter = Pure Stance Concordance, T* = 0.20)
bench_path = "results/calibration/stage2_cross_encoder_predictions_341.csv"
df_bench = pd.read_csv(bench_path)
df1700 = pd.read_csv('data/calibration/human_llm_benchmark_1700.csv')
df_bench = df_bench.merge(df1700[['pair_id', 'stance_gold', 'stance_generated']], on='pair_id', how='left')

with open('results/calibration/frozen_warrant_probe_model.json', 'r') as f:
    config = json.load(f)

scaler_mean = np.array(config['scaler_mean'])
scaler_scale = np.array(config['scaler_scale'])
w_cos = config['coefficients']['cosine_similarity']
w_ent = config['coefficients']['p_entail']
w_neu = config['coefficients']['p_neutral']
intercept = config['intercept']
t_probe = config['calibrated_threshold']

z_cos = (df_bench['cosine_similarity'] - scaler_mean[0]) / scaler_scale[0]
z_ent = (df_bench['p_entail'] - scaler_mean[1]) / scaler_scale[1]
z_neu = (df_bench['p_neutral'] - scaler_mean[2]) / scaler_scale[2]
probe_prob = 1.0 / (1.0 + np.exp(-(z_cos * w_cos + z_ent * w_ent + z_neu * w_neu + intercept)))
df_bench['probe_prob'] = probe_prob

# Stance filter: reject only explicit contradiction (support vs undermine)
g_st = df_bench['stance_gold'].astype(str).str.lower().str.strip()
h_st = df_bench['stance_generated'].astype(str).str.lower().str.strip()
is_contra = ((g_st == 'support') & (h_st == 'undermine')) | ((g_st == 'undermine') & (h_st == 'support'))
df_bench['stage2a_pass'] = ~is_contra

# Pipeline decision WITHOUT UNKNOWN safeguard:
# Accept iff stage2a_pass AND probe_prob >= t_probe
df_bench['pipeline_accept'] = df_bench['stage2a_pass'] & (df_bench['probe_prob'] >= t_probe)

print("\n--- PARTITION RESULTS ON 341 BENCHMARK (WITHOUT UNKNOWN SAFEGUARD) ---")
for part in ['train', 'calibration', 'test']:
    sub = df_bench[df_bench['partition'] == part]
    c2 = sub[sub['human_ground_truth_label'] == 'class_2']
    c1 = sub[sub['human_ground_truth_label'] == 'class_1']
    c0a = sub[sub['human_ground_truth_label'] == 'class_0a']
    c0b = sub[sub['human_ground_truth_label'] == 'class_0b']
    
    tp = c2['pipeline_accept'].sum()
    fp = sub[sub['human_ground_truth_label'] != 'class_2']['pipeline_accept'].sum()
    fn = (~c2['pipeline_accept']).sum()
    tn = (~sub[sub['human_ground_truth_label'] != 'class_2']['pipeline_accept']).sum()
    
    prec = tp / max(tp + fp, 1) * 100
    rec = tp / len(c2) * 100
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    
    c1_rej = (~c1['pipeline_accept']).mean() * 100
    c0a_rej = (~c0a['pipeline_accept']).mean() * 100
    c0b_rej = (~c0b['pipeline_accept']).mean() * 100
    
    print(f"\n{part.upper()} Partition (N = {len(sub)}):")
    print(f"  Class 2 Recall:    {rec:5.2f}% ({tp}/{len(c2)})")
    print(f"  Class 2 Precision: {prec:5.2f}% ({tp}/{tp+fp})")
    print(f"  Class 2 F1-Score:  {f1/100:6.4f}")
    print(f"  Class 1 Rejection: {c1_rej:5.2f}% ({(~c1['pipeline_accept']).sum()}/{len(c1)})")
    print(f"  Class 0a Rejection:{c0a_rej:5.2f}% ({(~c0a['pipeline_accept']).sum()}/{len(c0a)})")
    print(f"  Class 0b Rejection:{c0b_rej:5.2f}% ({(~c0b['pipeline_accept']).sum()}/{len(c0b)})")

# 2. EVALUATION ON 100 INDEPENDENT PRODUCTION AUDIT PAIRS
print("\n" + "=" * 85)
print("2. EVALUATION ON 100 AUDITED PRODUCTION PAIRS (WITHOUT UNKNOWN SAFEGUARD)")
print("=" * 85)

df_u100 = pd.read_csv('data/audit/unified_production_audit_benchmark_100.csv')
# Here final_decision is already the frozen pipeline output WITHOUT the safeguard
print(df_u100['final_decision'].value_counts())

tp = ((df_u100['final_decision'] == 'ACCEPT_CLASS2_EQUIVALENT') & (df_u100['human_audit_decision'] == 'class_2')).sum()
fp = ((df_u100['final_decision'] == 'ACCEPT_CLASS2_EQUIVALENT') & (df_u100['human_audit_decision'] != 'class_2')).sum()
fn = ((df_u100['final_decision'] != 'ACCEPT_CLASS2_EQUIVALENT') & (df_u100['human_audit_decision'] == 'class_2')).sum()
tn = ((df_u100['final_decision'] != 'ACCEPT_CLASS2_EQUIVALENT') & (df_u100['human_audit_decision'] != 'class_2')).sum()

prec = tp / (tp + fp) * 100
rec = tp / (tp + fn) * 100
f1 = 2 * prec * rec / (prec + rec)
spec = tn / (tn + fp) * 100

print(f"\nConfusion Matrix on 100 Audited Pairs:")
print(f"  TP = {tp}, FP = {fp}, FN = {fn}, TN = {tn}")
print(f"  Precision:   {prec:.2f}% ({tp}/{tp+fp})")
print(f"  Recall:      {rec:.2f}% ({tp}/{tp+fn})")
print(f"  F1-Score:    {f1/100:.4f}")
print(f"  Specificity: {spec:.2f}% ({tn}/{tn+fp})")

# Check UNKNOWN-stance performance specifically
df_u100['is_unknown'] = df_u100['stance_generated'].astype(str).str.lower().str.strip() == 'unknown'
unk_pairs = df_u100[df_u100['is_unknown']]
print(f"\nBreakdown of the {len(unk_pairs)} Audited Pairs with UNKNOWN Generated Stance:")
print(pd.crosstab(unk_pairs['final_decision'], unk_pairs['human_audit_decision'], margins=True))

unk_accepted = unk_pairs[unk_pairs['final_decision'] == 'ACCEPT_CLASS2_EQUIVALENT']
unk_tp = (unk_accepted['human_audit_decision'] == 'class_2').sum()
print(f"UNKNOWN Accepted Precision: {unk_tp} / {len(unk_accepted)} = {unk_tp/len(unk_accepted)*100:.2f}%")
