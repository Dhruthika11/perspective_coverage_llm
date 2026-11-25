import json
import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint

print("=" * 90)
print("COMPREHENSIVE END-TO-END EVALUATION ON 341 HUMAN GROUND-TRUTH BENCHMARK PAIRS")
print("PIPELINE: Stage 1 (Retrieval tau=0.55) -> Stage 2a (Stance Concordance) -> Stage 2b (Probe T*=0.20)")
print("SAFEGUARD STATUS: Pure clean pipeline (NO UNKNOWN safeguard, NO bidirectional safeguard)")
print("=" * 90)

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
t_probe = config['calibrated_threshold'] # 0.20

# 1. Stage 1: Dense Retrieval Filter
df['stage1_pass'] = df['cosine_similarity'] >= tau_s1

# 2. Stage 2a: Stance Concordance Filter (Preserves UNKNOWN, rejects explicit contradiction)
g_st = df['stance_gold'].astype(str).str.lower().str.strip()
h_st = df['stance_generated'].astype(str).str.lower().str.strip()
is_contra = ((g_st == 'support') & (h_st == 'undermine')) | ((g_st == 'undermine') & (h_st == 'support'))
df['stage2a_pass'] = df['stage1_pass'] & (~is_contra)

# 3. Stage 2b: Warrant Probe Probability
z_cos = (df['cosine_similarity'] - scaler_mean[0]) / scaler_scale[0]
z_ent = (df['p_entail'] - scaler_mean[1]) / scaler_scale[1]
z_neu = (df['p_neutral'] - scaler_mean[2]) / scaler_scale[2]
probe_prob = 1.0 / (1.0 + np.exp(-(z_cos * w_cos + z_ent * w_ent + z_neu * w_neu + intercept)))
df['probe_prob'] = probe_prob

df['stage2b_pass'] = df['stage2a_pass'] & (df['probe_prob'] >= t_probe)
df['final_accept'] = df['stage2b_pass']

# Wilson CI helper
def get_wilson_ci(k, n):
    if n == 0:
        return (0.0, 0.0)
    low, high = proportion_confint(k, n, alpha=0.05, method='wilson')
    return float(low), float(high)

# Fast Claim-Clustered Bootstrap CI helper
def fast_claim_clustered_ci(sub_df, n_boot=2000, seed=42):
    rng = np.random.RandomState(seed)
    # Pre-aggregate by claim
    claims = []
    c2_totals = []
    tps = []
    fps = []
    
    for c, grp in sub_df.groupby('claim_id'):
        claims.append(c)
        c2 = grp[grp['human_ground_truth_label'] == 'class_2']
        non_c2 = grp[grp['human_ground_truth_label'] != 'class_2']
        c2_totals.append(len(c2))
        tps.append(c2['final_accept'].sum())
        fps.append(non_c2['final_accept'].sum())
        
    c2_totals = np.array(c2_totals)
    tps = np.array(tps)
    fps = np.array(fps)
    n_claims = len(claims)
    
    boot_rec = []
    boot_prec = []
    boot_f1 = []
    
    for _ in range(n_boot):
        idx = rng.choice(n_claims, size=n_claims, replace=True)
        tot_c2 = c2_totals[idx].sum()
        if tot_c2 == 0:
            continue
        tot_tp = tps[idx].sum()
        tot_fp = fps[idx].sum()
        
        rec = tot_tp / tot_c2
        prec = tot_tp / max(tot_tp + tot_fp, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        
        boot_rec.append(rec)
        boot_prec.append(prec)
        boot_f1.append(f1)
        
    rec_ci = (float(np.percentile(boot_rec, 2.5)), float(np.percentile(boot_rec, 97.5)))
    prec_ci = (float(np.percentile(boot_prec, 2.5)), float(np.percentile(boot_prec, 97.5)))
    f1_ci = (float(np.percentile(boot_f1, 2.5)), float(np.percentile(boot_f1, 97.5)))
    return rec_ci, prec_ci, f1_ci

report_dict = {}

for part in ['train', 'calibration', 'test', 'overall']:
    if part == 'overall':
        sub = df.copy()
        part_name = "OVERALL BENCHMARK (65 CLAIMS, 341 PAIRS)"
    else:
        sub = df[df['partition'] == part].copy()
        n_c = sub['claim_id'].nunique()
        part_name = f"{part.upper()} PARTITION ({n_c} CLAIMS, {len(sub)} PAIRS)"
        
    c2 = sub[sub['human_ground_truth_label'] == 'class_2']
    c1 = sub[sub['human_ground_truth_label'] == 'class_1']
    c0a = sub[sub['human_ground_truth_label'] == 'class_0a']
    c0b = sub[sub['human_ground_truth_label'] == 'class_0b']
    non_c2 = sub[sub['human_ground_truth_label'] != 'class_2']
    
    # Counts
    tp = c2['final_accept'].sum()
    fn = len(c2) - tp
    fp = non_c2['final_accept'].sum()
    tn = len(non_c2) - fp
    
    rec = tp / max(len(c2), 1)
    prec = tp / max(tp + fp, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    spec = tn / max(len(non_c2), 1)
    
    c1_rej = (~c1['final_accept']).mean()
    c0a_rej = (~c0a['final_accept']).mean()
    c0b_rej = (~c0b['final_accept']).mean()
    
    rec_w = get_wilson_ci(tp, len(c2))
    prec_w = get_wilson_ci(tp, tp + fp)
    spec_w = get_wilson_ci(tn, len(non_c2))
    
    rec_boot, prec_boot, f1_boot = fast_claim_clustered_ci(sub)
    
    print("\n" + "=" * 90)
    print(f"--- {part_name} ---")
    print("=" * 90)
    print(f"Class Composition: Class 2 = {len(c2)}, Class 1 = {len(c1)}, Class 0a = {len(c0a)}, Class 0b = {len(c0b)}")
    print(f"Confusion Matrix:  TP = {tp}, FP = {fp}, FN = {fn}, TN = {tn}")
    print(f"Key Metrics:")
    print(f"  Class 2 Recall:    {rec*100:5.2f}% ({tp}/{len(c2)}) | 95% Wilson CI: [{rec_w[0]*100:.2f}%, {rec_w[1]*100:.2f}%] | Claim-Clustered CI: [{rec_boot[0]*100:.2f}%, {rec_boot[1]*100:.2f}%]")
    print(f"  Class 2 Precision: {prec*100:5.2f}% ({tp}/{tp+fp}) | 95% Wilson CI: [{prec_w[0]*100:.2f}%, {prec_w[1]*100:.2f}%] | Claim-Clustered CI: [{prec_boot[0]*100:.2f}%, {prec_boot[1]*100:.2f}%]")
    print(f"  Class 2 F1-Score:  {f1:6.4f}               | Claim-Clustered CI: [{f1_boot[0]:.4f}, {f1_boot[1]:.4f}]")
    print(f"  Negative Specificity:{spec*100:5.2f}% ({tn}/{len(non_c2)}) | 95% Wilson CI: [{spec_w[0]*100:.2f}%, {spec_w[1]*100:.2f}%]")
    print(f"Negative Rejection Breakdown:")
    print(f"  Class 1 Rejection (Different Warrant): {c1_rej*100:5.2f}% ({(~c1['final_accept']).sum()}/{len(c1)})")
    print(f"  Class 0a Rejection (Opposite Stance):  {c0a_rej*100:5.2f}% ({(~c0a['final_accept']).sum()}/{len(c0a)})")
    print(f"  Class 0b Rejection (Off-Topic):        {c0b_rej*100:5.2f}% ({(~c0b['final_accept']).sum()}/{len(c0b)})")
    
    # Stage Funnel
    s1_c2 = c2['stage1_pass'].sum()
    s2a_c2 = c2['stage2a_pass'].sum()
    s2b_c2 = c2['stage2b_pass'].sum()
    
    s1_all = sub['stage1_pass'].sum()
    s2a_all = sub['stage2a_pass'].sum()
    s2b_all = sub['stage2b_pass'].sum()
    
    print(f"\nStage Filtering Funnel:")
    print(f"  Total Candidates:      {len(sub):3d} | Class 2: {len(c2):2d} (100.0%)")
    print(f"  Passed Stage 1 (tau):  {s1_all:3d} ({s1_all/len(sub)*100:5.1f}%) | Class 2: {s1_c2:2d} ({s1_c2/len(c2)*100:5.1f}%)")
    print(f"  Passed Stage 2a (St):  {s2a_all:3d} ({s2a_all/len(sub)*100:5.1f}%) | Class 2: {s2a_c2:2d} ({s2a_c2/len(c2)*100:5.1f}%)")
    print(f"  Passed Stage 2b (Pr):  {s2b_all:3d} ({s2b_all/len(sub)*100:5.1f}%) | Class 2: {s2b_c2:2d} ({s2b_c2/len(c2)*100:5.1f}%)")
    
    report_dict[part] = {
        "claims_count": int(sub['claim_id'].nunique()),
        "pairs_count": int(len(sub)),
        "class_composition": {"c2": int(len(c2)), "c1": int(len(c1)), "c0a": int(len(c0a)), "c0b": int(len(c0b))},
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "recall": float(rec), "recall_ci_wilson": rec_w, "recall_ci_clustered": rec_boot,
        "precision": float(prec), "precision_ci_wilson": prec_w, "precision_ci_clustered": prec_boot,
        "f1": float(f1), "f1_ci_clustered": f1_boot,
        "specificity": float(spec), "specificity_ci_wilson": spec_w,
        "rejection_rates": {"c1": float(c1_rej), "c0a": float(c0a_rej), "c0b": float(c0b_rej)},
        "funnel": {
            "initial": {"total": int(len(sub)), "c2": int(len(c2))},
            "stage1": {"total": int(s1_all), "c2": int(s1_c2)},
            "stage2a": {"total": int(s2a_all), "c2": int(s2a_c2)},
            "stage2b": {"total": int(s2b_all), "c2": int(s2b_c2)}
        }
    }

out_path = "results/calibration/frozen_verifier_341_benchmark_evaluation.json"
with open(out_path, "w") as f:
    json.dump(report_dict, f, indent=2)
print(f"\nSaved complete benchmark evaluation report to: {out_path}")
