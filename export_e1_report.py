import json
import pandas as pd
import numpy as np

# Load data and config
df = pd.read_csv('results/calibration/stage2_cross_encoder_predictions_341.csv')
df1700 = pd.read_csv('data/calibration/human_llm_benchmark_1700.csv')
df = df.merge(df1700[['pair_id', 'stance_gold', 'stance_generated']], on='pair_id', how='left')

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

df['stage1_pass'] = df['cosine_similarity'] >= tau_s1
g_st = df['stance_gold'].astype(str).str.lower().str.strip()
h_st = df['stance_generated'].astype(str).str.lower().str.strip()
is_contra = ((g_st == 'support') & (h_st == 'undermine')) | ((g_st == 'undermine') & (h_st == 'support'))
df['stage2a_pass'] = df['stage1_pass'] & (~is_contra)

z_cos = (df['cosine_similarity'] - scaler_mean[0]) / scaler_scale[0]
z_ent = (df['p_entail'] - scaler_mean[1]) / scaler_scale[1]
z_neu = (df['p_neutral'] - scaler_mean[2]) / scaler_scale[2]
df['probe_prob'] = 1.0 / (1.0 + np.exp(-(z_cos * w_cos + z_ent * w_ent + z_neu * w_neu + intercept)))
df['final_accept'] = df['stage2a_pass'] & (df['probe_prob'] >= t_probe)

# 1. Five Class 2 misses
c2_misses = df[(df['human_ground_truth_label'] == 'class_2') & (~df['final_accept'])].copy()
misses_list = []
for idx, r in c2_misses.iterrows():
    misses_list.append({
        "pair_id": int(r.pair_id),
        "partition": str(r.partition),
        "claim_id": int(r.claim_id),
        "claim_text": str(r.claim_text),
        "gold_perspective": str(r.gold_perspective_text),
        "generated_viewpoint": str(r.generated_viewpoint_text),
        "stance_gold": str(r.stance_gold),
        "stance_generated": str(r.stance_generated),
        "cosine_similarity": float(r.cosine_similarity),
        "p_entail": float(r.p_entail),
        "p_neutral": float(r.p_neutral),
        "probe_score": float(r.probe_prob),
        "threshold": float(t_probe)
    })

# 2. Accepted Class 1 errors in calibration
cal_df = df[df['partition'] == 'calibration'].copy()
c1_cal_accepted = cal_df[(cal_df['human_ground_truth_label'] == 'class_1') & cal_df['final_accept']].copy()
c1_list = []
for idx, r in c1_cal_accepted.iterrows():
    c1_list.append({
        "pair_id": int(r.pair_id),
        "claim_id": int(r.claim_id),
        "claim_text": str(r.claim_text),
        "gold_perspective": str(r.gold_perspective_text),
        "generated_viewpoint": str(r.generated_viewpoint_text),
        "cosine_similarity": float(r.cosine_similarity),
        "p_entail": float(r.p_entail),
        "p_neutral": float(r.p_neutral),
        "probe_score": float(r.probe_prob)
    })

# 3. Fine threshold sweep
thresholds_fine = [0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30, 0.35, 0.40, 0.50]
sweep_dict = {}

for name, d_part in [("calibration", cal_df),
                     ("training", df[df['partition'] == 'train'].copy()),
                     ("train_plus_cal", df[df['partition'].isin(['train', 'calibration'])].copy())]:
    c2_p = d_part[d_part['human_ground_truth_label'] == 'class_2']
    c1_p = d_part[d_part['human_ground_truth_label'] == 'class_1']
    non_c2_p = d_part[d_part['human_ground_truth_label'] != 'class_2']
    
    rows = []
    for th in thresholds_fine:
        pass_mask = d_part['stage2a_pass'] & (d_part['probe_prob'] >= th)
        tp = int(pass_mask.loc[c2_p.index].sum())
        fp = int(pass_mask.loc[non_c2_p.index].sum())
        fn = int(len(c2_p) - tp)
        tn = int(len(non_c2_p) - fp)
        fp_c1 = int(pass_mask.loc[c1_p.index].sum())
        
        rec = float(tp / len(c2_p))
        prec = float(tp / max(tp + fp, 1))
        f1 = float(2 * prec * rec / max(prec + rec, 1e-8))
        c1_rej = float((~pass_mask.loc[c1_p.index]).mean())
        
        rows.append({
            "threshold": th,
            "c2_recall": rec, "tp": tp, "fn": fn,
            "c2_precision": prec, "fp": fp, "fp_c1": fp_c1,
            "f1": f1,
            "c1_rejection": c1_rej
        })
    sweep_dict[name] = rows

out = {
    "experiment": "E1 — Threshold sensitivity and error audit",
    "class2_misses_total": len(misses_list),
    "class2_misses": misses_list,
    "class1_accepted_calibration_total": len(c1_list),
    "class1_accepted_calibration": c1_list,
    "threshold_sweeps": sweep_dict
}

with open("results/calibration/experiment_e1_sensitivity_and_error_audit.json", "w") as f:
    json.dump(out, f, indent=2)

print("Saved results to results/calibration/experiment_e1_sensitivity_and_error_audit.json")
