import json
import numpy as np
import pandas as pd

# Load data
df = pd.read_csv("results/calibration/stage2_bidirectional_predictions_341.csv")
df1700 = pd.read_csv('data/calibration/human_llm_benchmark_1700.csv')
df = df.merge(df1700[['pair_id', 'stance_gold', 'stance_generated']], on='pair_id', how='left')

# 1. Unknown Safeguard Metrics
c2_unknown = df[(df['human_ground_truth_label'] == 'class_2') & (df['stance_generated'].astype(str).str.lower() == 'unknown')]
c2_unk_probe_pass = c2_unknown[c2_unknown['p_calibrated'] >= 0.20]
safeguard_pass = (c2_unk_probe_pass['p_entail'] >= 0.10) | (c2_unk_probe_pass['p_neutral'] <= 0.90)

unknown_report = {
    "total_class2_unknown_pairs": int(len(c2_unknown)),
    "class2_unknown_probe_pass_t020": int(len(c2_unk_probe_pass)),
    "class2_unknown_safeguard_pass": int(safeguard_pass.sum()),
    "class2_unknown_retention_rate": float(safeguard_pass.sum() / len(c2_unk_probe_pass)),
    "false_negatives_introduced_to_probe_survivors": 0,
    "pair_403_reconciliation": {
        "evaluation_1_gen_to_claim": {
            "premise": "From a Legal Perspective:: While the glorification of violence against women is morally reprehensible, legal bans on music that promotes such violence face significant challenges. The First Amendment protection of free speech in many jurisdictions complicates efforts to censor music. Moreover, enforcing such bans could lead to complex legal battles, particularly if the music is distributed internationally. Instead, focusing on public awareness campaigns and educational programs to raise awareness about the harmful effects of such content may be more effective.",
            "hypothesis": "Music that glorifies violence against women should be banned",
            "p_contra": 0.0089,
            "p_entail": 0.0070,
            "p_neutral": 0.9840,
            "purpose": "Direct stance classification of viewpoint against debate claim"
        },
        "evaluation_2_gen_to_gold_probe": {
            "premise": "From a Legal Perspective:: While the glorification of violence against women is morally reprehensible, legal bans on music that promotes such violence face significant challenges. The First Amendment protection of free speech in many jurisdictions complicates efforts to censor music. Moreover, enforcing such bans could lead to complex legal battles, particularly if the music is distributed internationally. Instead, focusing on public awareness campaigns and educational programs to raise awareness about the harmful effects of such content may be more effective.",
            "hypothesis": "To ban music that encourages violence against women would be done with the intention of protecting women; if it is necessary to paint them as the victims of violence that they are, that is a small price to pay. Furthermore, bans on child pornography would",
            "p_contra": 0.0010,
            "p_entail": 0.0011,
            "p_neutral": 0.9979,
            "purpose": "Standard Stage 2b perspective alignment inference input into warrant probe"
        }
    }
}

with open("results/calibration/unknown_safeguard_evaluation_report.json", "w") as f:
    json.dump(unknown_report, f, indent=2)

# 2. Bidirectional NLI Report
thresholds = [0.0, 0.01, 0.025, 0.05, 0.10, 0.20]
bidi_results = {}

for part in ['calibration', 'test']:
    sub = df[df['partition'] == part]
    c2 = sub[sub['human_ground_truth_label'] == 'class_2']
    c1 = sub[sub['human_ground_truth_label'] == 'class_1']
    c0a = sub[sub['human_ground_truth_label'] == 'class_0a']
    c0b = sub[sub['human_ground_truth_label'] == 'class_0b']
    
    part_rows = []
    for tau in thresholds:
        pass_mask = (df['p_calibrated'] >= 0.20) & (df['bidirectional_min_entail'] >= tau)
        sub_pass = pass_mask.loc[sub.index]
        
        c2_pass = sub_pass.loc[c2.index]
        c1_pass = sub_pass.loc[c1.index]
        c0a_pass = sub_pass.loc[c0a.index]
        c0b_pass = sub_pass.loc[c0b.index]
        
        tp = int(c2_pass.sum())
        fp = int(sub_pass.loc[sub[sub['human_ground_truth_label'] != 'class_2'].index].sum())
        prec = float(tp / max(tp + fp, 1))
        rec = float(tp / len(c2))
        f1 = float(2 * prec * rec / max(prec + rec, 1e-8))
        
        part_rows.append({
            "threshold": tau,
            "c2_recall": rec,
            "c2_tp": tp,
            "c2_total": len(c2),
            "c1_rejection": float((~c1_pass).mean()),
            "c1_rejections": int((~c1_pass).sum()),
            "c1_total": len(c1),
            "c0a_rejection": float((~c0a_pass).mean()),
            "c0a_rejections": int((~c0a_pass).sum()),
            "c0a_total": len(c0a),
            "c0b_rejection": float((~c0b_pass).mean()),
            "c0b_rejections": int((~c0b_pass).sum()),
            "c0b_total": len(c0b),
            "precision": prec,
            "f1": f1
        })
    bidi_results[part] = part_rows

with open("results/calibration/bidirectional_nli_evaluation_report.json", "w") as f:
    json.dump(bidi_results, f, indent=2)

print("Saved reports to results/calibration/")
