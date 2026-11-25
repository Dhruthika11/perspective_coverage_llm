import json
import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint

df = pd.read_csv('data/audit/production_verifier_audit_sample_50.csv')

ct = pd.crosstab(df['final_decision'], df['human_audit_decision'])

# Metrics
n_acc = int((df['final_decision'] == 'ACCEPT_CLASS2_EQUIVALENT').sum())
tp = int(((df['final_decision'] == 'ACCEPT_CLASS2_EQUIVALENT') & (df['human_audit_decision'] == 'class_2')).sum())
fp_1 = int(((df['final_decision'] == 'ACCEPT_CLASS2_EQUIVALENT') & (df['human_audit_decision'] == 'class_1')).sum())
fp_0a = int(((df['final_decision'] == 'ACCEPT_CLASS2_EQUIVALENT') & (df['human_audit_decision'] == 'class_0a')).sum())
fp_0b = int(((df['final_decision'] == 'ACCEPT_CLASS2_EQUIVALENT') & (df['human_audit_decision'] == 'class_0b')).sum())

tot_c2 = int((df['human_audit_decision'] == 'class_2').sum())
tot_c1 = int((df['human_audit_decision'] == 'class_1').sum())
tot_c0a = int((df['human_audit_decision'] == 'class_0a').sum())
tot_c0b = int((df['human_audit_decision'] == 'class_0b').sum())

prec = tp / n_acc if n_acc > 0 else 0.0
rec = tp / tot_c2 if tot_c2 > 0 else 0.0
f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

w_prec_l, w_prec_h = proportion_confint(tp, n_acc, alpha=0.05, method='wilson')
w_rec_l, w_rec_h = proportion_confint(tp, tot_c2, alpha=0.05, method='wilson')

# Filter accuracies
s2a_rej = int((df['final_decision'] == 'REJECT_STAGE2A_STANCE').sum())
s2a_true_0a = int(((df['final_decision'] == 'REJECT_STAGE2A_STANCE') & (df['human_audit_decision'] == 'class_0a')).sum())

s1_rej = int((df['final_decision'] == 'REJECT_STAGE1').sum())
s1_true_0b = int(((df['final_decision'] == 'REJECT_STAGE1') & (df['human_audit_decision'] == 'class_0b')).sum())

neg_tot = tot_c1 + tot_c0a + tot_c0b
neg_correctly_rej = int((df[df['human_audit_decision'] != 'class_2']['final_decision'] != 'ACCEPT_CLASS2_EQUIVALENT').sum())
neg_specificity = neg_correctly_rej / neg_tot

report = {
    "audit_sample_size": len(df),
    "strata_breakdown": {
        "accepted_class2_evaluated": n_acc,
        "stage2b_warrant_rejections_evaluated": int((df['final_decision'] == 'REJECT_STAGE2B_DIFFERENT_WARRANT').sum()),
        "stage2a_stance_rejections_evaluated": s2a_rej,
        "stage1_retrieval_rejections_evaluated": s1_rej
    },
    "human_ground_truth_counts": {
        "class_2": tot_c2,
        "class_1": tot_c1,
        "class_0a": tot_c0a,
        "class_0b": tot_c0b
    },
    "class_2_performance": {
        "true_positives": tp,
        "false_positives_class_1": fp_1,
        "false_positives_class_0a": fp_0a,
        "false_positives_class_0b": fp_0b,
        "precision": round(float(prec), 4),
        "precision_95_wilson_ci": [round(float(w_prec_l), 4), round(float(w_prec_h), 4)],
        "recall": round(float(rec), 4),
        "recall_95_wilson_ci": [round(float(w_rec_l), 4), round(float(w_rec_h), 4)],
        "f1_score": round(float(f1), 4)
    },
    "filter_precisions": {
        "stage_2a_stance_filter_precision": round(float(s2a_true_0a / s2a_rej), 4),
        "stage_1_retrieval_filter_precision": round(float(s1_true_0b / s1_rej), 4),
        "negative_class_specificity": round(float(neg_specificity), 4)
    },
    "confusion_matrix": ct.to_dict()
}

out_path = 'results/calibration/production_verifier_audit_50_report.json'
with open(out_path, 'w') as f:
    json.dump(report, f, indent=2)

print(f"Saved complete audit report to {out_path}")
print(json.dumps(report, indent=2))
