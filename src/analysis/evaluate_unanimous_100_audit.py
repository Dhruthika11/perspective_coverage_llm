"""
Comprehensive Evaluation of the 100-Pair Unanimous Jury Audit
Evaluates:
1. Overall 100-pair unanimous accuracy and agreement metrics (Accuracy, Cohen's Kappa, Macro-F1).
2. Subgroup 1: 60-pair natural distribution sample.
3. Subgroup 2: 40-pair class booster sample (focus on Class 2).
4. Per-class precision, recall, F1, and confusion matrices.
5. Cumulative metrics combining all human audits to date (Stage A N=50 + Stage B N=50 + Unanimous Audit N=100 = 200 Total Human Ground-Truth Pairs).
"""

import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, cohen_kappa_score

def eval_subset(df, name):
    classes = ["class_2", "class_1", "class_0a", "class_0b", "ambiguous"]
    y_true = df["human_label"].tolist()
    y_pred = df["proposed_label"].tolist()
    
    acc = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    
    report = classification_report(y_true, y_pred, labels=classes, output_dict=True, zero_division=0)
    
    return {
        "name": name,
        "n": len(df),
        "overall_accuracy": round(acc, 4),
        "cohen_kappa": round(kappa, 4),
        "macro_f1": round(macro_f1, 4),
        "matches": int((df["human_label"] == df["proposed_label"]).sum()),
        "mismatches": int((df["human_label"] != df["proposed_label"]).sum()),
        "per_class": {
            cls: {
                "precision": round(report[cls]["precision"], 4),
                "recall": round(report[cls]["recall"], 4),
                "f1": round(report[cls]["f1-score"], 4),
                "support": int(report[cls]["support"])
            }
            for cls in classes
        },
        "confusion_matrix": cm_df.to_dict()
    }

def main():
    csv_path = "unanimous_100_audit_queue.csv" if os.path.exists("unanimous_100_audit_queue.csv") else "data/calibration/unanimous_100_audit_queue.csv"
    df = pd.read_csv(csv_path)
    
    # 1. Full 100 pairs
    full_eval = eval_subset(df, "Full 100-Pair Unanimous Audit")
    
    # 2. Subgroup 1: 60 natural distribution
    df_60 = df[df["sampling_subgroup"] == "natural_distribution_60"]
    eval_60 = eval_subset(df_60, "Subgroup 1: 60-Pair Natural Distribution Sample")
    
    # 3. Subgroup 2: 40 booster
    df_40 = df[df["sampling_subgroup"] == "class_coverage_booster_40"]
    eval_40 = eval_subset(df_40, "Subgroup 2: 40-Pair Class Coverage Booster")
    
    # 4. Cumulative Unanimous Statistics Across Entire Project:
    # Previously audited unanimous pairs:
    # Stage A: 45 unanimous pairs, 45 agreed (100%)
    # Stage B: 6 unanimous pairs, 6 agreed (100%)
    # This Audit: 100 unanimous pairs, 99 agreed (99.0%)
    # Total Unanimous Human Ground Truth: 151 pairs, 150 agreed = 99.34%
    cumulative_unanimous = {
        "stage_a_unanimous_audited": 45,
        "stage_a_unanimous_correct": 45,
        "stage_b_unanimous_audited": 6,
        "stage_b_unanimous_correct": 6,
        "new_cohort_unanimous_audited": 100,
        "new_cohort_unanimous_correct": full_eval["matches"],
        "total_unanimous_audited_to_date": 45 + 6 + 100,
        "total_unanimous_correct_to_date": 45 + 6 + full_eval["matches"],
        "cumulative_unanimous_accuracy": round((45 + 6 + full_eval["matches"]) / (45 + 6 + 100), 4)
    }
    
    out_dict = {
        "full_100_audit": full_eval,
        "subgroup_60_natural": eval_60,
        "subgroup_40_booster": eval_40,
        "cumulative_unanimous_validation": cumulative_unanimous
    }
    
    out_path = "results/calibration/unanimous_100_audit_evaluation.json"
    with open(out_path, "w") as f:
        json.dump(out_dict, f, indent=2)
    print(f"Saved evaluation JSON to {out_path}")
    
    print("\n" + "="*80)
    print(f"FULL 100-PAIR UNANIMOUS AUDIT RESULTS:")
    print(f"Overall Accuracy: {full_eval['overall_accuracy']*100:.2f}% ({full_eval['matches']}/100)")
    print(f"Cohen's Kappa: {full_eval['cohen_kappa']:.4f} | Macro-F1: {full_eval['macro_f1']:.4f}")
    print("\nPer-Class Breakdown:")
    for cls, m in full_eval["per_class"].items():
        print(f"  {cls:10s} -> Precision: {m['precision']*100:.2f}% | Recall: {m['recall']*100:.2f}% | F1: {m['f1']:.4f} | Support: {m['support']}")
        
    print("\n" + "="*80)
    print(f"SUBGROUP 1 (60 Pairs - Natural Distribution):")
    print(f"Accuracy: {eval_60['overall_accuracy']*100:.2f}% ({eval_60['matches']}/60) | Macro-F1: {eval_60['macro_f1']:.4f}")
    
    print("\n" + "="*80)
    print(f"SUBGROUP 2 (40 Pairs - Class Booster, 22 Class 2 pairs):")
    print(f"Accuracy: {eval_40['overall_accuracy']*100:.2f}% ({eval_40['matches']}/40) | Macro-F1: {eval_40['macro_f1']:.4f}")
    
    print("\n" + "="*80)
    print(f"CUMULATIVE UNANIMOUS ACCURACY (Stage A + Stage B + New 100):")
    print(f"Total Unanimous Audited: {cumulative_unanimous['total_unanimous_audited_to_date']} pairs")
    print(f"Total Correct: {cumulative_unanimous['total_unanimous_correct_to_date']} / {cumulative_unanimous['total_unanimous_audited_to_date']}")
    print(f"Cumulative Unanimous Precision / Accuracy: {cumulative_unanimous['cumulative_unanimous_accuracy']*100:.2f}%")

if __name__ == "__main__":
    main()
