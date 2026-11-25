"""
Comprehensive Evaluation of Stage B Human Adjudication (N = 50 Pairs)
Calculates exact confusion matrices, per-class precision/recall/F1,
and compares Multi-LLM Jury consensus vs. Automated Heuristic against true Human Ground Truth.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score, f1_score, accuracy_score

def main():
    csv_path = "data/calibration/stage_b_50_human_adjudicated.csv"
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} human-adjudicated pairs from {csv_path}")
    
    # 1. Clean columns
    y_human = df["human_label"].str.lower().tolist()
    y_heur = df["true_heuristic_class"].str.lower().tolist()
    
    # Jury predictions: provisional / majority label
    # In df, we have provisional_label
    y_jury = df["provisional_label"].str.lower().tolist()
    
    y_gemini = [str(x).lower() if pd.notna(x) else "ambiguous" for x in df["Gemini-3-Flash_label"]]
    y_gpt = [str(x).lower() if pd.notna(x) else "ambiguous" for x in df["GPT-OSS-120B_label"]]
    y_qwen = [str(x).lower() if pd.notna(x) else "ambiguous" for x in df["Qwen-3.8-27B_label"]]
    
    all_classes = ["class_2", "class_1", "class_0a", "class_0b", "ambiguous"]
    concrete_classes = ["class_2", "class_1", "class_0a", "class_0b"]
    
    print("\n" + "=" * 80)
    print("1. MULTI-LLM JURY VS. HUMAN GROUND TRUTH (N = 50)")
    print("=" * 80)
    jury_acc = accuracy_score(y_human, y_jury)
    jury_macro_f1 = f1_score(y_human, y_jury, average="macro")
    jury_kappa = cohen_kappa_score(y_human, y_jury)
    
    print(f"Overall Jury Accuracy: {jury_acc * 100:.2f}% ({sum(np.array(y_human) == np.array(y_jury))} / {len(y_human)})")
    print(f"Jury Macro-F1: {jury_macro_f1:.4f}")
    print(f"Cohen's Kappa (Jury vs. Human): {jury_kappa:.4f}")
    
    print("\nJury Classification Report against Human Ground Truth:")
    print(classification_report(y_human, y_jury, labels=all_classes, digits=4, zero_division=0))
    
    cm_jury = confusion_matrix(y_human, y_jury, labels=all_classes)
    print("\nJury Confusion Matrix (Rows: Human Ground Truth, Cols: Jury Consensus):")
    print(pd.DataFrame(cm_jury, index=all_classes, columns=all_classes))
    
    print("\n" + "=" * 80)
    print("2. UNANIMOUS VS. MAJORITY BREAKDOWN")
    print("=" * 80)
    unanimous_mask = df["consensus_status"] == "3/3 UNANIMOUS"
    majority_mask = df["consensus_status"] == "2/3 MAJORITY"
    ambig_mask = df["consensus_status"].str.contains("AMBIGUOUS", na=False)
    
    print(f"Unanimous Pairs in 50 Sample: {unanimous_mask.sum()}")
    if unanimous_mask.sum() > 0:
        u_human = np.array(y_human)[unanimous_mask]
        u_jury = np.array(y_jury)[unanimous_mask]
        u_acc = accuracy_score(u_human, u_jury)
        print(f"  Unanimous Agreement with Human: {u_acc * 100:.2f}% ({sum(u_human == u_jury)} / {len(u_human)})")
        
    print(f"\nMajority Pairs in 50 Sample: {majority_mask.sum()}")
    if majority_mask.sum() > 0:
        m_human = np.array(y_human)[majority_mask]
        m_jury = np.array(y_jury)[majority_mask]
        m_acc = accuracy_score(m_human, m_jury)
        print(f"  Majority Agreement with Human: {m_acc * 100:.2f}% ({sum(m_human == m_jury)} / {len(m_human)})")
        
    print(f"\nAmbiguous / Flagged Pairs in 50 Sample: {ambig_mask.sum()}")
    if ambig_mask.sum() > 0:
        a_human = np.array(y_human)[ambig_mask]
        a_jury = np.array(y_jury)[ambig_mask]
        a_acc = accuracy_score(a_human, a_jury)
        print(f"  Ambiguous Agreement with Human: {a_acc * 100:.2f}% ({sum(a_human == a_jury)} / {len(a_human)})")
        
    print("\n" + "=" * 80)
    print("3. AUTOMATED HEURISTIC BASELINE VS. HUMAN GROUND TRUTH (N = 50)")
    print("=" * 80)
    heur_acc = accuracy_score(y_human, y_heur)
    heur_macro_f1 = f1_score(y_human, y_heur, labels=concrete_classes, average="macro")
    heur_kappa = cohen_kappa_score(y_human, y_heur)
    
    print(f"Heuristic Baseline Accuracy: {heur_acc * 100:.2f}% ({sum(np.array(y_human) == np.array(y_heur))} / {len(y_human)})")
    print(f"Heuristic Macro-F1: {heur_macro_f1:.4f}")
    print(f"Heuristic Cohen's Kappa: {heur_kappa:.4f}")
    
    print("\nHeuristic Classification Report against Human Ground Truth:")
    print(classification_report(y_human, y_heur, labels=all_classes, digits=4, zero_division=0))
    
    cm_heur = confusion_matrix(y_human, y_heur, labels=all_classes)
    print("\nHeuristic Confusion Matrix (Rows: Human Ground Truth, Cols: Heuristic):")
    print(pd.DataFrame(cm_heur, index=all_classes, columns=all_classes))
    
    print("\n" + "=" * 80)
    print("4. INDIVIDUAL JUDGE ACCURACY VS. HUMAN GROUND TRUTH")
    print("=" * 80)
    gem_acc = accuracy_score(y_human, y_gemini)
    gpt_acc = accuracy_score(y_human, y_gpt)
    qwen_acc = accuracy_score(y_human, y_qwen)
    
    print(f"Judge 1 (Gemini 3.1 Flash Lite): Accuracy = {gem_acc * 100:.2f}% ({sum(np.array(y_human) == np.array(y_gemini))}/{len(y_human)})")
    print(f"Judge 2 (GPT-OSS 120B):          Accuracy = {gpt_acc * 100:.2f}% ({sum(np.array(y_human) == np.array(y_gpt))}/{len(y_human)})")
    print(f"Judge 3 (Qwen 3.8-27B):          Accuracy = {qwen_acc * 100:.2f}% ({sum(np.array(y_human) == np.array(y_qwen))}/{len(y_human)})")
    
    # Save JSON report
    report = {
        "sample_size": len(df),
        "evaluation_stage": "Stage B Human Validation (N = 50)",
        "multi_llm_jury_performance": {
            "accuracy": round(float(jury_acc), 4),
            "macro_f1": round(float(jury_macro_f1), 4),
            "cohen_kappa_vs_human": round(float(jury_kappa), 4),
            "unanimous_accuracy": round(float(u_acc), 4) if unanimous_mask.sum() > 0 else None,
            "majority_accuracy": round(float(m_acc), 4) if majority_mask.sum() > 0 else None
        },
        "heuristic_baseline_performance": {
            "accuracy": round(float(heur_acc), 4),
            "macro_f1": round(float(heur_macro_f1), 4),
            "cohen_kappa_vs_human": round(float(heur_kappa), 4)
        },
        "individual_judge_accuracy": {
            "gemini_3_1_flash_lite": round(float(gem_acc), 4),
            "gpt_oss_120b": round(float(gpt_acc), 4),
            "qwen_3_8_27b": round(float(qwen_acc), 4)
        },
        "confusion_matrix_jury_vs_human": {
            "labels": all_classes,
            "matrix": cm_jury.tolist()
        },
        "confusion_matrix_heuristic_vs_human": {
            "labels": all_classes,
            "matrix": cm_heur.tolist()
        }
    }
    
    out_json = "results/calibration/stage_b_human_validation_50_report.json"
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved comprehensive validation report to {out_json}")

if __name__ == "__main__":
    main()
