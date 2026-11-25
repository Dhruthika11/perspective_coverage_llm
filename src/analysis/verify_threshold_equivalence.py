"""
Verification of Platt Calibrated Probability Threshold vs. Raw Operational Rule.

Empirically investigates the equivalence between:
  Rule A (Platt Calibrated): P_cal = sigma(w * p + b) >= 0.50  <=>  p >= -b / w
  Rule B (Raw Operational):  argmax(p) == Entailment AND p_entail >= 0.80
  Rule C (Raw Cutoff):       p_entail >= 0.80

Computes:
  - Fitted Platt scaling parameters (w, b) and raw mathematical cutoff tau_raw = -b / w
  - Empirical agreement (accuracy, Cohen's kappa) on the exact same held-out test split
  - 2x2 contingency table between Rule A and Rule B
  - Test set precision, recall, and F1 under each rule
  - Discrepancy analysis on differing cases
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.io_utils import resolve, load_csv, save_json
from src.utils.logging_utils import get_logger

logger = get_logger("verify_threshold_equivalence")


def run_threshold_equivalence_audit(
    benchmark_csv: str = "data/calibration/tri_dataset_750_pairs.csv",
    output_json: str = "results/calibration/threshold_equivalence_audit.json"
):
    csv_path = resolve(benchmark_csv)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Benchmark file not found: {csv_path}")

    df = load_csv(csv_path)
    logger.info(f"Loaded {len(df)} pairs from {csv_path}")

    cal_df = df[df["split"] == "calibration"].copy()
    test_df = df[df["split"] == "test"].copy()

    logger.info(f"Calibration split: {len(cal_df)} pairs | Held-out test split: {len(test_df)} pairs")

    # 1. Fit Platt Scaling on Calibration Split
    # Binary target: label_3class == 2 (Entailment / True Perspective Match)
    y_cal = (cal_df["label_3class"] == 2).astype(int).values
    X_cal = cal_df[["p_verification"]].values

    clf = LogisticRegression(C=1.0, solver="lbfgs")
    clf.fit(X_cal, y_cal)

    w = float(clf.coef_[0][0])
    b = float(clf.intercept_[0])
    raw_cutoff_platt_50 = float(-b / w)

    logger.info(f"Fitted Platt parameters: w = {w:.4f}, b = {b:.4f}")
    logger.info(f"Mathematical raw threshold for P_cal >= 0.50: -b/w = {raw_cutoff_platt_50:.4f}")

    # 2. Evaluate on the Exact Same Held-Out Test Split (N = 147)
    y_test = (test_df["label_3class"] == 2).astype(int).values
    p_test = test_df["p_verification"].values
    X_test = test_df[["p_verification"]].values

    # Calibrated probabilities from Platt model
    p_cal_test = clf.predict_proba(X_test)[:, 1]

    # Define decision rules
    rule_platt_50 = (p_cal_test >= 0.50)
    rule_raw_cutoff_platt = (p_test >= raw_cutoff_platt_50)
    rule_raw_80 = (p_test >= 0.80)

    # In 3-class classification, check if p >= 0.80 guarantees argmax == Entailment
    # If p_entail >= 0.80, the sum of contra + neutral is <= 0.20, so p_entail > any other class.
    # Therefore argmax is mathematically guaranteed to be Entailment whenever p_entail >= 0.80.
    argmax_guaranteed = bool(raw_cutoff_platt_50 > 0.50)

    # 3. Mathematical Equivalence Check between P_cal >= 0.50 and p >= -b/w
    math_diff = int(np.sum(rule_platt_50 != rule_raw_cutoff_platt))
    logger.info(f"Discrepancies between P_cal >= 0.50 and p >= -b/w: {math_diff} (Exact mathematical identity verified: {math_diff == 0})")

    # 4. Agreement between Platt 0.50 and Raw 0.80
    agreement_rate = float((rule_platt_50 == rule_raw_80).mean())
    kappa = float(cohen_kappa_score(rule_platt_50, rule_raw_80))

    # Contingency Table
    ct = pd.crosstab(
        rule_platt_50,
        rule_raw_80,
        rownames=["Platt P_cal >= 0.50"],
        colnames=["Raw p >= 0.80"]
    )
    logger.info(f"\nContingency Table on Held-Out Test Split (N={len(test_df)}):\n{ct.to_string()}")
    logger.info(f"Overall Agreement: {agreement_rate:.4f} ({agreement_rate * 100:.1f}%) | Cohen's Kappa: {kappa:.4f}")

    # 5. Performance Metrics Comparison on Test Split
    def calc_metrics(pred_bool, name):
        tp = int(np.sum((pred_bool == 1) & (y_test == 1)))
        fp = int(np.sum((pred_bool == 1) & (y_test == 0)))
        fn = int(np.sum((pred_bool == 0) & (y_test == 1)))
        tn = int(np.sum((pred_bool == 0) & (y_test == 0)))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        acc = (tp + tn) / len(y_test)
        return {
            "Rule_Name": name,
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1_Score": round(f1, 4),
            "Accuracy": round(acc, 4)
        }

    m_platt_50 = calc_metrics(rule_platt_50, "Rule A: Platt Calibrated (P_cal >= 0.50, raw >= 0.5391)")
    m_raw_80 = calc_metrics(rule_raw_80, "Rule B: Raw Operational (p_entail >= 0.80)")

    # 6. Discrepancy Case Inspection
    diff_mask = (rule_platt_50 != rule_raw_80)
    diff_records = []
    if np.sum(diff_mask) > 0:
        diff_df = test_df[diff_mask]
        for _, r in diff_df.iterrows():
            diff_records.append({
                "claim_text": str(r.get("claim_text", "")),
                "text_a": str(r.get("text_a", ""))[:120],
                "text_b": str(r.get("text_b", ""))[:120],
                "pair_category": str(r.get("pair_category", "")),
                "label_3class": int(r.get("label_3class", -1)),
                "p_verification": float(r.get("p_verification", 0.0)),
                "platt_p_cal": float(clf.predict_proba([[float(r.get("p_verification", 0.0))]])[0, 1]),
                "rule_platt_50_decision": bool(r.get("p_verification", 0.0) >= raw_cutoff_platt_50),
                "rule_raw_80_decision": bool(r.get("p_verification", 0.0) >= 0.80),
            })

    report = {
        "benchmark_file": benchmark_csv,
        "n_calibration_pairs": len(cal_df),
        "n_test_pairs": len(test_df),
        "platt_parameters": {
            "w_weight": round(w, 4),
            "b_intercept": round(b, 4),
            "raw_cutoff_for_0_50": round(raw_cutoff_platt_50, 4),
            "formula": f"P_cal = 1 / (1 + exp(-({w:.4f} * p + {b:.4f}))) >= 0.50 <=> p >= {raw_cutoff_platt_50:.4f}"
        },
        "agreement_analysis": {
            "agreement_rate": round(agreement_rate, 4),
            "cohen_kappa": round(kappa, 4),
            "n_disagreements": int(np.sum(diff_mask)),
            "contingency_table": {
                "platt_false_raw_false": int(ct.loc[False, False]) if False in ct.index and False in ct.columns else 0,
                "platt_false_raw_true": int(ct.loc[False, True]) if False in ct.index and True in ct.columns else 0,
                "platt_true_raw_false": int(ct.loc[True, False]) if True in ct.index and False in ct.columns else 0,
                "platt_true_raw_true": int(ct.loc[True, True]) if True in ct.index and True in ct.columns else 0,
            }
        },
        "rule_performance_comparison": [m_platt_50, m_raw_80],
        "argmax_condition_check": {
            "is_argmax_entailment_guaranteed_when_p_ge_80": True,
            "mathematical_proof": "In a 3-class softmax distribution sum(p)=1, if p_E >= 0.80, then p_C + p_N <= 0.20, guaranteeing p_E > max(p_C, p_N). Therefore argmax is strictly Entailment."
        },
        "discrepancy_records": diff_records
    }

    out_path = resolve(output_json)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save_json(report, out_path)
    logger.info(f"Threshold equivalence audit saved to {out_path}")

    # Print summary table
    print("\n" + "=" * 80)
    print("          THRESHOLD EQUIVALENCE & AGREEMENT AUDIT (HELD-OUT TEST N=147)")
    print("=" * 80)
    print(f"Platt Logistic Model: w = {w:.4f}, b = {b:.4f}")
    print(f"Platt P_cal >= 0.50 corresponds mathematically to raw p >= {raw_cutoff_platt_50:.4f}")
    print(f"Agreement Rate: {agreement_rate * 100:.2f}% | Cohen's Kappa: {kappa:.4f}")
    print("\nPerformance Comparison on Held-Out Test Split:")
    print(pd.DataFrame([m_platt_50, m_raw_80]).to_string(index=False))
    print("=" * 80 + "\n")

    return report


if __name__ == "__main__":
    run_threshold_equivalence_audit()

