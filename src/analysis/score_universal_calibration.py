"""
Scores the universal calibration dataset.
Calculates the single global similarity threshold (tau) that maximizes the F1-score
across all models and datasets, and computes performance metrics.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.io_utils import load_config, resolve, load_csv, save_json
from src.utils.logging_utils import get_logger

logger = get_logger("score_universal_calibration")


def main():
    parser = argparse.ArgumentParser(description="Score universal calibration pairs.")
    parser.add_argument(
        "--input", type=str, default="data/calibration/universal_calibration_pairs.csv",
        help="Path to the labeled universal calibration CSV file."
    )
    parser.add_argument(
        "--out-dir", type=str, default="results/calibration",
        help="Output directory for reports and figures."
    )
    args = parser.parse_args()

    input_path = resolve(args.input)
    if not os.path.exists(input_path):
        logger.error(f"Input file not found at: {input_path}")
        return

    df = load_csv(input_path)
    if "human_label" not in df.columns:
        logger.error("Missing 'human_label' column in CSV!")
        return

    # Filter for labeled rows
    labeled = df.dropna(subset=["human_label", "cosine_similarity"]).copy()
    labeled["human_label"] = pd.to_numeric(labeled["human_label"], errors="coerce")
    labeled = labeled.dropna(subset=["human_label"])

    if len(labeled) == 0:
        logger.error("No labeled rows found. Please fill in 1 or 0 under the human_label column.")
        return

    y_true = labeled["human_label"].astype(int).values
    y_score = labeled["cosine_similarity"].astype(float).values

    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())

    logger.info(f"Loaded {len(labeled)} labeled pairs: {n_pos} Positive (Matches), {n_neg} Negative (Non-matches)")

    if n_pos == 0 or n_neg == 0:
        logger.error(f"Both positive and negative labels required (got {n_pos} pos, {n_neg} neg).")
        return

    # 1. ROC Curve
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    youden_j = tpr - fpr
    best_youden_idx = int(np.argmax(youden_j))
    tau_youden = float(roc_thresholds[best_youden_idx])

    # 2. Precision-Recall Curve & F1-Max
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)

    f1_scores = np.where(
        (precision[:-1] + recall[:-1]) > 0,
        2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-12),
        0.0
    )
    best_f1_idx = int(np.argmax(f1_scores))
    tau_f1_max = float(pr_thresholds[best_f1_idx])
    best_f1 = float(f1_scores[best_f1_idx])
    best_prec = float(precision[best_f1_idx])
    best_rec = float(recall[best_f1_idx])

    # Plot ROC & PR
    out_dir = resolve(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    fig_path = os.path.join(out_dir, "universal_threshold_roc_pr.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
    axes[0].plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    axes[0].scatter([fpr[best_youden_idx]], [tpr[best_youden_idx]], color="red", zorder=5,
                    label=f"Youden-optimal tau = {tau_youden:.3f}")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("Universal ROC Curve")
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(recall, precision, color="teal", lw=2, label=f"PR (AP = {pr_auc:.3f})")
    axes[1].scatter([best_rec], [best_prec], color="crimson", zorder=5,
                    label=f"Max F1={best_f1:.3f} at tau = {tau_f1_max:.3f}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Universal Precision-Recall Curve")
    axes[1].legend(loc="lower left")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()

    # Per-Dataset Breakdown using the global tau
    dataset_breakdown = {}
    if "dataset" in labeled.columns:
        for ds_name, group in labeled.groupby("dataset"):
            g_true = group["human_label"].astype(int).values
            g_score = group["cosine_similarity"].astype(float).values
            g_pred = (g_score >= tau_f1_max).astype(int)
            acc = float((g_true == g_pred).mean())
            tp = int(((g_true == 1) & (g_pred == 1)).sum())
            fp = int(((g_true == 0) & (g_pred == 1)).sum())
            fn = int(((g_true == 1) & (g_pred == 0)).sum())
            p = tp / (tp + fp + 1e-12)
            r = tp / (tp + fn + 1e-12)
            f1 = 2 * p * r / (p + r + 1e-12)
            dataset_breakdown[ds_name] = {
                "n_samples": len(group),
                "accuracy": round(acc, 4),
                "precision": round(p, 4),
                "recall": round(r, 4),
                "f1": round(f1, 4),
            }

    report = {
        "n_pairs_total": len(labeled),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "recommended_tau": round(tau_f1_max, 4),
        "tau_f1_max": round(tau_f1_max, 4),
        "best_f1": round(best_f1, 4),
        "precision_at_tau": round(best_prec, 4),
        "recall_at_tau": round(best_rec, 4),
        "roc_auc": round(roc_auc, 4),
        "tau_youden_j": round(tau_youden, 4),
        "dataset_breakdown": dataset_breakdown,
        "figure_path": fig_path,
    }

    report_json_path = os.path.join(out_dir, "universal_calibration_summary.json")
    save_json(report, report_json_path)

    logger.info("==================================================================")
    logger.info(f"CALIBRATION COMPLETE!")
    logger.info(f"RECOMMENDED UNIVERSAL THRESHOLD (tau): {tau_f1_max:.4f}")
    logger.info(f"Overall Max F1: {best_f1:.4f} | Precision: {best_prec:.4f} | Recall: {best_rec:.4f}")
    logger.info(f"ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}")
    logger.info(f"Summary saved -> {report_json_path}")
    logger.info(f"Figure saved  -> {fig_path}")
    logger.info("==================================================================")


if __name__ == "__main__":
    main()
