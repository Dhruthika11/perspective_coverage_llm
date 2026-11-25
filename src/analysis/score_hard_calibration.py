"""
Sweeps all candidate similarity thresholds (0.40 to 0.95) on the 80% Train Set (240 pairs)
to determine the optimal threshold tau_train (maximizing F1-Score).
Then evaluates that exact frozen threshold on the 20% unseen Test Set (60 pairs)
to verify out-of-sample generalization across adversarial hard negatives.
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

logger = get_logger("score_hard_calibration")


def evaluate_cutoff_sweep(y_true, y_score, min_t=0.40, max_t=0.95, step=0.005):
    """Sweeps all candidate cutoffs between min_t and max_t and records metrics."""
    thresholds = np.arange(min_t, max_t + step / 2, step)
    sweep_results = []

    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())

        prec = tp / (tp + fp + 1e-12)
        rec = tp / (tp + fn + 1e-12)
        f1 = 2 * prec * rec / (prec + rec + 1e-12)
        acc = (tp + tn) / len(y_true)

        sweep_results.append({
            "threshold": round(float(t), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1": round(float(f1), 4),
            "accuracy": round(float(acc), 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn
        })

    sweep_df = pd.DataFrame(sweep_results)
    best_row = sweep_df.loc[sweep_df["f1"].idxmax()]
    return best_row["threshold"], best_row["f1"], sweep_df


def main():
    parser = argparse.ArgumentParser(description="Sweep thresholds on Train and evaluate on Test.")
    parser.add_argument(
        "--input", type=str, default="data/calibration/hard_calibration_pairs.csv",
        help="Path to hard calibration CSV containing 'split' column."
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
    if "split" not in df.columns or "human_label" not in df.columns:
        logger.error("CSV must contain 'split' (train/test) and 'human_label' columns.")
        return

    train_df = df[df["split"] == "train"].copy()
    test_df = df[df["split"] == "test"].copy()

    logger.info(f"Loaded dataset: {len(df)} total pairs | Train: {len(train_df)} | Test: {len(test_df)}")

    # -------------------------------------------------------------
    # 1. TRAIN SET OPTIMIZATION (80% -> 240 pairs)
    # -------------------------------------------------------------
    y_train = train_df["human_label"].astype(int).values
    s_train = train_df["cosine_similarity"].astype(float).values

    # Sweep all thresholds between 0.40 and 0.95
    logger.info("Sweeping cutoffs between 0.40 and 0.95 on Train Set...")
    tau_train, train_f1_max, sweep_df = evaluate_cutoff_sweep(y_train, s_train, min_t=0.40, max_t=0.95, step=0.005)

    # Train ROC and PR metrics
    fpr_tr, tpr_tr, roc_thresh_tr = roc_curve(y_train, s_train)
    roc_auc_tr = auc(fpr_tr, tpr_tr)
    youden_tr = tpr_tr - fpr_tr
    tau_youden_tr = float(roc_thresh_tr[int(np.argmax(youden_tr))])

    prec_tr_arr, rec_tr_arr, _ = precision_recall_curve(y_train, s_train)
    pr_auc_tr = average_precision_score(y_train, s_train)

    train_pred = (s_train >= tau_train).astype(int)
    train_prec = float(((y_train == 1) & (train_pred == 1)).sum() / (train_pred.sum() + 1e-12))
    train_rec = float(((y_train == 1) & (train_pred == 1)).sum() / (y_train.sum() + 1e-12))
    train_acc = float((train_pred == y_train).mean())

    logger.info("------------------------------------------------------------------")
    logger.info(f"TRAIN SET RESULTS (240 pairs):")
    logger.info(f"  Optimal Threshold (tau_train): {tau_train:.4f}")
    logger.info(f"  Train Max F1:                 {train_f1_max:.4f}")
    logger.info(f"  Train Precision at tau:       {train_prec:.4f}")
    logger.info(f"  Train Recall at tau:          {train_rec:.4f}")
    logger.info(f"  Train Accuracy:               {train_acc:.4f}")
    logger.info(f"  Train ROC-AUC:                {roc_auc_tr:.4f} | PR-AUC: {pr_auc_tr:.4f}")
    logger.info("------------------------------------------------------------------")

    # -------------------------------------------------------------
    # 2. UNSEEN TEST SET EVALUATION (20% -> 60 pairs)
    # -------------------------------------------------------------
    y_test = test_df["human_label"].astype(int).values
    s_test = test_df["cosine_similarity"].astype(float).values

    # Evaluate the FROZEN tau_train on unseen test data
    test_pred = (s_test >= tau_train).astype(int)
    tp_te = int(((y_test == 1) & (test_pred == 1)).sum())
    fp_te = int(((y_test == 0) & (test_pred == 1)).sum())
    fn_te = int(((y_test == 1) & (test_pred == 0)).sum())
    tn_te = int(((y_test == 0) & (test_pred == 0)).sum())

    test_prec = tp_te / (tp_te + fp_te + 1e-12)
    test_rec = tp_te / (tp_te + fn_te + 1e-12)
    test_f1 = 2 * test_prec * test_rec / (test_prec + test_rec + 1e-12)
    test_acc = (tp_te + tn_te) / len(y_test)

    fpr_te, tpr_te, _ = roc_curve(y_test, s_test)
    roc_auc_te = auc(fpr_te, tpr_te)
    prec_te_arr, rec_te_arr, _ = precision_recall_curve(y_test, s_test)
    pr_auc_te = average_precision_score(y_test, s_test)

    # Pair Type breakdown on Test
    pair_type_breakdown = {}
    for p_type, grp in test_df.groupby("pair_type"):
        g_true = grp["human_label"].astype(int).values
        g_score = grp["cosine_similarity"].astype(float).values
        g_pred = (g_score >= tau_train).astype(int)
        g_acc = float((g_true == g_pred).mean())
        pair_type_breakdown[p_type] = {
            "n_pairs": len(grp),
            "expected_label": int(g_true[0]),
            "accuracy": round(g_acc, 4),
            "mean_cosine_sim": round(float(g_score.mean()), 4),
        }

    logger.info("TEST SET GENERALIZATION (60 unseen pairs):")
    logger.info(f"  Applied tau:                  {tau_train:.4f}")
    logger.info(f"  Test F1-Score:                {test_f1:.4f}")
    logger.info(f"  Test Precision:               {test_prec:.4f}")
    logger.info(f"  Test Recall:                  {test_rec:.4f}")
    logger.info(f"  Test Accuracy:                {test_acc:.4f}")
    logger.info(f"  Test ROC-AUC:                 {roc_auc_te:.4f} | PR-AUC: {pr_auc_te:.4f}")
    logger.info("------------------------------------------------------------------")

    # -------------------------------------------------------------
    # 3. DUAL ROC & PR PLOT (Train vs. Test)
    # -------------------------------------------------------------
    out_dir = resolve(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    fig_path = os.path.join(out_dir, "hard_calibration_train_test_roc_pr.png")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # ROC Plot
    axes[0].plot(fpr_tr, tpr_tr, color="royalblue", lw=2.5, label=f"Train ROC (AUC = {roc_auc_tr:.3f})")
    axes[0].plot(fpr_te, tpr_te, color="darkorange", lw=2.5, linestyle="--", label=f"Test ROC (AUC = {roc_auc_te:.3f})")
    axes[0].plot([0, 1], [0, 1], color="gray", lw=1, linestyle=":")
    axes[0].set_xlabel("False Positive Rate", fontsize=11)
    axes[0].set_ylabel("True Positive Rate", fontsize=11)
    axes[0].set_title(f"ROC Curve (Train vs. Unseen Test)\nOptimal tau = {tau_train:.3f}", fontsize=12)
    axes[0].legend(loc="lower right", fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # PR Plot
    axes[1].plot(rec_tr_arr, prec_tr_arr, color="royalblue", lw=2.5, label=f"Train PR (AP = {pr_auc_tr:.3f})")
    axes[1].plot(rec_te_arr, prec_te_arr, color="darkorange", lw=2.5, linestyle="--", label=f"Test PR (AP = {pr_auc_te:.3f})")
    axes[1].scatter([train_rec], [train_prec], color="royalblue", s=70, zorder=5, label=f"Train tau={tau_train:.3f} (F1={train_f1_max:.3f})")
    axes[1].scatter([test_rec], [test_prec], color="darkorange", s=70, zorder=5, label=f"Test tau={tau_train:.3f} (F1={test_f1:.3f})")
    axes[1].set_xlabel("Recall", fontsize=11)
    axes[1].set_ylabel("Precision", fontsize=11)
    axes[1].set_title(f"Precision-Recall Curve (Train vs. Unseen Test)\nTrain F1={train_f1_max:.3f} -> Test F1={test_f1:.3f}", fontsize=12)
    axes[1].legend(loc="lower left", fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()

    # Save complete JSON summary
    summary = {
        "applied_threshold_tau": round(tau_train, 4),
        "train_metrics": {
            "n_pairs": len(train_df),
            "f1": round(train_f1_max, 4),
            "precision": round(train_prec, 4),
            "recall": round(train_rec, 4),
            "accuracy": round(train_acc, 4),
            "roc_auc": round(roc_auc_tr, 4),
            "pr_auc": round(pr_auc_tr, 4),
            "tau_youden": round(tau_youden_tr, 4),
        },
        "test_metrics": {
            "n_pairs": len(test_df),
            "f1": round(test_f1, 4),
            "precision": round(test_prec, 4),
            "recall": round(test_rec, 4),
            "accuracy": round(test_acc, 4),
            "roc_auc": round(roc_auc_te, 4),
            "pr_auc": round(pr_auc_te, 4),
        },
        "test_pair_type_breakdown": pair_type_breakdown,
        "figure_path": fig_path,
        "sweep_samples": sweep_df.iloc[::4].to_dict(orient="records")  # sample every 4th step for readability
    }

    report_path = os.path.join(out_dir, "hard_calibration_summary.json")
    save_json(summary, report_path)

    logger.info("==================================================================")
    logger.info("FINAL CONCLUSION:")
    logger.info(f"Optimal Train Threshold: tau = {tau_train:.4f}")
    logger.info(f"Train F1: {train_f1_max:.4f}  --->  Unseen Test F1: {test_f1:.4f}")
    logger.info(f"Summary saved: {report_path}")
    logger.info(f"Figure saved:  {fig_path}")
    logger.info("==================================================================")


if __name__ == "__main__":
    main()
