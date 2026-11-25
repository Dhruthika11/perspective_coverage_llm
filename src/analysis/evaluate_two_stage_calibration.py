"""
Comprehensive Statistical & Two-Stage Evaluation for Calibration:
1. 5-Fold Stratified Cross-Validation + 1,000 Bootstrap Confidence Intervals (95% CI)
2. Disaggregated Metrics across Strata:
   - Paraphrases (Positives)
   - Same Topic, Opposite Stance (Adversarial Negatives)
   - Same Topic, Different Concept (Hard Negatives)
   - Random Unrelated (Easy Negatives)
3. Hard-Negatives-Only AUC (excluding easy negatives to eliminate inflation)
4. Two-Stage System:
   - Stage 1: Cosine Similarity Threshold (tau = 0.75 or 0.80)
   - Stage 2: Stance / NLI Consistency Filter
5. Trade-off Analysis: 0.75 vs. 0.80 vs. Two-Stage Pipeline
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.model_selection import StratifiedKFold

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.io_utils import load_config, resolve, load_csv, save_json
from src.utils.logging_utils import get_logger

logger = get_logger("evaluate_two_stage_calibration")


def bootstrap_ci(y_true, y_score, tau, n_bootstraps=1000, seed=42):
    """Computes 95% bootstrap confidence intervals for F1, Precision, Recall, and Accuracy."""
    np.random.seed(seed)
    n = len(y_true)
    f1s, precs, recs, accs = [], [], [], []

    for _ in range(n_bootstraps):
        idx = np.random.choice(n, size=n, replace=True)
        yt_boot = y_true[idx]
        ys_boot = y_score[idx]

        yp_boot = (ys_boot >= tau).astype(int)
        tp = ((yt_boot == 1) & (yp_boot == 1)).sum()
        fp = ((yt_boot == 0) & (yp_boot == 1)).sum()
        fn = ((yt_boot == 1) & (yp_boot == 0)).sum()
        tn = ((yt_boot == 0) & (yp_boot == 0)).sum()

        p = tp / (tp + fp + 1e-12)
        r = tp / (tp + fn + 1e-12)
        f = 2 * p * r / (p + r + 1e-12)
        a = (tp + tn) / n

        f1s.append(f)
        precs.append(p)
        recs.append(r)
        accs.append(a)

    return {
        "f1_mean": float(np.mean(f1s)),
        "f1_ci": [float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))],
        "prec_mean": float(np.mean(precs)),
        "prec_ci": [float(np.percentile(precs, 2.5)), float(np.percentile(precs, 97.5))],
        "rec_mean": float(np.mean(recs)),
        "rec_ci": [float(np.percentile(recs, 2.5)), float(np.percentile(recs, 97.5))],
        "acc_mean": float(np.mean(accs)),
        "acc_ci": [float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))],
    }


def main():
    csv_path = resolve("data/calibration/hard_calibration_pairs.csv")
    if not os.path.exists(csv_path):
        logger.error(f"Missing {csv_path}")
        return

    df = load_csv(csv_path)
    logger.info(f"Loaded {len(df)} pairs from {csv_path}")

    # =========================================================================
    # PART 1: 5-FOLD STRATIFIED CROSS-VALIDATION
    # =========================================================================
    logger.info("--- PART 1: 5-Fold Stratified Cross-Validation ---")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    y = df["human_label"].astype(int).values
    s = df["cosine_similarity"].astype(float).values

    fold_metrics = []
    thresholds_grid = np.arange(0.40, 0.955, 0.005)

    for fold_i, (train_idx, val_idx) in enumerate(skf.split(df, y), start=1):
        y_tr, s_tr = y[train_idx], s[train_idx]
        y_val, s_val = y[val_idx], s[val_idx]

        # Find optimal tau on train fold
        best_f1_tr = -1
        best_tau_tr = 0.75
        for t in thresholds_grid:
            pred_tr = (s_tr >= t).astype(int)
            tp = ((y_tr == 1) & (pred_tr == 1)).sum()
            fp = ((y_tr == 0) & (pred_tr == 1)).sum()
            fn = ((y_tr == 1) & (pred_tr == 0)).sum()
            f1 = 2 * tp / (2 * tp + fp + fn + 1e-12)
            if f1 > best_f1_tr:
                best_f1_tr = f1
                best_tau_tr = t

        # Evaluate on validation fold
        pred_val = (s_val >= best_tau_tr).astype(int)
        tp = ((y_val == 1) & (pred_val == 1)).sum()
        fp = ((y_val == 0) & (pred_val == 1)).sum()
        fn = ((y_val == 1) & (pred_val == 0)).sum()
        tn = ((y_val == 0) & (pred_val == 0)).sum()

        p_val = tp / (tp + fp + 1e-12)
        r_val = tp / (tp + fn + 1e-12)
        f_val = 2 * p_val * r_val / (p_val + r_val + 1e-12)
        a_val = (tp + tn) / len(y_val)
        fpr, tpr, _ = roc_curve(y_val, s_val)
        auc_val = auc(fpr, tpr)

        fold_metrics.append({
            "fold": fold_i,
            "optimal_tau_train": float(best_tau_tr),
            "val_f1": float(f_val),
            "val_precision": float(p_val),
            "val_recall": float(r_val),
            "val_accuracy": float(a_val),
            "val_roc_auc": float(auc_val)
        })

    cv_df = pd.DataFrame(fold_metrics)
    mean_tau = cv_df["optimal_tau_train"].mean()
    mean_f1 = cv_df["val_f1"].mean()
    std_f1 = cv_df["val_f1"].std()

    logger.info(f"5-Fold CV Mean tau: {mean_tau:.4f} (std: {cv_df['optimal_tau_train'].std():.4f})")
    logger.info(f"5-Fold CV Validation F1: {mean_f1:.4f} ± {std_f1:.4f}")

    # =========================================================================
    # PART 2: BOOTSTRAP CONFIDENCE INTERVALS (95% CI)
    # =========================================================================
    logger.info("--- PART 2: 1,000-Iteration Bootstrap 95% Confidence Intervals ---")
    boot_res_075 = bootstrap_ci(y, s, tau=0.75, n_bootstraps=1000, seed=42)
    boot_res_080 = bootstrap_ci(y, s, tau=0.80, n_bootstraps=1000, seed=42)

    logger.info(f"At tau=0.75: F1 = {boot_res_075['f1_mean']:.4f} (95% CI: [{boot_res_075['f1_ci'][0]:.4f}, {boot_res_075['f1_ci'][1]:.4f}])")
    logger.info(f"At tau=0.75: Prec = {boot_res_075['prec_mean']:.4f} (95% CI: [{boot_res_075['prec_ci'][0]:.4f}, {boot_res_075['prec_ci'][1]:.4f}])")
    logger.info(f"At tau=0.75: Rec = {boot_res_075['rec_mean']:.4f} (95% CI: [{boot_res_075['rec_ci'][0]:.4f}, {boot_res_075['rec_ci'][1]:.4f}])")

    # =========================================================================
    # PART 3: DISAGGREGATED PERFORMANCE BY PAIR CATEGORY
    # =========================================================================
    logger.info("--- PART 3: Disaggregated Performance across Categories ---")
    category_rows = []
    for cat_name, grp in df.groupby("pair_type"):
        y_c = grp["human_label"].astype(int).values
        s_c = grp["cosine_similarity"].astype(float).values

        # Predictions at 0.75 and 0.80
        pred_075 = (s_c >= 0.75).astype(int)
        pred_080 = (s_c >= 0.80).astype(int)

        acc_075 = (pred_075 == y_c).mean()
        acc_080 = (pred_080 == y_c).mean()

        category_rows.append({
            "Category": cat_name,
            "Count": len(grp),
            "Expected_Label": int(y_c[0]),
            "Mean_Cosine_Sim": round(float(s_c.mean()), 4),
            "Min_Sim": round(float(s_c.min()), 4),
            "Max_Sim": round(float(s_c.max()), 4),
            "Accuracy_at_0.75": round(float(acc_075), 4),
            "Accuracy_at_0.80": round(float(acc_080), 4),
            "False_Accept_Rate_at_0.75": round(float((pred_075 == 1).mean() if y_c[0] == 0 else 0.0), 4),
            "False_Accept_Rate_at_0.80": round(float((pred_080 == 1).mean() if y_c[0] == 0 else 0.0), 4)
        })

    cat_df = pd.DataFrame(category_rows)
    print("\n" + cat_df.to_string(index=False) + "\n")

    # Hard-Negative Only AUC (excluding random_unrelated)
    hard_mask = df["pair_type"] != "random_unrelated"
    fpr_hard, tpr_hard, _ = roc_curve(y[hard_mask], s[hard_mask])
    auc_hard_only = auc(fpr_hard, tpr_hard)
    auc_all = auc(*roc_curve(y, s)[:2])

    logger.info(f"Overall ROC-AUC (including easy random negatives): {auc_all:.4f}")
    logger.info(f"HARD-NEGATIVES-ONLY ROC-AUC (without easy negatives): {auc_hard_only:.4f}")

    # =========================================================================
    # PART 4: TWO-STAGE EVALUATION (STAGE 1: COSINE + STAGE 2: STANCE/NLI)
    # =========================================================================
    logger.info("--- PART 4: Evaluating Two-Stage Pipeline (Cosine + Stance Filter) ---")
    # In our dataset, opposite-stance pairs have Same_Stance == 'No'.
    # In Stage 1: Cosine Threshold passes pairs where s >= tau.
    # In Stage 2: Stance Filter rejects any pair where stance opposes (Same_Stance == 'No').

    def evaluate_pipeline(tau_val, use_stance_filter=False):
        stage1_pass = (s >= tau_val)
        if use_stance_filter:
            # Stage 2: Only accept if Stance also matches
            stance_match = (df["same_stance"] == "Yes").values
            final_pred = (stage1_pass & stance_match).astype(int)
        else:
            final_pred = stage1_pass.astype(int)

        tp = ((y == 1) & (final_pred == 1)).sum()
        fp = ((y == 0) & (final_pred == 1)).sum()
        fn = ((y == 1) & (final_pred == 0)).sum()
        tn = ((y == 0) & (final_pred == 0)).sum()

        p = tp / (tp + fp + 1e-12)
        r = tp / (tp + fn + 1e-12)
        f = 2 * p * r / (p + r + 1e-12)
        a = (tp + tn) / len(y)

        # Specifically for opposite-stance pairs:
        opp_mask = (df["pair_type"] == "same_topic_opposite_stance").values
        opp_fp = ((final_pred == 1) & opp_mask).sum()
        opp_leak_rate = opp_fp / (opp_mask.sum() + 1e-12)

        return {
            "Precision": round(float(p), 4),
            "Recall": round(float(r), 4),
            "F1_Score": round(float(f), 4),
            "Accuracy": round(float(a), 4),
            "Opposite_Stance_False_Accepts": int(opp_fp),
            "Opposite_Stance_Leak_Rate": round(float(opp_leak_rate), 4)
        }

    p_075_single = evaluate_pipeline(0.75, use_stance_filter=False)
    p_080_single = evaluate_pipeline(0.80, use_stance_filter=False)
    p_075_twostage = evaluate_pipeline(0.75, use_stance_filter=True)
    p_072_twostage = evaluate_pipeline(0.72, use_stance_filter=True)

    comparison_df = pd.DataFrame([
        {"Method": "Single-Stage (Cosine tau=0.75)", **p_075_single},
        {"Method": "Single-Stage (Cosine tau=0.80)", **p_080_single},
        {"Method": "Two-Stage (Cosine tau=0.75 + Stance/NLI Filter)", **p_075_twostage},
        {"Method": "Two-Stage (Cosine tau=0.72 + Stance/NLI Filter)", **p_072_twostage},
    ])

    print("--- COMPARISON: Single-Stage vs. Two-Stage System ---")
    print(comparison_df.to_string(index=False) + "\n")

    # =========================================================================
    # PART 5: GENERATE COMPREHENSIVE VISUALIZATION
    # =========================================================================
    out_dir = resolve("results/calibration")
    os.makedirs(out_dir, exist_ok=True)
    fig_path = os.path.join(out_dir, "two_stage_vs_single_stage_calibration.png")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Panel 1: Similarity Distribution by Category
    for cat_name, grp in df.groupby("pair_type"):
        axes[0].hist(grp["cosine_similarity"], bins=20, alpha=0.5, label=f"{cat_name} (mean={grp['cosine_similarity'].mean():.2f})")
    axes[0].axvline(0.75, color="black", linestyle="--", lw=1.8, label="tau = 0.75")
    axes[0].axvline(0.80, color="gray", linestyle=":", lw=1.8, label="tau = 0.80")
    axes[0].set_xlabel("Cosine Similarity", fontsize=11)
    axes[0].set_ylabel("Pair Count", fontsize=11)
    axes[0].set_title("Distribution of Cosine Similarities Across Strata", fontsize=12)
    axes[0].legend(loc="upper left", fontsize=8.5)
    axes[0].grid(True, alpha=0.25)

    # Panel 2: ROC Curves: All vs. Hard Negatives Only
    axes[1].plot(*roc_curve(y, s)[:2], color="blue", lw=2, label=f"All Pairs (AUC = {auc_all:.3f})")
    axes[1].plot(fpr_hard, tpr_hard, color="crimson", lw=2, linestyle="--", label=f"Hard Negatives Only (AUC = {auc_hard_only:.3f})")
    axes[1].plot([0, 1], [0, 1], color="gray", lw=1, linestyle=":")
    axes[1].set_xlabel("False Positive Rate", fontsize=11)
    axes[1].set_ylabel("True Positive Rate", fontsize=11)
    axes[1].set_title("ROC Curve: Easy vs. Hard Negatives", fontsize=12)
    axes[1].legend(loc="lower right", fontsize=10)
    axes[1].grid(True, alpha=0.25)

    # Panel 3: F1 Comparison across Single-Stage vs Two-Stage
    methods = ["Single (0.75)", "Single (0.80)", "Two-Stage (0.75+NLI)", "Two-Stage (0.72+NLI)"]
    f1_vals = [p_075_single["F1_Score"], p_080_single["F1_Score"], p_075_twostage["F1_Score"], p_072_twostage["F1_Score"]]
    prec_vals = [p_075_single["Precision"], p_080_single["Precision"], p_075_twostage["Precision"], p_072_twostage["Precision"]]
    rec_vals = [p_075_single["Recall"], p_080_single["Recall"], p_075_twostage["Recall"], p_072_twostage["Recall"]]

    x_indices = np.arange(len(methods))
    w = 0.25
    axes[2].bar(x_indices - w, prec_vals, width=w, label="Precision", color="royalblue")
    axes[2].bar(x_indices, rec_vals, width=w, label="Recall", color="gold")
    axes[2].bar(x_indices + w, f1_vals, width=w, label="F1-Score", color="seagreen")
    axes[2].set_xticks(x_indices)
    axes[2].set_xticklabels(methods, rotation=15, ha="right", fontsize=9)
    axes[2].set_ylim(0.70, 1.02)
    axes[2].set_ylabel("Score", fontsize=11)
    axes[2].set_title("Performance: Single-Stage vs. Two-Stage", fontsize=12)
    axes[2].legend(loc="lower left", fontsize=10)
    axes[2].grid(True, alpha=0.25, axis="y")

    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()

    # Save summary JSON
    summary_report = {
        "cross_validation_5fold": {
            "mean_tau": round(float(mean_tau), 4),
            "tau_std": round(float(cv_df["optimal_tau_train"].std()), 4),
            "mean_f1": round(float(mean_f1), 4),
            "f1_std": round(float(std_f1), 4),
            "folds": fold_metrics
        },
        "bootstrap_confidence_intervals_tau_075": boot_res_075,
        "bootstrap_confidence_intervals_tau_080": boot_res_080,
        "disaggregated_categories": category_rows,
        "auc_comparison": {
            "all_pairs_auc": round(float(auc_all), 4),
            "hard_negatives_only_auc": round(float(auc_hard_only), 4)
        },
        "single_vs_two_stage_comparison": comparison_df.to_dict(orient="records"),
        "figure_path": fig_path
    }

    report_out = os.path.join(out_dir, "two_stage_calibration_report.json")
    save_json(summary_report, report_out)
    logger.info(f"Report saved -> {report_out}")
    logger.info(f"Figure saved -> {fig_path}")


if __name__ == "__main__":
    main()
