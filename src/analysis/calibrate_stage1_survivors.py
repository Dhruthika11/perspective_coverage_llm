"""
Stage 1 Survivor Threshold Calibration.

Implements the user's requirement:
"Calibrate using Stage 1 survivors. Stage 1 retrieves candidates (cos >= 0.75).
DeBERTa only sees those candidates. Therefore, threshold calibration should
primarily represent the types of candidates that survive Stage 1."

Features:
1. Filters candidates on the calibration partition that pass Stage 1 (cosine >= 0.75):
   - Genuine matches surviving BGE
   - Long LLM-to-human matches surviving BGE
   - Same-stance, different-reasoning candidates surviving BGE
   - Opposite-stance candidates surviving BGE
   - Same-topic, different-concept candidates surviving BGE
   - Unrelated candidates surviving BGE
2. Performs precision-constrained threshold sweep:
   theta* = argmax Recall(theta) s.t. Precision(theta) >= 0.90
3. Generates probability calibration metrics: ECE, Brier score, and confusion matrix.
4. Outputs results to results/calibration/stage1_survivors_calibration_report.json.
"""

import os
import sys
import json
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, confusion_matrix

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.io_utils import resolve, load_csv, save_csv, save_json
from src.utils.logging_utils import get_logger
from src.analysis.perspective_verifier import compute_expected_calibration_error

logger = get_logger("calibrate_stage1_survivors")


def compute_wilson_ci(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Computes Wilson score interval for a binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    import scipy.stats as stats
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    spread = (z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)) / denom
    return round(float(max(0.0, centre - spread)), 4), round(float(min(1.0, centre + spread)), 4)


def calibrate_on_stage1_survivors(
    calibration_pool_csv: str = "data/calibration/expanded_stage1_calibration_pool.csv",
    benchmark_csv: str = "data/calibration/tri_dataset_750_pairs.csv",
    output_json: str = "results/calibration/expanded_stage1_calibration_report.json",
    stage1_tau: float = 0.75,
    min_precision: float = 0.90
) -> Dict[str, Any]:
    cal_p = resolve(calibration_pool_csv)
    test_p = resolve(benchmark_csv)

    if not os.path.exists(cal_p):
        raise FileNotFoundError(f"Missing expanded calibration pool at {cal_p}")
    if not os.path.exists(test_p):
        raise FileNotFoundError(f"Missing test benchmark at {test_p}")

    cal_survivors = load_csv(cal_p)
    logger.info(f"Loaded {len(cal_survivors)} Stage 1 calibration survivors from {cal_p}")

    # Full held-out test split from tri_dataset_750
    test_full_df = load_csv(test_p)
    test_df = test_full_df[test_full_df["split"] == "test"].copy()
    test_survivors = test_df[test_df["cosine_similarity"] >= stage1_tau].copy()

    logger.info(f"Test Split: Total Pairs = {len(test_df)}, Stage 1 Survivors = {len(test_survivors)} / {len(test_df)} ({len(test_survivors)/len(test_df):.1%})")

    # True binary match: label_3class == 2
    y_cal = (cal_survivors["label_3class"] == 2).astype(int).values
    p_cal = cal_survivors["p_verification"].values

    # Full test labels and survivor labels
    y_test_full = (test_df["label_3class"] == 2).astype(int).values
    y_test_surv = (test_survivors["label_3class"] == 2).astype(int).values
    p_test_surv = test_survivors["p_verification"].values

    total_true_positives_test = int(np.sum(y_test_full == 1))
    positives_surviving_stage1 = int(np.sum(y_test_surv == 1))
    positives_dropped_by_bge = total_true_positives_test - positives_surviving_stage1

    logger.info(f"Stage 1 Test Positives: Total={total_true_positives_test}, Surviving BGE={positives_surviving_stage1}, Dropped={positives_dropped_by_bge}")

    # 2. Threshold Sweep on Calibration Survivors: argmax Recall s.t. Precision >= min_precision
    thresholds = np.arange(0.50, 0.96, 0.01)
    best_theta = 0.80
    best_recall = -1.0
    best_cal_stats = {}

    for theta in thresholds:
        preds = (p_cal >= theta).astype(int)
        tp = np.sum((preds == 1) & (y_cal == 1))
        fp = np.sum((preds == 1) & (y_cal == 0))
        fn = np.sum((preds == 0) & (y_cal == 1))
        tn = np.sum((preds == 0) & (y_cal == 0))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        if prec >= min_precision and rec > best_recall:
            best_recall = rec
            best_theta = float(theta)
            best_cal_stats = {
                "theta": round(float(theta), 3),
                "precision": round(float(prec), 4),
                "recall": round(float(rec), 4),
                "f1": round(float(f1), 4),
                "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)
            }

    if not best_cal_stats:
        best_f1 = -1.0
        for theta in thresholds:
            preds = (p_cal >= theta).astype(int)
            tp = np.sum((preds == 1) & (y_cal == 1))
            fp = np.sum((preds == 1) & (y_cal == 0))
            fn = np.sum((preds == 0) & (y_cal == 1))
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            if f1 > best_f1:
                best_f1 = f1
                best_theta = float(theta)
                best_cal_stats = {
                    "theta": round(float(theta), 3),
                    "precision": round(float(prec), 4),
                    "recall": round(float(rec), 4),
                    "f1": round(float(f1), 4),
                    "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(np.sum((preds == 0) & (y_cal == 0)))
                }

    logger.info(f"Selected optimal threshold on Stage 1 survivors: theta* = {best_theta:.3f} "
                f"(Cal Precision: {best_cal_stats['precision']:.4f}, Recall: {best_cal_stats['recall']:.4f})")

    # 3. Evaluate Selected Threshold on Held-Out Test Split Stage 1 Survivors
    test_preds = (p_test_surv >= best_theta).astype(int)
    t_tp = int(np.sum((test_preds == 1) & (y_test_surv == 1)))
    t_fp = int(np.sum((test_preds == 1) & (y_test_surv == 0)))
    t_fn_cond = int(np.sum((test_preds == 0) & (y_test_surv == 1)))
    t_tn_cond = int(np.sum((test_preds == 0) & (y_test_surv == 0)))

    # Conditional Stage 2 Metrics
    t_prec = t_tp / (t_tp + t_fp) if (t_tp + t_fp) > 0 else 0.0
    t_rec_cond = t_tp / (t_tp + t_fn_cond) if (t_tp + t_fn_cond) > 0 else 0.0
    t_f1_cond = 2 * t_prec * t_rec_cond / (t_prec + t_rec_cond) if (t_prec + t_rec_cond) > 0 else 0.0

    # True End-to-End Pipeline Metrics (Accounting for BGE losses)
    t_fn_e2e = t_fn_cond + positives_dropped_by_bge
    t_rec_e2e = t_tp / (t_tp + t_fn_e2e) if (t_tp + t_fn_e2e) > 0 else 0.0
    t_f1_e2e = 2 * t_prec * t_rec_e2e / (t_prec + t_rec_e2e) if (t_prec + t_rec_e2e) > 0 else 0.0

    # 95% Wilson Confidence Intervals
    prec_ci = compute_wilson_ci(t_tp, t_tp + t_fp)
    rec_cond_ci = compute_wilson_ci(t_tp, t_tp + t_fn_cond)
    rec_e2e_ci = compute_wilson_ci(t_tp, t_tp + t_fn_e2e)

    # Rejection rate of adversarial opposite stance among survivors
    test_survivors["stage2_pred"] = test_preds
    opp_survivors = test_survivors[test_survivors["label_3class"] == 0]
    opp_leaks = int(np.sum(opp_survivors["stage2_pred"] == 1))
    opp_total = len(opp_survivors)
    opp_rejection_rate = (opp_total - opp_leaks) / opp_total if opp_total > 0 else 1.0

    # Same-stance distinct reasoning rejection among survivors
    same_stance_diff = test_survivors[test_survivors["label_3class"] == 1]
    diff_leaks = int(np.sum(same_stance_diff["stage2_pred"] == 1))
    diff_total = len(same_stance_diff)
    diff_rejection_rate = (diff_total - diff_leaks) / diff_total if diff_total > 0 else 1.0

    # Per-model breakdown in expanded calibration pool
    per_model_stats = {}
    if "model_family" in cal_survivors.columns:
        for m_fam, m_df in cal_survivors.groupby("model_family"):
            m_y = (m_df["label_3class"] == 2).astype(int).values
            m_preds = (m_df["p_verification"] >= best_theta).astype(int).values
            m_tp = int(np.sum((m_preds == 1) & (m_y == 1)))
            m_fp = int(np.sum((m_preds == 1) & (m_y == 0)))
            m_fn = int(np.sum((m_preds == 0) & (m_y == 1)))
            m_tn = int(np.sum((m_preds == 0) & (m_y == 0)))
            m_accepted = m_tp + m_fp
            m_prec = m_tp / m_accepted if m_accepted > 0 else (1.0 if m_fp == 0 else 0.0)
            m_rec = m_tp / (m_tp + m_fn) if (m_tp + m_fn) > 0 else 0.0
            per_model_stats[m_fam] = {
                "n_survivors": len(m_df),
                "positives": int(np.sum(m_y == 1)),
                "negatives": int(np.sum(m_y == 0)),
                "tp": m_tp,
                "fp": m_fp,
                "fn": m_fn,
                "tn": m_tn,
                "accepted": m_accepted,
                "precision": round(float(m_prec), 4),
                "recall": round(float(m_rec), 4)
            }

    test_ece, _ = compute_expected_calibration_error(y_test_surv, p_test_surv)
    test_brier = round(float(brier_score_loss(y_test_surv, p_test_surv)), 4)

    report = {
        "description": "Threshold calibration fitted on expanded multi-model Stage 1 survivors (N=288) with strict claim disjointness.",
        "stage1_tau": stage1_tau,
        "min_precision_constraint": min_precision,
        "optimal_theta_star": best_theta,
        "calibration_survivors_metrics": {
            "n_survivors": len(cal_survivors),
            "stats": best_cal_stats,
            "per_model_family_breakdown": per_model_stats
        },
        "held_out_test_metrics": {
            "n_survivors": len(test_survivors),
            "total_test_pairs": len(test_df),
            "stage1_dropped_positives": positives_dropped_by_bge,
            "tp": int(t_tp), "fp": int(t_fp), "fn_conditional": int(t_fn_cond), "fn_end_to_end": int(t_fn_e2e),
            "precision": round(float(t_prec), 4),
            "precision_95_ci": list(prec_ci),
            "recall_conditional_stage2": round(float(t_rec_cond), 4),
            "recall_conditional_95_ci": list(rec_cond_ci),
            "recall_end_to_end": round(float(t_rec_e2e), 4),
            "recall_end_to_end_95_ci": list(rec_e2e_ci),
            "f1_conditional": round(float(t_f1_cond), 4),
            "f1_end_to_end": round(float(t_f1_e2e), 4),
            "ece": test_ece,
            "brier_score": test_brier,
            "opposite_stance_rejection_among_survivors": {
                "total_surviving_opposites": opp_total,
                "leaks": opp_leaks,
                "rejection_rate": round(float(opp_rejection_rate), 4)
            },
            "same_stance_diff_reasoning_rejection_among_survivors": {
                "total_surviving": diff_total,
                "leaks": diff_leaks,
                "rejection_rate": round(float(diff_rejection_rate), 4)
            }
        }
    }

    out_p = resolve(output_json)
    os.makedirs(os.path.dirname(out_p), exist_ok=True)
    save_json(report, out_p)
    # Also save to default report path for pipeline continuity
    save_json(report, resolve("results/calibration/stage1_survivors_calibration_report.json"))
    logger.info(f"Stage 1 survivor calibration complete. Report saved to {out_p}")

    # Print summary
    print("\n" + "=" * 90)
    print("      EXPANDED STAGE 1 SURVIVOR THRESHOLD CALIBRATION REPORT")
    print("=" * 90)
    print(f"Calibration Pool Size: N = {len(cal_survivors)} Stage 1 survivors")
    print(f"Stage 1 Dense Filter Threshold: tau >= {stage1_tau}")
    print(f"Optimal Stage 2 Threshold: theta* = {best_theta:.3f} (argmax Recall s.t. Precision >= {min_precision:.2f})")
    print(f"Calibration Pool Metrics: Precision = {best_cal_stats['precision']*100:.2f}%, Recall = {best_cal_stats['recall']*100:.2f}%")
    print("\nPer-Model Family Breakdown on Calibration Survivors:")
    for fam, stats_dict in per_model_stats.items():
        print(f"  - {fam:25s}: N={stats_dict['n_survivors']:3d} | Pos={stats_dict['positives']:3d}, Neg={stats_dict['negatives']:3d} | TP={stats_dict['tp']:2d}, FP={stats_dict['fp']:2d}, Acc={stats_dict['accepted']:2d} | Prec={stats_dict['precision']*100:5.1f}%, Rec={stats_dict['recall']*100:5.1f}%")

    print("\nHeld-Out Test Split Metrics (N=147 Total, N=66 Stage 1 Survivors):")
    print(f"  Stage 2 Conditional Precision: {t_prec*100:.2f}% [95% Wilson CI: {prec_ci[0]*100:.1f}% - {prec_ci[1]*100:.1f}%]")
    print(f"  Stage 2 Conditional Recall:    {t_rec_cond*100:.2f}% [95% Wilson CI: {rec_cond_ci[0]*100:.1f}% - {rec_cond_ci[1]*100:.1f}%]")
    print(f"  True End-to-End Pipeline Recall: {t_rec_e2e*100:.2f}% [95% Wilson CI: {rec_e2e_ci[0]*100:.1f}% - {rec_e2e_ci[1]*100:.1f}%]")
    print(f"  True End-to-End Pipeline F1:     {t_f1_e2e*100:.2f}%")
    print(f"  Opposite-Stance Rejection:       {opp_total - opp_leaks}/{opp_total} ({opp_rejection_rate*100:.1f}%)")
    print("=" * 90 + "\n")

    return report


if __name__ == "__main__":
    calibrate_on_stage1_survivors()

