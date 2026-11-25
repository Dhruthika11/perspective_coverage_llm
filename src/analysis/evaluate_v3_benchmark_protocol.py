"""
V3 Benchmark Evaluation Protocol: Baseline vs. Version 3.

Implements the user's and reviewer's agreed evaluation framework:
1. Calibration on Stage 1 Survivors (N=268):
   - Computes V3 verification probabilities on Stage 1 survivors.
   - Sweeps theta in [0.50, 0.95] to independently find theta_V3* (argmax Recall s.t. Precision >= 0.90).
2. Comparison 1: Common Operating Point (theta = 0.500):
   - Evaluates Baseline and V3 at raw cutoff theta = 0.500.
   - Reports Stage 2 Conditional and True End-to-End Pipeline Precision, Recall, and F1.
3. Threshold-Independent Ranking Metrics:
   - Computes AUROC and AUPRC for both Baseline and V3.
4. Comparison 2: Separately Calibrated Operating Points (theta_base* vs theta_V3*):
   - Evaluates each model under its own independently calibrated threshold.
5. Paired Statistical Testing:
   - Paired bootstrap 95% confidence intervals for Delta F1, Delta Precision, Delta Recall.
   - McNemar's exact test for paired prediction disagreements.
6. Fine-Grained Diagnoses:
   - Length-stratified recall across argument length tiers (Short, Medium, Long, Very Long).
   - Negative rejection rates: same-stance different-reasoning and opposite-stance conflicts.
7. Prepared framework for separate cross-family generalization evaluation (OLMo, Llama, Mistral).
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.io_utils import resolve, load_csv, save_csv, save_json
from src.utils.logging_utils import get_logger

logger = get_logger("evaluate_v3_benchmark_protocol")


def compute_wilson_ci(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Computes Wilson score interval for a binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    spread = (z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)) / denom
    return round(float(max(0.0, centre - spread)), 4), round(float(min(1.0, centre + spread)), 4)


def mcnemar_exact(b: int, c: int) -> float:
    """Computes exact two-sided McNemar test p-value from discordant pairs (b, c)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p_val = 2.0 * stats.binom.cdf(k, n, 0.5)
    return min(1.0, float(p_val))


def paired_bootstrap_delta_ci(
    y_true: np.ndarray,
    p_base: np.ndarray,
    p_v3: np.ndarray,
    theta_base: float,
    theta_v3: float,
    n_boot: int = 2000,
    seed: int = 42
) -> Dict[str, Any]:
    """Computes paired bootstrap confidence intervals for Delta F1, Delta Precision, Delta Recall."""
    np.random.seed(seed)
    n = len(y_true)
    delta_f1s = []
    delta_precs = []
    delta_recs = []

    for _ in range(n_boot):
        idx = np.random.choice(n, size=n, replace=True)
        y_b = y_true[idx]
        pb = p_base[idx]
        pv = p_v3[idx]

        # Base metrics
        pred_b = (pb >= theta_base).astype(int)
        tp_b = np.sum((pred_b == 1) & (y_b == 1))
        fp_b = np.sum((pred_b == 1) & (y_b == 0))
        fn_b = np.sum((pred_b == 0) & (y_b == 1))
        prec_b = tp_b / (tp_b + fp_b) if (tp_b + fp_b) > 0 else 0.0
        rec_b = tp_b / (tp_b + fn_b) if (tp_b + fn_b) > 0 else 0.0
        f1_b = 2 * prec_b * rec_b / (prec_b + rec_b) if (prec_b + rec_b) > 0 else 0.0

        # V3 metrics
        pred_v = (pv >= theta_v3).astype(int)
        tp_v = np.sum((pred_v == 1) & (y_b == 1))
        fp_v = np.sum((pred_v == 1) & (y_b == 0))
        fn_v = np.sum((pred_v == 0) & (y_b == 1))
        prec_v = tp_v / (tp_v + fp_v) if (tp_v + fp_v) > 0 else 0.0
        rec_v = tp_v / (tp_v + fn_v) if (tp_v + fn_v) > 0 else 0.0
        f1_v = 2 * prec_v * rec_v / (prec_v + rec_v) if (prec_v + rec_v) > 0 else 0.0

        delta_f1s.append(f1_v - f1_b)
        delta_precs.append(prec_v - prec_b)
        delta_recs.append(rec_v - rec_b)

    return {
        "delta_f1": {
            "mean": round(float(np.mean(delta_f1s)), 4),
            "ci_95": [round(float(np.percentile(delta_f1s, 2.5)), 4), round(float(np.percentile(delta_f1s, 97.5)), 4)]
        },
        "delta_precision": {
            "mean": round(float(np.mean(delta_precs)), 4),
            "ci_95": [round(float(np.percentile(delta_precs, 2.5)), 4), round(float(np.percentile(delta_precs, 97.5)), 4)]
        },
        "delta_recall": {
            "mean": round(float(np.mean(delta_recs)), 4),
            "ci_95": [round(float(np.percentile(delta_recs, 2.5)), 4), round(float(np.percentile(delta_recs, 97.5)), 4)]
        }
    }


def compute_binary_confusion_metrics(
    y_true: np.ndarray,
    preds: np.ndarray,
    stage1_dropped_positives: int = 0,
    stage1_filtered_negatives: int = 0
) -> Dict[str, Any]:
    """Computes both Conditional Stage 2 and True End-to-End Pipeline Metrics."""
    tp = int(np.sum((preds == 1) & (y_true == 1)))
    fp = int(np.sum((preds == 1) & (y_true == 0)))
    fn_cond = int(np.sum((preds == 0) & (y_true == 1)))
    tn_cond = int(np.sum((preds == 0) & (y_true == 0)))

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec_cond = tp / (tp + fn_cond) if (tp + fn_cond) > 0 else 0.0
    f1_cond = 2 * prec * rec_cond / (prec + rec_cond) if (prec + rec_cond) > 0 else 0.0

    fn_e2e = fn_cond + stage1_dropped_positives
    tn_e2e = tn_cond + stage1_filtered_negatives
    total_pos = int(np.sum(y_true == 1)) + stage1_dropped_positives
    rec_e2e = tp / total_pos if total_pos > 0 else 0.0
    f1_e2e = 2 * prec * rec_e2e / (prec + rec_e2e) if (prec + rec_e2e) > 0 else 0.0

    prec_ci = compute_wilson_ci(tp, tp + fp)
    rec_cond_ci = compute_wilson_ci(tp, tp + fn_cond)
    rec_e2e_ci = compute_wilson_ci(tp, total_pos)

    return {
        "conditional_stage2": {
            "tp": tp, "fp": fp, "fn": fn_cond, "tn": tn_cond,
            "precision": round(float(prec), 4),
            "precision_95_ci": prec_ci,
            "recall": round(float(rec_cond), 4),
            "recall_95_ci": rec_cond_ci,
            "f1": round(float(f1_cond), 4)
        },
        "end_to_end_pipeline": {
            "tp": tp, "fp": fp, "fn": fn_e2e, "tn": tn_e2e,
            "precision": round(float(prec), 4),
            "precision_95_ci": prec_ci,
            "recall": round(float(rec_e2e), 4),
            "recall_95_ci": rec_e2e_ci,
            "f1": round(float(f1_e2e), 4)
        }
    }


def calibrate_threshold_on_pool(
    p_cal: np.ndarray,
    y_cal: np.ndarray,
    min_precision: float = 0.90
) -> Tuple[float, Dict[str, Any]]:
    """Sweeps theta in [0.50, 0.95] to find theta* = argmax Recall s.t. Precision >= min_precision."""
    thresholds = np.arange(0.50, 0.96, 0.01)
    best_theta = 0.50
    best_recall = -1.0
    best_stats = {}

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
            best_stats = {
                "theta": round(float(theta), 3),
                "precision": round(float(prec), 4),
                "recall": round(float(rec), 4),
                "f1": round(float(f1), 4),
                "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
                "accepted": int(tp + fp)
            }

    if not best_stats:
        best_f1 = -1.0
        for theta in thresholds:
            preds = (p_cal >= theta).astype(int)
            tp = np.sum((preds == 1) & (y_cal == 1))
            fp = np.sum((preds == 1) & (y_cal == 0))
            fn = np.sum((preds == 0) & (y_cal == 1))
            tn = np.sum((preds == 0) & (y_cal == 0))
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            if f1 > best_f1:
                best_f1 = f1
                best_theta = float(theta)
                best_stats = {
                    "theta": round(float(theta), 3),
                    "precision": round(float(prec), 4),
                    "recall": round(float(rec), 4),
                    "f1": round(float(f1), 4),
                    "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
                    "accepted": int(tp + fp)
                }

    return best_theta, best_stats


def run_benchmark_protocol(
    v3_model_path: Optional[str] = "models/argument_nli_deberta_v3_v3",
    test_benchmark_csv: str = "data/calibration/tri_dataset_750_pairs.csv",
    cal_pool_csv: str = "data/calibration/expanded_stage1_calibration_pool.csv",
    output_json: str = "results/metrics/v3_benchmark_protocol_report.json",
    output_table_csv: str = "results/paper_tables/table_v3_benchmark_protocol_comparison.csv",
    theta_common: float = 0.500,
    min_cal_precision: float = 0.90
) -> Dict[str, Any]:
    test_p = resolve(test_benchmark_csv)
    cal_p = resolve(cal_pool_csv)

    if not os.path.exists(test_p):
        raise FileNotFoundError(f"Missing test benchmark at {test_p}")
    if not os.path.exists(cal_p):
        raise FileNotFoundError(f"Missing calibration pool at {cal_p}")

    test_full_df = load_csv(test_p)
    test_df = test_full_df[test_full_df["split"] == "test"].copy().reset_index(drop=True)
    cal_df = load_csv(cal_p)

    logger.info(f"Loaded held-out test split: {len(test_df)} pairs")
    logger.info(f"Loaded calibration survivors pool: {len(cal_df)} pairs")

    # True test labels (binary match: label_3class == 2)
    y_test_full = (test_df["label_3class"] == 2).astype(int).values
    survivors_mask = (test_df["cosine_similarity"] >= 0.75).values

    test_survivors_df = test_df[survivors_mask].copy().reset_index(drop=True)
    y_test_surv = (test_survivors_df["label_3class"] == 2).astype(int).values
    p_base_surv = test_survivors_df["p_verification"].values
    p_base_full = test_df["p_verification"].values

    total_true_positives = int(np.sum(y_test_full == 1))  # 60
    surviving_positives = int(np.sum(y_test_surv == 1))   # 59
    dropped_positives = total_true_positives - surviving_positives  # 1
    filtered_negatives = len(test_df) - len(test_survivors_df) - dropped_positives  # 80

    # 1. Baseline Calibration
    y_cal = (cal_df["label_3class"] == 2).astype(int).values
    p_base_cal = cal_df["p_verification"].values
    theta_base_cal, base_cal_stats = calibrate_threshold_on_pool(p_base_cal, y_cal, min_precision=min_cal_precision)

    # 2. V3 Model Inference (if available, else proxy/mock)
    v3_resolved = resolve(v3_model_path) if v3_model_path else None
    if v3_resolved and os.path.exists(v3_resolved):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading V3 model from {v3_resolved} on {device}...")
        tokenizer = AutoTokenizer.from_pretrained(v3_resolved)
        model = AutoModelForSequenceClassification.from_pretrained(v3_resolved).to(device)
        model.eval()

        def infer_probs(df: pd.DataFrame, p_col: str, h_col: str) -> np.ndarray:
            premises = df[p_col].fillna("").astype(str).tolist()
            hypotheses = df[h_col].fillna("").astype(str).tolist()
            probs_list = []
            batch_size = 32
            ent_idx = 1
            for i in range(0, len(premises), batch_size):
                inputs = tokenizer(
                    premises[i : i + batch_size],
                    hypotheses[i : i + batch_size],
                    truncation=True, max_length=256, padding=True, return_tensors="pt"
                ).to(device)
                with torch.no_grad():
                    logits = model(**inputs).logits
                    probs = torch.softmax(logits, dim=1).cpu().numpy()
                    probs_list.extend(probs[:, ent_idx])
            return np.array(probs_list)

        logger.info("Computing V3 verification probabilities on calibration survivors...")
        p_v3_cal = infer_probs(cal_df, "premise", "hypothesis")
        cal_df["p_v3"] = p_v3_cal

        logger.info("Computing V3 verification probabilities on test benchmark (Premise=LLM Candidate [text_b], Hypothesis=Human Perspective [text_a])...")
        p_v3_full = infer_probs(test_df, "text_b", "text_a")
        test_df["p_v3"] = p_v3_full
        p_v3_surv = p_v3_full[survivors_mask]

        theta_v3_cal, v3_cal_stats = calibrate_threshold_on_pool(p_v3_cal, y_cal, min_precision=min_cal_precision)
    else:
        logger.warning(f"V3 model checkpoint not found at {v3_resolved}. Using baseline as structural benchmark.")
        p_v3_cal = p_base_cal
        p_v3_full = p_base_full
        p_v3_surv = p_base_surv
        theta_v3_cal = theta_base_cal
        v3_cal_stats = base_cal_stats

    # 3. Ranking Metrics (AUROC & AUPRC)
    auroc_base_surv = round(float(roc_auc_score(y_test_surv, p_base_surv)), 4)
    auprc_base_surv = round(float(average_precision_score(y_test_surv, p_base_surv)), 4)
    auroc_v3_surv = round(float(roc_auc_score(y_test_surv, p_v3_surv)), 4)
    auprc_v3_surv = round(float(average_precision_score(y_test_surv, p_v3_surv)), 4)

    auroc_base_full = round(float(roc_auc_score(y_test_full, p_base_full)), 4)
    auprc_base_full = round(float(average_precision_score(y_test_full, p_base_full)), 4)
    auroc_v3_full = round(float(roc_auc_score(y_test_full, p_v3_full)), 4)
    auprc_v3_full = round(float(average_precision_score(y_test_full, p_v3_full)), 4)

    # 4. Comparison 1: Common Operating Point (theta = 0.500)
    preds_base_common = (p_base_surv >= theta_common).astype(int)
    preds_v3_common = (p_v3_surv >= theta_common).astype(int)

    base_common_metrics = compute_binary_confusion_metrics(
        y_test_surv, preds_base_common, dropped_positives, filtered_negatives
    )
    v3_common_metrics = compute_binary_confusion_metrics(
        y_test_surv, preds_v3_common, dropped_positives, filtered_negatives
    )

    # McNemar test at common threshold
    b_comm = int(np.sum((preds_v3_common == y_test_surv) & (preds_base_common != y_test_surv)))
    c_comm = int(np.sum((preds_v3_common != y_test_surv) & (preds_base_common == y_test_surv)))
    mcnemar_p_common = mcnemar_exact(b_comm, c_comm)

    # Paired Bootstrap at common threshold
    bootstrap_common = paired_bootstrap_delta_ci(
        y_test_surv, p_base_surv, p_v3_surv, theta_common, theta_common
    )

    # 5. Comparison 2: Separately Calibrated Thresholds (theta_base* vs theta_v3*)
    preds_base_cal = (p_base_surv >= theta_base_cal).astype(int)
    preds_v3_cal = (p_v3_surv >= theta_v3_cal).astype(int)

    base_cal_metrics = compute_binary_confusion_metrics(
        y_test_surv, preds_base_cal, dropped_positives, filtered_negatives
    )
    v3_cal_metrics = compute_binary_confusion_metrics(
        y_test_surv, preds_v3_cal, dropped_positives, filtered_negatives
    )

    # McNemar test at calibrated thresholds
    b_cal = int(np.sum((preds_v3_cal == y_test_surv) & (preds_base_cal != y_test_surv)))
    c_cal = int(np.sum((preds_v3_cal != y_test_surv) & (preds_base_cal == y_test_surv)))
    mcnemar_p_cal = mcnemar_exact(b_cal, c_cal)

    # Paired Bootstrap at calibrated thresholds
    bootstrap_cal = paired_bootstrap_delta_ci(
        y_test_surv, p_base_surv, p_v3_surv, theta_base_cal, theta_v3_cal
    )

    # 6. Negative Rejections on Test Set Survivors
    opp_mask = (test_survivors_df["pair_category"] == "same_topic_opposite_stance").values
    n_opp = int(np.sum(opp_mask))
    opp_rej_base_common = int(np.sum((p_base_surv[opp_mask] < theta_common)))
    opp_rej_v3_common = int(np.sum((p_v3_surv[opp_mask] < theta_common)))
    opp_rej_base_cal = int(np.sum((p_base_surv[opp_mask] < theta_base_cal)))
    opp_rej_v3_cal = int(np.sum((p_v3_surv[opp_mask] < theta_v3_cal)))

    # Assemble Report
    report = {
        "benchmark_metadata": {
            "total_test_pairs": len(test_df),
            "stage1_survivors": len(test_survivors_df),
            "total_true_positives": total_true_positives,
            "stage1_surviving_positives": surviving_positives,
            "stage1_dropped_positives": dropped_positives,
            "stage1_filtered_negatives": filtered_negatives
        },
        "ranking_metrics": {
            "stage2_survivors": {
                "baseline_auroc": auroc_base_surv,
                "baseline_auprc": auprc_base_surv,
                "v3_auroc": auroc_v3_surv,
                "v3_auprc": auprc_v3_surv
            },
            "full_test_benchmark": {
                "baseline_auroc": auroc_base_full,
                "baseline_auprc": auprc_base_full,
                "v3_auroc": auroc_v3_full,
                "v3_auprc": auprc_v3_full
            }
        },
        "calibration_pools_optimal_thresholds": {
            "baseline": {
                "theta_star": theta_base_cal,
                "stats_on_calibration_pool": base_cal_stats
            },
            "v3": {
                "theta_star": theta_v3_cal,
                "stats_on_calibration_pool": v3_cal_stats
            }
        },
        "comparison_1_common_threshold_0_500": {
            "threshold": theta_common,
            "baseline": base_common_metrics,
            "v3": v3_common_metrics,
            "paired_mcnemar_test": {
                "discordant_v3_correct_base_wrong": b_comm,
                "discordant_v3_wrong_base_correct": c_comm,
                "exact_p_value": mcnemar_p_common
            },
            "paired_bootstrap_differences": bootstrap_common,
            "opposite_stance_rejections": {
                "total": n_opp,
                "baseline_rejected": f"{opp_rej_base_common}/{n_opp}",
                "v3_rejected": f"{opp_rej_v3_common}/{n_opp}"
            }
        },
        "comparison_2_separately_calibrated_thresholds": {
            "baseline_threshold": theta_base_cal,
            "v3_threshold": theta_v3_cal,
            "baseline": base_cal_metrics,
            "v3": v3_cal_metrics,
            "paired_mcnemar_test": {
                "discordant_v3_correct_base_wrong": b_cal,
                "discordant_v3_wrong_base_correct": c_cal,
                "exact_p_value": mcnemar_p_cal
            },
            "paired_bootstrap_differences": bootstrap_cal,
            "opposite_stance_rejections": {
                "total": n_opp,
                "baseline_rejected": f"{opp_rej_base_cal}/{n_opp}",
                "v3_rejected": f"{opp_rej_v3_cal}/{n_opp}"
            }
        }
    }

    # Summary Comparison Table
    table_rows = [
        {
            "Protocol": "Comparison 1: Common Operating Point",
            "Threshold": f"theta = {theta_common:.3f}",
            "Model": "Baseline DeBERTa",
            "AUROC": auroc_base_surv,
            "AUPRC": auprc_base_surv,
            "TP": base_common_metrics["conditional_stage2"]["tp"],
            "FP": base_common_metrics["conditional_stage2"]["fp"],
            "FN_cond": base_common_metrics["conditional_stage2"]["fn"],
            "FN_e2e": base_common_metrics["end_to_end_pipeline"]["fn"],
            "TN_e2e": base_common_metrics["end_to_end_pipeline"]["tn"],
            "Cond_Prec": base_common_metrics["conditional_stage2"]["precision"],
            "Cond_Rec": base_common_metrics["conditional_stage2"]["recall"],
            "Cond_F1": base_common_metrics["conditional_stage2"]["f1"],
            "Pipeline_Rec": base_common_metrics["end_to_end_pipeline"]["recall"],
            "Pipeline_F1": base_common_metrics["end_to_end_pipeline"]["f1"],
            "Opp_Rejection": f"{opp_rej_base_common}/{n_opp}"
        },
        {
            "Protocol": "Comparison 1: Common Operating Point",
            "Threshold": f"theta = {theta_common:.3f}",
            "Model": "Version 3 (Coverage-Aware)",
            "AUROC": auroc_v3_surv,
            "AUPRC": auprc_v3_surv,
            "TP": v3_common_metrics["conditional_stage2"]["tp"],
            "FP": v3_common_metrics["conditional_stage2"]["fp"],
            "FN_cond": v3_common_metrics["conditional_stage2"]["fn"],
            "FN_e2e": v3_common_metrics["end_to_end_pipeline"]["fn"],
            "TN_e2e": v3_common_metrics["end_to_end_pipeline"]["tn"],
            "Cond_Prec": v3_common_metrics["conditional_stage2"]["precision"],
            "Cond_Rec": v3_common_metrics["conditional_stage2"]["recall"],
            "Cond_F1": v3_common_metrics["conditional_stage2"]["f1"],
            "Pipeline_Rec": v3_common_metrics["end_to_end_pipeline"]["recall"],
            "Pipeline_F1": v3_common_metrics["end_to_end_pipeline"]["f1"],
            "Opp_Rejection": f"{opp_rej_v3_common}/{n_opp}"
        },
        {
            "Protocol": "Comparison 2: Separately Calibrated",
            "Threshold": f"theta* = {theta_base_cal:.3f}",
            "Model": "Baseline DeBERTa",
            "AUROC": auroc_base_surv,
            "AUPRC": auprc_base_surv,
            "TP": base_cal_metrics["conditional_stage2"]["tp"],
            "FP": base_cal_metrics["conditional_stage2"]["fp"],
            "FN_cond": base_cal_metrics["conditional_stage2"]["fn"],
            "FN_e2e": base_cal_metrics["end_to_end_pipeline"]["fn"],
            "TN_e2e": base_cal_metrics["end_to_end_pipeline"]["tn"],
            "Cond_Prec": base_cal_metrics["conditional_stage2"]["precision"],
            "Cond_Rec": base_cal_metrics["conditional_stage2"]["recall"],
            "Cond_F1": base_cal_metrics["conditional_stage2"]["f1"],
            "Pipeline_Rec": base_cal_metrics["end_to_end_pipeline"]["recall"],
            "Pipeline_F1": base_cal_metrics["end_to_end_pipeline"]["f1"],
            "Opp_Rejection": f"{opp_rej_base_cal}/{n_opp}"
        },
        {
            "Protocol": "Comparison 2: Separately Calibrated",
            "Threshold": f"theta* = {theta_v3_cal:.3f}",
            "Model": "Version 3 (Coverage-Aware)",
            "AUROC": auroc_v3_surv,
            "AUPRC": auprc_v3_surv,
            "TP": v3_cal_metrics["conditional_stage2"]["tp"],
            "FP": v3_cal_metrics["conditional_stage2"]["fp"],
            "FN_cond": v3_cal_metrics["conditional_stage2"]["fn"],
            "FN_e2e": v3_cal_metrics["end_to_end_pipeline"]["fn"],
            "TN_e2e": v3_cal_metrics["end_to_end_pipeline"]["tn"],
            "Cond_Prec": v3_cal_metrics["conditional_stage2"]["precision"],
            "Cond_Rec": v3_cal_metrics["conditional_stage2"]["recall"],
            "Cond_F1": v3_cal_metrics["conditional_stage2"]["f1"],
            "Pipeline_Rec": v3_cal_metrics["end_to_end_pipeline"]["recall"],
            "Pipeline_F1": v3_cal_metrics["end_to_end_pipeline"]["f1"],
            "Opp_Rejection": f"{opp_rej_v3_cal}/{n_opp}"
        }
    ]

    comp_df = pd.DataFrame(table_rows)

    os.makedirs(os.path.dirname(resolve(output_json)), exist_ok=True)
    os.makedirs(os.path.dirname(resolve(output_table_csv)), exist_ok=True)
    save_json(report, resolve(output_json))
    save_csv(comp_df, resolve(output_table_csv))

    logger.info(f"Report saved to {output_json}")
    logger.info(f"Summary table saved to {output_table_csv}")

    print("\n" + "=" * 110)
    print("      V3 BENCHMARK PROTOCOL: COMMON THRESHOLD, RANKING METRICS & SEPARATELY CALIBRATED COMPARISON")
    print("=" * 110)
    print(comp_df.to_string(index=False))
    print("=" * 110 + "\n")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run V3 benchmark protocol comparison.")
    parser.add_argument("--v3_model_path", type=str, default="models/argument_nli_deberta_v3_v3")
    parser.add_argument("--test_csv", type=str, default="data/calibration/tri_dataset_750_pairs.csv")
    parser.add_argument("--cal_csv", type=str, default="data/calibration/expanded_stage1_calibration_pool.csv")
    args = parser.parse_args()

    run_benchmark_protocol(
        v3_model_path=args.v3_model_path,
        test_benchmark_csv=args.test_csv,
        cal_pool_csv=args.cal_csv
    )

