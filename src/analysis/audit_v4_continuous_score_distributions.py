"""
Continuous Score Distribution and Mann-Whitney Concordance Audit for Argument-NLI V4.

Mathematically audits why V4's global AUROC is 0.3907:
1. Computes the exact Mann-Whitney U statistic: AUROC = P(S_pos > S_neg) on raw continuous scores.
2. Decomposes global AUROC into the exact weighted sum of subpopulation AUROCs:
     AUROC_global = sum_{p, n} w_{p, n} * AUROC(P_p, N_n)
   where w_{p, n} = (|P_p| * |N_n|) / (N_pos * N_neg).
3. Evaluates the full continuous quantile distribution (Min, 25%, 50%, 75%, 90%, Max, Mean, Std)
   of P(full_coverage) across all 8 ground-truth test categories.
4. Generates the full Pairwise Concordance Matrix P(S_pos_i > S_neg_j).
"""

import os
import sys
import json
import argparse
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.io_utils import resolve, load_csv, save_json
from src.utils.logging_utils import get_logger

logger = get_logger("audit_v4_continuous_scores")


def mann_whitney_concordance(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    """
    Computes exact P(S_pos > S_neg) + 0.5 * P(S_pos == S_neg).
    Equivalent to Mann-Whitney U / (N_pos * N_neg) and roc_auc_score.
    """
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return 0.5
    # Broadcasting for exact pairwise comparisons
    diff = pos_scores[:, None] - neg_scores[None, :]
    greater = np.sum(diff > 0)
    equal = np.sum(diff == 0)
    return float((greater + 0.5 * equal) / (len(pos_scores) * len(neg_scores)))


def compute_quantile_summary(scores: np.ndarray) -> Dict[str, float]:
    """Computes descriptive summary of continuous score distribution."""
    if len(scores) == 0:
        return {}
    return {
        "count": int(len(scores)),
        "mean": round(float(np.mean(scores)), 4),
        "std": round(float(np.std(scores)), 4),
        "min": round(float(np.min(scores)), 4),
        "p25": round(float(np.percentile(scores, 25)), 4),
        "median": round(float(np.median(scores)), 4),
        "p75": round(float(np.percentile(scores, 75)), 4),
        "p90": round(float(np.percentile(scores, 90)), 4),
        "max": round(float(np.max(scores)), 4),
    }


def run_continuous_score_audit(
    v4_mono_csv: str = "results/metrics/v4_1118_predictions.csv",
    v4_chunk_csv: str = "results/metrics/v4_chunked_1118_predictions.csv",
    test_csv: str = "data/nli_training_v3/test.csv",
    output_report_json: str = "results/metrics/v4_continuous_score_audit.json"
) -> Dict[str, Any]:
    test_p = resolve(test_csv)
    mono_p = resolve(v4_mono_csv)
    chunk_p = resolve(v4_chunk_csv)
    out_json = resolve(output_report_json)

    if not os.path.exists(test_p):
        raise FileNotFoundError(f"Missing test CSV at {test_p}")
    if not os.path.exists(mono_p):
        raise FileNotFoundError(f"Missing monolithic V4 predictions at {mono_p}")

    df_test = load_csv(test_p)
    df_mono = load_csv(mono_p)
    df_chunk = load_csv(chunk_p) if os.path.exists(chunk_p) else None

    # Merge scores
    df_merged = df_test.copy()
    df_merged["mono_p_full"] = df_mono["p_v4_full_cov"]
    if df_chunk is not None and "p_v4_chunk_full_cov" in df_chunk.columns:
        df_merged["chunk_max_p_full"] = df_chunk["p_v4_chunk_full_cov"]

    is_pos = (df_merged["label"].str.lower() == "entailment").to_numpy()
    is_neg = ~is_pos

    pos_cats = [
        "within_cluster_paraphrase",
        "long_llm_to_human_positive_qwen2_5_1_5b_instruct",
        "long_llm_to_human_positive_qwen2_5_32b_instruct",
        "long_llm_to_human_positive_qwen2_5_3b_instruct"
    ]
    neg_cats = [
        "same_stance_different_reasoning",
        "opposite_stance_conflict",
        "same_topic_diff_concept",
        "unrelated_different_claim"
    ]

    total_pos = int(is_pos.sum())  # 540
    total_neg = int(is_neg.sum())  # 578

    # 1. Continuous Distribution Summary per Category
    dist_summaries = {}
    for cat in df_merged["pair_category"].unique():
        cat_scores = df_merged[df_merged["pair_category"] == cat]["mono_p_full"].to_numpy()
        dist_summaries[cat] = compute_quantile_summary(cat_scores)

    # 2. Pairwise Mann-Whitney Concordance Matrix
    concordance_matrix = {}
    weighted_decomposed_auc = 0.0

    print("\n" + "=" * 115)
    print("CONTINUOUS SCORE DISTRIBUTIONS ACROSS TEST CATEGORIES (P(full_coverage))")
    print("=" * 115)
    print(f"{'Category':<50} | {'Count':<5} | {'Mean':<6} | {'Std':<6} | {'Min':<6} | {'Median':<6} | {'P75':<6} | {'Max':<6}")
    print("-" * 115)
    for cat, s in dist_summaries.items():
        print(f"{cat:<50} | {s['count']:<5} | {s['mean']:<6.4f} | {s['std']:<6.4f} | {s['min']:<6.4f} | {s['median']:<6.4f} | {s['p75']:<6.4f} | {s['max']:<6.4f}")

    print("\n" + "=" * 115)
    print("EXACT MANN-WHITNEY PAIRWISE CONCORDANCE MATRIX: P(S_pos > S_neg)")
    print("=" * 115)
    header = f"{'Positive Category':<40} | " + " | ".join([f"{nc[:15]:<15}" for nc in neg_cats]) + " | Overall vs Negs"
    print(header)
    print("-" * 115)

    pairwise_table = {}
    for p_cat in pos_cats:
        p_scores = df_merged[df_merged["pair_category"] == p_cat]["mono_p_full"].to_numpy()
        row_vals = []
        pairwise_table[p_cat] = {}
        for n_cat in neg_cats:
            n_scores = df_merged[df_merged["pair_category"] == n_cat]["mono_p_full"].to_numpy()
            prob = mann_whitney_concordance(p_scores, n_scores)
            pairwise_table[p_cat][n_cat] = round(prob, 4)
            row_vals.append(prob)

            # Accumulate in weighted global decomposition
            weight = (len(p_scores) * len(n_scores)) / (total_pos * total_neg)
            weighted_decomposed_auc += weight * prob

        # vs all negatives
        all_neg_scores = df_merged[is_neg]["mono_p_full"].to_numpy()
        vs_all_neg = mann_whitney_concordance(p_scores, all_neg_scores)
        pairwise_table[p_cat]["vs_all_negatives"] = round(vs_all_neg, 4)

        row_str = f"{p_cat:<40} | " + " | ".join([f"{v:>15.4f}" for v in row_vals]) + f" | {vs_all_neg:>15.4f}"
        print(row_str)

    # Global AUROC verification
    all_pos_scores = df_merged[is_pos]["mono_p_full"].to_numpy()
    all_neg_scores = df_merged[is_neg]["mono_p_full"].to_numpy()
    direct_auc = mann_whitney_concordance(all_pos_scores, all_neg_scores)
    sklearn_auc = roc_auc_score(is_pos.astype(int), df_merged["mono_p_full"].to_numpy())

    print("-" * 115)
    print(f"Direct Mann-Whitney AUROC:      {direct_auc:.4f}")
    print(f"Weighted Decomposed AUROC:     {weighted_decomposed_auc:.4f}")
    print(f"Scikit-Learn roc_auc_score:    {sklearn_auc:.4f}")
    print("=" * 115 + "\n")

    report = {
        "continuous_score_distributions": dist_summaries,
        "pairwise_mann_whitney_concordance": pairwise_table,
        "overall_metrics": {
            "direct_mann_whitney_auroc": round(float(direct_auc), 4),
            "weighted_decomposed_auroc": round(float(weighted_decomposed_auc), 4),
            "sklearn_auroc": round(float(sklearn_auc), 4)
        }
    }

    save_json(report, out_json)
    logger.info(f"Saved continuous score audit report to {out_json}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit V4 Continuous Score Distributions")
    parser.add_argument("--v4_mono_csv", type=str, default="results/metrics/v4_1118_predictions.csv")
    parser.add_argument("--v4_chunk_csv", type=str, default="results/metrics/v4_chunked_1118_predictions.csv")
    parser.add_argument("--test_csv", type=str, default="data/nli_training_v3/test.csv")
    parser.add_argument("--output_report", type=str, default="results/metrics/v4_continuous_score_audit.json")

    args = parser.parse_args()
    run_continuous_score_audit(
        v4_mono_csv=args.v4_mono_csv,
        v4_chunk_csv=args.v4_chunk_csv,
        test_csv=args.test_csv,
        output_report_json=args.output_report
    )

