"""
Tests for Human-LLM Transfer Validation and Cluster Aggregation Scoring.
"""
import pytest
import numpy as np
import pandas as pd
from src.analysis.bipartite_matching import (
    aggregate_cluster_scores,
    compute_claim_cluster_bipartite_coverage,
    solve_maximum_bipartite_matching
)
from src.analysis.human_llm_transfer_calibration import (
    compute_text_length_stats,
    aggregate_cluster_pair_scores,
    analyze_length_bias,
    analyze_cluster_size_bias,
    run_human_llm_transfer_validation
)


def test_aggregate_cluster_scores():
    scores = [0.85, 0.70, 0.40]
    
    # Test max
    assert aggregate_cluster_scores(scores, method="max") == 0.85
    
    # Test top_k (k=2) -> (0.85 + 0.70) / 2 = 0.775
    assert abs(aggregate_cluster_scores(scores, method="top_k", top_k=2) - 0.775) < 1e-4
    
    # Test mean -> (0.85 + 0.70 + 0.40) / 3 = 0.65
    assert abs(aggregate_cluster_scores(scores, method="mean") - 0.65) < 1e-4

    # Test edge cases
    assert aggregate_cluster_scores([], method="max") == 0.0
    assert aggregate_cluster_scores([None], method="max") == 0.0


def test_compute_text_length_stats():
    texts = [
        "Short perspective sentence.",
        "Another relatively short sentence for testing.",
        "This is an LLM generated argument that goes into much greater depth and explains the complete economic reasoning behind the proposal."
    ]
    stats = compute_text_length_stats(texts)
    assert stats["count"] == 3
    assert stats["mean"] > 0
    assert stats["median"] > 0
    assert stats["p75"] >= stats["median"]


def test_analyze_length_bias():
    df = pd.DataFrame({
        "llm_word_count": [10, 20, 30, 40, 50, 60, 70, 80],
        "sim_max": [0.75, 0.76, 0.74, 0.78, 0.75, 0.77, 0.76, 0.79]
    })
    bias_audit = analyze_length_bias(df, length_col="llm_word_count", score_col="sim_max")
    assert "pearson_r" in bias_audit
    assert "has_significant_length_bias" in bias_audit
    assert "length_tiers" in bias_audit
    assert not bias_audit["has_significant_length_bias"]


def test_analyze_cluster_size_bias():
    df = pd.DataFrame({
        "cluster_size": [1, 2, 3, 4, 6, 8],
        "sim_max": [0.70, 0.72, 0.74, 0.73, 0.75, 0.76],
        "ground_truth_label": [1, 0, 1, 0, 1, 0]
    })
    size_audit = analyze_cluster_size_bias(df, size_col="cluster_size", score_col="sim_max", eval_threshold=0.73)
    assert "has_cluster_size_bias" in size_audit
    assert "size_tiers" in size_audit
    assert "pearson_r_size_vs_score" in size_audit


def test_cluster_bipartite_matching_with_aggregation():
    viewpoints = [
        "Eliminates extreme poverty through basic monthly income.",
        "Robots and AI replace human workers in manufacturing."
    ]
    clusters = [
        ["UBI eradicates poverty.", "No one will starve under UBI."],
        ["Automation destroys routine jobs.", "AI takes factory employment."]
    ]
    cluster_ids = ["C1", "C2"]

    # Mock match function
    def mock_match(vp, p):
        if "poverty" in vp.lower() and "poverty" in p.lower():
            return 0.90
        if "starve" in p.lower() and "income" in vp.lower():
            return 0.70
        if "robot" in vp.lower() and "automation" in p.lower():
            return 0.85
        if "ai" in vp.lower() and "ai" in p.lower():
            return 0.88
        return 0.10

    # Max aggregation
    res_max = compute_claim_cluster_bipartite_coverage(
        viewpoints, clusters, cluster_ids, mock_match, cluster_agg="max"
    )
    assert res_max["n_matched"] == 2
    assert res_max["coverage_score"] == 1.0

    # Top-K aggregation
    res_topk = compute_claim_cluster_bipartite_coverage(
        viewpoints, clusters, cluster_ids, mock_match, cluster_agg="top_k", top_k=2
    )
    assert res_topk["n_matched"] == 2
    assert res_topk["coverage_score"] == 1.0


def test_transfer_validation_flow():
    mock_benchmark = pd.DataFrame({
        "pair_id": [f"pair_{i}" for i in range(10)],
        "ground_truth_label": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        "llm_word_count": [45, 50, 55, 60, 48, 42, 52, 58, 61, 47],
        "sim_max": [0.88, 0.85, 0.82, 0.89, 0.84, 0.25, 0.30, 0.22, 0.19, 0.28],
        "sim_top2": [0.85, 0.82, 0.80, 0.86, 0.81, 0.20, 0.25, 0.18, 0.15, 0.24],
        "sim_mean": [0.82, 0.80, 0.78, 0.83, 0.79, 0.18, 0.22, 0.15, 0.12, 0.20],
    })
    report = run_human_llm_transfer_validation(mock_benchmark, human_tau=0.77, human_theta=0.80)
    assert report["n_pairs"] == 10
    assert report["status"] == "STABLE_TRANSFER"
    assert report["aggregation_comparison"]["sim_max"]["recall_at_human_tau"] == 1.0

