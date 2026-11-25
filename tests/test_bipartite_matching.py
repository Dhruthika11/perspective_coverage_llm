"""
Unit tests for Maximum Bipartite Matching and Phase 3 Evaluator Calibration.
"""
import pytest
import numpy as np
from src.analysis.bipartite_matching import (
    solve_maximum_bipartite_matching,
    solve_sparse_claim_bipartite_matching
)
from src.analysis.calibrate_dense_retrieval import evaluate_cosine_sweep, select_calibrated_threshold
from src.analysis.perspective_verifier import compute_expected_calibration_error, select_precision_constrained_threshold


def test_bipartite_matching_injective_rule():
    """Test that multiple viewpoints matching the same cluster are only credited once."""
    # 3 generated viewpoints, 3 reference clusters
    # All 3 viewpoints strongly match Cluster 0, but have 0 match to Clusters 1 and 2
    match_matrix = np.array([
        [0.92, 0.0, 0.0],
        [0.88, 0.0, 0.0],
        [0.85, 0.0, 0.0],
    ])
    cluster_ids = ["cluster_0", "cluster_1", "cluster_2"]
    res = solve_maximum_bipartite_matching(match_matrix, cluster_ids)

    # Under greedy matching, an evaluator might falsely credit 3 matches.
    # Under Bipartite Matching, Cluster 0 can only be matched ONCE!
    assert res["n_matched"] == 1
    assert res["coverage_score"] == pytest.approx(1 / 3, abs=1e-3)
    assert len(res["covered_cluster_ids"]) == 1
    assert res["covered_cluster_ids"][0] == "cluster_0"
    assert len(res["unmatched_viewpoint_ids"]) == 2


def test_bipartite_matching_full_coverage():
    """Test that distinct viewpoints matching distinct clusters achieve 100% coverage."""
    match_matrix = np.array([
        [0.85, 0.0, 0.0],
        [0.0, 0.90, 0.0],
        [0.0, 0.0, 0.88],
    ])
    cluster_ids = ["c1", "c2", "c3"]
    res = solve_maximum_bipartite_matching(match_matrix, cluster_ids)

    assert res["n_matched"] == 3
    assert res["coverage_score"] == 1.0
    assert set(res["covered_cluster_ids"]) == {"c1", "c2", "c3"}
    assert len(res["unmatched_viewpoint_ids"]) == 0


def test_bipartite_matching_empty_cases():
    # Empty clusters
    res_empty_c = solve_maximum_bipartite_matching(np.zeros((3, 0)), [])
    assert res_empty_c["coverage_score"] == 0.0

    # Empty viewpoints
    res_empty_v = solve_maximum_bipartite_matching(np.zeros((0, 3)), ["c1", "c2", "c3"])
    assert res_empty_v["coverage_score"] == 0.0


def test_dense_retrieval_candidate_recall_rule():
    sims = np.array([0.85, 0.82, 0.79, 0.75, 0.70, 0.65, 0.60, 0.40])
    labels = np.array([2, 2, 2, 2, 1, 0, 0, 0])  # 4 positives (Class 2), 1 related, 3 negatives

    sweep_df = evaluate_cosine_sweep(sims, labels, thresholds=np.array([0.70, 0.75, 0.79, 0.82]))
    tau_star, stats = select_calibrated_threshold(sweep_df, target_recall=0.98)

    # To get 100% recall (4/4 positives), threshold must be <= 0.75
    assert tau_star <= 0.75
    assert stats["candidate_recall"] >= 0.98


def test_expected_calibration_error_bounds():
    y_true = np.array([1, 1, 0, 0, 1, 0, 1, 0, 1, 0])
    y_prob = np.array([0.9, 0.8, 0.2, 0.1, 0.7, 0.3, 0.85, 0.15, 0.95, 0.05])

    ece, bins = compute_expected_calibration_error(y_true, y_prob, n_bins=5)
    assert 0.0 <= ece <= 1.0
    assert len(bins) > 0


def test_sparse_claim_bipartite_matching_equivalence():
    """Verify exact equivalence between dense and sparse bipartite matching."""
    # 4 viewpoints, 3 clusters
    # Edge (0, 0) = 0.85, Edge (1, 0) = 0.90, Edge (1, 1) = 0.75, Edge (3, 2) = 0.60
    # Viewpoint 2 has 0 valid edges; Cluster 2 has 1 edge
    cluster_ids = ["c0", "c1", "c2"]
    candidate_edges = {
        (0, 0): 0.85,
        (1, 0): 0.90,
        (1, 1): 0.75,
        (3, 2): 0.60,
        (2, 1): 0.15  # below threshold 0.20 -> must be excluded!
    }

    # Dense matrix
    dense_matrix = np.array([
        [0.85, 0.0, 0.0],
        [0.90, 0.75, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.60]
    ])

    res_dense = solve_maximum_bipartite_matching(dense_matrix, cluster_ids)
    res_sparse = solve_sparse_claim_bipartite_matching(
        n_viewpoints=4,
        n_clusters=3,
        cluster_ids=cluster_ids,
        candidate_edges=candidate_edges,
        threshold=0.20
    )

    assert res_dense["n_matched"] == res_sparse["n_matched"] == 3
    assert res_dense["coverage_score"] == res_sparse["coverage_score"] == 1.0
    assert set(res_dense["covered_cluster_ids"]) == set(res_sparse["covered_cluster_ids"]) == {"c0", "c1", "c2"}


def test_sparse_claim_matching_zero_edges():
    """Verify O(1) early exit when no pairs passed Stage 2."""
    cluster_ids = ["c0", "c1", "c2"]
    candidate_edges = {
        (0, 1): 0.12,  # all below threshold
        (1, 0): 0.05
    }

    res = solve_sparse_claim_bipartite_matching(
        n_viewpoints=5,
        n_clusters=3,
        cluster_ids=cluster_ids,
        candidate_edges=candidate_edges,
        threshold=0.20
    )

    assert res["n_matched"] == 0
    assert res["coverage_score"] == 0.0
    assert len(res["covered_cluster_ids"]) == 0
    assert len(res["uncovered_cluster_ids"]) == 3


