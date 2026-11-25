"""
Maximum Bipartite Matching on Equivalence Clusters (Phase 3).

Replaces invalid greedy maximum similarity with Maximum Bipartite Matching
using the Kuhn-Munkres / Hungarian algorithm (scipy.optimize.linear_sum_assignment).

Mathematical Guarantees:
  1. Injective Mapping: One generated viewpoint covers at most one reference cluster.
  2. Non-Redundant Credit: One reference cluster is counted at most once per evaluation.
  3. Strict Bounded Metric: Coverage is strictly bounded in [0.0, 1.0].
     Coverage(C) = |M*| / M_clusters
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Dict, List, Set, Tuple, Optional, Any


def solve_maximum_bipartite_matching(
    match_matrix: np.ndarray,
    cluster_ids: List[Any],
    viewpoint_ids: Optional[List[Any]] = None
) -> Dict[str, Any]:
    """Solves maximum weight bipartite matching between generated viewpoints and reference clusters.
    
    Optimized to:
      1. Exclude zero-weight edges upfront.
      2. Avoid solving full dense matrix when edges are sparse.
      3. Solve on compact active bipartite subgraph.
      4. Early-exit in O(1) for claims with 0 or 1 active edges.
    
    Args:
        match_matrix: 2D array of shape (n_viewpoints, n_clusters) containing match scores in [0, 1].
                      Entries below verification threshold should be 0.0 or negative.
        cluster_ids: List of ground-truth cluster identifiers of length n_clusters.
        viewpoint_ids: Optional identifiers for the generated viewpoints of length n_viewpoints.
        
    Returns:
        Dictionary with matching results.
    """
    n_viewpoints, n_clusters = match_matrix.shape

    if viewpoint_ids is None:
        viewpoint_ids = list(range(n_viewpoints))

    if n_clusters == 0:
        return {
            "n_viewpoints": n_viewpoints,
            "n_clusters": 0,
            "n_matched": 0,
            "coverage_score": 0.0,
            "matched_pairs": [],
            "covered_cluster_ids": [],
            "uncovered_cluster_ids": [],
            "unmatched_viewpoint_ids": list(viewpoint_ids),
        }

    if n_viewpoints == 0:
        return {
            "n_viewpoints": 0,
            "n_clusters": n_clusters,
            "n_matched": 0,
            "coverage_score": 0.0,
            "matched_pairs": [],
            "covered_cluster_ids": [],
            "uncovered_cluster_ids": list(cluster_ids),
            "unmatched_viewpoint_ids": [],
        }

    # Extract non-zero candidate edges (exclude zero-weight pairs)
    valid_r, valid_c = np.where(match_matrix > 0)
    
    if len(valid_r) == 0:
        # Zero valid edges: O(1) early exit
        return {
            "n_viewpoints": n_viewpoints,
            "n_clusters": n_clusters,
            "n_matched": 0,
            "coverage_score": 0.0,
            "matched_pairs": [],
            "covered_cluster_ids": [],
            "uncovered_cluster_ids": list(cluster_ids),
            "unmatched_viewpoint_ids": list(viewpoint_ids),
        }
        
    if len(valid_r) == 1:
        # Single valid edge: O(1) direct match
        r, c = int(valid_r[0]), int(valid_c[0])
        score = float(match_matrix[r, c])
        vp_id = viewpoint_ids[r]
        cl_id = cluster_ids[c]
        covered_clusters = [cl_id]
        matched_pairs = [(vp_id, cl_id, round(score, 4))]
        uncovered_clusters = [cid for cid in cluster_ids if cid != cl_id]
        unmatched_viewpoints = [vpid for vpid in viewpoint_ids if vpid != vp_id]
        return {
            "n_viewpoints": n_viewpoints,
            "n_clusters": n_clusters,
            "n_matched": 1,
            "coverage_score": round(1.0 / n_clusters, 4),
            "matched_pairs": matched_pairs,
            "covered_cluster_ids": covered_clusters,
            "uncovered_cluster_ids": uncovered_clusters,
            "unmatched_viewpoint_ids": unmatched_viewpoints,
        }

    # Compact active bipartite subgraph: index only active rows and columns
    active_rows = np.unique(valid_r)
    active_cols = np.unique(valid_c)

    if len(active_rows) == 1 or len(active_cols) == 1:
        # Trivial star-graph: pick the single edge with the highest score
        best_idx = np.argmax([match_matrix[r, c] for r, c in zip(valid_r, valid_c)])
        r, c = int(valid_r[best_idx]), int(valid_c[best_idx])
        score = float(match_matrix[r, c])
        vp_id = viewpoint_ids[r]
        cl_id = cluster_ids[c]
        matched_pairs = [(vp_id, cl_id, round(score, 4))]
        covered_clusters = [cl_id]
        uncovered_clusters = [cid for cid in cluster_ids if cid != cl_id]
        unmatched_viewpoints = [vpid for vpid in viewpoint_ids if vpid != vp_id]
        return {
            "n_viewpoints": n_viewpoints,
            "n_clusters": n_clusters,
            "n_matched": 1,
            "coverage_score": round(1.0 / n_clusters, 4),
            "matched_pairs": matched_pairs,
            "covered_cluster_ids": covered_clusters,
            "uncovered_cluster_ids": uncovered_clusters,
            "unmatched_viewpoint_ids": unmatched_viewpoints,
        }

    row_map = {r: i for i, r in enumerate(active_rows)}
    col_map = {c: j for j, c in enumerate(active_cols)}

    n_act_rows = len(active_rows)
    n_act_cols = len(active_cols)

    cost_matrix = np.zeros((n_act_rows, n_act_cols), dtype=np.float64)
    for r, c in zip(valid_r, valid_c):
        s = float(match_matrix[r, c])
        cost_matrix[row_map[r], col_map[c]] = -1000.0 - s

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matched_pairs = []
    covered_clusters: Set[Any] = set()
    matched_viewpoints: Set[Any] = set()

    for r_idx, c_idx in zip(row_ind, col_ind):
        if cost_matrix[r_idx, c_idx] < 0:
            orig_r = int(active_rows[r_idx])
            orig_c = int(active_cols[c_idx])
            score = float(match_matrix[orig_r, orig_c])
            vp_id = viewpoint_ids[orig_r]
            cl_id = cluster_ids[orig_c]
            matched_pairs.append((vp_id, cl_id, round(score, 4)))
            covered_clusters.add(cl_id)
            matched_viewpoints.add(vp_id)

    n_matched = len(matched_pairs)
    coverage_score = n_matched / n_clusters if n_clusters > 0 else 0.0

    uncovered_clusters = [cid for cid in cluster_ids if cid not in covered_clusters]
    unmatched_viewpoints = [vpid for vpid in viewpoint_ids if vpid not in matched_viewpoints]

    return {
        "n_viewpoints": n_viewpoints,
        "n_clusters": n_clusters,
        "n_matched": n_matched,
        "coverage_score": round(coverage_score, 4),
        "matched_pairs": matched_pairs,
        "covered_cluster_ids": list(covered_clusters),
        "uncovered_cluster_ids": uncovered_clusters,
        "unmatched_viewpoint_ids": unmatched_viewpoints,
    }


def solve_sparse_claim_bipartite_matching(
    n_viewpoints: int,
    n_clusters: int,
    cluster_ids: List[Any],
    candidate_edges: Dict[Tuple[int, int], float],
    threshold: float = 0.20,
    viewpoint_ids: Optional[List[Any]] = None
) -> Dict[str, Any]:
    """Solves claim-level maximum weight bipartite matching on a sparse candidate graph.

    Optimizations:
      1. Constructs sparse candidate graphs: only pairs passing Stage 2 are considered.
      2. Excludes zero-weight and below-threshold edges (never added to cost structure).
      3. Solves independent claim-level assignment problems.
      4. Avoids constructing dense matrices for non-candidate pairs.
      5. O(1) early exit for claims with 0 or 1 candidate edges.

    Args:
        n_viewpoints: Total number of generated viewpoints for this claim.
        n_clusters: Total number of gold perspective clusters for this claim.
        cluster_ids: List of ground-truth cluster identifiers of length n_clusters.
        candidate_edges: Mapping (vp_idx, cluster_col_idx) -> verification_score.
                         Pairs that failed Stage 1 or Stage 2 are simply omitted from this dict.
        threshold: Decision threshold for acceptance (default: T* = 0.20).
        viewpoint_ids: Optional identifiers for the generated viewpoints of length n_viewpoints.

    Returns:
        Dictionary with matching results.
    """
    if viewpoint_ids is None:
        viewpoint_ids = list(range(n_viewpoints))

    if n_clusters == 0:
        return {
            "n_viewpoints": n_viewpoints,
            "n_clusters": 0,
            "n_matched": 0,
            "coverage_score": 0.0,
            "matched_pairs": [],
            "covered_cluster_ids": [],
            "uncovered_cluster_ids": [],
            "unmatched_viewpoint_ids": list(viewpoint_ids),
        }

    if n_viewpoints == 0:
        return {
            "n_viewpoints": 0,
            "n_clusters": n_clusters,
            "n_matched": 0,
            "coverage_score": 0.0,
            "matched_pairs": [],
            "covered_cluster_ids": [],
            "uncovered_cluster_ids": list(cluster_ids),
            "unmatched_viewpoint_ids": [],
        }

    # 1. Filter out edges below acceptance threshold (exclude zero-weight edges)
    filtered_edges = {
        (int(r), int(c)): float(score)
        for (r, c), score in candidate_edges.items()
        if score >= threshold
    }

    # 2. Early exit: 0 edges -> O(1)
    if not filtered_edges:
        return {
            "n_viewpoints": n_viewpoints,
            "n_clusters": n_clusters,
            "n_matched": 0,
            "coverage_score": 0.0,
            "matched_pairs": [],
            "covered_cluster_ids": [],
            "uncovered_cluster_ids": list(cluster_ids),
            "unmatched_viewpoint_ids": list(viewpoint_ids),
        }

    # 3. Early exit: 1 edge -> O(1)
    if len(filtered_edges) == 1:
        (r, c), s = next(iter(filtered_edges.items()))
        vp_id = viewpoint_ids[r]
        cl_id = cluster_ids[c]
        matched_pairs = [(vp_id, cl_id, round(s, 4))]
        covered_clusters = [cl_id]
        uncovered_clusters = [cid for cid in cluster_ids if cid != cl_id]
        unmatched_viewpoints = [vpid for vpid in viewpoint_ids if vpid != vp_id]
        return {
            "n_viewpoints": n_viewpoints,
            "n_clusters": n_clusters,
            "n_matched": 1,
            "coverage_score": round(1.0 / n_clusters, 4),
            "matched_pairs": matched_pairs,
            "covered_cluster_ids": covered_clusters,
            "uncovered_cluster_ids": uncovered_clusters,
            "unmatched_viewpoint_ids": unmatched_viewpoints,
        }

    # 4. Extract active vertices only
    active_rows = sorted(list(set(r for r, _ in filtered_edges.keys())))
    active_cols = sorted(list(set(c for _, c in filtered_edges.keys())))

    # Fast path: star graph
    if len(active_rows) == 1 or len(active_cols) == 1:
        (best_r, best_c), best_score = max(filtered_edges.items(), key=lambda item: item[1])
        vp_id = viewpoint_ids[best_r]
        cl_id = cluster_ids[best_c]
        matched_pairs = [(vp_id, cl_id, round(best_score, 4))]
        covered_clusters = [cl_id]
        uncovered_clusters = [cid for cid in cluster_ids if cid != cl_id]
        unmatched_viewpoints = [vpid for vpid in viewpoint_ids if vpid != vp_id]
        return {
            "n_viewpoints": n_viewpoints,
            "n_clusters": n_clusters,
            "n_matched": 1,
            "coverage_score": round(1.0 / n_clusters, 4),
            "matched_pairs": matched_pairs,
            "covered_cluster_ids": covered_clusters,
            "uncovered_cluster_ids": uncovered_clusters,
            "unmatched_viewpoint_ids": unmatched_viewpoints,
        }

    # 5. Compact active submatrix: size |V_active| x |C_active|
    row_map = {r: i for i, r in enumerate(active_rows)}
    col_map = {c: j for j, c in enumerate(active_cols)}

    n_act_rows = len(active_rows)
    n_act_cols = len(active_cols)

    cost_matrix = np.zeros((n_act_rows, n_act_cols), dtype=np.float64)
    for (r, c), score in filtered_edges.items():
        cost_matrix[row_map[r], col_map[c]] = -1000.0 - score

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matched_pairs = []
    covered_clusters: Set[Any] = set()
    matched_viewpoints: Set[Any] = set()

    for r_idx, c_idx in zip(row_ind, col_ind):
        if cost_matrix[r_idx, c_idx] < 0:
            orig_r = active_rows[r_idx]
            orig_c = active_cols[c_idx]
            score = filtered_edges[(orig_r, orig_c)]
            vp_id = viewpoint_ids[orig_r]
            cl_id = cluster_ids[orig_c]
            matched_pairs.append((vp_id, cl_id, round(score, 4)))
            covered_clusters.add(cl_id)
            matched_viewpoints.add(vp_id)

    n_matched = len(matched_pairs)
    coverage_score = n_matched / n_clusters if n_clusters > 0 else 0.0

    uncovered_clusters = [cid for cid in cluster_ids if cid not in covered_clusters]
    unmatched_viewpoints = [vpid for vpid in viewpoint_ids if vpid not in matched_viewpoints]

    return {
        "n_viewpoints": n_viewpoints,
        "n_clusters": n_clusters,
        "n_matched": n_matched,
        "coverage_score": round(coverage_score, 4),
        "matched_pairs": matched_pairs,
        "covered_cluster_ids": list(covered_clusters),
        "uncovered_cluster_ids": uncovered_clusters,
        "unmatched_viewpoint_ids": unmatched_viewpoints,
    }


def aggregate_cluster_scores(
    scores: List[float],
    method: str = "max",
    top_k: int = 2
) -> float:
    """Aggregates similarity/match scores between a viewpoint and multiple paraphrases in a cluster.
    
    Methods:
      - 'max': Maximum similarity across all paraphrases in the cluster (standard).
      - 'top_k': Average of the top-k highest similarity scores (reduces cluster-size bias).
      - 'mean': Average across all paraphrases in the cluster.
    """
    if not scores:
        return 0.0
    
    valid_scores = [float(s) for s in scores if s is not None]
    if not valid_scores:
        return 0.0
        
    if method == "max":
        return max(valid_scores)
    elif method == "top_k":
        sorted_scores = sorted(valid_scores, reverse=True)
        k = min(top_k, len(sorted_scores))
        return sum(sorted_scores[:k]) / k
    elif method == "mean":
        return sum(valid_scores) / len(valid_scores)
    else:
        raise ValueError(f"Unknown cluster aggregation method: {method}")


def compute_claim_cluster_bipartite_coverage(
    generated_viewpoints: List[str],
    cluster_paraphrase_lists: List[List[str]],
    cluster_ids: List[Any],
    match_fn: Any,
    cluster_agg: str = "max",
    top_k: int = 2
) -> Dict[str, Any]:
    """Computes cluster reference coverage for a single claim using bipartite matching.
    
    Args:
        generated_viewpoints: List of K generated viewpoint strings.
        cluster_paraphrase_lists: List of paraphrase lists, one per reference cluster.
        cluster_ids: List of ground-truth cluster identifiers.
        match_fn: Callable (viewpoint_text, paraphrase_text) -> float score (0.0 if not matched).
        cluster_agg: Strategy to aggregate scores across cluster paraphrases ('max', 'top_k', 'mean').
        top_k: K parameter if cluster_agg == 'top_k'.
    """
    n_vps = len(generated_viewpoints)
    n_cls = len(cluster_paraphrase_lists)

    match_matrix = np.zeros((n_vps, n_cls), dtype=np.float64)

    for i, vp_text in enumerate(generated_viewpoints):
        for j, paraphrases in enumerate(cluster_paraphrase_lists):
            p_scores = [float(match_fn(vp_text, p_text)) for p_text in paraphrases]
            match_matrix[i, j] = aggregate_cluster_scores(p_scores, method=cluster_agg, top_k=top_k)

    return solve_maximum_bipartite_matching(match_matrix, cluster_ids)

