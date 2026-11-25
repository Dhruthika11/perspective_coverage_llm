"""
Marginal Coverage & Generation Dynamics Metrics (Phase 2).

Implements:
  - Step-by-step marginal coverage: Delta C_k = C_k - C_{k-1}
  - Distinct perspective count
  - Redundancy rate (arguments re-covering already covered reference clusters)
  - Abstention rate and output validity
  - Average argument length (word count)
  - Stance distribution across generation budget K
"""
import os
import json
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd


def compute_marginal_coverage_series(
    step_coverages: List[float]
) -> List[float]:
    """Computes marginal coverage gain Delta C_k = C_k - C_{k-1}."""
    if not step_coverages:
        return []
    deltas = []
    prev = 0.0
    for cov in step_coverages:
        delta = max(0.0, cov - prev)
        deltas.append(round(delta, 4))
        prev = cov
    return deltas


def analyze_generation_logs(
    raw_jsonl_path: str,
    k_budget: int = 5
) -> Dict[str, Any]:
    """Analyzes a generation manifest/jsonl log file to compute generation-level metrics:
      - Total attempts
      - Output validity rate
      - Abstention rate (overall and per aspect)
      - Stance distribution
      - Average argument length (words)
      - Criterion diversity
    """
    if not os.path.exists(raw_jsonl_path):
        raise FileNotFoundError(f"Log file not found: {raw_jsonl_path}")

    records = []
    with open(raw_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except Exception:
                pass

    if not records:
        return {"total_attempts": 0, "error": "empty log file"}

    df = pd.DataFrame(records)
    total_attempts = len(df)
    valid_count = int(df["is_valid"].sum()) if "is_valid" in df.columns else 0
    abstain_count = int(df["is_abstention"].sum()) if "is_abstention" in df.columns else 0

    validity_rate = valid_count / total_attempts if total_attempts > 0 else 0.0
    abstention_rate = abstain_count / total_attempts if total_attempts > 0 else 0.0

    # Stance distribution
    stances = df["stance"].value_counts().to_dict()

    # Word count of generated arguments (excluding abstentions)
    word_counts = []
    unique_criteria = set()
    for _, r in df.iterrows():
        if r.get("is_valid") and not r.get("is_abstention"):
            txt = str(r.get("parsed_output", {}).get("argument", ""))
            if txt:
                word_counts.append(len(txt.split()))
            crit = str(r.get("reasoning_criterion", "")).strip().lower()
            if crit and crit != "none" and crit != "unspecified":
                unique_criteria.add(crit)

    avg_words = float(np.mean(word_counts)) if word_counts else 0.0

    # Per-aspect abstention breakdown
    aspect_abstentions = {}
    if "aspect" in df.columns:
        for asp, grp in df.groupby("aspect"):
            aspect_abstentions[asp] = {
                "attempts": len(grp),
                "abstained": int(grp["is_abstention"].sum()),
                "abstention_rate": round(float(grp["is_abstention"].mean()), 3),
            }

    return {
        "total_attempts": total_attempts,
        "valid_generations": valid_count,
        "validity_rate": round(validity_rate, 4),
        "abstained_generations": abstain_count,
        "abstention_rate": round(abstention_rate, 4),
        "stance_distribution": stances,
        "average_argument_words": round(avg_words, 2),
        "distinct_reasoning_criteria_count": len(unique_criteria),
        "aspect_breakdown": aspect_abstentions,
    }


def compute_stepwise_cluster_coverage(
    claim_id: int,
    step_predictions: List[Dict[str, Any]],
    gold_clusters: List[Dict[str, Any]],
    similarity_fn: Any
) -> Dict[str, Any]:
    """Computes cumulative and marginal cluster coverage across sequential steps for a single claim.
    
    Args:
        claim_id: Claim ID.
        step_predictions: List of dicts representing generated viewpoints ordered by step (0..K-1).
        gold_clusters: List of ground-truth cluster objects for this claim.
        similarity_fn: Function (viewpoint_text, cluster_paraphrases) -> bool (returns True if matches).
    """
    total_clusters = len(gold_clusters)
    if total_clusters == 0:
        return {"claim_id": claim_id, "cumulative_coverage": [], "marginal_coverage": []}

    covered_cluster_ids = set()
    cumulative_cov = []
    redundancy_count = 0

    for step, pred in enumerate(step_predictions):
        v_text = pred.get("argument", pred.get("combined_text", ""))
        newly_covered = set()

        for c in gold_clusters:
            cid = c["cluster_id"]
            if cid in covered_cluster_ids:
                continue
            # Check if viewpoint matches any paraphrase in cluster
            paraphrases = c.get("paraphrases", [c.get("text", "")])
            if similarity_fn(v_text, paraphrases):
                newly_covered.add(cid)

        if not newly_covered and covered_cluster_ids:
            # If model didn't cover new clusters, check if it was redundant with already covered clusters
            is_redundant = any(similarity_fn(v_text, c.get("paraphrases", [c.get("text", "")])) for c in gold_clusters if c["cluster_id"] in covered_cluster_ids)
            if is_redundant:
                redundancy_count += 1

        covered_cluster_ids.update(newly_covered)
        current_cov = len(covered_cluster_ids) / total_clusters
        cumulative_cov.append(round(current_cov, 4))

    marginal_cov = compute_marginal_coverage_series(cumulative_cov)

    return {
        "claim_id": claim_id,
        "total_gold_clusters": total_clusters,
        "cumulative_coverage_by_step": cumulative_cov,
        "marginal_coverage_by_step": marginal_cov,
        "final_coverage": cumulative_cov[-1] if cumulative_cov else 0.0,
        "redundant_generations": redundancy_count,
        "redundancy_rate": round(redundancy_count / len(step_predictions), 4) if step_predictions else 0.0,
    }


def summarize_prompt_ablations(ablation_log_paths: Dict[str, str]) -> pd.DataFrame:
    """Compiles a comparative table across prompt ablation conditions (P1, P2, P3, P4).
    
    Args:
        ablation_log_paths: Dict mapping condition name (e.g. 'P1', 'P2') to raw_generations.jsonl path.
        
    Returns:
        DataFrame containing all 8 target metrics per condition.
    """
    rows = []
    for cond_name, log_path in ablation_log_paths.items():
        if not os.path.exists(log_path):
            continue
        stats = analyze_generation_logs(log_path)
        stances = stats.get("stance_distribution", {})
        sup = stances.get("support", 0)
        und = stances.get("undermine", 0)
        stance_ratio = round(sup / und, 2) if und > 0 else (float("inf") if sup > 0 else 1.0)

        rows.append({
            "Condition": cond_name,
            "Total Attempts": stats.get("total_attempts", 0),
            "Output Validity Rate": stats.get("validity_rate", 0.0),
            "Invalid Output Rate": round(1.0 - stats.get("validity_rate", 0.0), 4),
            "Abstention Rate": stats.get("abstention_rate", 0.0),
            "Unique Reasoning Criteria": stats.get("distinct_reasoning_criteria_count", 0),
            "Average Word Count": stats.get("average_argument_words", 0.0),
            "Support / Undermine Ratio": stance_ratio,
        })

    return pd.DataFrame(rows)


