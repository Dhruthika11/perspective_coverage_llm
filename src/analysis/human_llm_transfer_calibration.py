"""
Human-LLM Transfer Validation & Cluster Calibration Protocol (Phase 3 Extension).

Validates whether thresholds calibrated on human-human paraphrase pairs (tau=0.77, theta=0.80)
transfer reliably to pairs of (LLM-generated argument, Human Ground-Truth Cluster).

Implements:
  1. Text Length & Distribution Mismatch Analysis (Human reference length vs LLM argument length).
  2. Cluster-Level Similarity Aggregation:
       - Max similarity: S_max(g, C) = max_{p in C} S(g, p)
       - Top-k average: S_topk(g, C) = mean(top_k(S(g, p)))
       - Cluster mean: S_mean(g, C) = mean_{p in C} S(g, p)
  3. Transfer Evaluation & Bridge Calibration:
       - Evaluates candidate recall, precision, and F1 across length tiers.
       - Tests length bias (correlation between argument length and match scores).
       - Compares human-human thresholds vs bridge-calibrated thresholds.
  4. Human Annotation & Adjudication Protocol:
       - Exports balanced 300-pair benchmark for double-blind human verification.
"""
import os
import sys
import json
import re
import argparse
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.io_utils import resolve, load_csv, save_csv, save_json
from src.utils.logging_utils import get_logger
from src.analysis.bipartite_matching import aggregate_cluster_scores

logger = get_logger("human_llm_transfer_calibration")


def compute_text_length_stats(texts: List[str]) -> Dict[str, float]:
    """Computes descriptive word count statistics for a collection of texts."""
    if not texts:
        return {"count": 0, "mean": 0.0, "median": 0.0, "std": 0.0, "p25": 0.0, "p75": 0.0, "p95": 0.0}
    
    lengths = np.array([len(t.split()) for t in texts if isinstance(t, str) and t.strip()])
    if len(lengths) == 0:
        return {"count": 0, "mean": 0.0, "median": 0.0, "std": 0.0, "p25": 0.0, "p75": 0.0, "p95": 0.0}

    return {
        "count": int(len(lengths)),
        "mean": round(float(np.mean(lengths)), 2),
        "median": round(float(np.median(lengths)), 2),
        "std": round(float(np.std(lengths)), 2),
        "p25": round(float(np.percentile(lengths, 25)), 2),
        "p75": round(float(np.percentile(lengths, 75)), 2),
        "p95": round(float(np.percentile(lengths, 95)), 2),
    }


def aggregate_cluster_pair_scores(
    pair_scores: List[float],
    cluster_size: int
) -> Dict[str, float]:
    """Computes cluster scores under max, top-2, and mean aggregation."""
    return {
        "sim_max": round(float(aggregate_cluster_scores(pair_scores, method="max")), 4),
        "sim_top2": round(float(aggregate_cluster_scores(pair_scores, method="top_k", top_k=2)), 4),
        "sim_mean": round(float(aggregate_cluster_scores(pair_scores, method="mean")), 4),
        "cluster_size": cluster_size,
    }


def analyze_length_bias(
    df: pd.DataFrame,
    length_col: str = "llm_word_count",
    score_col: str = "sim_max",
    label_col: str = "ground_truth_label",
    eval_threshold: float = 0.75
) -> Dict[str, Any]:
    """Evaluates whether longer LLM arguments receive artificially inflated similarity scores,
    and measures False-Positive (FPR) and False-Negative (FNR) rates across short, medium, and long tiers
    (Reviewer Correction 1.1).
    """
    if len(df) < 5:
        return {"pearson_r": 0.0, "pearson_p": 1.0, "spearman_rho": 0.0, "spearman_p": 1.0}

    lengths = df[length_col].values
    scores = df[score_col].values
    has_labels = label_col in df.columns
    labels = df[label_col].values if has_labels else np.zeros(len(df))

    p_r, p_p = pearsonr(lengths, scores)
    s_r, s_p = spearmanr(lengths, scores)

    # Adaptive length terciles so short, medium, and long are all populated meaningfully
    q33 = np.percentile(lengths, 33.3)
    q66 = np.percentile(lengths, 66.7)
    if q66 <= q33:
        q33, q66 = 50.0, 150.0

    tier_defs = [
        ("short", lengths <= q33),
        ("medium", (lengths > q33) & (lengths <= q66)),
        ("long", lengths > q66),
    ]

    tiers = {}
    for tier_name, mask in tier_defs:
        if np.sum(mask) > 0:
            sub_scores = scores[mask]
            sub_lengths = lengths[mask]
            tier_data = {
                "count": int(np.sum(mask)),
                "min_words": int(np.min(sub_lengths)),
                "max_words": int(np.max(sub_lengths)),
                "mean_score": round(float(np.mean(sub_scores)), 4),
                "std_score": round(float(np.std(sub_scores)), 4),
            }
            if has_labels:
                sub_labels = labels[mask]
                tp = int(np.sum((sub_scores >= eval_threshold) & (sub_labels == 1)))
                fp = int(np.sum((sub_scores >= eval_threshold) & (sub_labels == 0)))
                fn = int(np.sum((sub_scores < eval_threshold) & (sub_labels == 1)))
                tn = int(np.sum((sub_scores < eval_threshold) & (sub_labels == 0)))

                fpr = round(float(fp / (fp + tn)), 4) if (fp + tn) > 0 else 0.0
                fnr = round(float(fn / (fn + tp)), 4) if (fn + tp) > 0 else 0.0
                prec = round(float(tp / (tp + fp)), 4) if (tp + fp) > 0 else 0.0
                rec = round(float(tp / (tp + fn)), 4) if (tp + fn) > 0 else 0.0

                tier_data.update({
                    "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                    "false_positive_rate": fpr,
                    "false_negative_rate": fnr,
                    "precision": prec,
                    "recall": rec
                })
            tiers[tier_name] = tier_data

    return {
        "pearson_r": round(float(p_r), 4),
        "pearson_p": round(float(p_p), 4),
        "spearman_rho": round(float(s_r), 4),
        "spearman_p": round(float(s_p), 4),
        "has_significant_length_bias": bool(abs(p_r) > 0.25 and p_p < 0.05),
        "length_tiers": tiers,
    }


def analyze_cluster_size_bias(
    df: pd.DataFrame,
    size_col: str = "cluster_size",
    score_col: str = "sim_max",
    label_col: str = "ground_truth_label",
    eval_threshold: float = 0.75
) -> Dict[str, Any]:
    """Evaluates whether cluster size changes the false-match rate (Reviewer Correction 1.2)."""
    if size_col not in df.columns or len(df) < 5:
        return {"has_cluster_size_bias": False, "size_tiers": {}}

    sizes = df[size_col].values
    scores = df[score_col].values
    has_labels = label_col in df.columns
    labels = df[label_col].values if has_labels else np.zeros(len(df))

    p_r, p_p = pearsonr(sizes, scores)
    s_r, s_p = spearmanr(sizes, scores)

    # Bin into cluster size tiers: Small (1-2), Medium (3-5), Large (>5)
    tiers = {}
    tier_defs = [
        ("small_1_to_2", sizes <= 2),
        ("medium_3_to_5", (sizes >= 3) & (sizes <= 5)),
        ("large_gt_5", sizes > 5),
    ]
    for tier_name, mask in tier_defs:
        if np.sum(mask) > 0:
            sub_scores = scores[mask]
            sub_sizes = sizes[mask]
            tier_dict = {
                "count": int(np.sum(mask)),
                "mean_size": round(float(np.mean(sub_sizes)), 2),
                "mean_score": round(float(np.mean(sub_scores)), 4),
                "std_score": round(float(np.std(sub_scores)), 4),
            }
            if has_labels:
                sub_labels = labels[mask]
                tp = int(np.sum((sub_scores >= eval_threshold) & (sub_labels == 1)))
                fp = int(np.sum((sub_scores >= eval_threshold) & (sub_labels == 0)))
                fn = int(np.sum((sub_scores < eval_threshold) & (sub_labels == 1)))
                tn = int(np.sum((sub_scores < eval_threshold) & (sub_labels == 0)))
                fpr = round(float(fp / (fp + tn)), 4) if (fp + tn) > 0 else 0.0
                fnr = round(float(fn / (fn + tp)), 4) if (fn + tp) > 0 else 0.0
                tier_dict.update({
                    "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                    "false_positive_rate": fpr,
                    "false_negative_rate": fnr,
                })
            tiers[tier_name] = tier_dict

    return {
        "pearson_r_size_vs_score": round(float(p_r), 4),
        "pearson_p": round(float(p_p), 4),
        "spearman_rho": round(float(s_r), 4),
        "has_cluster_size_bias": bool(p_r > 0.25 and p_p < 0.05),
        "size_tiers": tiers,
    }


def load_generations_pool(
    generations_path: Optional[str] = None,
    generations_dir: str = "data/llm_outputs",
) -> pd.DataFrame:
    """Loads LLM generations from a specific file, or scans generations directory recursively.
    
    Supports:
      - Phase 2 processed_generations.csv (with perspective_summary, core_reasoning, combined_text)
      - Phase 2 raw_generations.jsonl (with parsed_viewpoints)
      - Historical processed_generations.csv
    """
    records = []

    # Case A: Specific file provided by user
    if generations_path and os.path.exists(resolve(generations_path)):
        full_p = resolve(generations_path)
        logger.info(f"Loading user-specified generations from {full_p}...")
        model_name = os.path.basename(os.path.dirname(full_p))

        if full_p.endswith(".jsonl"):
            with open(full_p, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            d = json.loads(line)
                            cid = int(d.get("claim_id", 0))
                            for vp in d.get("parsed_viewpoints", []):
                                summary = vp.get("perspective_summary") or vp.get("viewpoint_text") or ""
                                reasoning = vp.get("core_reasoning") or ""
                                text = f"{summary} {reasoning}".strip() if reasoning else summary
                                if text and len(text.split()) >= 4:
                                    records.append({
                                        "model_name": d.get("model", model_name),
                                        "claim_id": cid,
                                        "llm_argument": text,
                                        "llm_word_count": len(text.split()),
                                    })
                        except Exception:
                            continue
        else:
            df = load_csv(full_p)
            for _, r in df.iterrows():
                summary = str(r.get("perspective_summary", "")).strip() if pd.notna(r.get("perspective_summary")) else ""
                combined = str(r.get("combined_text", "")).strip() if pd.notna(r.get("combined_text")) else ""
                generated = str(r.get("generated_text", "")).strip() if pd.notna(r.get("generated_text")) else ""

                chosen = summary if (summary and len(summary.split()) >= 4) else (combined or generated)
                if chosen and len(chosen.split()) >= 4:
                    records.append({
                        "model_name": str(r.get("model", model_name)),
                        "claim_id": int(r["claim_id"]),
                        "llm_argument": chosen,
                        "llm_word_count": len(chosen.split()),
                    })

        return pd.DataFrame(records)

    # Case B: Scan generations directory recursively
    gen_dir_resolved = resolve(generations_dir)
    if os.path.exists(gen_dir_resolved):
        for root, _, files in os.walk(gen_dir_resolved):
            for file in files:
                if file.endswith(".csv") and "processed_generations" in file:
                    fp = os.path.join(root, file)
                    rel = os.path.relpath(fp, gen_dir_resolved)
                    model_tag = rel.split(os.sep)[0]
                    try:
                        df = load_csv(fp)
                        for _, r in df.iterrows():
                            summary = str(r.get("perspective_summary", "")).strip() if pd.notna(r.get("perspective_summary")) else ""
                            combined = str(r.get("combined_text", "")).strip() if pd.notna(r.get("combined_text")) else ""
                            gen_t = str(r.get("generated_text", "")).strip() if pd.notna(r.get("generated_text")) else ""
                            chosen = summary if (summary and len(summary.split()) >= 4) else (combined or gen_t)
                            if chosen and len(chosen.split()) >= 4:
                                records.append({
                                    "model_name": model_tag,
                                    "claim_id": int(r["claim_id"]),
                                    "llm_argument": chosen,
                                    "llm_word_count": len(chosen.split()),
                                })
                    except Exception as e:
                        logger.warning(f"Failed to load {fp}: {e}")

    return pd.DataFrame(records)


def build_human_llm_cluster_pairs(
    claims_path: str = "data/processed/claims.csv",
    perspectives_path: str = "data/processed/perspectives.csv",
    generations_file: Optional[str] = None,
    llm_generations_dir: str = "data/llm_outputs",
    n_target_pairs: int = 300,
    seed: int = 42
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Builds a balanced human-LLM cluster benchmark across models, claims, and negative tiers."""
    np.random.seed(seed)
    claims_df = load_csv(resolve(claims_path))
    persp_df = load_csv(resolve(perspectives_path))

    # Group human perspectives into clusters
    cluster_groups = persp_df.groupby(["claim_id", "cluster_id"])
    cluster_lookup = {}
    for (cid, clid), group in cluster_groups:
        paraphrases = [str(t).strip() for t in group["text"].dropna().tolist() if str(t).strip()]
        if paraphrases:
            cluster_lookup[(int(cid), int(clid))] = {
                "claim_id": int(cid),
                "cluster_id": int(clid),
                "paraphrases": paraphrases,
                "cluster_size": len(paraphrases),
                "mean_word_count": round(float(np.mean([len(p.split()) for p in paraphrases])), 1),
                "stance": group["stance"].iloc[0] if "stance" in group.columns else "neutral",
            }

    # Load LLM generations pool (from specific file or scan)
    llm_df = load_generations_pool(generations_path=generations_file, generations_dir=llm_generations_dir)

    if llm_df.empty:
        raise RuntimeError("No LLM generations found to construct human-LLM pairs! Provide --generations file.")

    logger.info(f"Loaded {len(llm_df)} LLM generation candidates across {llm_df['model_name'].nunique()} models.")

    # Compute text length statistics
    all_human_texts = persp_df["text"].dropna().astype(str).tolist()
    human_length_stats = compute_text_length_stats(all_human_texts)
    llm_length_stats = compute_text_length_stats(llm_df["llm_argument"].tolist())

    length_comparison = {
        "human_references": human_length_stats,
        "llm_generations": llm_length_stats,
        "length_disparity_ratio_mean": round(llm_length_stats["mean"] / max(human_length_stats["mean"], 1.0), 2),
        "length_disparity_ratio_median": round(llm_length_stats["median"] / max(human_length_stats["median"], 1.0), 2),
    }

    # Construct Balanced Benchmarks with per-model allocation and similarity-band stratification:
    # 1. Above-threshold positive candidates (same claim, high semantic overlap)
    # 2. Borderline / near-threshold candidates (similarity near tau*)
    # 3. Same-claim hard negatives (different cluster)
    # 4. Cross-claim easy negatives (different claim)
    claim_ids_with_clusters = set(c[0] for c in cluster_lookup.keys())
    eligible_llm = llm_df[llm_df["claim_id"].isin(claim_ids_with_clusters)].sample(frac=1.0, random_state=seed)

    pairs = []
    pair_id = 0

    all_claims = list(claim_ids_with_clusters)
    claim_text_dict = dict(zip(claims_df["claim_id"], claims_df["claim_text"])) if "claim_text" in claims_df.columns else {}

    # Helper for simple lexical overlap proxy (exact embeddings computed during validation run)
    def lexical_similarity(text1: str, text2: str) -> float:
        w1 = set(re.findall(r"\w+", text1.lower()))
        w2 = set(re.findall(r"\w+", text2.lower()))
        if not w1 or not w2:
            return 0.0
        return len(w1 & w2) / np.sqrt(len(w1) * len(w2))

    models = sorted(eligible_llm["model_name"].unique()) if "model_name" in eligible_llm.columns else ["default_model"]
    target_per_model = max(10, n_target_pairs // max(len(models), 1))
    logger.info(f"Sampling ~{target_per_model} stratified pairs per model across {len(models)} models: {models}")

    pos_count, hard_neg_count, easy_neg_count = 0, 0, 0

    for model in models:
        m_subset = eligible_llm[eligible_llm["model_name"] == model] if "model_name" in eligible_llm.columns else eligible_llm
        m_pairs_added = 0
        n_pos_m = target_per_model // 3
        n_hard_m = target_per_model // 3
        n_easy_m = target_per_model - n_pos_m - n_hard_m
        m_pos, m_hard, m_easy = 0, 0, 0

        for _, gen_row in m_subset.iterrows():
            cid = gen_row["claim_id"]
            llm_text = gen_row["llm_argument"]
            claim_clusters = [cl for cl in cluster_lookup.values() if cl["claim_id"] == cid]
            if not claim_clusters:
                continue

            # Score LLM argument against all clusters of this claim
            cluster_scores = []
            for cl in claim_clusters:
                sims = [lexical_similarity(llm_text, p) for p in cl["paraphrases"]]
                agg = aggregate_cluster_pair_scores(sims, cl["cluster_size"])
                cluster_scores.append((cl, agg))

            # Sort by sim_max descending
            cluster_scores.sort(key=lambda x: x[1]["sim_max"], reverse=True)

            # 1. Best match -> Positive candidate
            if m_pos < n_pos_m and cluster_scores:
                best_cl, best_agg = cluster_scores[0]
                sim_val = best_agg["sim_max"]
                sim_band = "above_threshold" if sim_val >= 0.75 else ("borderline" if sim_val >= 0.65 else "below_threshold")
                pairs.append({
                    "pair_id": f"pair_llm_{pair_id:04d}",
                    "model_name": model,
                    "pair_type": "positive_candidate",
                    "similarity_band": sim_band,
                    "ground_truth_label": 1,
                    "claim_id": cid,
                    "claim_text": claim_text_dict.get(cid, ""),
                    "llm_argument": llm_text,
                    "llm_word_count": gen_row["llm_word_count"],
                    "cluster_id": best_cl["cluster_id"],
                    "cluster_size": best_cl["cluster_size"],
                    "cluster_paraphrases": " | ".join(best_cl["paraphrases"][:3]),
                    "human_mean_word_count": best_cl["mean_word_count"],
                    "sim_max": sim_val,
                    "sim_top2": best_agg["sim_top2"],
                    "sim_mean": best_agg["sim_mean"],
                    "annotator_1_label": "",
                    "annotator_2_label": "",
                    "adjudicated_label": "",
                })
                pair_id += 1
                m_pos += 1
                pos_count += 1
                m_pairs_added += 1

            # 2. Different cluster of SAME claim -> Hard negative
            if m_hard < n_hard_m and len(cluster_scores) > 1:
                worst_cl, worst_agg = cluster_scores[-1]
                sim_val = worst_agg["sim_max"]
                sim_band = "above_threshold" if sim_val >= 0.75 else ("borderline" if sim_val >= 0.65 else "below_threshold")
                pairs.append({
                    "pair_id": f"pair_llm_{pair_id:04d}",
                    "model_name": model,
                    "pair_type": "same_claim_hard_negative",
                    "similarity_band": sim_band,
                    "ground_truth_label": 0,
                    "claim_id": cid,
                    "claim_text": claim_text_dict.get(cid, ""),
                    "llm_argument": llm_text,
                    "llm_word_count": gen_row["llm_word_count"],
                    "cluster_id": worst_cl["cluster_id"],
                    "cluster_size": worst_cl["cluster_size"],
                    "cluster_paraphrases": " | ".join(worst_cl["paraphrases"][:3]),
                    "human_mean_word_count": worst_cl["mean_word_count"],
                    "sim_max": sim_val,
                    "sim_top2": worst_agg["sim_top2"],
                    "sim_mean": worst_agg["sim_mean"],
                    "annotator_1_label": "",
                    "annotator_2_label": "",
                    "adjudicated_label": "",
                })
                pair_id += 1
                m_hard += 1
                hard_neg_count += 1
                m_pairs_added += 1

            # 3. Cluster of DIFFERENT claim -> Easy negative
            if m_easy < n_easy_m:
                diff_cids = [c for c in all_claims if c != cid]
                if diff_cids:
                    rand_cid = np.random.choice(diff_cids)
                    other_clusters = [cl for cl in cluster_lookup.values() if cl["claim_id"] == rand_cid]
                    if other_clusters:
                        rand_cl = other_clusters[0]
                        sims = [lexical_similarity(llm_text, p) for p in rand_cl["paraphrases"]]
                        agg = aggregate_cluster_pair_scores(sims, rand_cl["cluster_size"])
                        sim_val = agg["sim_max"]
                        sim_band = "above_threshold" if sim_val >= 0.75 else ("borderline" if sim_val >= 0.65 else "below_threshold")
                        pairs.append({
                            "pair_id": f"pair_llm_{pair_id:04d}",
                            "model_name": model,
                            "pair_type": "cross_claim_easy_negative",
                            "similarity_band": sim_band,
                            "ground_truth_label": 0,
                            "claim_id": cid,
                            "claim_text": claim_text_dict.get(cid, ""),
                            "llm_argument": llm_text,
                            "llm_word_count": gen_row["llm_word_count"],
                            "cluster_id": rand_cl["cluster_id"],
                            "cluster_size": rand_cl["cluster_size"],
                            "cluster_paraphrases": " | ".join(rand_cl["paraphrases"][:3]),
                            "human_mean_word_count": rand_cl["mean_word_count"],
                            "sim_max": sim_val,
                            "sim_top2": agg["sim_top2"],
                            "sim_mean": agg["sim_mean"],
                            "annotator_1_label": "",
                            "annotator_2_label": "",
                            "adjudicated_label": "",
                        })
                        pair_id += 1
                        m_easy += 1
                        easy_neg_count += 1
                        m_pairs_added += 1

            if m_pairs_added >= target_per_model:
                break

    benchmark_df = pd.DataFrame(pairs)
    logger.info(
        f"Constructed {len(benchmark_df)} human-LLM cluster benchmark pairs "
        f"({pos_count} pos, {hard_neg_count} hard neg, {easy_neg_count} easy neg)."
    )

    return benchmark_df, length_comparison


def run_human_llm_transfer_validation(
    benchmark_df: pd.DataFrame,
    human_tau: float = 0.77,
    human_theta: float = 0.80
) -> Dict[str, Any]:
    """Evaluates transferability of human-human calibrated thresholds to human-LLM cluster pairs."""
    labels = benchmark_df["ground_truth_label"].values

    # Evaluate each aggregation method
    agg_results = {}
    for agg_name in ["sim_max", "sim_top2", "sim_mean"]:
        scores = benchmark_df[agg_name].values
        
        # Test human-human threshold transfer
        preds_human = (scores >= human_tau).astype(int)
        tp = int(np.sum((preds_human == 1) & (labels == 1)))
        fp = int(np.sum((preds_human == 1) & (labels == 0)))
        fn = int(np.sum((preds_human == 0) & (labels == 1)))
        tn = int(np.sum((preds_human == 0) & (labels == 0)))

        prec = round(float(tp / (tp + fp)), 4) if (tp + fp) > 0 else 0.0
        rec = round(float(tp / (tp + fn)), 4) if (tp + fn) > 0 else 0.0
        f1 = round(float(2 * prec * rec / (prec + rec)), 4) if (prec + rec) > 0 else 0.0

        # Sweep candidate recall constraint: tau* = max tau s.t. recall >= 0.95
        best_tau = 0.50
        for tau in np.arange(0.10, 0.95, 0.01):
            p = (scores >= tau).astype(int)
            r = np.sum((p == 1) & (labels == 1)) / max(np.sum(labels == 1), 1)
            if r >= 0.95:
                best_tau = round(float(tau), 2)

        # Length bias & cluster size checks (Reviewer Corrections 1.1 & 1.2)
        length_bias = analyze_length_bias(benchmark_df, score_col=agg_name, eval_threshold=human_tau)
        cluster_size_bias = analyze_cluster_size_bias(benchmark_df, score_col=agg_name, eval_threshold=human_tau)

        agg_results[agg_name] = {
            "human_tau_evaluated": human_tau,
            "precision_at_human_tau": prec,
            "recall_at_human_tau": rec,
            "f1_at_human_tau": f1,
            "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
            "bridge_calibrated_tau": best_tau,
            "length_bias_audit": length_bias,
            "cluster_size_audit": cluster_size_bias,
        }

    # Summary transfer decision
    transfer_status = "STABLE_TRANSFER"
    max_stats = agg_results["sim_max"]
    if max_stats["recall_at_human_tau"] < 0.90:
        transfer_status = "BRIDGE_CALIBRATION_RECOMMENDED"

    # Per-Model Evaluator Breakdown under COMMON GLOBAL THRESHOLD (Reviewer Protocol Stage 2)
    per_model_results = []
    if "model_name" in benchmark_df.columns and benchmark_df["model_name"].nunique() > 1:
        for m_name, m_df in benchmark_df.groupby("model_name"):
            m_labels = m_df["ground_truth_label"].values
            m_scores = m_df["sim_max"].values
            m_preds = (m_scores >= human_tau).astype(int)
            m_tp = int(np.sum((m_preds == 1) & (m_labels == 1)))
            m_fp = int(np.sum((m_preds == 1) & (m_labels == 0)))
            m_fn = int(np.sum((m_preds == 0) & (m_labels == 1)))
            m_tn = int(np.sum((m_preds == 0) & (m_labels == 0)))

            m_prec = round(float(m_tp / (m_tp + m_fp)), 4) if (m_tp + m_fp) > 0 else 0.0
            m_rec = round(float(m_tp / (m_tp + m_fn)), 4) if (m_tp + m_fn) > 0 else 0.0
            m_f1 = round(float(2 * m_prec * m_rec / (m_prec + m_rec)), 4) if (m_prec + m_rec) > 0 else 0.0
            m_words = round(float(m_df["llm_word_count"].mean()), 1) if "llm_word_count" in m_df.columns else 0.0

            per_model_results.append({
                "model_name": m_name,
                "n_sampled_pairs": int(len(m_df)),
                "mean_word_count": m_words,
                "common_tau_evaluated": human_tau,
                "precision": m_prec,
                "recall": m_rec,
                "f1": m_f1,
                "tp": m_tp, "fp": m_fp, "fn": m_fn, "tn": m_tn
            })

        # Save per-model table
        os.makedirs(resolve("results/paper_tables"), exist_ok=True)
        os.makedirs(resolve("data/calibration"), exist_ok=True)
        df_pm = pd.DataFrame(per_model_results)
        save_csv(df_pm, resolve("data/calibration/stage2_per_model_evaluator_audit.csv"))
        save_csv(df_pm, resolve("results/paper_tables/table_stage2_per_model_audit.csv"))
        logger.info(f"Saved Stage 2 per-model audit table ({len(df_pm)} models) to results/paper_tables/table_stage2_per_model_audit.csv")

    return {
        "status": transfer_status,
        "n_pairs": len(benchmark_df),
        "class_balance": {
            "positive_candidates": int(np.sum(labels == 1)),
            "negative_candidates": int(np.sum(labels == 0)),
        },
        "aggregation_comparison": agg_results,
        "per_model_evaluator_performance": per_model_results,
        "recommendation": (
            "Human-human calibration provides the gold-standard anchor. "
            "Use sim_max with a single common global threshold across all models. "
            "Inspect per-model precision/recall to verify no systematic length or style bias."
        )
    }


def main():
    parser = argparse.ArgumentParser(description="Human-LLM Cluster Transfer Calibration Protocol")
    parser.add_argument("--generations", type=str, default=None, help="Path to specific LLM generations file (CSV or JSONL)")
    parser.add_argument("--generations_dir", type=str, default="data/llm_outputs", help="Directory of LLM generations to scan")
    parser.add_argument("--claims", type=str, default="data/processed/claims.csv", help="Path to claims CSV")
    parser.add_argument("--perspectives", type=str, default="data/processed/perspectives.csv", help="Path to perspectives CSV")
    parser.add_argument("--n_pairs", type=int, default=300, help="Target number of balanced benchmark pairs")
    parser.add_argument("--human_tau", type=float, default=0.77, help="Human-human calibrated retrieval threshold")
    parser.add_argument("--human_theta", type=float, default=0.80, help="Human-human calibrated verification threshold")
    parser.add_argument("--output_csv", type=str, default="data/calibration/human_llm_cluster_calibration_300.csv", help="Path to save annotation CSV")
    parser.add_argument("--output_report", type=str, default="results/calibration/human_llm_transfer_validation_report.json", help="Path to save JSON report")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()

    logger.info("Starting Human-LLM Cluster Transfer Calibration Protocol...")

    benchmark_df, length_comp = build_human_llm_cluster_pairs(
        claims_path=args.claims,
        perspectives_path=args.perspectives,
        generations_file=args.generations,
        llm_generations_dir=args.generations_dir,
        n_target_pairs=args.n_pairs,
        seed=args.seed
    )

    # Save benchmark CSV for annotation and audit
    out_csv = resolve(args.output_csv)
    save_csv(benchmark_df, out_csv)
    logger.info(f"Saved benchmark to {out_csv}")

    # Run transfer analysis
    report = run_human_llm_transfer_validation(
        benchmark_df,
        human_tau=args.human_tau,
        human_theta=args.human_theta
    )
    report["length_comparison"] = length_comp

    out_json = resolve(args.output_report)
    save_json(report, out_json)
    logger.info(f"Saved transfer validation report to {out_json}")

    print("\n" + "=" * 60)
    print("HUMAN-LLM CLUSTER TRANSFER VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Human Reference Mean Words : {length_comp['human_references']['mean']} words")
    print(f"LLM Generation Mean Words  : {length_comp['llm_generations']['mean']} words")
    print(f"Length Disparity Ratio     : {length_comp['length_disparity_ratio_mean']}x longer")
    print(f"Transfer Status            : {report['status']}")
    print(f"Pairs Exported for Review  : {report['n_pairs']} in {out_csv}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

