"""
Direct V5 Perspective Coverage Evaluation for Unseen / Out-of-Family Generator Models.

Scope:
  Evaluates all models OUTSIDE the 4 primary generator families used to train V5
  (excluding Qwen family, Llama-3.1-8B, Gemma-2-9B, and OLMo-2-7B).

Target Unseen Architectures:
  1. Mistral-7B-Instruct-v0.3    (Mistral AI - completely unseen model family)
  2. Gemma-3-12B-IT              (Google - unseen next-generation architecture)
  3. Llama-3.2-1B-Instruct       (Meta - unseen 1B edge architecture)
  4. Llama-3.2-3B-Instruct       (Meta - unseen 3B edge architecture)
  5. Gemma-2-2B-IT               (Google - unseen 2B compact architecture)
  6. SmolLM2-1.7B-Instruct       (HuggingFace - sub-2B mobile architecture)
  7. SmolLM2-360M-Instruct       (HuggingFace - sub-billion compact architecture)
  8. SmolLM2-135M-Instruct       (HuggingFace - extreme sub-billion edge architecture)
  9. OLMoE-1B-7B-Instruct        (AllenAI - Sparse Mixture-of-Experts architecture)

Verifier Configuration:
  - Evaluator: Frozen Main V5 Argument-NLI Cross-Encoder (models/ensemble_data_only)
  - Temperature Scaling: T* = 1.9104
  - Operating Points: Optimal F1 (theta_F1* = 0.52) & High-Precision (theta_prec90* = 0.83)
  - Pure Single-Stage: NO Stage 1 retrieval filter (evaluates within-claim pairs directly)
  - Evaluates across all claims of Perspectrum
"""

import os
import sys
import json
import time
import argparse
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.io_utils import resolve, save_csv, save_json, load_csv
from src.utils.logging_utils import get_logger
from src.analysis.run_direct_v5_coverage import (
    load_gold_perspectives,
    load_v5_ensemble,
    ensure_processed_generations,
)

logger = get_logger("unseen_models_coverage")

UNSEEN_MODELS = [
    # 1. Out-of-Family Generalization Models (Architectures completely unseen during V5 tuning)
    {"key": "mistral_7b_instruct_v0_3", "name": "Mistral-7B",      "family": "Mistral",    "params_b": 7.0,  "category": "Out-of-Family"},
    {"key": "gemma_3_12b_it",          "name": "Gemma-3-12B",     "family": "Gemma-3",    "params_b": 12.0, "category": "Out-of-Family"},

    # 2. Compact / Edge Architecture Variants
    {"key": "llama_3_2_1b_instruct",   "name": "Llama-3.2-1B",   "family": "Llama-3.2",  "params_b": 1.0,  "category": "Edge-Variants"},
    {"key": "llama_3_2_3b_instruct",   "name": "Llama-3.2-3B",   "family": "Llama-3.2",  "params_b": 3.0,  "category": "Edge-Variants"},
    {"key": "gemma_2_2b_it",           "name": "Gemma-2-2B",     "family": "Gemma-2",    "params_b": 2.0,  "category": "Edge-Variants"},

    # 3. SmolLM2 Sub-Billion Family
    {"key": "smollm2_1_7b_instruct",   "name": "SmolLM2-1.7B",   "family": "SmolLM2",    "params_b": 1.7,  "category": "Sub-Billion"},
    {"key": "smollm2_360m_instruct",   "name": "SmolLM2-360M",   "family": "SmolLM2",    "params_b": 0.36, "category": "Sub-Billion"},
    {"key": "smollm2_135m_instruct",   "name": "SmolLM2-135M",   "family": "SmolLM2",    "params_b": 0.135,"category": "Sub-Billion"},

    # 4. Sparse Mixture-of-Experts
    {"key": "olmoe_1b_7b_instruct",    "name": "OLMoE-1B-7B",    "family": "OLMoE",      "params_b": 7.0,  "category": "Sparse-MoE"},
]


def run_unseen_models_coverage(
    v5_model_dir: str = "models/ensemble_data_only",
    temperature: float = 1.9104,
    threshold_f1: float = 0.52,
    threshold_prec90: float = 0.83,
    batch_size: int = 128,
    max_claims: int = None,
    target_models: List[str] = None,
    mock_run: bool = False,
    output_report: str = "results/unseen_models_coverage_report.json",
    output_md: str = "results/unseen_models_coverage_summary.md"
) -> Dict[str, Any]:
    """
    Evaluates perspective coverage for models outside the 4 V5 training families.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("=" * 95)
    logger.info("DIRECT V5 PERSPECTIVE COVERAGE: UNSEEN & OUT-OF-FAMILY GENERATORS")
    logger.info("Scope: All models OUTSIDE the 4 V5 training families (Mistral, Gemma-3, Edge, SmolLM2, MoE)")
    logger.info(f"Device: {device} | Temperature T*: {temperature}")
    logger.info(f"Calibrated Thresholds: Optimal F1 = {threshold_f1} | High-Precision = {threshold_prec90}")
    logger.info("=" * 95)

    persps_df = load_gold_perspectives()
    persp_dict_by_claim = {}
    for cid, grp in persps_df.groupby("claim_id"):
        persp_dict_by_claim[int(cid)] = grp.to_dict(orient="records")

    # Load V5 Verifier
    v5_models = []
    v5_tokenizer = None
    if not mock_run:
        v5_models, v5_tokenizer = load_v5_ensemble(v5_model_dir, device)

    # Filter target models if requested
    models_to_run = UNSEEN_MODELS
    if target_models:
        filtered = []
        for t in target_models:
            t_clean = t.lower().replace("-", "_")
            matched = [m for m in UNSEEN_MODELS if m["key"] == t_clean or m["name"].lower() == t.lower() or t_clean in m["key"]]
            if matched:
                filtered.extend(matched)
            else:
                filtered.append({"key": t, "name": t, "family": "Other", "params_b": 0.0, "category": "Custom"})
        # De-duplicate
        seen = set()
        models_to_run = [m for m in filtered if not (m["key"] in seen or seen.add(m["key"]))]

    master_summaries = []

    for m_info in models_to_run:
        m_key = m_info["key"]
        m_name = m_info["name"]
        m_fam = m_info["family"]
        m_params = m_info["params_b"]
        m_cat = m_info.get("category", "Unseen")

        logger.info("-" * 85)
        logger.info(f"Evaluating Model: {m_name} ({m_fam}, {m_params}B params, Category: {m_cat})")

        try:
            gen_path = ensure_processed_generations(m_key)
            gen_df = load_csv(gen_path)
            logger.info(f"  Loaded {len(gen_df)} generations across {gen_df['claim_id'].nunique()} claims from {gen_path}")
        except Exception as e:
            logger.warning(f"  [SKIPPED] Could not load generations for {m_key}: {e}. (Run generation first).")
            continue

        claim_ids = sorted(set(persp_dict_by_claim.keys()).intersection(set(gen_df["claim_id"])))
        if max_claims and len(claim_ids) > max_claims:
            claim_ids = claim_ids[:max_claims]

        logger.info(f"  Evaluating across {len(claim_ids)} claims of Perspectrum...")

        # Construct pairs directly
        pairs_to_evaluate = []
        pair_mapping = {}

        for cid in claim_ids:
            c_persps = persp_dict_by_claim[cid]
            c_gens = gen_df[gen_df["claim_id"] == cid]

            arg_texts = []
            for _, r in c_gens.iterrows():
                text = str(r.get("combined_text") or r.get("generated_text") or "").strip()
                if text:
                    arg_texts.append(text)

            if not arg_texts:
                continue

            for p_rec in c_persps:
                pid = int(p_rec["perspective_id"])
                p_text = str(p_rec["text"]).strip()
                pair_mapping[(cid, pid)] = []

                for arg in arg_texts:
                    pair_idx = len(pairs_to_evaluate)
                    pairs_to_evaluate.append({
                        "cid": cid,
                        "pid": pid,
                        "premise": arg,
                        "hypothesis": p_text
                    })
                    pair_mapping[(cid, pid)].append(pair_idx)

        total_pairs = len(pairs_to_evaluate)
        logger.info(f"  Direct verification pairs: {total_pairs} (avg {total_pairs/max(1, len(claim_ids)):.1f} pairs/claim)")

        pair_probs = np.zeros(total_pairs, dtype=np.float32)

        if mock_run:
            logger.info("  Running in mock mode (simulated calibrated predictions)...")
            rng = np.random.RandomState(abs(hash(m_key)) % 10000)
            pair_probs = rng.beta(1.8, 3.5, size=total_pairs).astype(np.float32)
        elif total_pairs > 0:
            from torch.utils.data import DataLoader, TensorDataset

            logger.info(f"  Tokenizing {total_pairs} pairs for V5...")
            enc = v5_tokenizer(
                [p["premise"] for p in pairs_to_evaluate],
                [p["hypothesis"] for p in pairs_to_evaluate],
                max_length=256,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            ds = TensorDataset(enc["input_ids"], enc["attention_mask"])
            loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

            total_batches = len(loader)
            logger.info(f"  Running V5 inference ({total_batches} batches of {batch_size})...")
            ensemble_probs_list = []
            t0 = time.time()

            for s_idx, m in enumerate(v5_models):
                m_probs = []
                with torch.no_grad():
                    for b_idx, (b_ids, b_mask) in enumerate(loader):
                        b_ids = b_ids.to(device)
                        b_mask = b_mask.to(device)
                        logits = m(input_ids=b_ids, attention_mask=b_mask).logits
                        logits = logits / max(temperature, 1e-4)
                        entail_idx = 1 if logits.shape[-1] >= 2 else 0
                        probs = torch.softmax(logits, dim=-1)[:, entail_idx].cpu().numpy()
                        m_probs.append(probs)

                        if (b_idx + 1) % 50 == 0 or (b_idx + 1) == total_batches:
                            elapsed = time.time() - t0
                            rate = ((b_idx + 1) * batch_size) / max(elapsed, 1e-3)
                            logger.info(f"    [Model {s_idx+1}/{len(v5_models)}] Batch {b_idx + 1}/{total_batches} ({(b_idx + 1)/total_batches*100:.1f}%) | {rate:.0f} pairs/sec")
                ensemble_probs_list.append(np.concatenate(m_probs, axis=0))
                logger.info(f"    Checkpoint {s_idx + 1}/{len(v5_models)} done in {time.time() - t0:.1f}s")

            pair_probs = np.mean(ensemble_probs_list, axis=0)

        # Aggregate coverage per perspective
        perspective_records = []
        for cid in claim_ids:
            c_persps = persp_dict_by_claim[cid]
            for p_rec in c_persps:
                pid = int(p_rec["perspective_id"])
                cluster_id = p_rec.get("cluster_id", 0)
                stance = str(p_rec.get("stance", "neutral")).lower()
                pair_idxs = pair_mapping.get((cid, pid), [])

                if pair_idxs:
                    max_prob = float(pair_probs[pair_idxs].max())
                    best_pair_idx = int(pair_idxs[int(pair_probs[pair_idxs].argmax())])
                    best_arg_text = pairs_to_evaluate[best_pair_idx]["premise"]
                else:
                    max_prob = 0.0
                    best_arg_text = ""

                cov_f1 = bool(max_prob >= threshold_f1)
                cov_prec = bool(max_prob >= threshold_prec90)

                perspective_records.append({
                    "claim_id": cid,
                    "perspective_id": pid,
                    "cluster_id": cluster_id,
                    "stance": stance,
                    "max_v5_probability": round(max_prob, 4),
                    "covered_f1": cov_f1,
                    "covered_prec90": cov_prec,
                    "best_supporting_argument": best_arg_text[:200]
                })

        persp_df = pd.DataFrame(perspective_records)

        # Aggregate Per-Claim Coverage
        claim_rows = []
        for cid, grp in persp_df.groupby("claim_id"):
            tot_atomic = len(grp)
            cov_atomic_f1 = grp["covered_f1"].mean()
            cov_atomic_prec = grp["covered_prec90"].mean()

            cluster_f1_covs = []
            cluster_prec_covs = []
            for _, cl_grp in grp.groupby("cluster_id"):
                cluster_f1_covs.append(cl_grp["covered_f1"].any())
                cluster_prec_covs.append(cl_grp["covered_prec90"].any())
            cov_cluster_f1 = np.mean(cluster_f1_covs) if cluster_f1_covs else 0.0
            cov_cluster_prec = np.mean(cluster_prec_covs) if cluster_prec_covs else 0.0

            sup_grp = grp[grp["stance"].str.contains("sup", na=False)]
            und_grp = grp[grp["stance"].str.contains("und", na=False)]
            sup_cov = sup_grp["covered_f1"].mean() if len(sup_grp) > 0 else np.nan
            und_cov = und_grp["covered_f1"].mean() if len(und_grp) > 0 else np.nan

            claim_rows.append({
                "claim_id": cid,
                "n_perspectives": tot_atomic,
                "n_clusters": len(cluster_f1_covs),
                "atomic_coverage_f1": round(float(cov_atomic_f1), 4),
                "atomic_coverage_prec90": round(float(cov_atomic_prec), 4),
                "cluster_coverage_f1": round(float(cov_cluster_f1), 4),
                "cluster_coverage_prec90": round(float(cov_cluster_prec), 4),
                "support_coverage": round(float(sup_cov), 4) if not np.isnan(sup_cov) else 0.0,
                "undermine_coverage": round(float(und_cov), 4) if not np.isnan(und_cov) else 0.0,
                "stance_gap": round(float(sup_cov - und_cov), 4) if (not np.isnan(sup_cov) and not np.isnan(und_cov)) else 0.0
            })

        claim_df = pd.DataFrame(claim_rows)

        out_m_dir = resolve(f"results/{m_key}/metrics")
        os.makedirs(out_m_dir, exist_ok=True)
        persp_df.to_csv(os.path.join(out_m_dir, "direct_v5_per_perspective_coverage.csv"), index=False)
        claim_df.to_csv(os.path.join(out_m_dir, "direct_v5_per_claim_coverage.csv"), index=False)

        mean_atomic_f1 = float(claim_df["atomic_coverage_f1"].mean())
        mean_atomic_prec = float(claim_df["atomic_coverage_prec90"].mean())
        mean_cluster_f1 = float(claim_df["cluster_coverage_f1"].mean())
        mean_cluster_prec = float(claim_df["cluster_coverage_prec90"].mean())
        mean_sup = float(claim_df["support_coverage"].mean())
        mean_und = float(claim_df["undermine_coverage"].mean())
        mean_gap = float(claim_df["stance_gap"].mean())

        logger.info(f"  Summary for {m_name}:")
        logger.info(f"    Atomic Coverage (F1 theta=0.52):      {mean_atomic_f1*100:.1f}%")
        logger.info(f"    Cluster Coverage (F1 theta=0.52):     {mean_cluster_f1*100:.1f}%")
        logger.info(f"    High-Precision Coverage (theta=0.83): {mean_atomic_prec*100:.1f}%")
        logger.info(f"    Support Cov: {mean_sup*100:.1f}% | Undermine Cov: {mean_und*100:.1f}% | Stance Gap: {mean_gap*100:+.1f}%")

        master_summaries.append({
            "model_key": m_key,
            "display_name": m_name,
            "family": m_fam,
            "params_b": m_params,
            "category": m_cat,
            "evaluated_claims": len(claim_df),
            "atomic_coverage_f1": mean_atomic_f1,
            "atomic_coverage_prec90": mean_atomic_prec,
            "cluster_coverage_f1": mean_cluster_f1,
            "cluster_coverage_prec90": mean_cluster_prec,
            "support_coverage": mean_sup,
            "undermine_coverage": mean_und,
            "stance_gap_sup_minus_und": mean_gap
        })

    # Save Reports
    os.makedirs(os.path.dirname(resolve(output_report)), exist_ok=True)
    report_dict = {
        "title": "Direct V5 Perspective Coverage: Unseen & Out-of-Family Generator Models",
        "verifier_configuration": {
            "v5_model_dir": v5_model_dir,
            "temperature_T_applied": temperature,
            "frozen_theta_f1": threshold_f1,
            "frozen_theta_prec90": threshold_prec90,
            "scope": "Models outside the 4 V5 training families"
        },
        "models_evaluated": master_summaries
    }
    save_json(report_dict, resolve(output_report))
    logger.info(f"Saved JSON coverage report to {resolve(output_report)}")

    # Markdown Table
    md_lines = [
        "# Perspective Coverage Report: Unseen & Out-of-Family Generator Models",
        "> **Evaluator:** Frozen Main V5 Argument-NLI Cross-Encoder ($T^* = 1.9104$)",
        "> **Scope:** Generator Architectures OUTSIDE the 4 V5 training families (excluding Qwen, Llama-3.1-8B, Gemma-2-9B, OLMo-2-7B)",
        "> **Verification Mode:** Direct Within-Claim Cross-Encoder (**No Stage 1 Retrieval Filter**)",
        "",
        "---",
        "",
        "## Unseen Models Coverage Benchmark",
        "",
        "| Generator Model | Family | Category | Params (B) | Claims | Atomic Coverage ($\\theta_{F1}^* = 0.52$) | Cluster Coverage ($\\theta_{F1}^* = 0.52$) | High-Precision Coverage ($\\theta^* = 0.83$) | Stance Gap |",
        "|---|---|---|---|---|---|---|---|---|"
    ]

    for m in master_summaries:
        md_lines.append(
            f"| **{m['display_name']}** | {m['family']} | {m['category']} | {m['params_b']}B | {m['evaluated_claims']} | "
            f"**`{m['atomic_coverage_f1']*100:.1f}%`** | **`{m['cluster_coverage_f1']*100:.1f}%`** | "
            f"`{m['atomic_coverage_prec90']*100:.1f}%` | "
            f"`{m['stance_gap_sup_minus_und']*100:+.1f}%` |"
        )

    with open(resolve(output_md), "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    logger.info(f"Saved Markdown summary to {resolve(output_md)}")

    print("\n" + "=" * 135)
    print("FINAL DIRECT V5 COVERAGE: UNSEEN & OUT-OF-FAMILY GENERATORS")
    print("=" * 135)
    print(f"{'Model':<16} | {'Family':<10} | {'Category':<14} | {'Params':<7} | {'Claims':<7} | {'Atomic Cov (F1)':<16} | {'Cluster Cov (F1)':<18} | {'High-Prec':<12}")
    print("-" * 135)
    for m in master_summaries:
        print(f"{m['display_name']:<16} | {m['family']:<10} | {m['category']:<14} | {m['params_b']:<5.1f}B  | {m['evaluated_claims']:<7} | {m['atomic_coverage_f1']*100:>14.1f}% | {m['cluster_coverage_f1']*100:>16.1f}% | {m['atomic_coverage_prec90']*100:>10.1f}%")
    print("=" * 135 + "\n")

    return report_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Direct V5 Coverage for Unseen Models")
    parser.add_argument("--v5_model_dir", type=str, default="models/ensemble_data_only")
    parser.add_argument("--temperature", type=float, default=1.9104)
    parser.add_argument("--threshold_f1", type=float, default=0.52)
    parser.add_argument("--threshold_prec90", type=float, default=0.83)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_claims", type=int, default=None)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--mock_run", action="store_true")
    parser.add_argument("--output_report", type=str, default="results/unseen_models_coverage_report.json")
    parser.add_argument("--output_md", type=str, default="results/unseen_models_coverage_summary.md")

    args = parser.parse_args()

    run_unseen_models_coverage(
        v5_model_dir=args.v5_model_dir,
        temperature=args.temperature,
        threshold_f1=args.threshold_f1,
        threshold_prec90=args.threshold_prec90,
        batch_size=args.batch_size,
        max_claims=args.max_claims,
        target_models=args.models,
        mock_run=args.mock_run,
        output_report=args.output_report,
        output_md=args.output_md
    )

