#!/usr/bin/env python3
"""
Full Perspectrum Perspective Coverage Pipeline across All 34 Models on Flair2 GPU.

Enforces:
  1. Frozen M0 Verifier (Official):
     - Stage 1: BGE-Large dense retrieval (tau* = 0.55)
     - Stage 2a: Stance concordance (reject explicit contradiction, keep UNKNOWN)
     - Stage 2b: DeBERTa-v3-large NLI + Logistic Probe (T* = 0.20)
  2. Shadow M_soft Verifier:
     - Gated residual reasoning penalty: S_soft = clip(P_M0 - 0.40 * max(0, 0.65 - reasoning_cos), 0, 1)
  3. Strict Equivalence Cluster Matching:
     - Kuhn-Munkres Maximum Bipartite Matching on human perspective clusters
     - Claim-level coverage = Matched Clusters / Total Gold Clusters
  4. Resumable Checkpointing:
     - Per-model results saved in results/coverage/per_model/
     - Master summary in results/coverage/master_coverage_all_models.csv
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

# Project root setup
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.extraction.argument_component_extractor import ArgumentComponentExtractor, EXTRACTION_VERSION
from src.analysis.bipartite_matching import solve_sparse_claim_bipartite_matching

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
TAU_S1 = 0.55
T_PROBE = 0.20
TAU_DIV = 0.65
LAM_PENALTY = 0.40

BGE_MODEL_NAME = "BAAI/bge-large-en-v1.5"
DEBERTA_MODEL_PATH = "models/argument_nli_deberta_v3_v3"
CONFIG_PROBE_PATH = "results/calibration/frozen_warrant_probe_model.json"
PERSPECTIVES_PATH = "data/processed/perspectives.csv"
VIEWPOINTS_DIR = "data/viewpoint_separated"
OUTPUT_DIR = "results/coverage"

def load_probe_weights():
    with open(CONFIG_PROBE_PATH, "r") as f:
        cfg = json.load(f)
    scaler_mean = np.array(cfg['scaler_mean'], dtype=np.float32)
    scaler_scale = np.array(cfg['scaler_scale'], dtype=np.float32)
    w = np.array([
        cfg['coefficients']['cosine_similarity'],
        cfg['coefficients']['p_entail'],
        cfg['coefficients']['p_neutral']
    ], dtype=np.float32)
    b = float(cfg['intercept'])
    return scaler_mean, scaler_scale, w, b

def embed_texts(texts: list, model, tokenizer, batch_size=256) -> np.ndarray:
    """Computes normalized BGE-Large embeddings in GPU batches."""
    embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs)[0][:, 0]
            norm_out = F.normalize(out, p=2, dim=1)
            embs.append(norm_out.cpu().numpy())
    return np.vstack(embs)

def run_deberta_batch(pairs: list, model, tokenizer, batch_size=128) -> np.ndarray:
    """Runs DeBERTa NLI cross-encoder inference returning (p_contra, p_entail, p_neutral)."""
    if len(pairs) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    
    all_probs = []
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i + batch_size]
        premises = [p[0] for p in batch]
        hypotheses = [p[1] for p in batch]
        inputs = tokenizer(premises, hypotheses, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
    return np.vstack(all_probs)

def main():
    parser = argparse.ArgumentParser(description="Full Perspectrum Coverage on Flair2")
    parser.add_argument("--models", nargs="*", default=None, help="Specific model keys to evaluate (default: all)")
    parser.add_argument("--batch_size_bge", type=int, default=256, help="BGE embedding batch size")
    parser.add_argument("--batch_size_deb", type=int, default=128, help="DeBERTa inference batch size")
    parser.add_argument("--force", action="store_true", help="Force re-run on already completed models")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "per_model"), exist_ok=True)

    print("=" * 100)
    print(f"FULL PERSPECTRUM COVERAGE ENGINE: FROZEN M0 (OFFICIAL) + M_SOFT (SHADOW)")
    print(f"Device: {device} | tau_s1 = {TAU_S1:.2f} | T_probe = {T_PROBE:.2f}")
    print("=" * 100)

    # 1. Load Probe Weights
    scaler_mean, scaler_scale, probe_w, probe_b = load_probe_weights()
    print("Loaded Frozen M0 Probe Weights:")
    print(f"  w = {probe_w}, b = {probe_b:.4f}")

    # 2. Load Perspectrum Gold Perspectives
    df_persp = pd.read_csv(PERSPECTIVES_PATH)
    print(f"Loaded {len(df_persp)} gold perspectives across {df_persp['claim_id'].nunique()} claims.")
    
    # 3. Load Models on GPU
    print(f"\nLoading BGE-Large from {BGE_MODEL_NAME}...")
    bge_tok = AutoTokenizer.from_pretrained(BGE_MODEL_NAME)
    bge_mod = AutoModel.from_pretrained(BGE_MODEL_NAME).to(device).eval()

    print(f"Loading DeBERTa-v3 from {DEBERTA_MODEL_PATH}...")
    deb_tok = AutoTokenizer.from_pretrained(DEBERTA_MODEL_PATH)
    deb_mod = AutoModelForSequenceClassification.from_pretrained(DEBERTA_MODEL_PATH).to(device).eval()

    extractor = ArgumentComponentExtractor(version=EXTRACTION_VERSION)

    # 4. Precompute & Cache Gold Perspective Embeddings
    gold_cache_path = os.path.join("data", "processed", "gold_perspectives_bge_embeddings.npy")
    gold_reas_cache_path = os.path.join("data", "processed", "gold_perspectives_reas_embeddings.npy")
    
    if os.path.exists(gold_cache_path) and os.path.exists(gold_reas_cache_path):
        print(f"Loading cached gold embeddings from {gold_cache_path}...")
        gold_embs = np.load(gold_cache_path)
        gold_reas_embs = np.load(gold_reas_cache_path)
    else:
        print(f"Computing BGE embeddings for {len(df_persp)} gold perspectives...")
        gold_texts = df_persp["text"].tolist()
        gold_embs = embed_texts(gold_texts, bge_mod, bge_tok, batch_size=args.batch_size_bge)
        np.save(gold_cache_path, gold_embs)
        
        # Extract gold reasoning components
        print(f"Extracting reasoning components for {len(df_persp)} gold perspectives...")
        gold_reas_texts = [extractor.extract_single(t)['core_reasoning'] for t in gold_texts]
        gold_reas_embs = embed_texts(gold_reas_texts, bge_mod, bge_tok, batch_size=args.batch_size_bge)
        np.save(gold_reas_cache_path, gold_reas_embs)
        print(f"Saved gold embeddings cache to {gold_cache_path}")

    df_persp["emb_idx"] = np.arange(len(df_persp))

    # Index gold perspectives by claim
    gold_by_claim = {}
    for cid, group in df_persp.groupby("claim_id"):
        gold_by_claim[cid] = {
            "indices": group["emb_idx"].values,
            "clusters": group["cluster_id"].values,
            "unique_clusters": sorted(group["cluster_id"].unique()),
            "stances": group["stance"].astype(str).str.lower().str.strip().values,
            "texts": group["text"].values
        }

    # 5. Determine Models to Process
    manifest_path = os.path.join(VIEWPOINTS_DIR, "extraction_manifest.csv")
    if os.path.exists(manifest_path):
        m_df = pd.read_csv(manifest_path)
        all_models = m_df["model_key"].tolist()
    else:
        all_models = sorted([d for d in os.listdir(VIEWPOINTS_DIR) if os.path.isdir(os.path.join(VIEWPOINTS_DIR, d)) and not d.startswith(".")])

    target_models = args.models if args.models else all_models
    print(f"\nTotal models to evaluate: {len(target_models)}")

    master_summary = []
    
    total_t0 = time.time()
    for m_idx, m_key in enumerate(target_models, 1):
        m_out_csv = os.path.join(OUTPUT_DIR, "per_model", f"{m_key}_coverage.csv")
        if os.path.exists(m_out_csv) and not args.force:
            print(f"[{m_idx}/{len(target_models)}] Model '{m_key}' already completed -> Skipping (use --force to re-run).")
            # Load existing summary metrics
            prev_df = pd.read_csv(m_out_csv)
            master_summary.append({
                "model_key": m_key,
                "n_claims": len(prev_df),
                "n_viewpoints": int(prev_df["n_viewpoints"].sum()),
                "m0_covered_clusters": int(prev_df["m0_matched_clusters"].sum()),
                "total_gold_clusters": int(prev_df["n_gold_clusters"].sum()),
                "m0_coverage_pct": float(prev_df["m0_coverage_score"].mean() * 100),
                "m_soft_coverage_pct": float(prev_df["m_soft_coverage_score"].mean() * 100),
                "viewpoint_efficiency": float(prev_df["m0_matched_clusters"].sum() / max(1, prev_df["n_viewpoints"].sum()))
            })
            continue

        vp_path = os.path.join(VIEWPOINTS_DIR, m_key, "viewpoints.csv")
        if not os.path.exists(vp_path):
            print(f"[-] Viewpoints not found for '{m_key}' ({vp_path}) -> Skipping.")
            continue

        print(f"\n{'='*100}")
        print(f"[{m_idx}/{len(target_models)}] PROCESSING: {m_key}")
        print(f"{'='*100}")
        t_model_start = time.time()

        df_vp = pd.read_csv(vp_path)
        n_vps_total = len(df_vp)
        print(f"Loaded {n_vps_total} viewpoints across {df_vp['claim_id'].nunique()} claims.")

        # Embed model viewpoints
        t_emb0 = time.time()
        vp_texts = df_vp["viewpoint_text"].tolist()
        vp_embs = embed_texts(vp_texts, bge_mod, bge_tok, batch_size=args.batch_size_bge)
        
        # Reasoning embeddings for shadow model
        vp_reas_texts = [extractor.extract_single(t)['core_reasoning'] for t in vp_texts]
        vp_reas_embs = embed_texts(vp_reas_texts, bge_mod, bge_tok, batch_size=args.batch_size_bge)
        df_vp["vp_emb_idx"] = np.arange(len(df_vp))
        print(f"Embedded {len(df_vp)} viewpoints in {time.time() - t_emb0:.2f}s.")

        # Group viewpoints by claim
        vp_by_claim = {cid: group for cid, group in df_vp.groupby("claim_id")}

        # ---------------------------------------------------------------------
        # STAGE 1 & 2a: CANDIDATE FILTERING ACROSS ALL CLAIMS
        # ---------------------------------------------------------------------
        candidate_pairs_meta = [] # (claim_id, vp_idx_in_claim, gold_idx_in_claim, vp_text, gold_text, s1_cos, reas_cos)
        deberta_pairs = []

        for cid, g_info in gold_by_claim.items():
            if cid not in vp_by_claim:
                continue
            v_group = vp_by_claim[cid]
            v_indices = v_group["vp_emb_idx"].values
            v_stances = v_group["normalized_stance"].astype(str).str.lower().str.strip().values
            v_texts_c = v_group["viewpoint_text"].values

            g_indices = g_info["indices"]
            g_stances = g_info["stances"]
            g_texts_c = g_info["texts"]

            # Cosine similarity matrix: (n_vps, n_gold)
            v_mat = vp_embs[v_indices]
            g_mat = gold_embs[g_indices]
            sim_mat = np.dot(v_mat, g_mat.T) # shape: (n_vps, n_gold)

            # Reasoning similarity matrix
            v_reas_mat = vp_reas_embs[v_indices]
            g_reas_mat = gold_reas_embs[g_indices]
            reas_sim_mat = np.dot(v_reas_mat, g_reas_mat.T)

            # Mask Stage 1 (cosine >= 0.55)
            s1_mask = sim_mat >= TAU_S1

            # Mask Stage 2a (reject opposite stance)
            for vi in range(len(v_indices)):
                v_st = v_stances[vi]
                for gi in range(len(g_indices)):
                    if not s1_mask[vi, gi]:
                        continue
                    g_st = g_stances[gi]
                    # Check contradiction
                    is_contra = (
                        (v_st == "support" and g_st == "undermine") or
                        (v_st == "undermine" and g_st == "support")
                    )
                    if is_contra:
                        continue
                    
                    s1_cos = float(sim_mat[vi, gi])
                    r_cos = float(reas_sim_mat[vi, gi])
                    candidate_pairs_meta.append({
                        "claim_id": cid,
                        "vp_idx": vi,
                        "gold_idx": gi,
                        "s1_cos": s1_cos,
                        "reas_cos": r_cos
                    })
                    deberta_pairs.append((v_texts_c[vi], g_texts_c[gi]))

        print(f"Stage 1 & 2a Candidates: {len(candidate_pairs_meta)} pairs to evaluate with DeBERTa.")

        # ---------------------------------------------------------------------
        # STAGE 2b: DEBERTA NLI INFERENCE & PROBE SCORING
        # ---------------------------------------------------------------------
        t_deb0 = time.time()
        if len(deberta_pairs) > 0:
            probs = run_deberta_batch(deberta_pairs, deb_mod, deb_tok, batch_size=args.batch_size_deb)
            p_ent = probs[:, 1]
            p_neu = probs[:, 2]
        else:
            p_ent = np.array([])
            p_neu = np.array([])
        print(f"DeBERTa inference completed in {time.time() - t_deb0:.2f}s ({len(deberta_pairs)/max(0.01, time.time() - t_deb0):.1f} pairs/sec).")

        # Evaluate M0 Probe and Shadow M_soft
        claim_cand_map = {} # claim_id -> list of candidate dicts
        for idx, meta in enumerate(candidate_pairs_meta):
            cid = meta["claim_id"]
            cos_val = meta["s1_cos"]
            ent_val = float(p_ent[idx])
            neu_val = float(p_neu[idx])
            r_cos = meta["reas_cos"]

            # Standardize features
            feat_raw = np.array([cos_val, ent_val, neu_val], dtype=np.float32)
            feat_scaled = (feat_raw - scaler_mean) / scaler_scale
            logit = np.dot(probe_w, feat_scaled) + probe_b
            p_m0 = float(1.0 / (1.0 + np.exp(-logit)))

            # Shadow M_soft
            diff = max(0.0, TAU_DIV - r_cos)
            score_soft = float(np.clip(p_m0 - LAM_PENALTY * diff, 0.0, 1.0))

            meta["p_m0"] = p_m0
            meta["score_soft"] = score_soft
            meta["pass_m0"] = (p_m0 >= T_PROBE)
            meta["pass_soft"] = (score_soft >= T_PROBE)

            if cid not in claim_cand_map:
                claim_cand_map[cid] = []
            claim_cand_map[cid].append(meta)

        # ---------------------------------------------------------------------
        # STAGE 3: MAXIMUM BIPARTITE MATCHING PER CLAIM
        # ---------------------------------------------------------------------
        claim_coverage_records = []
        for cid, g_info in gold_by_claim.items():
            clusters = g_info["clusters"]
            unique_clusters = g_info["unique_clusters"]
            n_clusters = len(unique_clusters)
            cluster_to_col = {clust_id: col_idx for col_idx, clust_id in enumerate(unique_clusters)}

            if cid not in vp_by_claim:
                claim_coverage_records.append({
                    "claim_id": cid,
                    "n_viewpoints": 0,
                    "n_gold_clusters": n_clusters,
                    "m0_matched_clusters": 0,
                    "m0_coverage_score": 0.0,
                    "m_soft_matched_clusters": 0,
                    "m_soft_coverage_score": 0.0
                })
                continue

            v_group = vp_by_claim[cid]
            n_vps = len(v_group)

            # 1. Construct sparse candidate graphs (exclude zero-weight edges & avoid dense matrices)
            edges_m0 = {}
            edges_soft = {}

            if cid in claim_cand_map:
                for cand in claim_cand_map[cid]:
                    vi = cand["vp_idx"]
                    gi = cand["gold_idx"]
                    c_id = clusters[gi]
                    col_idx = cluster_to_col[c_id]

                    # Retain only verified pairs (exclude zero-weight and below-threshold pairs)
                    if cand["pass_m0"]:
                        edges_m0[(vi, col_idx)] = max(edges_m0.get((vi, col_idx), 0.0), cand["p_m0"])
                    if cand["pass_soft"]:
                        edges_soft[(vi, col_idx)] = max(edges_soft.get((vi, col_idx), 0.0), cand["score_soft"])

            # 2. Solve independent claim-level assignment problem on sparse candidate graphs
            res_m0 = solve_sparse_claim_bipartite_matching(
                n_viewpoints=n_vps,
                n_clusters=n_clusters,
                cluster_ids=unique_clusters,
                candidate_edges=edges_m0,
                threshold=T_PROBE
            )
            matched_m0 = res_m0["n_matched"]
            cov_m0 = res_m0["coverage_score"]

            res_soft = solve_sparse_claim_bipartite_matching(
                n_viewpoints=n_vps,
                n_clusters=n_clusters,
                cluster_ids=unique_clusters,
                candidate_edges=edges_soft,
                threshold=T_PROBE
            )
            matched_s = res_soft["n_matched"]
            cov_s = res_soft["coverage_score"]

            claim_coverage_records.append({
                "claim_id": cid,
                "n_viewpoints": n_vps,
                "n_gold_clusters": n_clusters,
                "m0_matched_clusters": int(matched_m0),
                "m0_coverage_score": round(cov_m0, 6),
                "m_soft_matched_clusters": int(matched_s),
                "m_soft_coverage_score": round(cov_s, 6)
            })

        df_model_cov = pd.DataFrame(claim_coverage_records)
        df_model_cov.to_csv(m_out_csv, index=False)

        mean_cov_m0 = df_model_cov["m0_coverage_score"].mean() * 100
        mean_cov_soft = df_model_cov["m_soft_coverage_score"].mean() * 100
        total_matched_m0 = df_model_cov["m0_matched_clusters"].sum()
        total_clusters = df_model_cov["n_gold_clusters"].sum()
        efficiency = total_matched_m0 / max(1, n_vps_total)

        print(f"\nRESULTS FOR {m_key}:")
        print(f"  * Mean M0 Perspective Coverage:     {mean_cov_m0:.2f}%")
        print(f"  * Mean M_soft Perspective Coverage: {mean_cov_soft:.2f}%")
        print(f"  * Total Matched Clusters:          {total_matched_m0} / {total_clusters}")
        print(f"  * Viewpoint Efficiency (Match/VP): {efficiency:.4f}")
        print(f"  * Model Duration:                  {time.time() - t_model_start:.2f}s")
        print(f"  * Saved Per-Claim Coverage to:     {m_out_csv}")

        master_summary.append({
            "model_key": m_key,
            "n_claims": len(df_model_cov),
            "n_viewpoints": n_vps_total,
            "m0_matched_clusters": int(total_matched_m0),
            "total_gold_clusters": int(total_clusters),
            "m0_coverage_pct": round(mean_cov_m0, 4),
            "m_soft_coverage_pct": round(mean_cov_soft, 4),
            "viewpoint_efficiency": round(efficiency, 6)
        })

        # Save cumulative master summary
        df_master = pd.DataFrame(master_summary)
        df_master.to_csv(os.path.join(OUTPUT_DIR, "master_coverage_all_models.csv"), index=False)

    print("\n" + "=" * 100)
    print(f"ALL {len(target_models)} MODELS PROCESSED IN {time.time() - total_t0:.2f}s")
    print(f"Master Coverage Summary saved to: {os.path.join(OUTPUT_DIR, 'master_coverage_all_models.csv')}")
    print("=" * 100)

if __name__ == "__main__":
    main()
