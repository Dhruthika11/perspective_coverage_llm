#!/usr/bin/env python3
"""
Fixed-Budget vs Native-Budget Perspective Coverage Engine on Flair2 GPU.

Separates generation budget from model capability:
  1. Computes Kuhn-Munkres maximum bipartite matching across budget thresholds:
     - K = 1 viewpoint per claim
     - K = 2 viewpoints per claim
     - K = 3 viewpoints per claim
     - K = 4 viewpoints per claim
     - K = 5 viewpoints per claim (Fixed Standardized Benchmark)
     - K = Native (Full unconstrained extracted viewpoints)
  2. Generates Coverage-versus-Budget Curves (K=1 to 5 to Native).
  3. Direct Capability Comparison:
     Ranks all models at IDENTICAL fixed budget (K=5), removing volume bias.
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

from src.analysis.bipartite_matching import solve_sparse_claim_bipartite_matching

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

TAU_S1 = 0.55
T_PROBE = 0.20
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
    parser = argparse.ArgumentParser(description="Fixed-Budget vs Native Perspective Coverage")
    parser.add_argument("--models", nargs="*", default=None, help="Specific models to evaluate (default: all)")
    parser.add_argument("--batch_size_bge", type=int, default=256)
    parser.add_argument("--batch_size_deb", type=int, default=128)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 80)
    print("FIXED-BUDGET VS NATIVE-BUDGET PERSPECTIVE COVERAGE ENGINE")
    print(f"Device: {device} | Budgets: K in [1, 2, 3, 4, 5, Native]")
    print("=" * 80)

    scaler_mean, scaler_scale, probe_w, probe_b = load_probe_weights()
    df_persp = pd.read_csv(PERSPECTIVES_PATH)
    
    bge_tok = AutoTokenizer.from_pretrained(BGE_MODEL_NAME)
    bge_mod = AutoModel.from_pretrained(BGE_MODEL_NAME).to(device).eval()
    deb_tok = AutoTokenizer.from_pretrained(DEBERTA_MODEL_PATH)
    deb_mod = AutoModelForSequenceClassification.from_pretrained(DEBERTA_MODEL_PATH).to(device).eval()

    gold_cache_path = os.path.join("data", "processed", "gold_perspectives_bge_embeddings.npy")
    gold_embs = np.load(gold_cache_path)
    df_persp["emb_idx"] = np.arange(len(df_persp))

    gold_by_claim = {}
    for cid, group in df_persp.groupby("claim_id"):
        gold_by_claim[cid] = {
            "indices": group["emb_idx"].values,
            "clusters": group["cluster_id"].values,
            "unique_clusters": sorted(group["cluster_id"].unique()),
            "stances": group["stance"].astype(str).str.lower().str.strip().values,
            "texts": group["text"].values
        }

    manifest_path = os.path.join(VIEWPOINTS_DIR, "extraction_manifest.csv")
    if os.path.exists(manifest_path):
        m_df = pd.read_csv(manifest_path)
        all_models = m_df["model_key"].tolist()
    else:
        all_models = sorted([d for d in os.listdir(VIEWPOINTS_DIR) if os.path.isdir(os.path.join(VIEWPOINTS_DIR, d)) and not d.startswith(".")])

    target_models = args.models if args.models else all_models
    print(f"Total models to evaluate: {len(target_models)}")

    budget_matrix_records = []

    for idx, m_key in enumerate(target_models, 1):
        vp_path = os.path.join(VIEWPOINTS_DIR, m_key, "viewpoints.csv")
        if not os.path.exists(vp_path):
            continue

        print(f"\n[{idx}/{len(target_models)}] PROCESSING: {m_key}")
        t0 = time.time()
        df_vp = pd.read_csv(vp_path)
        texts = df_vp["viewpoint_text"].fillna("").astype(str).tolist()
        vp_embs = embed_texts(texts, bge_mod, bge_tok, batch_size=args.batch_size_bge)

        df_vp["vp_emb_idx"] = np.arange(len(df_vp))
        vp_by_claim = {cid: grp for cid, grp in df_vp.groupby("claim_id")}

        # Filter Stage 1 & 2a candidates
        candidate_meta = []
        deb_pairs = []

        for cid, g_info in gold_by_claim.items():
            if cid not in vp_by_claim:
                continue
            v_group = vp_by_claim[cid]
            v_indices = v_group["vp_emb_idx"].values
            v_st_col = "normalized_stance" if "normalized_stance" in v_group.columns else ("stance" if "stance" in v_group.columns else None)
            v_stances = v_group[v_st_col].astype(str).str.lower().str.strip().values if v_st_col else np.array(["unknown"] * len(v_group))
            v_texts_c = v_group["viewpoint_text"].values

            g_indices = g_info["indices"]
            g_stances = g_info["stances"]
            g_texts_c = g_info["texts"]

            sim_mat = np.dot(vp_embs[v_indices], gold_embs[g_indices].T)
            s1_mask = sim_mat >= TAU_S1

            for vi in range(len(v_indices)):
                v_st = v_stances[vi]
                for gi in range(len(g_indices)):
                    if not s1_mask[vi, gi]:
                        continue
                    g_st = g_stances[gi]
                    is_contra = (
                        (v_st == "support" and g_st == "undermine") or
                        (v_st == "undermine" and g_st == "support")
                    )
                    if is_contra:
                        continue
                    candidate_meta.append({
                        "claim_id": cid,
                        "vp_idx": vi,
                        "gold_idx": gi,
                        "s1_cos": float(sim_mat[vi, gi])
                    })
                    deb_pairs.append((v_texts_c[vi], g_texts_c[gi]))

        # DeBERTa inference
        if len(deb_pairs) > 0:
            probs = run_deberta_batch(deb_pairs, deb_mod, deb_tok, batch_size=args.batch_size_deb)
            p_ent = probs[:, 1]
            p_neu = probs[:, 2]
        else:
            p_ent = np.array([])
            p_neu = np.array([])

        claim_cand_map = {}
        for c_idx, meta in enumerate(candidate_meta):
            cid = meta["claim_id"]
            cos_val = meta["s1_cos"]
            ent_val = float(p_ent[c_idx])
            neu_val = float(p_neu[c_idx])

            feat_raw = np.array([cos_val, ent_val, neu_val], dtype=np.float32)
            feat_scaled = (feat_raw - scaler_mean) / scaler_scale
            logit = np.dot(probe_w, feat_scaled) + probe_b
            p_m0 = float(1.0 / (1.0 + np.exp(-logit)))

            meta["p_m0"] = p_m0
            meta["pass_m0"] = (p_m0 >= T_PROBE)
            if cid not in claim_cand_map:
                claim_cand_map[cid] = []
            claim_cand_map[cid].append(meta)

        # Kuhn-Munkres across Budgets K = 1, 2, 3, 4, 5, and Native
        budgets = [1, 2, 3, 4, 5, "native"]
        budget_claim_covs = {b: [] for b in budgets}

        for cid, g_info in gold_by_claim.items():
            clusters = g_info["clusters"]
            unique_clusters = g_info["unique_clusters"]
            n_clusters = len(unique_clusters)
            cluster_to_col = {clust_id: col_idx for col_idx, clust_id in enumerate(unique_clusters)}

            if cid not in vp_by_claim:
                for b in budgets:
                    budget_claim_covs[b].append(0.0)
                continue

            n_vps = len(vp_by_claim[cid])
            valid_edges = {}

            if cid in claim_cand_map:
                for cand in claim_cand_map[cid]:
                    vi = cand["vp_idx"]
                    gi = cand["gold_idx"]
                    c_id = clusters[gi]
                    col_idx = cluster_to_col[c_id]
                    if cand["pass_m0"]:
                        valid_edges[(vi, col_idx)] = max(valid_edges.get((vi, col_idx), 0.0), cand["p_m0"])

            # Evaluate each budget on sparse candidate subgraphs
            for b in budgets:
                k_limit = n_vps if b == "native" else min(b, n_vps)
                if k_limit == 0:
                    budget_claim_covs[b].append(0.0)
                    continue
                # Restrict sparse graph to viewpoints < k_limit
                b_edges = {edge: score for edge, score in valid_edges.items() if edge[0] < k_limit}
                res = solve_sparse_claim_bipartite_matching(
                    n_viewpoints=k_limit,
                    n_clusters=n_clusters,
                    cluster_ids=unique_clusters,
                    candidate_edges=b_edges,
                    threshold=T_PROBE
                )
                budget_claim_covs[b].append(res["coverage_score"])

        mean_covs = {b: round(float(np.mean(budget_claim_covs[b])) * 100.0, 4) for b in budgets}
        rec = {
            "model_key": m_key,
            "native_viewpoints": len(df_vp),
            "cov_k1": mean_covs[1],
            "cov_k2": mean_covs[2],
            "cov_k3": mean_covs[3],
            "cov_k4": mean_covs[4],
            "cov_k5": mean_covs[5],
            "cov_native": mean_covs["native"],
            "delta_native_vs_k5": round(mean_covs["native"] - mean_covs[5], 4),
            "duration_sec": round(time.time() - t0, 2)
        }
        budget_matrix_records.append(rec)
        print(f"  * Budget Curve: K1={mean_covs[1]}% -> K2={mean_covs[2]}% -> K3={mean_covs[3]}% -> K4={mean_covs[4]}% -> K5={mean_covs[5]}% | Native={mean_covs['native']}% (Total VPs: {len(df_vp)})")

    df_budget = pd.DataFrame(budget_matrix_records)
    out_csv = os.path.join(OUTPUT_DIR, "budget_vs_capability_matrix.csv")
    df_budget.to_csv(out_csv, index=False)
    print(f"\nSaved budget analysis to {out_csv}")

    out_json = os.path.join(OUTPUT_DIR, "budget_vs_capability_report.json")
    with open(out_json, "w") as f:
        json.dump(budget_matrix_records, f, indent=2)
    print(f"Saved JSON report to {out_json}")

if __name__ == "__main__":
    main()
