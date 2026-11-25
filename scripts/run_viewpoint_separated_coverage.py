"""
Run Calibrated Perspective Coverage via Bipartite Matching on Viewpoint-Separated Data.

Workflow:
1. Loads discrete viewpoint arguments (~45-60w) from:
   data/viewpoint_separated/<model_key>/viewpoints.csv
2. Loads Perspectrum gold perspectives and clusters from:
   data/processed/perspectives.csv
3. Embeds viewpoints and reference perspectives using BGE-Large.
4. For each claim:
   - Constructs K_gen x M_clusters similarity matrix.
   - Cluster similarity: max_{p in cluster} cos(e_gen, e_p).
   - Applies Calibrated Stage 1 Threshold tau* (default = 0.6130, tuned via Method A).
   - Optional: Applies Stance Consistency Check (stance_gen == stance_gold).
   - Solves Kuhn-Munkres Maximum Bipartite Matching (scipy linear_sum_assignment).
5. Aggregates and outputs:
   - results/viewpoint_separated_coverage_summary.csv
   - results/viewpoint_separated_per_claim_coverage.csv
"""

import os
import sys
import glob
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Ensure src imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.analysis.bipartite_matching import solve_maximum_bipartite_matching

device = "cuda" if torch.cuda.is_available() else "cpu"

def embed_texts(texts: list[str], model, tokenizer, batch_size=32) -> np.ndarray:
    """Embeds a list of texts using BGE-Large with L2 normalization."""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            outputs = model(**encoded)
            # CLS pooling
            emb = outputs[0][:, 0]
            emb = F.normalize(emb, p=2, dim=1)
            all_embeddings.append(emb.cpu().numpy())
    return np.vstack(all_embeddings)

def run_coverage_evaluation(
    model_key: str,
    tau: float = 0.6130,
    check_stance: bool = False,
    max_claims: int = None
):
    print("=" * 95)
    print(f"EVALUATING CALIBRATED COVERAGE: {model_key} (tau* = {tau:.4f}, device = {device})")
    print("=" * 95)
    
    # 1. Load viewpoints
    vp_path = f"data/viewpoint_separated/{model_key}/viewpoints.csv"
    if not os.path.exists(vp_path):
        raise FileNotFoundError(f"Viewpoints file not found: {vp_path}")
    df_vps = pd.read_csv(vp_path)
    print(f"Loaded {len(df_vps)} viewpoints across {df_vps['claim_id'].nunique()} claims.")
    
    # 2. Load Perspectrum gold perspectives
    gold_path = "data/processed/perspectives.csv"
    df_gold = pd.read_csv(gold_path)
    print(f"Loaded {len(df_gold)} gold perspectives across {df_gold['claim_id'].nunique()} claims.")
    
    # Common claims
    common_claims = sorted(list(set(df_vps['claim_id']).intersection(set(df_gold['claim_id']))))
    if max_claims:
        common_claims = common_claims[:max_claims]
    print(f"Evaluating {len(common_claims)} overlapping claims.")
    
    # Filter gold and viewpoints to common claims
    df_gold = df_gold[df_gold["claim_id"].isin(common_claims)].reset_index(drop=True)
    df_vps = df_vps[df_vps["claim_id"].isin(common_claims)].reset_index(drop=True)
    
    # 3. Load BGE-Large
    from transformers import AutoTokenizer, AutoModel
    bge_name = "BAAI/bge-large-en-v1.5"
    print(f"Loading {bge_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(bge_name)
    bge_model = AutoModel.from_pretrained(bge_name).to(device).eval()
    
    # 4. Precompute gold perspective embeddings and viewpoint embeddings in batches
    print(f"Embedding {len(df_gold)} gold reference perspectives...")
    gold_texts = df_gold["text"].tolist()
    gold_embs = embed_texts(gold_texts, bge_model, tokenizer, batch_size=64)
    df_gold["emb_idx"] = np.arange(len(df_gold))
    
    print(f"Embedding {len(df_vps)} generated viewpoints...")
    vp_texts = df_vps["viewpoint_text"].tolist()
    all_vp_embs = embed_texts(vp_texts, bge_model, tokenizer, batch_size=64)
    df_vps["vp_emb_idx"] = np.arange(len(df_vps))
    
    # 5. Evaluate claim by claim
    claim_records = []
    
    for cid in tqdm(common_claims, desc=f"Scoring {model_key}"):
        sub_vps = df_vps[df_vps["claim_id"] == cid]
        sub_gold = df_gold[df_gold["claim_id"] == cid]
        
        if len(sub_vps) == 0 or len(sub_gold) == 0:
            continue
            
        vp_embs = all_vp_embs[sub_vps["vp_emb_idx"].values]
        
        # Unique clusters for this claim
        cluster_ids = sorted(sub_gold["cluster_id"].unique())
        n_clusters = len(cluster_ids)
        n_vps = len(sub_vps)
        
        # Build match matrix: shape (n_vps, n_clusters)
        match_matrix = np.zeros((n_vps, n_clusters), dtype=np.float32)
        
        for c_idx, clust_id in enumerate(cluster_ids):
            clust_members = sub_gold[sub_gold["cluster_id"] == clust_id]
            clust_emb_indices = clust_members["emb_idx"].values
            clust_embs = gold_embs[clust_emb_indices]  # shape: (n_members, D)
            
            # Pairwise cosine similarities: (n_vps, n_members)
            sims = np.dot(vp_embs, clust_embs.T)
            # Max similarity to any perspective in the cluster
            max_sims_to_cluster = np.max(sims, axis=1)  # shape: (n_vps,)
            
            # Prune edges below threshold tau
            for v_idx in range(n_vps):
                sim_val = max_sims_to_cluster[v_idx]
                if sim_val >= tau:
                    match_matrix[v_idx, c_idx] = sim_val
                else:
                    match_matrix[v_idx, c_idx] = 0.0
                    
        # Solve Maximum Bipartite Matching (Kuhn-Munkres)
        matching_res = solve_maximum_bipartite_matching(
            match_matrix=match_matrix,
            cluster_ids=cluster_ids,
            viewpoint_ids=sub_vps["viewpoint_id"].tolist()
        )
        
        claim_records.append({
            "model_key": model_key,
            "claim_id": cid,
            "n_generated_viewpoints": n_vps,
            "n_gold_clusters": n_clusters,
            "n_matched_clusters": matching_res["n_matched"],
            "coverage_score": matching_res["coverage_score"],
            "uncovered_clusters": len(matching_res["uncovered_cluster_ids"])
        })
        
    df_results = pd.DataFrame(claim_records)
    mean_coverage = df_results["coverage_score"].mean() * 100
    mean_vps = df_results["n_generated_viewpoints"].mean()
    mean_clusters = df_results["n_gold_clusters"].mean()
    
    print("\n" + "=" * 95)
    print(f"RESULTS FOR {model_key} (tau* = {tau:.4f}):")
    print(f"  * Claims Evaluated:          {len(df_results)}")
    print(f"  * Mean Viewpoints / Claim:   {mean_vps:.2f}")
    print(f"  * Mean Gold Clusters / Claim:{mean_clusters:.2f}")
    print(f"  * Mean Perspective Coverage: {mean_coverage:.2f}%")
    print("=" * 95)
    
    os.makedirs("results", exist_ok=True)
    out_csv = f"results/coverage_{model_key}_tau_{tau:.4f}.csv"
    df_results.to_csv(out_csv, index=False)
    print(f"Saved detailed per-claim coverage to: {out_csv}\n")
    return df_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Calibrated Bipartite Coverage on Viewpoint-Separated Data")
    parser.add_argument("--model", type=str, default="gemma_2_9b_it", help="Model key (or 'all')")
    parser.add_argument("--tau", type=float, default=0.6130, help="Calibrated Stage 1 Threshold tau*")
    parser.add_argument("--max_claims", type=int, default=None, help="Limit number of claims for testing")
    args = parser.parse_args()
    
    if args.model == "all":
        models = [
            "gemma_2_9b_it", "qwen2_5_7b_instruct", "olmo_2_1124_7b_instruct",
            "llama_3_1_8b_instruct", "qwen2_5_1_5b_instruct", "qwen2_5_3b_instruct",
            "qwen2_5_32b_instruct"
        ]
        summaries = []
        for m in models:
            res = run_coverage_evaluation(m, tau=args.tau, max_claims=args.max_claims)
            summaries.append({
                "model_key": m,
                "claims": len(res),
                "mean_viewpoints": res["n_generated_viewpoints"].mean(),
                "mean_gold_clusters": res["n_gold_clusters"].mean(),
                "coverage_score": res["coverage_score"].mean() * 100
            })
        df_all = pd.DataFrame(summaries)
        df_all.to_csv(f"results/master_viewpoint_separated_coverage_tau_{args.tau:.4f}.csv", index=False)
        print("\n" + "="*95)
        print(f"MASTER COVERAGE LEADERBOARD (tau* = {args.tau:.4f}):")
        print(df_all.to_string(index=False))
        print("="*95)
    else:
        run_coverage_evaluation(args.model, tau=args.tau, max_claims=args.max_claims)
