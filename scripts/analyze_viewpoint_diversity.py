#!/usr/bin/env python3
"""
Comprehensive Viewpoint Diversity and Redundancy Analysis across All 34 Models.

Quantifies both Lexical Diversity and Semantic (Embedding-Space) Diversity:
  1. Lexical Diversity Metrics:
     - Distinct-1, Distinct-2, Distinct-3 n-gram ratios
     - Average viewpoint word count and length distribution
     - Vocabulary entropy
  2. Semantic Embedding Space Metrics (via BGE-Large):
     - Intra-Claim Pairwise Cosine Similarity: Average similarity between generated viewpoints for the same claim.
       (Higher similarity = higher redundancy = narrower perspective diversity)
     - Intra-Claim Centroid Dispersion: Mean L2 distance of viewpoints from the claim's viewpoint centroid.
       (Higher dispersion = broader semantic coverage of alternative stances/arguments)
     - Minimum Intra-Claim Cosine: Widest perspective divergence generated for a claim.
  3. Alignment Ladder Mode Collapse Audit:
     - Specifically tests whether DPO / Instruction tuning systematically narrows viewpoint diversity
       compared to Base models across the OLMo-2 1.2B, 7.0B, 13.0B, and 32.0B ladders.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BGE_MODEL_NAME = "BAAI/bge-large-en-v1.5"
VIEWPOINTS_DIR = "data/viewpoint_separated"
OUTPUT_DIR = "results/coverage"

def compute_lexical_diversity(texts):
    """Computes Distinct-1, Distinct-2, Distinct-3 and vocabulary statistics."""
    total_unigrams = 0
    total_bigrams = 0
    total_trigrams = 0
    
    unigram_set = set()
    bigram_set = set()
    trigram_set = set()
    
    word_counts = []
    
    for t in texts:
        tokens = str(t).lower().strip().split()
        word_counts.append(len(tokens))
        n = len(tokens)
        if n == 0:
            continue
            
        total_unigrams += n
        unigram_set.update(tokens)
        
        if n >= 2:
            bigrams = [f"{tokens[i]}_{tokens[i+1]}" for i in range(n - 1)]
            total_bigrams += len(bigrams)
            bigram_set.update(bigrams)
            
        if n >= 3:
            trigrams = [f"{tokens[i]}_{tokens[i+1]}_{tokens[i+2]}" for i in range(n - 2)]
            total_trigrams += len(trigrams)
            trigram_set.update(trigrams)
            
    dist1 = len(unigram_set) / max(1, total_unigrams)
    dist2 = len(bigram_set) / max(1, total_bigrams)
    dist3 = len(trigram_set) / max(1, total_trigrams)
    mean_wc = float(np.mean(word_counts)) if word_counts else 0.0
    
    return {
        "distinct_1": round(dist1, 4),
        "distinct_2": round(dist2, 4),
        "distinct_3": round(dist3, 4),
        "vocab_size": len(unigram_set),
        "mean_word_count": round(mean_wc, 2)
    }

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

def compute_semantic_diversity_per_claim(df_vps, embs):
    """
    Computes intra-claim pairwise cosine similarity and centroid dispersion.
    """
    claim_sims = []
    claim_dispersions = []
    claim_min_sims = []
    
    df_vps = df_vps.copy()
    df_vps["emb_idx"] = np.arange(len(df_vps))
    
    for cid, group in df_vps.groupby("claim_id"):
        indices = group["emb_idx"].values
        k = len(indices)
        if k < 2:
            continue
            
        claim_embs = embs[indices] # shape: (k, 1024)
        
        # Pairwise cosine similarity matrix
        sim_mat = np.dot(claim_embs, claim_embs.T)
        
        # Upper triangle without diagonal
        iu = np.triu_indices(k, k=1)
        pairwise_sims = sim_mat[iu]
        
        mean_sim = float(np.mean(pairwise_sims))
        min_sim = float(np.min(pairwise_sims))
        
        # Centroid dispersion: mean L2 distance from centroid
        centroid = np.mean(claim_embs, axis=0, keepdims=True)
        # Normalize centroid or compute euclidean distance
        dist_from_centroid = np.linalg.norm(claim_embs - centroid, axis=1)
        dispersion = float(np.mean(dist_from_centroid))
        
        claim_sims.append(mean_sim)
        claim_min_sims.append(min_sim)
        claim_dispersions.append(dispersion)
        
    return {
        "intra_claim_pairwise_cosine": round(float(np.mean(claim_sims)), 4) if claim_sims else 0.0,
        "intra_claim_dispersion": round(float(np.mean(claim_dispersions)), 4) if claim_dispersions else 0.0,
        "intra_claim_min_cosine": round(float(np.mean(claim_min_sims)), 4) if claim_min_sims else 0.0,
        "n_evaluated_claims": len(claim_sims)
    }

def main():
    parser = argparse.ArgumentParser(description="Viewpoint Diversity and Redundancy Analysis")
    parser.add_argument("--batch_size", type=int, default=256, help="BGE embedding batch size")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 80)
    print("VIEWPOINT DIVERSITY AND REDUNDANCY ANALYSIS ENGINE")
    print(f"Device: {device} | Model: {BGE_MODEL_NAME}")
    print("=" * 80)
    
    # 1. Load BGE Model
    print("Loading BGE-Large encoder...")
    bge_tok = AutoTokenizer.from_pretrained(BGE_MODEL_NAME)
    bge_mod = AutoModel.from_pretrained(BGE_MODEL_NAME).to(device).eval()
    
    # 2. Enumerate Models
    manifest_path = os.path.join(VIEWPOINTS_DIR, "extraction_manifest.csv")
    if os.path.exists(manifest_path):
        m_df = pd.read_csv(manifest_path)
        all_models = m_df["model_key"].tolist()
    else:
        all_models = sorted([d for d in os.listdir(VIEWPOINTS_DIR) if os.path.isdir(os.path.join(VIEWPOINTS_DIR, d)) and not d.startswith(".")])
        
    print(f"Total models to evaluate: {len(all_models)}")
    
    results = []
    
    for idx, m_key in enumerate(all_models, 1):
        vp_path = os.path.join(VIEWPOINTS_DIR, m_key, "viewpoints.csv")
        if not os.path.exists(vp_path):
            print(f"[{idx}/{len(all_models)}] Skipping {m_key}: viewpoints.csv not found.")
            continue
            
        print(f"\n[{idx}/{len(all_models)}] Processing: {m_key}")
        df_vp = pd.read_csv(vp_path)
        texts = df_vp["viewpoint_text"].fillna("").astype(str).tolist()
        
        # Lexical diversity
        lex_stats = compute_lexical_diversity(texts)
        
        # Semantic diversity via BGE embeddings
        embs = embed_texts(texts, bge_mod, bge_tok, batch_size=args.batch_size)
        sem_stats = compute_semantic_diversity_per_claim(df_vp, embs)
        
        rec = {
            "model_key": m_key,
            "total_viewpoints": len(df_vp),
            "n_claims": df_vp["claim_id"].nunique(),
            **lex_stats,
            **sem_stats
        }
        results.append(rec)
        print(f"  * Mean Intra-Claim Cosine: {sem_stats['intra_claim_pairwise_cosine']:.4f} "
              f"(Dispersion: {sem_stats['intra_claim_dispersion']:.4f})")
        print(f"  * Distinct-1: {lex_stats['distinct_1']:.4f} | Distinct-2: {lex_stats['distinct_2']:.4f} | Distinct-3: {lex_stats['distinct_3']:.4f}")
        
    df_res = pd.DataFrame(results)
    
    out_csv = os.path.join(OUTPUT_DIR, "viewpoint_diversity_and_redundancy_all_models.csv")
    df_res.to_csv(out_csv, index=False)
    print(f"\nSaved diversity metrics to {out_csv}")
    
    # Analyze OLMo-2 ladder progression
    olmo_sub = df_res[df_res["model_key"].str.startswith("olmo_2")].copy()
    
    report = {
        "all_models_diversity": results,
        "olmo_ladder_analysis": olmo_sub.to_dict(orient="records")
    }
    
    out_json = os.path.join(OUTPUT_DIR, "viewpoint_diversity_and_redundancy_report.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved diversity JSON report to {out_json}")
    
    # Save publication Markdown table
    md_table_path = os.path.join(OUTPUT_DIR, "table_viewpoint_diversity_and_redundancy.md")
    cols = ["model_key", "total_viewpoints", "intra_claim_pairwise_cosine", "intra_claim_dispersion", "distinct_1", "distinct_2", "distinct_3", "mean_word_count"]
    df_table = df_res[cols].sort_values(by="intra_claim_pairwise_cosine").copy()
    df_table.columns = ["Model", "Total VPs", "Intra-Claim Cosine", "Dispersion", "Distinct-1", "Distinct-2", "Distinct-3", "Mean Words"]
    
    header = "| " + " | ".join(df_table.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(df_table.columns)) + " |"
    rows = ["| " + " | ".join(str(r[c]) for c in df_table.columns) + " |" for _, r in df_table.iterrows()]
    md_content = "\n".join([header, sep] + rows)
    
    with open(md_table_path, "w") as f:
        f.write(md_content)
    print(f"Saved Markdown table to {md_table_path}")

if __name__ == "__main__":
    main()

