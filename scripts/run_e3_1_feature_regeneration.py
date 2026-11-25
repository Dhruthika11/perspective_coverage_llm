"""
E3.1 Experiment: Extraction Correction and Feature Stability Analysis
Measures:
  1. Feature drift between v1.0 and v1.1 (MAE, max diff, Pearson/Spearman corr).
  2. Impact on M2 model predictions, calibration threshold, and test set performance.
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

sys.path.append(os.path.abspath("."))
from src.extraction.argument_component_extractor import ArgumentComponentExtractor, compute_stable_text_id, EXTRACTION_VERSION
from src.embeddings.embed_model import BGEEmbedder

def main():
    print("=" * 90)
    print("EXPERIMENT E3.1: EXTRACTION CORRECTION & FEATURE STABILITY AUDIT")
    print(f"Extraction Version: {EXTRACTION_VERSION}")
    print("=" * 90)
    
    # 1. Load v1.0 features
    old_csv = "results/calibration/e3_pairwise_features_341.csv"
    old_df = pd.read_csv(old_csv)
    print(f"Loaded v1.0 feature dataset with {len(old_df)} pairs.")
    
    # 2. Extract v1.1 components for all unique texts
    extractor = ArgumentComponentExtractor(version=EXTRACTION_VERSION)
    unique_texts = {}
    for _, row in old_df.iterrows():
        g_txt = row['gold_perspective_text']
        h_txt = row['generated_viewpoint_text']
        c_topic = row['claim_text']
        if g_txt not in unique_texts:
            unique_texts[g_txt] = {"text": g_txt, "claim_topic": c_topic, "source": "gold"}
        if h_txt not in unique_texts:
            unique_texts[h_txt] = {"text": h_txt, "claim_topic": c_topic, "source": "generated"}
            
    print(f"Extracting v1.1 components for {len(unique_texts)} unique texts...")
    t0 = time.time()
    v1_1_records = {}
    for text, info in unique_texts.items():
        rec = extractor.extract_single(text, claim_topic=info["claim_topic"])
        v1_1_records[rec["text_id"]] = rec
    print(f"Extraction completed in {time.time() - t0:.3f}s.")
    
    # Save v1.1 cache
    v1_1_cache_path = "data/calibration/e3_1_extracted_components_cache.json"
    with open(v1_1_cache_path, "w", encoding="utf-8") as f:
        json.dump(v1_1_records, f, indent=2)
        
    # 3. Embed v1.1 components
    print("\nLoading BGE Embedder...")
    t1 = time.time()
    embedder = BGEEmbedder(model_name="BAAI/bge-large-en-v1.5", normalize=True)
    print(f"Embedder loaded in {time.time() - t1:.2f}s on device: {embedder.device}")
    
    text_ids_ordered = list(v1_1_records.keys())
    claim_texts = [v1_1_records[tid]["central_claim"] for tid in text_ids_ordered]
    reasoning_texts = [v1_1_records[tid]["core_reasoning"] for tid in text_ids_ordered]
    warrant_texts = [v1_1_records[tid]["underlying_warrant"] for tid in text_ids_ordered]
    
    t2 = time.time()
    claim_embs = embedder.embed(claim_texts, batch_size=32)
    reasoning_embs = embedder.embed(reasoning_texts, batch_size=32)
    warrant_embs = embedder.embed(warrant_texts, batch_size=32)
    print(f"Embedding completed in {time.time() - t2:.2f}s.")
    
    # Save v1.1 embeddings
    v1_1_npz_path = "data/calibration/e3_1_component_embeddings_cache.npz"
    np.savez_compressed(
        v1_1_npz_path,
        text_ids=np.array(text_ids_ordered),
        claim_embeddings=claim_embs,
        reasoning_embeddings=reasoning_embs,
        warrant_embeddings=warrant_embs
    )
    
    id_to_idx = {tid: i for i, tid in enumerate(text_ids_ordered)}
    
    # 4. Compute pairwise v1.1 features
    v1_1_claim_cos = []
    v1_1_reason_cos = []
    v1_1_warrant_cos = []
    
    for _, row in old_df.iterrows():
        g_id = compute_stable_text_id(row['gold_perspective_text'])
        h_id = compute_stable_text_id(row['generated_viewpoint_text'])
        g_idx = id_to_idx[g_id]
        h_idx = id_to_idx[h_id]
        
        c_cos = float(np.dot(claim_embs[g_idx], claim_embs[h_idx]))
        r_cos = float(np.dot(reasoning_embs[g_idx], reasoning_embs[h_idx]))
        w_cos = float(np.dot(warrant_embs[g_idx], warrant_embs[h_idx]))
        
        v1_1_claim_cos.append(max(-1.0, min(1.0, c_cos)))
        v1_1_reason_cos.append(max(-1.0, min(1.0, r_cos)))
        v1_1_warrant_cos.append(max(-1.0, min(1.0, w_cos)))
        
    old_df['claim_cosine_v1_1'] = [round(x, 4) for x in v1_1_claim_cos]
    old_df['reasoning_cosine_v1_1'] = [round(x, 4) for x in v1_1_reason_cos]
    old_df['warrant_cosine_v1_1'] = [round(x, 4) for x in v1_1_warrant_cos]
    
    # 5. Measure Feature Stability / Drift
    print("\n" + "=" * 90)
    print("FEATURE STABILITY ANALYSIS (v1.0 vs v1.1 across 341 pairs)")
    print("=" * 90)
    
    stability_results = []
    for feat_name, old_col, new_col in [
        ("Central Claim Cosine", "claim_cosine", "claim_cosine_v1_1"),
        ("Core Reasoning Cosine", "reasoning_cosine", "reasoning_cosine_v1_1"),
        ("Underlying Warrant Cosine", "warrant_cosine", "warrant_cosine_v1_1")
    ]:
        v0 = old_df[old_col].values
        v1 = old_df[new_col].values
        diff = np.abs(v1 - v0)
        p_corr, _ = pearsonr(v0, v1)
        s_corr, _ = spearmanr(v0, v1)
        
        res = {
            "Feature": feat_name,
            "Pearson_r": round(float(p_corr), 5),
            "Spearman_rho": round(float(s_corr), 5),
            "Mean_Abs_Diff": round(float(np.mean(diff)), 5),
            "Median_Abs_Diff": round(float(np.median(diff)), 5),
            "Max_Abs_Diff": round(float(np.max(diff)), 5),
            "Pairs_Shift_gt_0_01": int(np.sum(diff > 0.01)),
            "Pairs_Shift_gt_0_05": int(np.sum(diff > 0.05)),
            "Pairs_Identical": int(np.sum(diff < 1e-4))
        }
        stability_results.append(res)
        
    stab_df = pd.DataFrame(stability_results)
    print(stab_df[['Feature', 'Pearson_r', 'Spearman_rho', 'Mean_Abs_Diff', 'Max_Abs_Diff', 'Pairs_Shift_gt_0_01', 'Pairs_Identical']].to_string(index=False))
    
    # Save stability report
    stab_df.to_csv("results/calibration/e3_1_feature_stability_report.csv", index=False)
    
    # Save combined features
    out_csv = "results/calibration/e3_1_pairwise_features_341.csv"
    old_df.to_csv(out_csv, index=False)
    print(f"\nSaved v1.1 pairwise features to: {out_csv}")

if __name__ == "__main__":
    main()
