"""
E3 Experiment: Feature Generation and Component Caching
Processes the 341 human ground-truth benchmark pairs:
  1. Decomposes each unique gold perspective and generated viewpoint into:
     - Central Claim
     - Core Reasoning
     - Underlying Warrant
  2. Caches extracted components with stable text IDs and extraction version.
  3. Computes normalized BGE-large embeddings for all extracted components.
  4. Computes pairwise cosine similarities (Claim, Reasoning, Warrant).
  5. Assembles complete feature dataset with DeBERTa entailment and probe scores.
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath("."))
from src.extraction.argument_component_extractor import ArgumentComponentExtractor, compute_stable_text_id, EXTRACTION_VERSION
from src.embeddings.embed_model import BGEEmbedder

def main():
    print("=" * 90)
    print("PHASE 2 & 3: REASONING-AWARE AUTOMATIC FEATURE EXTRACTION & CACHING")
    print("=" * 90)
    
    # 1. Load frozen 341-pair benchmark predictions
    input_csv = "results/calibration/stage2_cross_encoder_predictions_341.csv"
    bench_csv = "data/calibration/human_llm_benchmark_1700.csv"
    
    df = pd.read_csv(input_csv)
    bench = pd.read_csv(bench_csv)
    
    # Ensure stances are merged
    if "stance_gold" not in df.columns or "stance_generated" not in df.columns:
        df = df.merge(bench[['pair_id', 'stance_gold', 'stance_generated']], on='pair_id', how='left')
        
    print(f"Loaded {len(df)} pairs across {df['claim_id'].nunique()} claims.")
    print(f"Partitions: {df['partition'].value_counts().to_dict()}")
    
    # 2. Extract unique texts
    extractor = ArgumentComponentExtractor(version=EXTRACTION_VERSION)
    
    unique_texts = {} # text -> {"text_id": ..., "claim_topic": ..., "source": ...}
    
    for _, row in df.iterrows():
        g_txt = row['gold_perspective_text']
        h_txt = row['generated_viewpoint_text']
        cid = row['claim_id']
        c_topic = row['claim_text']
        
        if g_txt not in unique_texts:
            unique_texts[g_txt] = {
                "text": g_txt,
                "claim_topic": c_topic,
                "source": "gold",
                "claim_id": int(cid)
            }
        if h_txt not in unique_texts:
            unique_texts[h_txt] = {
                "text": h_txt,
                "claim_topic": c_topic,
                "source": "generated",
                "claim_id": int(cid)
            }
            
    print(f"\nUnique texts found:")
    print(f"  Gold perspectives:   {sum(1 for v in unique_texts.values() if v['source'] == 'gold')}")
    print(f"  Generated viewpoints:{sum(1 for v in unique_texts.values() if v['source'] == 'generated')}")
    print(f"  Total unique texts:  {len(unique_texts)}")
    
    # 3. Extract argument components
    t0 = time.time()
    extracted_records = {}
    for text, info in unique_texts.items():
        rec = extractor.extract_single(text, claim_topic=info["claim_topic"], metadata={"source": info["source"], "claim_id": info["claim_id"]})
        extracted_records[rec["text_id"]] = rec
        
    ext_time = time.time() - t0
    print(f"Argument component extraction completed in {ext_time:.3f}s ({ext_time/len(unique_texts)*1000:.2f}ms/text).")
    
    # Save extracted components cache
    cache_json_path = "data/calibration/e3_extracted_components_cache.json"
    os.makedirs("data/calibration", exist_ok=True)
    with open(cache_json_path, "w", encoding="utf-8") as f:
        json.dump(extracted_records, f, indent=2)
    print(f"Saved extracted components cache to: {cache_json_path}")
    
    # 4. Embed all components using BGE-large
    print("\nLoading BGE Embedder (BAAI/bge-large-en-v1.5)...")
    t1 = time.time()
    embedder = BGEEmbedder(model_name="BAAI/bge-large-en-v1.5", normalize=True)
    print(f"BGE Embedder loaded in {time.time() - t1:.2f}s")
    
    # Prepare lists to embed
    text_ids_ordered = list(extracted_records.keys())
    claim_texts = [extracted_records[tid]["central_claim"] for tid in text_ids_ordered]
    reasoning_texts = [extracted_records[tid]["core_reasoning"] for tid in text_ids_ordered]
    warrant_texts = [extracted_records[tid]["underlying_warrant"] for tid in text_ids_ordered]
    
    print(f"Embedding {len(text_ids_ordered) * 3} components ({len(text_ids_ordered)} unique texts x 3 components)...")
    t2 = time.time()
    
    claim_embs = embedder.embed(claim_texts, batch_size=32)
    reasoning_embs = embedder.embed(reasoning_texts, batch_size=32)
    warrant_embs = embedder.embed(warrant_texts, batch_size=32)
    
    embed_time = time.time() - t2
    print(f"Embedding completed in {embed_time:.2f}s ({embed_time / (len(text_ids_ordered)*3):.4f}s per embedding).")
    
    # Save embeddings cache (npz)
    cache_npz_path = "data/calibration/e3_component_embeddings_cache.npz"
    np.savez_compressed(
        cache_npz_path,
        text_ids=np.array(text_ids_ordered),
        claim_embeddings=claim_embs,
        reasoning_embeddings=reasoning_embs,
        warrant_embeddings=warrant_embs
    )
    print(f"Saved component embeddings cache to: {cache_npz_path}")
    
    # Index embeddings by text_id for fast pairwise lookup
    id_to_idx = {tid: i for i, tid in enumerate(text_ids_ordered)}
    
    # 5. Construct pairwise features for all 341 pairs
    print("\nConstructing pairwise feature dataset for 341 benchmark pairs...")
    pairwise_rows = []
    
    for _, row in df.iterrows():
        g_txt = row['gold_perspective_text']
        h_txt = row['generated_viewpoint_text']
        
        g_id = compute_stable_text_id(g_txt)
        h_id = compute_stable_text_id(h_txt)
        
        g_idx = id_to_idx[g_id]
        h_idx = id_to_idx[h_id]
        
        # Cosines (dot products since vectors are L2-normalized)
        cos_claim = float(np.dot(claim_embs[g_idx], claim_embs[h_idx]))
        cos_reasoning = float(np.dot(reasoning_embs[g_idx], reasoning_embs[h_idx]))
        cos_warrant = float(np.dot(warrant_embs[g_idx], warrant_embs[h_idx]))
        
        # Clip cosines to [-1.0, 1.0] for numerical safety
        cos_claim = max(-1.0, min(1.0, cos_claim))
        cos_reasoning = max(-1.0, min(1.0, cos_reasoning))
        cos_warrant = max(-1.0, min(1.0, cos_warrant))
        
        pair_dict = {
            "pair_id": int(row["pair_id"]),
            "claim_id": int(row["claim_id"]),
            "claim_text": row["claim_text"],
            "gold_perspective_text": g_txt,
            "generated_viewpoint_text": h_txt,
            "gold_text_id": g_id,
            "gen_text_id": h_id,
            "stance_gold": row.get("stance_gold", ""),
            "stance_generated": row.get("stance_generated", ""),
            "human_ground_truth_label": row["human_ground_truth_label"],
            "partition": row["partition"],
            # Base features
            "cosine_similarity": float(row["cosine_similarity"]),
            "p_entail": float(row["p_entail"]),
            "p_neutral": float(row["p_neutral"]),
            "p_contra": float(row["p_contra"]),
            "logit_entail": float(row["logit_entail"]),
            "p_calibrated": float(row.get("p_calibrated", 0.0)),
            # E3 Reasoning-Aware Features
            "claim_cosine": round(cos_claim, 4),
            "reasoning_cosine": round(cos_reasoning, 4),
            "warrant_cosine": round(cos_warrant, 4),
        }
        pairwise_rows.append(pair_dict)
        
    pairwise_df = pd.DataFrame(pairwise_rows)
    
    # Save pairwise features
    out_features_csv = "results/calibration/e3_pairwise_features_341.csv"
    os.makedirs("results/calibration", exist_ok=True)
    pairwise_df.to_csv(out_features_csv, index=False)
    print(f"Saved pairwise feature dataset to: {out_features_csv}")
    
    # 6. Feature statistics and correlation check
    print("\nFeature Summary Statistics (Overall Benchmark N=341):")
    stat_cols = ["cosine_similarity", "claim_cosine", "reasoning_cosine", "warrant_cosine", "p_entail"]
    print(pairwise_df[stat_cols].describe().round(4).to_string())
    
    print("\nFeature Correlation Matrix:")
    print(pairwise_df[stat_cols].corr().round(4).to_string())
    
    print("\nMean Feature Values by Human Label:")
    print(pairwise_df.groupby("human_ground_truth_label")[stat_cols].mean().round(4).to_string())
    
    # 7. Metadata provenance
    metadata = {
        "experiment": "E3_Reasoning_Aware_Feature_Extraction",
        "extraction_version": EXTRACTION_VERSION,
        "embedding_model": "BAAI/bge-large-en-v1.5",
        "total_benchmark_pairs": len(pairwise_df),
        "unique_gold_texts": sum(1 for v in unique_texts.values() if v['source'] == 'gold'),
        "unique_gen_texts": sum(1 for v in unique_texts.values() if v['source'] == 'generated'),
        "total_unique_texts": len(unique_texts),
        "total_component_embeddings": len(text_ids_ordered) * 3,
        "extraction_time_seconds": round(ext_time, 3),
        "embedding_time_seconds": round(embed_time, 3),
        "feature_columns": [
            "cosine_similarity",
            "p_entail",
            "p_neutral",
            "p_contra",
            "logit_entail",
            "claim_cosine",
            "reasoning_cosine",
            "warrant_cosine"
        ],
        "output_files": {
            "components_cache_json": cache_json_path,
            "embeddings_cache_npz": cache_npz_path,
            "pairwise_features_csv": out_features_csv
        }
    }
    
    metadata_path = "results/calibration/e3_feature_generation_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved feature-generation metadata to: {metadata_path}")
    print("\nPhase 2 & 3 Completed successfully!")

if __name__ == "__main__":
    main()
