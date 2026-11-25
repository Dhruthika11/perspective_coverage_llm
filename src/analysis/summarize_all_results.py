"""
Master Summary Aggregator:
Finds all `final_report_summary*.json` across all models and embedding models,
and outputs a unified comparative table.
"""
import os
import sys
import json
import glob
import pandas as pd

def summarize_all(results_dir="results"):
    rows = []
    
    # 1. Gather all final_report_summary*.json files
    pattern = os.path.join(results_dir, "**", "final_report_summary*.json")
    summary_files = glob.glob(pattern, recursive=True)
    
    for fpath in summary_files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        
        rel_dir = os.path.dirname(os.path.relpath(fpath, results_dir))
        model_name = data.get("llm_model", rel_dir.split(os.sep)[0])
        emb_model = data.get("embedding_model", "Unknown")
        
        # Clean display names
        short_llm = model_name.split("/")[-1]
        short_emb = emb_model.split("/")[-1]
        
        rows.append({
            "LLM Model": short_llm,
            "Embedding": short_emb,
            "Tau": round(data.get("similarity_threshold", 0.0), 3),
            "Coverage": round(data.get("mean_coverage_score", 0.0), 4),
            "Support Cov": round(data.get("mean_support_coverage", 0.0), 4),
            "Undermine Cov": round(data.get("mean_undermine_coverage", 0.0), 4),
            "JS Divergence": round(data.get("mean_js_divergence", 0.0), 4),
            "Diversity": round(data.get("mean_diversity_score", 0.0), 4),
            "Zero-Cov Claims": int(data.get("claims_with_zero_coverage", 0)),
            "Full-Cov Claims": int(data.get("claims_with_full_coverage", 0)),
            "Path": fpath
        })
    
    if not rows:
        print("No final_report_summary*.json files found in results/.")
        return None
        
    df = pd.DataFrame(rows)
    df = df.sort_values(by=["LLM Model", "Embedding"])
    
    # Print formatted master table
    print("\n" + "=" * 120)
    print("                 MASTER EXPERIMENT RESULTS (ALL MODELS x ALL EMBEDDINGS)")
    print("=" * 120)
    display_cols = ["LLM Model", "Embedding", "Tau", "Coverage", "Support Cov", "Undermine Cov", "JS Divergence", "Diversity", "Zero-Cov Claims"]
    print(df[display_cols].to_string(index=False))
    print("=" * 120)
    
    # Save unified CSV and JSON
    out_csv = os.path.join(results_dir, "master_all_experiments_summary.csv")
    out_json = os.path.join(results_dir, "master_all_experiments_summary.json")
    df.to_csv(out_csv, index=False)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"\n[Saved master aggregated table to: {out_csv}]")
    
    # 2. Also check and display True Two-Stage Cross-Model Comparison if available
    two_stage_csv = os.path.join(results_dir, "two_stage_cross_model_comparison.csv")
    if os.path.exists(two_stage_csv):
        print("\n" + "=" * 120)
        print("          CALIBRATED TWO-STAGE COVERAGE COMPARISON (STAGE 1 COSINE + STAGE 2 DeBERTa-v3)")
        print("=" * 120)
        two_stage_df = pd.read_csv(two_stage_csv)
        print(two_stage_df.to_string(index=False))
        print("=" * 120)
        
    return df

if __name__ == "__main__":
    summarize_all()
