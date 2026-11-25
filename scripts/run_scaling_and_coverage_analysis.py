#!/usr/bin/env python3
"""
Comprehensive Scaling and Coverage Analysis for Perspective Deliberation.

Analyzes master coverage results across all 34 models:
  1. Scaling Law Fitting: Coverage vs Log10(Params) by Family and Overall
  2. Alignment Progression: Base vs SFT vs DPO vs Instruct across OLMo-2 ladders (1B, 7B, 13B, 32B)
  3. Viewpoint Generation Efficiency: Matched Clusters per Viewpoint
  4. Official M0 vs Shadow M_soft Comparison across Scales
  5. Publication Tables (Markdown & LaTeX)
"""

import os
import sys
import glob
import json
import numpy as np
import pandas as pd
from scipy import stats

MODEL_METADATA = {
    # Gemma 2
    "gemma_2_2b_it": {"family": "Gemma 2", "params_b": 2.6, "stage": "Instruct", "tier": "Small (1-4B)"},
    "gemma_2_9b_it": {"family": "Gemma 2", "params_b": 9.2, "stage": "Instruct", "tier": "Medium (7-14B)"},
    "gemma_2_27b_it": {"family": "Gemma 2", "params_b": 27.2, "stage": "Instruct", "tier": "Large (27-32B)"},
    
    # LLaMA 3.x
    "llama_3_2_1b_instruct": {"family": "LLaMA 3.x", "params_b": 1.2, "stage": "Instruct", "tier": "Small (1-4B)"},
    "llama_3_2_3b_instruct": {"family": "LLaMA 3.x", "params_b": 3.2, "stage": "Instruct", "tier": "Small (1-4B)"},
    "llama_3_1_8b_instruct": {"family": "LLaMA 3.x", "params_b": 8.0, "stage": "Instruct", "tier": "Medium (7-14B)"},
    
    # SmolLM2
    "smollm2_135m_instruct": {"family": "SmolLM2", "params_b": 0.135, "stage": "Instruct", "tier": "Sub-Billion (<1B)"},
    "smollm2_360m_instruct": {"family": "SmolLM2", "params_b": 0.36, "stage": "Instruct", "tier": "Sub-Billion (<1B)"},
    "smollm2_1_7b_instruct": {"family": "SmolLM2", "params_b": 1.71, "stage": "Instruct", "tier": "Small (1-4B)"},

    # Qwen 2.5
    "qwen2_5_0_5b_instruct": {"family": "Qwen 2.5", "params_b": 0.49, "stage": "Instruct", "tier": "Sub-Billion (<1B)"},
    "qwen2_5_1_5b_instruct": {"family": "Qwen 2.5", "params_b": 1.54, "stage": "Instruct", "tier": "Small (1-4B)"},
    "qwen2_5_3b_instruct": {"family": "Qwen 2.5", "params_b": 3.09, "stage": "Instruct", "tier": "Small (1-4B)"},
    "qwen2_5_3b_instruct_p2": {"family": "Qwen 2.5", "params_b": 3.09, "stage": "Instruct-P2", "tier": "Small (1-4B)"},
    "qwen2_5_7b_instruct": {"family": "Qwen 2.5", "params_b": 7.61, "stage": "Instruct", "tier": "Medium (7-14B)"},
    "qwen2_5_7b_instruct_p2": {"family": "Qwen 2.5", "params_b": 7.61, "stage": "Instruct-P2", "tier": "Medium (7-14B)"},
    "qwen2_5_14b_instruct": {"family": "Qwen 2.5", "params_b": 14.77, "stage": "Instruct", "tier": "Medium (7-14B)"},
    "qwen2_5_14b_instruct_p2": {"family": "Qwen 2.5", "params_b": 14.77, "stage": "Instruct-P2", "tier": "Medium (7-14B)"},
    "qwen2_5_32b_instruct": {"family": "Qwen 2.5", "params_b": 32.76, "stage": "Instruct", "tier": "Large (27-32B)"},

    # OLMoE
    "olmoe_1b_7b_0924_instruct": {"family": "OLMoE", "params_b": 6.9, "stage": "Instruct", "tier": "Medium (7-14B)"},

    # OLMo 2 1B
    "olmo_2_0425_1b": {"family": "OLMo 2", "params_b": 1.2, "stage": "Base", "tier": "Small (1-4B)"},
    "olmo_2_0425_1b_sft": {"family": "OLMo 2", "params_b": 1.2, "stage": "SFT", "tier": "Small (1-4B)"},
    "olmo_2_0425_1b_dpo": {"family": "OLMo 2", "params_b": 1.2, "stage": "DPO", "tier": "Small (1-4B)"},
    "olmo_2_0425_1b_instruct": {"family": "OLMo 2", "params_b": 1.2, "stage": "Instruct", "tier": "Small (1-4B)"},

    # OLMo 2 7B
    "olmo_2_1124_7b": {"family": "OLMo 2", "params_b": 7.0, "stage": "Base", "tier": "Medium (7-14B)"},
    "olmo_2_1124_7b_sft": {"family": "OLMo 2", "params_b": 7.0, "stage": "SFT", "tier": "Medium (7-14B)"},
    "olmo_2_1124_7b_dpo": {"family": "OLMo 2", "params_b": 7.0, "stage": "DPO", "tier": "Medium (7-14B)"},
    "olmo_2_1124_7b_instruct": {"family": "OLMo 2", "params_b": 7.0, "stage": "Instruct", "tier": "Medium (7-14B)"},

    # OLMo 2 13B
    "olmo_2_1124_13b": {"family": "OLMo 2", "params_b": 13.0, "stage": "Base", "tier": "Medium (7-14B)"},
    "olmo_2_1124_13b_sft": {"family": "OLMo 2", "params_b": 13.0, "stage": "SFT", "tier": "Medium (7-14B)"},
    "olmo_2_1124_13b_dpo": {"family": "OLMo 2", "params_b": 13.0, "stage": "DPO", "tier": "Medium (7-14B)"},
    "olmo_2_1124_13b_instruct": {"family": "OLMo 2", "params_b": 13.0, "stage": "Instruct", "tier": "Medium (7-14B)"},

    # OLMo 2 32B
    "olmo_2_0325_32b": {"family": "OLMo 2", "params_b": 32.0, "stage": "Base", "tier": "Large (27-32B)"},
    "olmo_2_0325_32b_dpo": {"family": "OLMo 2", "params_b": 32.0, "stage": "DPO", "tier": "Large (27-32B)"},
    "olmo_2_0325_32b_instruct": {"family": "OLMo 2", "params_b": 32.0, "stage": "Instruct", "tier": "Large (27-32B)"},
}

def build_master_summary_from_per_model(output_dir="results/coverage"):
    per_model_dir = os.path.join(output_dir, "per_model")
    if not os.path.exists(per_model_dir):
        return None
    
    files = sorted(glob.glob(os.path.join(per_model_dir, "*_coverage.csv")))
    if not files:
        return None
        
    records = []
    for f in files:
        m_key = os.path.basename(f).replace("_coverage.csv", "")
        df_m = pd.read_csv(f)
        total_vps = int(df_m["n_viewpoints"].sum())
        m0_matched = int(df_m["m0_matched_clusters"].sum())
        total_clusters = int(df_m["n_gold_clusters"].sum())
        cov_m0 = float(df_m["m0_coverage_score"].mean() * 100)
        cov_soft = float(df_m["m_soft_coverage_score"].mean() * 100)
        eff = float(m0_matched / max(1, total_vps))
        
        records.append({
            "model_key": m_key,
            "n_claims": len(df_m),
            "n_viewpoints": total_vps,
            "m0_matched_clusters": m0_matched,
            "total_gold_clusters": total_clusters,
            "m0_coverage_pct": round(cov_m0, 4),
            "m_soft_coverage_pct": round(cov_soft, 4),
            "viewpoint_efficiency": round(eff, 6)
        })
    df_all = pd.DataFrame(records)
    out_master = os.path.join(output_dir, "master_coverage_all_models.csv")
    df_all.to_csv(out_master, index=False)
    print(f"Compiled master summary from {len(records)} completed models to {out_master}")
    return df_all

def analyze_scaling(summary_csv_path="results/coverage/master_coverage_all_models.csv"):
    import glob
    # Recompile from per-model files to ensure all finished models are present
    build_master_summary_from_per_model()
    
    if not os.path.exists(summary_csv_path):
        raise FileNotFoundError(f"Missing master summary: {summary_csv_path}")

    df = pd.read_csv(summary_csv_path)
    
    # Merge metadata
    meta_rows = []
    for _, r in df.iterrows():
        k = r['model_key']
        m = MODEL_METADATA.get(k, {"family": "Other", "params_b": 1.0, "stage": "Unknown", "tier": "Unknown"})
        meta_rows.append(m)
    df_meta = pd.DataFrame(meta_rows)
    df = pd.concat([df, df_meta], axis=1)

    print("=" * 100)
    print("PERSPECTIVE COVERAGE AND SCALING ANALYSIS")
    print("=" * 100)
    print(f"Total Models Evaluated: {len(df)}")
    print(f"Parameter Range: {df['params_b'].min():.2f}B to {df['params_b'].max():.2f}B")
    print(f"Families: {df['family'].unique().tolist()}")

    # 1. Scaling Law Fitting across Instruct models
    instruct_df = df[df['stage'].isin(['Instruct', 'Instruct-P2'])].copy()
    x_log = np.log10(instruct_df['params_b'].values)
    y_cov = instruct_df['m0_coverage_pct'].values

    slope, intercept, r_value, p_value, std_err = stats.linregress(x_log, y_cov)
    r_squared = r_value ** 2

    scaling_laws = {
        "overall_instruct": {
            "n_models": len(instruct_df),
            "slope_alpha": float(slope),
            "intercept_beta": float(intercept),
            "r_squared": float(r_squared),
            "p_value": float(p_value),
            "std_err": float(std_err),
            "formula": f"Coverage(%) = {slope:.2f} * log10(Params_B) + {intercept:.2f}"
        }
    }

    print("\n--- 1. SCALING LAW FIT (INSTRUCT MODELS) ---")
    print(f"Formula:   Coverage(%) = {slope:.2f} * log10(Params_B) + {intercept:.2f}")
    print(f"R^2 Score: {r_squared:.4f} (p = {p_value:.6e})")

    # Per-family scaling
    family_scaling = {}
    for fam in ['Qwen 2.5', 'LLaMA 3.x', 'Gemma 2', 'OLMo 2']:
        fam_df = instruct_df[instruct_df['family'] == fam]
        if len(fam_df) >= 3:
            s, ic, r, p, se = stats.linregress(np.log10(fam_df['params_b'].values), fam_df['m0_coverage_pct'].values)
            family_scaling[fam] = {
                "n_models": len(fam_df),
                "slope": float(s),
                "intercept": float(ic),
                "r_squared": float(r**2),
                "p_value": float(p)
            }
            print(f"  * {fam:<12}: Slope = {s:+5.2f}%/decade | R^2 = {r**2:.4f}")
    scaling_laws["by_family"] = family_scaling

    # 2. Alignment Progression (OLMo-2 Ladder)
    print("\n--- 2. ALIGNMENT PROGRESSION (OLMo-2 BASE -> SFT -> DPO -> INSTRUCT) ---")
    olmo_matrix = []
    for scale in [1.2, 7.0, 13.0, 32.0]:
        scale_df = df[(df['family'] == 'OLMo 2') & (df['params_b'] == scale)]
        row = {"params_b": scale}
        for stg in ['Base', 'SFT', 'DPO', 'Instruct']:
            match = scale_df[scale_df['stage'] == stg]
            if len(match) > 0:
                row[stg] = float(match['m0_coverage_pct'].values[0])
            else:
                row[stg] = None
        olmo_matrix.append(row)
        print(f"  * OLMo-2 {scale:4.1f}B: Base={row.get('Base', 'N/A')}% -> SFT={row.get('SFT', 'N/A')}% -> DPO={row.get('DPO', 'N/A')}% -> Instruct={row.get('Instruct', 'N/A')}%")
    
    # 3. Viewpoint Generation Efficiency & Redundancy
    print("\n--- 3. VIEWPOINT EFFICIENCY (MATCHED CLUSTERS PER VIEWPOINT) ---")
    tier_summary = df.groupby("tier").agg(
        mean_cov=("m0_coverage_pct", "mean"),
        mean_vps=("n_viewpoints", "mean"),
        mean_efficiency=("viewpoint_efficiency", "mean"),
        n_models=("model_key", "count")
    ).reset_index()
    print(tier_summary.to_string(index=False))

    # 4. M0 vs Shadow M_soft Gap Analysis
    df['m_soft_delta'] = df['m_soft_coverage_pct'] - df['m0_coverage_pct']
    print("\n--- 4. M0 VS SHADOW M_SOFT COVERAGE GAP ---")
    print(f"Mean Coverage under M0:       {df['m0_coverage_pct'].mean():.2f}%")
    print(f"Mean Coverage under M_soft:   {df['m_soft_coverage_pct'].mean():.2f}%")
    print(f"Mean Delta (M_soft - M0):     {df['m_soft_delta'].mean():+.2f}% [Min: {df['m_soft_delta'].min():+.2f}%, Max: {df['m_soft_delta'].max():+.2f}%]")

    # Save comprehensive report
    results = {
        "scaling_laws": scaling_laws,
        "olmo_alignment_matrix": olmo_matrix,
        "tier_summary": tier_summary.to_dict(orient="records"),
        "m_soft_gap": {
            "mean_m0": float(df['m0_coverage_pct'].mean()),
            "mean_m_soft": float(df['m_soft_coverage_pct'].mean()),
            "mean_delta": float(df['m_soft_delta'].mean()),
            "min_delta": float(df['m_soft_delta'].min()),
            "max_delta": float(df['m_soft_delta'].max())
        },
        "all_models_table": df.sort_values(by="m0_coverage_pct", ascending=False).to_dict(orient="records")
    }

    out_json = "results/coverage/perspectrum_scaling_and_coverage_report.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved analysis to {out_json}")

    # Generate Markdown Table for Paper
    df_sorted = df.sort_values(by=["family", "params_b"])
    try:
        table_md = df_sorted.to_markdown(index=False)
    except Exception:
        # Fallback without tabulate
        cols = list(df_sorted.columns)
        header = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join(["---"] * len(cols)) + " |"
        rows = []
        for _, r in df_sorted.iterrows():
            rows.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
        table_md = "\n".join([header, sep] + rows)

    with open("results/coverage/table_main_perspective_coverage.md", "w") as f:
        f.write(table_md)
    print("Saved Markdown table to results/coverage/table_main_perspective_coverage.md")

if __name__ == "__main__":
    analyze_scaling()
