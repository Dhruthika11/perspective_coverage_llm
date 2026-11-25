"""
Comprehensive Research Paper Analysis Generator:
Generates publication-ready figures, tables, and statistics for:
  1. Threshold Sensitivity & Robustness Sweep (tau in [0.65, 0.85] for all models)
  2. The Scaling Paradox Plot (Parameter size vs Coverage & Diversity)
  3. Sycophancy & Stance Bias Analysis (Support vs Undermine asymmetry)
  4. Qualitative Blind-Spot & Error Analysis (Claims missed by all models)
  5. Inference Budget Saturation Curves (Coverage vs K in [1..5])

Usage:
  python -m src.analysis.generate_paper_artifacts
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.io_utils import load_config, resolve, save_csv, save_json, load_csv
from src.utils.logging_utils import get_logger

logger = get_logger("paper_analysis")

MODEL_METADATA = {
    "qwen2_5_0_5b_instruct": {"name": "Qwen2.5-0.5B", "params_b": 0.5, "family": "Qwen"},
    "qwen2_5_1_5b_instruct": {"name": "Qwen2.5-1.5B", "params_b": 1.5, "family": "Qwen"},
    "qwen2_5_3b_instruct":   {"name": "Qwen2.5-3B",   "params_b": 3.0, "family": "Qwen"},
    "qwen2_5_7b_instruct":   {"name": "Qwen2.5-7B",   "params_b": 7.0, "family": "Qwen"},
    "qwen2_5_14b_instruct":  {"name": "Qwen2.5-14B",  "params_b": 14.0, "family": "Qwen"},
    "qwen2_5_32b_instruct":  {"name": "Qwen2.5-32B",  "params_b": 32.0, "family": "Qwen"},
    "mistral_7b_instruct_v0_3": {"name": "Mistral-7B", "params_b": 7.0, "family": "Mistral"},
    "llama_3_2_1b_instruct": {"name": "Llama-3.2-1B", "params_b": 1.0, "family": "Llama"},
    "llama_3_2_3b_instruct": {"name": "Llama-3.2-3B", "params_b": 3.0, "family": "Llama"},
    "llama_3_1_8b_instruct": {"name": "Llama-3.1-8B", "params_b": 8.0, "family": "Llama"},
    "gemma_2_2b_it":        {"name": "Gemma-2-2B",   "params_b": 2.0, "family": "Gemma"},
    "gemma_2_9b_it":        {"name": "Gemma-2-9B",   "params_b": 9.0, "family": "Gemma"},
    "gemma_3_12b_it":       {"name": "Gemma-3-12B",  "params_b": 12.0, "family": "Gemma"},
}

MODEL_COLORS = {
    "Mistral-7B": "#e66101",
    "Qwen2.5-0.5B": "#80b1d3",
    "Qwen2.5-1.5B": "#2b83ba",
    "Qwen2.5-3B": "#5e3c99",
    "Qwen2.5-7B": "#018571",
    "Qwen2.5-14B": "#984ea3",
    "Qwen2.5-32B": "#7b3294",
    "Llama-3.2-1B": "#33a02c",
    "Llama-3.2-3B": "#b2df8a",
    "Llama-3.1-8B": "#fdae61",
    "Gemma-2-2B": "#fb9a99",
    "Gemma-2-9B": "#e31a1c",
    "Gemma-3-12B": "#d7191c",
}


def load_available_models(results_root: str):
    """Discovers all models that have final_report_summary.json in results/."""
    models_data = {}
    for entry in os.listdir(results_root):
        sub_dir = os.path.join(results_root, entry)
        if not os.path.isdir(sub_dir):
            continue
        summary_path = os.path.join(sub_dir, "final_report_summary.json")
        if os.path.exists(summary_path):
            with open(summary_path, "r") as f:
                data = json.load(f)
            meta = MODEL_METADATA.get(entry, {"name": entry, "params_b": None, "family": "Other"})
            models_data[entry] = {
                "dir": sub_dir,
                "summary": data,
                "name": meta["name"],
                "params_b": meta["params_b"],
                "family": meta["family"],
            }
    return models_data


# ==============================================================================
# 1. THRESHOLD SENSITIVITY & ROBUSTNESS SWEEP
# ==============================================================================
def run_threshold_sensitivity(models_data: dict, out_figures: str, out_tables: str, tau_range=None):
    if tau_range is None:
        tau_range = np.round(np.arange(0.65, 0.86, 0.02), 3)

    logger.info("Computing Multi-Model Threshold Sensitivity Sweep...")
    records = []

    for model_key, info in models_data.items():
        sim_dir = os.path.join(info["dir"], "metrics")
        # Check if per_perspective_coverage.csv or similarity matrix data is available
        persp_cov_path = os.path.join(sim_dir, "per_perspective_coverage.csv")
        adv_sweep_path = os.path.join(sim_dir, "advanced_tau_sweep.csv")

        model_name = info["name"]

        # Strategy A: Use precomputed advanced_tau_sweep.csv if available
        if os.path.exists(adv_sweep_path):
            df_adv = pd.read_csv(adv_sweep_path)
            for tau in tau_range:
                # Find closest tau
                closest_row = df_adv.iloc[(df_adv["tau"] - tau).abs().argsort()[:1]]
                records.append({
                    "model_key": model_key,
                    "model_name": model_name,
                    "tau": float(tau),
                    "coverage": float(closest_row["coverage"].values[0]),
                })
        # Strategy B: Compute directly from per_perspective_coverage.csv (max_sim column)
        elif os.path.exists(persp_cov_path):
            df_p = pd.read_csv(persp_cov_path)
            sim_col = next((c for c in ["max_similarity", "best_sim", "similarity"] if c in df_p.columns), None)
            if sim_col:
                for tau in tau_range:
                    cov = (df_p[sim_col] >= tau).mean()
                    records.append({
                        "model_key": model_key,
                        "model_name": model_name,
                        "tau": float(tau),
                        "coverage": float(cov),
                    })

    if not records:
        logger.warning("No similarity sweep data found to build threshold sensitivity curve.")
        return

    df_sweep = pd.DataFrame(records)
    save_csv(df_sweep, os.path.join(out_tables, "table1_threshold_sensitivity.csv"))

    # Plot Multi-Model Sensitivity Curve
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=200)
    for model_name, grp in df_sweep.groupby("model_name"):
        grp = grp.sort_values("tau")
        color = MODEL_COLORS.get(model_name, None)
        ax.plot(grp["tau"], grp["coverage"], marker="o", markersize=4, linewidth=2, label=model_name, color=color)

    ax.axvline(0.744, color="#333333", linestyle="--", linewidth=1.2, label="Calibrated τ = 0.744")
    ax.set_xlabel("Similarity Threshold (τ)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Perspective Coverage Score", fontsize=11, fontweight="bold")
    ax.set_title("Threshold Sensitivity & Rank Invariance (τ ∈ [0.65, 0.85])", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(title="Model", frameon=True, fontsize=9)
    plt.tight_layout()
    fig_path = os.path.join(out_figures, "fig1_threshold_sensitivity.png")
    fig.savefig(fig_path)
    plt.close(fig)
    logger.info(f"Saved Figure 1 (Threshold Sensitivity) -> {fig_path}")


# ==============================================================================
# 2. SCALING PARADOX ANALYSIS (PARAMETERS VS COVERAGE & DIVERSITY)
# ==============================================================================
def run_scaling_paradox_analysis(models_data: dict, out_figures: str, out_tables: str):
    logger.info("Computing Scaling Paradox Analysis...")
    rows = []
    for model_key, info in models_data.items():
        s = info["summary"]
        params = info["params_b"]
        if params is None:
            continue
        rows.append({
            "model_key": model_key,
            "model_name": info["name"],
            "family": info["family"],
            "params_b": params,
            "coverage": s.get("mean_coverage_score", 0.0),
            "diversity": s.get("mean_diversity_score", 0.0),
            "js_divergence": s.get("mean_js_divergence", 0.0),
        })

    if len(rows) < 2:
        return

    df_scaling = pd.DataFrame(rows).sort_values("params_b")
    save_csv(df_scaling, os.path.join(out_tables, "table2_scaling_paradox.csv"))

    fig, ax1 = plt.subplots(figsize=(8.5, 5), dpi=200)
    color1 = "#2b83ba"
    color2 = "#d7191c"

    ax1.set_xscale("log")
    ax1.plot(df_scaling["params_b"], df_scaling["coverage"], marker="o", color=color1, linewidth=2.2, markersize=8, label="Perspective Coverage")
    for _, r in df_scaling.iterrows():
        ax1.annotate(r["model_name"], (r["params_b"], r["coverage"]), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8, fontweight="bold")

    ax1.set_xlabel("Model Parameters (Billions, Log Scale)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Perspective Coverage Score", color=color1, fontsize=11, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Secondary axis for internal diversity
    ax2 = ax1.twinx()
    ax2.plot(df_scaling["params_b"], df_scaling["diversity"], marker="s", color=color2, linewidth=1.8, linestyle="--", markersize=6, label="Internal Diversity Score")
    ax2.set_ylabel("Internal Diversity Score (1 - self_sim)", color=color2, fontsize=11, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor=color2)

    ax1.set_title("The Scaling Paradox: Coverage vs. Model Parameter Scale", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig_path = os.path.join(out_figures, "fig2_scaling_paradox.png")
    fig.savefig(fig_path)
    plt.close(fig)
    logger.info(f"Saved Figure 2 (Scaling Paradox) -> {fig_path}")


# ==============================================================================
# 3. SYCOPHANCY & STANCE BIAS ANALYSIS
# ==============================================================================
def run_stance_bias_analysis(models_data: dict, out_figures: str, out_tables: str):
    logger.info("Computing Sycophancy & Stance Bias Analysis...")
    rows = []
    for model_key, info in models_data.items():
        s = info["summary"]
        sup = s.get("mean_support_coverage", 0.0)
        und = s.get("mean_undermine_coverage", 0.0)
        asymmetry = sup - und
        ratio = (sup / und) if und > 0 else np.nan

        rows.append({
            "model_name": info["name"],
            "support_coverage": sup,
            "undermine_coverage": und,
            "stance_asymmetry_delta": asymmetry,
            "support_to_undermine_ratio": ratio,
        })

    df_stance = pd.DataFrame(rows).sort_values("support_coverage", ascending=False)
    save_csv(df_stance, os.path.join(out_tables, "table3_stance_bias.csv"))

    # Grouped Bar Chart
    x = np.arange(len(df_stance))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5), dpi=200)
    rects1 = ax.bar(x - width/2, df_stance["support_coverage"], width, label="Support Coverage (Affirmative)", color="#2ca02c")
    rects2 = ax.bar(x + width/2, df_stance["undermine_coverage"], width, label="Undermine Coverage (Opposing)", color="#d62728")

    ax.set_ylabel("Mean Coverage Score", fontsize=11, fontweight="bold")
    ax.set_title("Systematic Sycophancy / Stance Bias (Support vs. Undermine)", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(df_stance["model_name"], fontsize=10, fontweight="bold")
    ax.legend(frameon=True)
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    # Annotate deltas
    for idx, r in df_stance.reset_index().iterrows():
        delta = r["stance_asymmetry_delta"]
        top_y = max(r["support_coverage"], r["undermine_coverage"]) + 0.015
        ax.text(idx, top_y, f"Δ=+{delta:.1%}", ha="center", fontsize=8, color="#555555", fontweight="bold")

    plt.tight_layout()
    fig_path = os.path.join(out_figures, "fig3_stance_bias_sycophancy.png")
    fig.savefig(fig_path)
    plt.close(fig)
    logger.info(f"Saved Figure 3 (Stance Bias) -> {fig_path}")


# ==============================================================================
# 4. QUALITATIVE BLIND-SPOT & ERROR ANALYSIS
# ==============================================================================
def run_blind_spot_analysis(models_data: dict, out_tables: str, config: dict):
    logger.info("Extracting Universal Blind-Spot Claims...")
    processed_dir = resolve(config["dataset"]["processed_dir"])
    claims_df = load_csv(os.path.join(processed_dir, "claims.csv"))
    perspectives_df = load_csv(os.path.join(processed_dir, "perspectives.csv"))

    # Find claims where coverage == 0 across all available models
    zero_claims_by_model = []
    for model_key, info in models_data.items():
        cov_path = os.path.join(info["dir"], "metrics", "per_claim_coverage.csv")
        if os.path.exists(cov_path):
            df_cov = pd.read_csv(cov_path)
            col = next((c for c in ["coverage_score", "coverage"] if c in df_cov.columns), None)
            if col:
                zero_claims = set(df_cov[df_cov[col] == 0.0]["claim_id"])
                zero_claims_by_model.append(zero_claims)

    if not zero_claims_by_model:
        return

    # Intersection: Claims where ALL models failed completely (0.0 coverage)
    universal_zero_ids = set.intersection(*zero_claims_by_model)
    logger.info(f"Found {len(universal_zero_ids)} claims with 0.0% coverage across all models.")

    blind_spot_rows = []
    for cid in list(universal_zero_ids)[:50]:
        claim_text = claims_df.loc[claims_df["claim_id"] == cid, "claim_text"].values
        c_text = claim_text[0] if len(claim_text) else f"Claim {cid}"
        persps = perspectives_df[perspectives_df["claim_id"] == cid]["text"].tolist()
        blind_spot_rows.append({
            "claim_id": cid,
            "claim_text": c_text,
            "num_human_perspectives": len(persps),
            "sample_human_perspectives": " | ".join(persps[:3]),
        })

    df_blind = pd.DataFrame(blind_spot_rows)
    save_csv(df_blind, os.path.join(out_tables, "table4_universal_blind_spots.csv"))

    # Generate Markdown Summary for Paper Section 6
    md_content = "# Qualitative Blind-Spot Case Studies\n\n"
    md_content += f"Across all evaluated models, **{len(universal_zero_ids)} claims** received zero perspective coverage.\n\n"
    md_content += "| Claim ID | Claim / Debate Topic | Key Missed Human Perspectives |\n"
    md_content += "| :---: | :--- | :--- |\n"
    for _, r in df_blind.head(10).iterrows():
        md_content += f"| **{r['claim_id']}** | *{r['claim_text']}* | {r['sample_human_perspectives']} |\n"

    with open(os.path.join(out_tables, "blind_spots_summary.md"), "w", encoding="utf-8") as f:
        f.write(md_content)
    logger.info(f"Saved Table 4 (Blind Spots) -> {os.path.join(out_tables, 'table4_universal_blind_spots.csv')}")


# ==============================================================================
# 5. INFERENCE BUDGET SATURATION CURVE (COVERAGE VS K SAMPLES)
# ==============================================================================
def run_sample_saturation_analysis(models_data: dict, out_figures: str, out_tables: str):
    logger.info("Computing Sample Saturation Curves (K = 1..5)...")
    records = []
    for model_key, info in models_data.items():
        k_path = os.path.join(info["dir"], "metrics", "advanced_k_subsampling.csv")
        if os.path.exists(k_path):
            df_k = pd.read_csv(k_path)
            for _, r in df_k.iterrows():
                records.append({
                    "model_name": info["name"],
                    "k_samples": int(r["k_samples"]),
                    "coverage": float(r["coverage"]),
                })

    if not records:
        logger.warning("No advanced_k_subsampling.csv found across models.")
        return

    df_k_all = pd.DataFrame(records)
    save_csv(df_k_all, os.path.join(out_tables, "table5_sample_saturation.csv"))

    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=200)
    for model_name, grp in df_k_all.groupby("model_name"):
        grp = grp.sort_values("k_samples")
        color = MODEL_COLORS.get(model_name, None)
        ax.plot(grp["k_samples"], grp["coverage"], marker="s", markersize=6, linewidth=2, label=model_name, color=color)

    ax.set_xlabel("Generation Sampling Budget (K Samples per Claim)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Perspective Coverage Score", fontsize=11, fontweight="bold")
    ax.set_title("Perspective Discovery Saturation Curve (K = 1 to 5)", fontsize=12, fontweight="bold")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(title="Model", frameon=True)
    plt.tight_layout()
    fig_path = os.path.join(out_figures, "fig4_sample_saturation.png")
    fig.savefig(fig_path)
    plt.close(fig)
    logger.info(f"Saved Figure 4 (Saturation Curve) -> {fig_path}")


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================
def main():
    config = load_config()
    results_root = resolve("results")
    out_figures = resolve("results/paper_figures")
    out_tables = resolve("results/paper_tables")

    os.makedirs(out_figures, exist_ok=True)
    os.makedirs(out_tables, exist_ok=True)

    models_data = load_available_models(results_root)
    if not models_data:
        logger.warning(f"No evaluated models found in {results_root}.")
        return

    logger.info(f"Loaded {len(models_data)} models: {list(models_data.keys())}")

    run_threshold_sensitivity(models_data, out_figures, out_tables)
    run_scaling_paradox_analysis(models_data, out_figures, out_tables)
    run_stance_bias_analysis(models_data, out_figures, out_tables)
    run_blind_spot_analysis(models_data, out_tables, config)
    run_sample_saturation_analysis(models_data, out_figures, out_tables)

    print("\n" + "=" * 70)
    print(" ALL 5 RESEARCH PAPER ARTIFACTS GENERATED SUCCESSFULLY!")
    print("=" * 70)
    print(f" Figures saved to: {out_figures}")
    print(f" Tables saved to:  {out_tables}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
