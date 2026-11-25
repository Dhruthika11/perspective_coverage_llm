#!/usr/bin/env python3
"""
Paired Claim-Level Clustered Bootstrap Uncertainty Analysis for Perspective Coverage.

Evaluates 10,000 paired claim-clustered bootstrap resamples across all 907 claims:
  1. Computes mean coverage and 95% Confidence Intervals (Percentile & BCa) for all 34 models.
  2. Evaluates paired hypothesis tests (Model A vs Model B) on identical claims:
     - OLMo-2 7B: DPO vs Base, DPO vs Instruct, SFT vs Base
     - OLMo-2 1.2B: Instruct vs Base, SFT vs Base, DPO vs Base
     - OLMo-2 13B: SFT vs Base, SFT vs Instruct, DPO vs Base
     - OLMo-2 32B: Base vs Instruct, Base vs DPO
     - LLaMA 3.2 3B vs LLaMA 3.1 8B, LLaMA 3.2 3B vs 3.2 1B
     - Gemma 2: 9B vs 27B, 9B vs 2B
     - Official M0 vs Shadow M_soft paired test for all model families
  3. Outputs exact two-tailed paired p-values and 95% CIs on delta.
"""

import os
import glob
import json
import numpy as np
import pandas as pd
from scipy import stats

def compute_bca_ci(resamples, original_stat, data_matrix, alpha=0.05):
    """Computes Bias-Corrected and Accelerated (BCa) bootstrap confidence interval."""
    B = len(resamples)
    # Acceleration parameter a via jackknife
    n = len(data_matrix)
    # Mean of jackknife samples
    jack_means = np.zeros(n)
    total_sum = np.sum(data_matrix, axis=0)
    for i in range(n):
        jack_means[i] = (total_sum - data_matrix[i]) / (n - 1)
    
    mean_jack = np.mean(jack_means)
    num = np.sum((mean_jack - jack_means) ** 3)
    den = 6.0 * (np.sum((mean_jack - jack_means) ** 2) ** 1.5)
    a = num / max(1e-12, den)
    
    # Bias correction z0
    prop_less = np.mean(resamples < original_stat)
    prop_less = np.clip(prop_less, 1e-6, 1.0 - 1e-6)
    z0 = stats.norm.ppf(prop_less)
    
    # Standard normal quantiles
    z_lo = stats.norm.ppf(alpha / 2.0)
    z_hi = stats.norm.ppf(1.0 - alpha / 2.0)
    
    a1 = stats.norm.cdf(z0 + (z0 + z_lo) / (1.0 - a * (z0 + z_lo)))
    a2 = stats.norm.cdf(z0 + (z0 + z_hi) / (1.0 - a * (z0 + z_hi)))
    
    ci_lo = np.percentile(resamples, 100.0 * a1)
    ci_hi = np.percentile(resamples, 100.0 * a2)
    return float(ci_lo), float(ci_hi)

def run_paired_claim_bootstrap(per_model_dir="results/coverage/per_model", n_boot=10000, seed=42):
    files = sorted(glob.glob(os.path.join(per_model_dir, "*_coverage.csv")))
    if not files:
        raise FileNotFoundError(f"No coverage CSVs found in {per_model_dir}")
        
    print(f"Loading claim-level coverage for {len(files)} models...")
    model_data = {}
    claim_ids = None
    
    for f in files:
        m_key = os.path.basename(f).replace("_coverage.csv", "")
        df = pd.read_csv(f).sort_values("claim_id")
        if claim_ids is None:
            claim_ids = df["claim_id"].values
        else:
            assert np.array_equal(claim_ids, df["claim_id"].values), f"Claim IDs mismatch in {m_key}"
            
        model_data[m_key] = {
            "m0": df["m0_coverage_score"].values * 100.0,
            "m_soft": df["m_soft_coverage_score"].values * 100.0
        }
        
    n_claims = len(claim_ids)
    print(f"Loaded {n_claims} synchronized claims across all {len(model_data)} models.")
    
    rng = np.random.default_rng(seed)
    # Generate 10,000 resample indices (shape: n_boot, n_claims)
    boot_indices = rng.integers(0, n_claims, size=(n_boot, n_claims))
    
    # 1. Per-Model Bootstrap CI
    model_stats = {}
    for m_key, scores in model_data.items():
        m0_arr = scores["m0"]
        orig_mean = float(np.mean(m0_arr))
        boot_means = np.mean(m0_arr[boot_indices], axis=1)
        
        ci_lo_perc = float(np.percentile(boot_means, 2.5))
        ci_hi_perc = float(np.percentile(boot_means, 97.5))
        bca_lo, bca_hi = compute_bca_ci(boot_means, orig_mean, m0_arr)
        
        # Shadow M_soft
        soft_arr = scores["m_soft"]
        orig_soft_mean = float(np.mean(soft_arr))
        boot_soft_means = np.mean(soft_arr[boot_indices], axis=1)
        soft_ci_lo = float(np.percentile(boot_soft_means, 2.5))
        soft_ci_hi = float(np.percentile(boot_soft_means, 97.5))
        
        # Paired delta M_soft - M0
        delta_arr = soft_arr - m0_arr
        orig_delta = float(np.mean(delta_arr))
        boot_deltas = np.mean(delta_arr[boot_indices], axis=1)
        delta_ci_lo = float(np.percentile(boot_deltas, 2.5))
        delta_ci_hi = float(np.percentile(boot_deltas, 97.5))
        p_val_delta = float(2.0 * min(np.mean(boot_deltas <= 0), np.mean(boot_deltas >= 0)))
        
        model_stats[m_key] = {
            "m0_mean": round(orig_mean, 4),
            "m0_ci_95_percentile": [round(ci_lo_perc, 4), round(ci_hi_perc, 4)],
            "m0_ci_95_bca": [round(bca_lo, 4), round(bca_hi, 4)],
            "m0_std_err": round(float(np.std(boot_means)), 4),
            "m_soft_mean": round(orig_soft_mean, 4),
            "m_soft_ci_95": [round(soft_ci_lo, 4), round(soft_ci_hi, 4)],
            "delta_mean": round(orig_delta, 4),
            "delta_ci_95": [round(delta_ci_lo, 4), round(delta_ci_hi, 4)],
            "p_val_m_soft_vs_m0": round(p_val_delta, 6)
        }
        
    # 2. Key Pairwise Comparisons (Hypothesis Tests on Claims)
    pairwise_comparisons = [
        # OLMo 2 7B ladder
        ("olmo_2_1124_7b_dpo", "olmo_2_1124_7b", "OLMo-2 7B: DPO vs Base"),
        ("olmo_2_1124_7b_dpo", "olmo_2_1124_7b_instruct", "OLMo-2 7B: DPO vs Instruct"),
        ("olmo_2_1124_7b_sft", "olmo_2_1124_7b", "OLMo-2 7B: SFT vs Base"),
        ("olmo_2_1124_7b_dpo", "olmo_2_1124_7b_sft", "OLMo-2 7B: DPO vs SFT"),
        
        # OLMo 2 1.2B ladder
        ("olmo_2_0425_1b_instruct", "olmo_2_0425_1b", "OLMo-2 1.2B: Instruct vs Base"),
        ("olmo_2_0425_1b_sft", "olmo_2_0425_1b", "OLMo-2 1.2B: SFT vs Base"),
        ("olmo_2_0425_1b_dpo", "olmo_2_0425_1b", "OLMo-2 1.2B: DPO vs Base"),
        
        # OLMo 2 13B ladder
        ("olmo_2_1124_13b_sft", "olmo_2_1124_13b", "OLMo-2 13B: SFT vs Base"),
        ("olmo_2_1124_13b_sft", "olmo_2_1124_13b_instruct", "OLMo-2 13B: SFT vs Instruct"),
        ("olmo_2_1124_13b_dpo", "olmo_2_1124_13b_instruct", "OLMo-2 13B: DPO vs Instruct"),
        
        # OLMo 2 32B ladder
        ("olmo_2_0325_32b", "olmo_2_0325_32b_instruct", "OLMo-2 32B: Base vs Instruct"),
        ("olmo_2_0325_32b", "olmo_2_0325_32b_dpo", "OLMo-2 32B: Base vs DPO"),
        ("olmo_2_0325_32b_dpo", "olmo_2_0325_32b_instruct", "OLMo-2 32B: DPO vs Instruct"),
        
        # LLaMA 3.x
        ("llama_3_2_3b_instruct", "llama_3_1_8b_instruct", "LLaMA: 3.2 3B vs 3.1 8B"),
        ("llama_3_2_3b_instruct", "llama_3_2_1b_instruct", "LLaMA: 3.2 3B vs 3.2 1B"),
        
        # Gemma 2
        ("gemma_2_9b_it", "gemma_2_27b_it", "Gemma 2: 9B vs 27B"),
        ("gemma_2_9b_it", "gemma_2_2b_it", "Gemma 2: 9B vs 2B"),
        
        # Cross-family at ~7-8B scale
        ("olmo_2_1124_7b_dpo", "llama_3_1_8b_instruct", "OLMo-2 7B DPO vs LLaMA-3.1 8B"),
        ("olmo_2_1124_7b_dpo", "gemma_2_9b_it", "OLMo-2 7B DPO vs Gemma-2 9B"),
        ("olmo_2_1124_7b_dpo", "qwen2_5_7b_instruct", "OLMo-2 7B DPO vs Qwen-2.5 7B")
    ]
    
    pairwise_results = []
    for m_a, m_b, label in pairwise_comparisons:
        if m_a not in model_data or m_b not in model_data:
            continue
            
        diff_arr = model_data[m_a]["m0"] - model_data[m_b]["m0"]
        orig_diff = float(np.mean(diff_arr))
        boot_diffs = np.mean(diff_arr[boot_indices], axis=1)
        
        ci_lo = float(np.percentile(boot_diffs, 2.5))
        ci_hi = float(np.percentile(boot_diffs, 97.5))
        bca_lo, bca_hi = compute_bca_ci(boot_diffs, orig_diff, diff_arr)
        
        # Two-tailed paired p-value
        p_val = float(2.0 * min(np.mean(boot_diffs <= 0), np.mean(boot_diffs >= 0)))
        p_val = min(1.0, max(1.0 / n_boot, p_val))
        
        pairwise_results.append({
            "comparison": label,
            "model_a": m_a,
            "model_b": m_b,
            "delta_mean_pct": round(orig_diff, 4),
            "ci_95_percentile": [round(ci_lo, 4), round(ci_hi, 4)],
            "ci_95_bca": [round(bca_lo, 4), round(bca_hi, 4)],
            "std_err": round(float(np.std(boot_diffs)), 4),
            "p_value": round(p_val, 6),
            "significant_at_005": bool(p_val < 0.05)
        })
        
    return model_stats, pairwise_results

def main():
    print("=" * 80)
    print("PAIRED CLAIM-LEVEL CLUSTERED BOOTSTRAP UNCERTAINTY ANALYSIS (10,000 RESAMPLES)")
    print("=" * 80)
    
    model_stats, pairwise = run_paired_claim_bootstrap(n_boot=10000)
    
    print("\nKEY PAIRWISE CLAIM-LEVEL COMPARISONS (95% BCa Confidence Intervals):")
    for r in pairwise:
        sig_marker = "***" if r['p_value'] < 0.001 else ("**" if r['p_value'] < 0.01 else ("*" if r['p_value'] < 0.05 else "n.s."))
        print(f"  * {r['comparison']:<35}: Delta = {r['delta_mean_pct']:+6.2f}% "
              f"[95% CI: {r['ci_95_bca'][0]:+6.2f}%, {r['ci_95_bca'][1]:+6.2f}%] "
              f"| p = {r['p_value']:.4e} ({sig_marker})")
              
    report = {
        "n_bootstrap_resamples": 10000,
        "cluster_unit": "claim_id (N=907 independent clusters)",
        "pairwise_comparisons": pairwise,
        "per_model_confidence_intervals": model_stats
    }
    
    out_path = os.path.join("results", "coverage", "claim_level_bootstrap_uncertainty_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved claim-level bootstrap report to {out_path}")
    
    # Save clean Markdown table
    df_pw = pd.DataFrame(pairwise)
    df_pw["95% CI (BCa)"] = df_pw["ci_95_bca"].apply(lambda x: f"[{x[0]:+.2f}%, {x[1]:+.2f}%]")
    df_pw_table = df_pw[["comparison", "delta_mean_pct", "95% CI (BCa)", "std_err", "p_value", "significant_at_005"]].copy()
    df_pw_table.columns = ["Comparison", "Mean Delta (%)", "95% BCa CI", "SE", "p-value", "p < 0.05"]
    
    md_table_path = os.path.join("results", "coverage", "table_claim_level_bootstrap_pairwise.md")
    cols = list(df_pw_table.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = ["| " + " | ".join(str(r[c]) for c in cols) + " |" for _, r in df_pw_table.iterrows()]
    md_content = "\n".join([header, sep] + rows)
    
    with open(md_table_path, "w") as f:
        f.write(md_content)
    print(f"Saved pairwise Markdown table to {md_table_path}")

if __name__ == "__main__":
    main()

