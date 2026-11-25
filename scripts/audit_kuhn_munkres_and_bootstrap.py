#!/usr/bin/env python3
"""
Methodological Audit: Kuhn-Munkres Verification, Corrected Budget Ladders, and Holm-Adjusted Bootstrap.

Audits:
  1. Kuhn-Munkres Mechanics:
     - Discarding below-threshold assignments vs pre-thresholding weight matrix
     - Proving zero-weight edges cannot displace above-threshold matches
     - Slot consumption and objective function alignment
  2. Corrected OLMo-2 Ladders (Separating Native from K=5 Budget):
     - Side-by-side comparison of Base, SFT, DPO, Instruct at K=5 and Native
     - Reconciling the 32B comparison (Base vs Instruct vs DPO)
  3. Holm-Bonferroni Multiple Comparisons Correction:
     - Replaces numerical resolution floor (1.0e-4) with p < 0.0001
     - Applies Holm-Bonferroni step-down adjustment to all pairwise claim tests
"""

import os
import glob
import json
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

def audit_kuhn_munkres_mechanics():
    """
    Audits whether below-threshold pairs can displace or prevent valid matches.
    Specifically tests:
      Scenario A: Weight matrix with zero-weight entries vs pre-thresholded matrix.
      Scenario B: Can a weak pair (w < T) prevent a valid pair (w >= T) from matching?
    """
    rng = np.random.default_rng(42)
    T = 0.20
    discrepancies = 0
    n_trials = 1000

    for _ in range(n_trials):
        n_rows = rng.integers(2, 15)
        n_cols = rng.integers(2, 15)
        # Random weights in [0, 1]
        w_raw = rng.uniform(0.0, 1.0, size=(n_rows, n_cols)).astype(np.float32)
        
        # Method 1: Current implementation in pipeline
        # Entries < T are set to 0.0 in candidate generation
        w_pipeline = np.where(w_raw >= T, w_raw, 0.0)
        row_ind1, col_ind1 = linear_sum_assignment(-w_pipeline)
        matched1 = sum(w_pipeline[r, c] >= T for r, c in zip(row_ind1, col_ind1))
        
        # Method 2: Maximum Cardinality Bipartite Matching on thresholded adjacency
        # Adjacency matrix: 1 if w >= T else 0
        w_adj = (w_raw >= T).astype(np.float32)
        row_ind2, col_ind2 = linear_sum_assignment(-w_adj)
        matched2 = sum(w_adj[r, c] == 1.0 for r, c in zip(row_ind2, col_ind2))
        
        # Method 3: Strictly positive thresholding with large negative cost for < T
        w_strict = np.where(w_raw >= T, w_raw, -1e6)
        row_ind3, col_ind3 = linear_sum_assignment(-w_strict)
        matched3 = sum(w_strict[r, c] >= T for r, c in zip(row_ind3, col_ind3))
        
        if matched1 != matched2 or matched1 != matched3:
            discrepancies += 1

    return {
        "n_trials": n_trials,
        "discrepancies_between_weight_max_and_cardinality_max": discrepancies,
        "is_equivalent_to_maximum_cardinality": (discrepancies == 0)
    }

def audit_holm_bonferroni_bootstrap(report_path="results/coverage/claim_level_bootstrap_uncertainty_report.json"):
    with open(report_path, "r") as f:
        rep = json.load(f)
        
    pairwise = rep["pairwise_comparisons"]
    m = len(pairwise)
    
    # Sort by unadjusted p-value ascending
    sorted_pairs = sorted(pairwise, key=lambda x: x["p_value"])
    
    # Holm-Bonferroni step-down: p_holm[k] = min(1.0, max(p_holm[k-1], p_val[k] * (m - k)))
    prev_holm = 0.0
    for k, item in enumerate(sorted_pairs):
        p_raw = item["p_value"]
        # If p_raw == 1.0e-4 (our bootstrap resolution limit), treat as 0.0001
        p_for_calc = p_raw if p_raw > 1.0e-4 else 0.0001
        multiplier = m - k
        p_holm = min(1.0, p_for_calc * multiplier)
        p_holm = max(p_holm, prev_holm)
        prev_holm = p_holm
        
        item["p_raw"] = p_raw
        item["p_raw_display"] = "< 0.0001" if p_raw <= 1.0e-4 else f"{p_raw:.4f}"
        item["p_holm"] = round(p_holm, 6)
        item["p_holm_display"] = "< 0.001" if p_holm < 0.001 else f"{p_holm:.4f}"
        item["significant_holm_005"] = bool(p_holm < 0.05)
        item["significant_holm_001"] = bool(p_holm < 0.01)
        
    return sorted_pairs

def build_corrected_olmo_ladder_analysis():
    df_budget = pd.read_csv("results/coverage/budget_vs_capability_matrix.csv")
    df_main = pd.read_csv("results/coverage/master_coverage_all_models.csv")
    df_div = pd.read_csv("results/coverage/viewpoint_diversity_and_redundancy_all_models.csv")
    
    olmo_models = [
        # 1.2B
        "olmo_2_0425_1b", "olmo_2_0425_1b_sft", "olmo_2_0425_1b_dpo", "olmo_2_0425_1b_instruct",
        # 7.0B
        "olmo_2_1124_7b", "olmo_2_1124_7b_sft", "olmo_2_1124_7b_dpo", "olmo_2_1124_7b_instruct",
        # 13.0B
        "olmo_2_1124_13b", "olmo_2_1124_13b_sft", "olmo_2_1124_13b_dpo", "olmo_2_1124_13b_instruct",
        # 32.0B
        "olmo_2_0325_32b", "olmo_2_0325_32b_dpo", "olmo_2_0325_32b_instruct"
    ]
    
    records = []
    for m in olmo_models:
        r_b = df_budget[df_budget["model_key"] == m].iloc[0]
        r_m = df_main[df_main["model_key"] == m].iloc[0]
        r_d = df_div[df_div["model_key"] == m].iloc[0]
        
        # Parse scale and stage
        if "1b" in m:
            scale = "1.2B"
        elif "7b" in m:
            scale = "7.0B"
        elif "13b" in m:
            scale = "13.0B"
        else:
            scale = "32.0B"
            
        if "sft" in m:
            stage = "SFT"
        elif "dpo" in m:
            stage = "DPO"
        elif "instruct" in m:
            stage = "Instruct"
        else:
            stage = "Base"
            
        records.append({
            "model_key": m,
            "scale": scale,
            "stage": stage,
            "native_viewpoints": int(r_b["native_viewpoints"]),
            "cov_k1": round(float(r_b["cov_k1"]), 2),
            "cov_k3": round(float(r_b["cov_k3"]), 2),
            "cov_k5": round(float(r_b["cov_k5"]), 2),
            "cov_native": round(float(r_b["cov_native"]), 2),
            "delta_native_vs_k5": round(float(r_b["delta_native_vs_k5"]), 2),
            "intra_claim_cosine": round(float(r_d["intra_claim_pairwise_cosine"]), 4),
            "dispersion": round(float(r_d["intra_claim_dispersion"]), 4)
        })
        
    return pd.DataFrame(records)

def main():
    print("=" * 80)
    print("METHODOLOGICAL AUDIT & RECONCILIATION")
    print("=" * 80)
    
    # 1. Kuhn-Munkres Audit
    print("\n1. Auditing Kuhn-Munkres Optimization & Cardinality Equivalence...")
    km_audit = audit_kuhn_munkres_mechanics()
    print(f"  * 1,000 Trial Test: Discrepancies between weighted assignment and maximum cardinality: {km_audit['discrepancies_between_weight_max_and_cardinality_max']}")
    print(f"  * Exact Cardinality Equivalence: {km_audit['is_equivalent_to_maximum_cardinality']}")
    
    # 2. Holm-Bonferroni Bootstrap
    print("\n2. Applying Holm-Bonferroni Adjustment to Pairwise Claim Tests...")
    holm_pairs = audit_holm_bonferroni_bootstrap()
    print(f"  * Evaluated {len(holm_pairs)} pairwise claim-clustered hypothesis tests.")
    for p in holm_pairs:
        sig = "***" if p['p_holm'] < 0.001 else ("**" if p['p_holm'] < 0.01 else ("*" if p['p_holm'] < 0.05 else "n.s."))
        print(f"  * {p['comparison']:<35}: Delta = {p['delta_mean_pct']:+6.2f}% "
              f"[95% BCa: {p['ci_95_bca'][0]:+6.2f}%, {p['ci_95_bca'][1]:+6.2f}%] "
              f"| p_raw = {p['p_raw_display']:<8} | p_holm = {p['p_holm_display']:<8} ({sig})")
              
    out_holm_json = "results/coverage/claim_level_bootstrap_holm_adjusted_report.json"
    with open(out_holm_json, "w") as f:
        json.dump(holm_pairs, f, indent=2)
    print(f"\nSaved Holm-adjusted report to {out_holm_json}")
    
    # Generate Holm Markdown Table
    df_h = pd.DataFrame(holm_pairs)
    df_h["95% BCa CI"] = df_h["ci_95_bca"].apply(lambda x: f"[{x[0]:+.2f}%, {x[1]:+.2f}%]")
    table_cols = ["comparison", "delta_mean_pct", "95% BCa CI", "std_err", "p_raw_display", "p_holm_display", "significant_holm_005"]
    df_h_table = df_h[table_cols].copy()
    df_h_table.columns = ["Comparison", "Mean Delta (%)", "95% BCa CI", "SE (%)", "p (Raw)", "p (Holm-Adjusted)", "Signif. (p < 0.05)"]
    
    cols = list(df_h_table.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = ["| " + " | ".join(str(r[c]) for c in cols) + " |" for _, r in df_h_table.iterrows()]
    md_holm = "\n".join([header, sep] + rows)
    
    with open("results/coverage/table_claim_level_bootstrap_holm_adjusted.md", "w") as f:
        f.write(md_holm)
    print("Saved Holm-adjusted Markdown table to results/coverage/table_claim_level_bootstrap_holm_adjusted.md")
    
    # 3. Corrected OLMo Ladder Table
    print("\n3. Generating Reconciled OLMo-2 Ladder Table (K=5 vs Native)...")
    df_olmo = build_corrected_olmo_ladder_analysis()
    df_olmo.to_csv("results/coverage/reconciled_olmo2_ladder_analysis.csv", index=False)
    
    cols = list(df_olmo.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = ["| " + " | ".join(str(r[c]) for c in cols) + " |" for _, r in df_olmo.iterrows()]
    md_olmo = "\n".join([header, sep] + rows)
    
    with open("results/coverage/table_reconciled_olmo2_ladders.md", "w") as f:
        f.write(md_olmo)
    print("Saved Reconciled OLMo-2 Ladder Markdown table to results/coverage/table_reconciled_olmo2_ladders.md")
    
    # Save combined methodological audit summary
    summary = {
        "kuhn_munkres_mechanics_audit": km_audit,
        "reconciled_olmo_ladders": df_olmo.to_dict(orient="records"),
        "holm_adjusted_hypotheses": holm_pairs
    }
    with open("results/coverage/methodological_audit_and_reconciliation_report.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved full Methodological Audit Report to results/coverage/methodological_audit_and_reconciliation_report.json")

if __name__ == "__main__":
    main()
