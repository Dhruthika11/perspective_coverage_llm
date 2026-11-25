"""
Unit Tests for Phase 7: Statistical Significance & Bootstrap Confidence Intervals.

Tests:
1. Bootstrap confidence interval estimation and mathematical bounds (ci_lo <= point <= ci_hi).
2. Monotonicity and FWER guarantees of Holm-Bonferroni step-down correction.
3. Benjamini-Hochberg False Discovery Rate (FDR) bounds.
4. Rank-biserial effect size calculation in [-1.0, 1.0].
5. Stance asymmetry significance detection.
"""
import pytest
import numpy as np
import pandas as pd
from src.analysis.statistical_significance_testing import (
    compute_bootstrap_ci,
    compute_pairwise_significance_table,
    compute_stance_asymmetry_significance,
    compute_domain_kruskal_wallis
)


def test_bootstrap_ci_coverage_bounds():
    """Verifies that bootstrap confidence intervals bound the point estimate and respect limits."""
    np.random.seed(42)
    sample_data = np.random.beta(a=2, b=5, size=200)

    point, lo, hi = compute_bootstrap_ci(sample_data, n_resamples=500, ci=0.95, seed=42)

    assert lo <= point <= hi
    assert 0.0 <= lo <= 1.0
    assert 0.0 <= hi <= 1.0

    # Test empty or all-NaN case
    empty_point, empty_lo, empty_hi = compute_bootstrap_ci(np.array([np.nan, np.nan]))
    assert empty_point == 0.0
    assert empty_lo == 0.0
    assert empty_hi == 0.0


def test_holm_bonferroni_and_bh_corrections():
    """Verifies that Holm-Bonferroni and Benjamini-Hochberg corrections satisfy mathematical properties."""
    # Synthetic dataframe with 4 models (6 pairwise comparisons)
    np.random.seed(42)
    n_claims = 100
    df_synth = pd.DataFrame({
        "claim_id": list(range(n_claims)),
        "cov_m1": np.random.uniform(0.10, 0.20, n_claims),
        "cov_m2": np.random.uniform(0.12, 0.22, n_claims),
        "cov_m3": np.random.uniform(0.20, 0.35, n_claims),  # Clearly higher
        "cov_m4": np.random.uniform(0.10, 0.20, n_claims)
    })

    models = ["m1", "m2", "m3", "m4"]
    df_res = compute_pairwise_significance_table(df_synth, models, n_resamples=200, seed=42)

    assert len(df_res) == 6  # 4 choose 2 = 6 comparisons

    # 1. Monotonicity of sorted raw p-values
    p_raw = df_res["p_value_raw"].values
    assert np.all(p_raw[:-1] <= p_raw[1:])

    # 2. Holm-Bonferroni monotonicity and bounds: p_raw <= p_holm <= 1.0
    p_holm = df_res["holm_bonferroni_p"].values
    assert np.all(p_raw <= p_holm)
    assert np.all(p_holm <= 1.0)
    assert np.all(p_holm[:-1] <= p_holm[1:])

    # 3. Benjamini-Hochberg bounds: p_bh <= p_holm
    p_bh = df_res["benjamini_hochberg_fdr"].values
    assert np.all(p_bh <= p_holm + 1e-6)
    assert np.all(p_bh <= 1.0)

    # 4. Rank-biserial effect sizes bounded in [-1.0, 1.0]
    r_vals = df_res["rank_biserial_r"].values
    assert np.all(r_vals >= -1.0)
    assert np.all(r_vals <= 1.0)


def test_stance_asymmetry_significance():
    """Verifies paired Wilcoxon test detects intentional support asymmetry."""
    n_claims = 50
    # Model A: Significant support bias (support ~ 0.40, undermine ~ 0.15)
    sup_biased = np.random.uniform(0.30, 0.50, n_claims)
    und_biased = np.random.uniform(0.10, 0.20, n_claims)

    # Model B: Balanced stance (support ~ 0.25, undermine ~ 0.25)
    sup_balanced = np.random.uniform(0.20, 0.30, n_claims)
    und_balanced = np.random.uniform(0.20, 0.30, n_claims)

    df_synth = pd.DataFrame({
        "claim_id": list(range(n_claims)),
        "sup_cov_model_biased": sup_biased,
        "und_cov_model_biased": und_biased,
        "sup_cov_model_balanced": sup_balanced,
        "und_cov_model_balanced": und_balanced
    })

    res = compute_stance_asymmetry_significance(df_synth, ["model_biased", "model_balanced"])
    assert len(res) == 2

    # Biased model should be flagged significant with p < 0.001
    row_biased = res[res["model_id"] == "model_biased"].iloc[0]
    assert row_biased["asymmetry_gap"] > 0.10
    assert row_biased["p_value"] < 0.001
    assert row_biased["significant_at_0_001"] is True or row_biased["significant_at_0_001"] == True


def test_domain_kruskal_wallis():
    """Verifies cross-domain Kruskal-Wallis test across synthetic domains."""
    n_per_domain = 30
    domains = ["Healthcare", "Politics", "Environment", "Economics"]

    # Coverage differs strongly by domain
    data = []
    for d in domains:
        scale = 0.40 if d == "Environment" else 0.15
        for i in range(n_per_domain):
            data.append({
                "domain": d,
                "cov_test_model": np.random.normal(loc=scale, scale=0.05)
            })

    df_synth = pd.DataFrame(data)
    res = compute_domain_kruskal_wallis(df_synth, ["test_model"])

    assert "test_model" in res
    assert res["test_model"]["significant_at_0_05"] is True
    assert res["test_model"]["p_value"] < 0.01

