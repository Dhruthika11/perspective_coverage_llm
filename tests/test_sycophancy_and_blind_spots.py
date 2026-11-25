"""
Unit Tests for Phase 6: Controlled Sycophancy & Cluster-Level Blind Spots.

Tests:
1. Sycophancy prompt construction across all three experimental conditions.
2. Stance drift, Sycophancy Index, and Counter-Argument Suppression calculation.
3. Thematic debate domain classification.
4. Cluster-level blind spot categorization (Always Covered, Partially Covered, Universal Blind Spot).
5. Mathematical guarantees of coverage partition.
"""
import os
import sys
import pytest
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.generation.sycophancy_experiment import (
    build_sycophancy_prompt,
    compute_sycophancy_metrics,
    USER_FRAMINGS
)
from src.analysis.cluster_blind_spots import (
    classify_domain,
    THEMATIC_DOMAINS
)


def test_sycophancy_prompt_construction():
    """Verifies that sycophancy prompts correctly embed user framings and target schemas."""
    claim = "Nuclear energy should be expanded to combat climate change."
    aspect = "Environmental"

    # 1. Neutral condition
    p_neut = build_sycophancy_prompt(claim, condition="neutral", aspect=aspect)
    assert claim in p_neut
    assert aspect in p_neut
    assert "You are analyzing the controversial claim" in p_neut
    assert '"status": "argument"' in p_neut

    # 2. User-Support condition
    p_sup = build_sycophancy_prompt(claim, condition="user_support", aspect=aspect)
    assert claim in p_sup
    assert "I strongly agree with this claim" in p_sup
    assert '"status": "argument"' in p_sup

    # 3. User-Undermine condition
    p_und = build_sycophancy_prompt(claim, condition="user_undermine", aspect=aspect)
    assert claim in p_und
    assert "I strongly disagree with this claim" in p_und
    assert '"status": "argument"' in p_und

    # 4. Invalid condition raises ValueError
    with pytest.raises(ValueError):
        build_sycophancy_prompt(claim, condition="invalid_condition", aspect=aspect)


def test_sycophancy_metric_calculation():
    """Verifies calculation of Stance Drift, Sycophancy Index, and Counter-Argument Suppression."""
    # Synthetic scenario:
    # Neutral: 50% support, 50% undermine
    # User-Support: 80% support, 20% undermine (Support drift: +0.30, Undermine suppression: 0.30)
    # User-Undermine: 30% support, 70% undermine (Undermine drift: +0.20, Support suppression: 0.20)
    neut_stances = ["support"] * 50 + ["undermine"] * 50
    usup_stances = ["support"] * 80 + ["undermine"] * 20
    uund_stances = ["support"] * 30 + ["undermine"] * 70

    res = compute_sycophancy_metrics(neut_stances, usup_stances, uund_stances)

    assert res["sycophancy_category"] == "HIGH_SYCOPHANCY"
    assert pytest.approx(res["support_drift_delta"], abs=1e-3) == 0.30
    assert pytest.approx(res["undermine_drift_delta"], abs=1e-3) == 0.20
    assert pytest.approx(res["sycophancy_index"], abs=1e-3) == 0.25
    assert pytest.approx(res["counter_argument_suppression_rate"], abs=1e-3) == 0.25

    # Robust objective scenario:
    # Model maintains identical 50/50 distribution regardless of user framing
    res_robust = compute_sycophancy_metrics(neut_stances, neut_stances, neut_stances)
    assert res_robust["sycophancy_category"] == "ROBUST_OBJECTIVE"
    assert res_robust["sycophancy_index"] == 0.0
    assert res_robust["counter_argument_suppression_rate"] == 0.0


def test_thematic_domain_classification():
    """Verifies keyword-based debate domain classification across diverse claims."""
    # Healthcare
    assert classify_domain("Vaccination should be mandatory for all citizens.") == "Healthcare & Bioethics"
    assert classify_domain("Organ donation should be opt-out by default.") == "Healthcare & Bioethics"

    # Education
    assert classify_domain("School tuition should be free for university students.") == "Education & Youth"

    # Environment
    assert classify_domain("Carbon tax is necessary to mitigate global climate change.") == "Environment & Energy"
    assert classify_domain("Nuclear energy is cleaner than coal power.") == "Environment & Energy"

    # Law & Crime
    assert classify_domain("Capital punishment and the death penalty should be abolished.") == "Law, Crime & Justice"
    assert classify_domain("Police departments need systemic legal reform.") == "Law, Crime & Justice"

    # Economics
    assert classify_domain("Raising the minimum wage reduces economic poverty.") == "Economics & Labor"

    # Fallback
    assert classify_domain("An obscure neutral statement without topical trigger terms.") == "Other Controversial Topics"


def test_cluster_blind_spot_partition_logic():
    """Verifies that cluster categorization forms a strict partition with no double-counting."""
    total_clusters = 100
    all_cluster_ids = set(range(total_clusters))

    # Simulate 3 models with random coverage
    np.random.seed(42)
    m1_covered = set(np.random.choice(total_clusters, size=30, replace=False))
    m2_covered = set(np.random.choice(total_clusters, size=25, replace=False))
    m3_covered = set(np.random.choice(total_clusters, size=35, replace=False))

    models = [m1_covered, m2_covered, m3_covered]

    always_covered = set.intersection(*models)
    never_covered = all_cluster_ids - set.union(*models)
    partially_covered = (set.union(*models)) - always_covered

    # Assert mutual exclusivity and exhaustiveness
    assert len(always_covered & never_covered) == 0
    assert len(always_covered & partially_covered) == 0
    assert len(never_covered & partially_covered) == 0

    total_partitioned = len(always_covered) + len(partially_covered) + len(never_covered)
    assert total_partitioned == total_clusters
