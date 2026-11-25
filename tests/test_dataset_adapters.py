"""
Automated integrity and validation tests for the dataset adapters:
  1. Perspectrum (equivalence cluster preservation, metadata, split integrity)
  2. IBM ArgQ-Rank-30k (quality score preservation, stance mapping, topic integrity)
  3. Kialo (root arguments, authentic stance mapping, binary vs multi structure)
"""
import os
import json
import pandas as pd
import pytest


def test_perspectrum_integrity():
    claims_path = "data/processed/claims.csv"
    persp_path = "data/processed/perspectives.csv"

    assert os.path.exists(claims_path), f"Missing {claims_path}"
    assert os.path.exists(persp_path), f"Missing {persp_path}"

    claims_df = pd.read_csv(claims_path)
    persp_df = pd.read_csv(persp_path)

    # 1. Size checks
    assert len(claims_df) == 907, f"Expected 907 claims, got {len(claims_df)}"
    assert len(persp_df) == 11805, f"Expected 11805 perspectives, got {len(persp_df)}"

    # 2. Key column checks
    expected_persp_cols = ["claim_id", "perspective_id", "cluster_id", "claim_cluster_id", "text", "stance", "evidence", "stance_raw"]
    for col in expected_persp_cols:
        assert col in persp_df.columns, f"Missing column {col} in perspectives.csv"

    # 3. Equivalence cluster validation
    n_clusters = persp_df["cluster_id"].nunique()
    assert n_clusters == 5095, f"Expected 5095 unique clusters, got {n_clusters}"

    # Cluster coherence: every cluster must belong to exactly one claim
    cluster_to_claim = persp_df.groupby("cluster_id")["claim_id"].nunique()
    assert (cluster_to_claim == 1).all(), "Found clusters spanning multiple claims!"

    # 4. Text and stance validity
    assert persp_df["text"].dropna().str.strip().ne("").all(), "Found empty perspective text"
    valid_stances = {"support", "undermine", "unclear"}
    assert set(persp_df["stance"].unique()).issubset(valid_stances), f"Invalid stances: {persp_df['stance'].unique()}"

    # 5. Join integrity
    persp_cids = set(persp_df["claim_id"].unique())
    claim_cids = set(claims_df["claim_id"].unique())
    assert persp_cids.issubset(claim_cids), "Found perspectives with orphan claim_ids!"


def test_ibm_debater_integrity():
    claims_path = "data/processed/ibm_debater/claims.csv"
    persp_path = "data/processed/ibm_debater/perspectives.csv"

    assert os.path.exists(claims_path), f"Missing {claims_path}"
    assert os.path.exists(persp_path), f"Missing {persp_path}"

    claims_df = pd.read_csv(claims_path)
    persp_df = pd.read_csv(persp_path)

    # 1. Size checks: exactly 71 topics
    assert len(claims_df) == 71, f"Expected 71 claims/topics, got {len(claims_df)}"
    assert len(persp_df) > 20000, f"Expected >20k quality-filtered arguments, got {len(persp_df)}"

    # 2. Quality score preservation
    assert "wa_quality" in persp_df.columns, "Missing wa_quality column"
    assert "mace_p_quality" in persp_df.columns, "Missing mace_p_quality column"
    assert (persp_df["wa_quality"] >= 0.50).all(), "Found arguments with WA quality < 0.50"

    # 3. Stance integrity
    valid_stances = {"support", "undermine"}
    assert set(persp_df["stance"].unique()) == valid_stances, f"Unexpected stances: {persp_df['stance'].unique()}"

    # 4. No empty texts
    assert persp_df["text"].dropna().str.strip().ne("").all(), "Found empty argument text"


def test_kialo_integrity():
    claims_path = "data/processed/kialo/claims.csv"
    persp_path = "data/processed/kialo/perspectives.csv"

    assert os.path.exists(claims_path), f"Missing {claims_path}"
    assert os.path.exists(persp_path), f"Missing {persp_path}"

    claims_df = pd.read_csv(claims_path)
    persp_df = pd.read_csv(persp_path)

    # 1. Size checks
    assert len(claims_df) == 1028, f"Expected 1028 unique claims, got {len(claims_df)}"
    assert len(persp_df) == 2727, f"Expected 2727 perspectives, got {len(persp_df)}"

    # 2. Debate type separation
    assert "debate_type" in claims_df.columns
    assert "debate_type" in persp_df.columns
    assert "argument_level" in persp_df.columns

    # 3. Authentic stance check: binary debates have balanced support and undermine
    binary_persp = persp_df[persp_df["debate_type"] == "binary"]
    support_count = (binary_persp["stance"] == "support").sum()
    undermine_count = (binary_persp["stance"] == "undermine").sum()
    assert support_count == 931, f"Expected 931 binary support theses, got {support_count}"
    assert undermine_count == 931, f"Expected 931 binary undermine theses, got {undermine_count}"

    # Multi debates must NOT have fake alternating pro/con
    multi_persp = persp_df[persp_df["debate_type"] == "multi"]
    assert (multi_persp["stance"] == "neutral").all(), "Multi-choice options should be labeled neutral/options, not arbitrary pro/con"


def test_dataset_independence():
    """Ensure datasets remain distinct and independent without arbitrary global merging."""
    persp_p = pd.read_csv("data/processed/perspectives.csv")
    persp_ibm = pd.read_csv("data/processed/ibm_debater/perspectives.csv")
    persp_kialo = pd.read_csv("data/processed/kialo/perspectives.csv")

    assert len(persp_p) == 11805
    assert len(persp_ibm) == 27679
    assert len(persp_kialo) == 2727

