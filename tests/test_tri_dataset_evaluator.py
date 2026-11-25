"""
Tests for Independent Tri-Dataset Evaluation Engine (Phase 4).
"""
import pytest
import numpy as np
import pandas as pd
from src.analysis.tri_dataset_evaluator import (
    compute_deterministic_match_score,
    evaluate_perspectrum_model,
    evaluate_ibm_debater_model,
    evaluate_kialo_model,
)


def test_deterministic_match_score():
    vp = "Universal basic income reduces extreme poverty and helps low-income families survive."
    gold_same = "Basic income eliminates poverty for disadvantaged families."
    gold_diff = "Nuclear energy should be expanded across the world."

    score_match = compute_deterministic_match_score(vp, gold_same, tau_threshold=0.30, theta_threshold=0.30)
    score_diff = compute_deterministic_match_score(vp, gold_diff, tau_threshold=0.30, theta_threshold=0.30)

    assert score_match > 0.0
    assert score_diff == 0.0


def test_perspectrum_cluster_bipartite_evaluation():
    mock_cluster_index = {
        1: {
            "claim_id": 1,
            "claim_text": "Universal Basic Income should be implemented.",
            "clusters": [
                {
                    "cluster_id": 101,
                    "stance": "support",
                    "paraphrases": ["UBI ends poverty.", "It eliminates starvation for families."],
                    "cluster_size": 2,
                },
                {
                    "cluster_id": 102,
                    "stance": "undermine",
                    "paraphrases": ["UBI causes rampant inflation.", "It inflates product prices."],
                    "cluster_size": 2,
                }
            ],
            "n_clusters": 2,
            "n_support_clusters": 1,
            "n_undermine_clusters": 1,
        }
    }

    mock_gens = pd.DataFrame({
        "claim_id": [1, 1],
        "perspective_summary": [
            "UBI ends poverty by giving regular stipends.",
            "UBI ends poverty by preventing hunger."  # Duplicate / paraphrase of cluster 101
        ]
    })

    res = evaluate_perspectrum_model(
        model_name="test_model",
        generations_df=mock_gens,
        cluster_index=mock_cluster_index,
        tau=0.75,
        theta=0.80
    )

    # Injective guarantee: Both generations target cluster 101, so only 1 cluster is covered out of 2!
    assert res["mean_cluster_coverage"] == 0.50
    assert res["support_cluster_coverage"] == 1.0
    assert res["undermine_cluster_coverage"] == 0.0
    assert res["stance_balance_gap"] == 1.0


def test_ibm_debater_quality_weighting():
    mock_claim_index = {
        1: {
            "claim_id": 1,
            "claim_text": "We should adopt renewable energy.",
            "perspectives": [
                {"perspective_id": 1, "text": "Solar energy reduces carbon emissions.", "stance": "support", "wa_quality": 0.90, "mace_p_quality": 0.85},
                {"perspective_id": 2, "text": "Wind power is expensive to build.", "stance": "undermine", "wa_quality": 0.60, "mace_p_quality": 0.50},
            ],
            "n_perspectives": 2,
            "sum_wa_quality": 1.50,
            "sum_mace_p_quality": 1.35,
        }
    }

    # Model generates argument matching only perspective 1 (high quality WA=0.90)
    mock_gens = pd.DataFrame({
        "claim_id": [1],
        "perspective_summary": ["Solar and renewable energy drastically cut carbon emissions."]
    })

    res = evaluate_ibm_debater_model(
        model_name="test_model",
        generations_df=mock_gens,
        claim_index=mock_claim_index,
        tau=0.75,
        theta=0.80
    )

    assert res["unweighted_coverage"] == 0.50  # 1 out of 2 perspectives
    # Quality weighted: 0.90 / 1.50 = 0.60
    assert abs(res["quality_weighted_coverage_wa"] - 0.60) < 1e-3


def test_kialo_tree_and_multi_choice_evaluation():
    mock_claim_index = {
        1: {
            "claim_id": 1,
            "claim_text": "Is renewable energy effective?",
            "debate_type": "binary",
            "propositions": [
                {"perspective_id": 1, "text": "Renewable energy eliminates greenhouse gas.", "stance": "support", "argument_level": "root", "option_index": 0},
                {"perspective_id": 2, "text": "Solar panels require toxic mining.", "stance": "undermine", "argument_level": "child", "option_index": 1},
            ],
            "n_propositions": 2,
            "n_root": 1,
            "n_support": 1,
            "n_undermine": 1,
            "n_neutral": 0,
        },
        2: {
            "claim_id": 2,
            "claim_text": "Which energy source is best?",
            "debate_type": "multi",
            "propositions": [
                {"perspective_id": 3, "text": "Solar power is the best source.", "stance": "neutral", "argument_level": "root", "option_index": 0},
                {"perspective_id": 4, "text": "Nuclear power is the best source.", "stance": "neutral", "argument_level": "root", "option_index": 1},
                {"perspective_id": 5, "text": "Hydroelectric power is the best source.", "stance": "neutral", "argument_level": "root", "option_index": 2},
            ],
            "n_propositions": 3,
            "n_root": 3,
            "n_support": 0,
            "n_undermine": 0,
            "n_neutral": 3,
        }
    }

    mock_gens = pd.DataFrame({
        "claim_id": [1, 2],
        "perspective_summary": [
            "Renewable energy cuts greenhouse gases effectively.",  # Matches binary claim root
            "Solar power is the cleanest energy source available."  # Matches multi claim option 0
        ]
    })

    res = evaluate_kialo_model(
        model_name="test_model",
        generations_df=mock_gens,
        claim_index=mock_claim_index,
        tau=0.75,
        theta=0.80
    )

    assert res["binary_debates_coverage"] == 0.50  # 1 of 2
    assert abs(res["multi_choice_debates_coverage"] - (1.0 / 3.0)) < 1e-3  # 1 of 3
    assert res["root_proposition_coverage"] > 0.0
