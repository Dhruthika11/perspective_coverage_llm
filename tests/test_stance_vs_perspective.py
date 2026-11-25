"""
Unit tests for stance vs. perspective equivalence decoupling.
"""
import pytest
from src.analysis.stance_vs_perspective import PerspectiveMatcher, StanceDetector


def test_perspective_matcher_decision_rules():
    matcher = PerspectiveMatcher()

    # Case 1: Same stance, high cosine, high verification -> EQUIVALENT (2)
    res = matcher.classify_perspective_pair(
        stance_a="support",
        stance_b="support",
        cosine_sim=0.82,
        reasoning_entailment_prob=0.91,
        tau_cosine=0.75,
        theta_verify=0.80
    )
    assert res == PerspectiveMatcher.EQUIVALENT

    # Case 2: Opposite stances -> DIFFERENT (0) even with high cosine
    res = matcher.classify_perspective_pair(
        stance_a="support",
        stance_b="undermine",
        cosine_sim=0.85,
        reasoning_entailment_prob=0.95,
        tau_cosine=0.75,
        theta_verify=0.80
    )
    assert res == PerspectiveMatcher.DIFFERENT

    # Case 3: Same stance, low cosine (< 0.75) -> DIFFERENT (0)
    res = matcher.classify_perspective_pair(
        stance_a="support",
        stance_b="support",
        cosine_sim=0.68,
        reasoning_entailment_prob=0.85,
        tau_cosine=0.75,
        theta_verify=0.80
    )
    assert res == PerspectiveMatcher.DIFFERENT

    # Case 4: Same stance, high cosine (0.78), but moderate verification (0.65) -> RELATED (1)
    res = matcher.classify_perspective_pair(
        stance_a="support",
        stance_b="support",
        cosine_sim=0.78,
        reasoning_entailment_prob=0.65,
        tau_cosine=0.75,
        theta_verify=0.80
    )
    assert res == PerspectiveMatcher.RELATED

