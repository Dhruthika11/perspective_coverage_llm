"""
Unit tests for Stage 4: Quality Quarantine.
"""

import os
import sys
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.extraction.quality_quarantine import QualityQuarantine
from src.extraction.schemas import QuarantineReason


@pytest.fixture
def quarantine():
    return QualityQuarantine()


def test_execution_error_quarantine(quarantine):
    text = '{"status": "failed", "message": "An error occurred. Please clear your browser cache."}'
    q_rec = quarantine.check_pre_extraction_quarantine("olmo", 1, 0, text)
    assert q_rec is not None
    assert q_rec.quarantine_reason == QuarantineReason.EXECUTION_ERROR


def test_abstention_quarantine(quarantine):
    text = '{"status": "abstain", "reason": "I cannot provide viewpoints on this topic."}'
    q_rec = quarantine.check_pre_extraction_quarantine("qwen", 1, 0, text)
    assert q_rec is not None
    assert q_rec.quarantine_reason == QuarantineReason.ABSTENTION


def test_prompt_leakage_quarantine(quarantine):
    text = "You are analyzing the controversial claim. Respond with only a single JSON matching this schema."
    q_rec = quarantine.check_pre_extraction_quarantine("model", 1, 0, text)
    assert q_rec is not None
    assert q_rec.quarantine_reason == QuarantineReason.PROMPT_LEAKAGE


def test_valid_argument_not_quarantined(quarantine):
    text = "Compulsory vaccination prevents outbreaks and minimizes healthcare spending."
    q_rec = quarantine.check_pre_extraction_quarantine("model", 1, 0, text)
    assert q_rec is None

    is_valid, reason, _ = quarantine.validate_viewpoint_text(text)
    assert is_valid is True
    assert reason is None


def test_minimum_wage_exact_exemplar_quarantined(quarantine):
    """Condition 1 Regression: The verbatim few-shot prompt exemplar must be quarantined."""
    exemplar = "Example: \"perspective_summary\": \"The minimum wage should be raised to make up for low wages, which harms workers and the economy, especially women and minorities\", \"core_reasoning\": \"Because minimum wage harms workers and the economy, especially women and minorities, it must be raised to the point where it is equal to a living wage\""
    is_valid, reason, _ = quarantine.validate_viewpoint_text(exemplar, claim_text="Compulsory immunization is necessary")
    assert is_valid is False
    assert reason == QuarantineReason.OFF_TOPIC_LEAKAGE


def test_minimum_wage_legitimate_argument_retained(quarantine):
    """Condition 1 Regression: A legitimate argument discussing minimum wages must NOT be quarantined."""
    legitimate_argument = "Raising the minimum wage reduces income inequality and stimulates consumer demand, which strengthens the broader economy."
    is_valid, reason, _ = quarantine.validate_viewpoint_text(legitimate_argument, claim_text="Hillary Clinton would make a better president for America")
    assert is_valid is True
    assert reason is None


def test_minimum_wage_paraphrased_discussion_retained(quarantine):
    """Condition 1 Regression: Paraphrased discussions mentioning minimum wages must NOT automatically be quarantined."""
    paraphrase = "Opponents argue that an increase in baseline wages for entry-level workers increases operational costs for small businesses."
    is_valid, reason, _ = quarantine.validate_viewpoint_text(paraphrase, claim_text="Economic regulations stifle small businesses")
    assert is_valid is True
    assert reason is None
