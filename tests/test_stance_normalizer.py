"""
Unit tests for Stage 3: Stance Normalizer.
"""

import os
import sys
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.extraction.stance_normalizer import StanceNormalizer
from src.extraction.schemas import StanceStatus


@pytest.fixture
def normalizer():
    return StanceNormalizer()


def test_explicit_json_stances(normalizer):
    norm, src, conf, stat = normalizer.normalize("support", source="json_explicit")
    assert norm == "support"
    assert src == "json_explicit"
    assert conf == 1.0
    assert stat == StanceStatus.EXPLICIT

    norm2, src2, conf2, stat2 = normalizer.normalize("undermine", source="json_explicit")
    assert norm2 == "undermine"
    assert conf2 == 1.0
    assert stat2 == StanceStatus.EXPLICIT


def test_synonym_normalization(normalizer):
    norm, _, _, _ = normalizer.normalize("pro", source="json_explicit")
    assert norm == "support"

    norm2, _, _, _ = normalizer.normalize("against", source="json_explicit")
    assert norm2 == "undermine"

    norm3, _, _, _ = normalizer.normalize("critical", source="heading_explicit")
    assert norm3 == "undermine"


def test_unambiguous_prefix_detection(normalizer):
    norm, src, conf, stat = normalizer.normalize(
        None, 
        source="missing_or_ambiguous", 
        text_context="Supportive Viewpoint: Vaccines protect community health."
    )
    assert norm == "support"
    assert src == "text_label_unambiguous"
    assert stat == StanceStatus.INFERRED


def test_no_sentiment_inference_rule(normalizer):
    # Text contains negative sentiment words, but lacks explicit stance label -> must normalize to unknown
    text = "The policy results in disastrous economic collapse, rising poverty, and severe harms."
    norm, src, conf, stat = normalizer.normalize(None, source="missing_or_ambiguous", text_context=text)
    assert norm == "unknown"
    assert conf == 0.0
    assert stat == StanceStatus.UNKNOWN
