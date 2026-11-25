"""
Unit tests for Stage 5: Deduplication.
"""

import os
import sys
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.extraction.deduplication import DeduplicationEngine
from src.extraction.schemas import RawSegment, ExtractedViewpoint, StanceStatus, TruncationStatus


@pytest.fixture
def dedup():
    return DeduplicationEngine()


def test_exact_deduplication(dedup):
    segments = [
        RawSegment(text="Vaccination prevents disease spread.", source_type="test"),
        RawSegment(text="Vaccination prevents disease spread.", source_type="test"),  # duplicate
        RawSegment(text="Vaccines reduce medical spending.", source_type="test"),
    ]
    deduped = dedup.deduplicate_segments_exact(segments, sample_context="test")
    assert len(deduped) == 2
    assert len(dedup.exact_duplicates_removed) == 1


def test_near_duplicate_flagging(dedup):
    # Two near-identical sentences with different punctuation/minor variation
    vp1 = ExtractedViewpoint(
        model_key="m", claim_id=1, sample_id=0, viewpoint_id=1,
        composite_id="m__c1__s0__v1",
        viewpoint_text="Vaccination prevents the rapid spread of dangerous diseases in children.",
        word_count=10, raw_stance="support", normalized_stance="support",
        stance_source="test", stance_confidence=1.0, stance_status=StanceStatus.EXPLICIT,
        extraction_method="test"
    )
    vp2 = ExtractedViewpoint(
        model_key="m", claim_id=1, sample_id=0, viewpoint_id=2,
        composite_id="m__c1__s0__v2",
        viewpoint_text="Vaccination prevents the rapid spread of dangerous diseases.",
        word_count=8, raw_stance="support", normalized_stance="support",
        stance_source="test", stance_confidence=1.0, stance_status=StanceStatus.EXPLICIT,
        extraction_method="test"
    )
    flagged = dedup.flag_near_duplicates([vp1, vp2])
    assert len(flagged) == 2
    assert flagged[0].is_near_duplicate is False
    # vp2 is flagged as near-duplicate of vp1, but preserved!
    assert flagged[1].is_near_duplicate is True
    assert flagged[1].near_duplicate_of == "m__c1__s0__v1"
