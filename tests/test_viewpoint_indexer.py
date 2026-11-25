"""
Unit tests for Stage 6: Viewpoint Indexer.
"""

import os
import sys
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.extraction.viewpoint_indexer import ViewpointIndexer
from src.extraction.stance_normalizer import StanceNormalizer
from src.extraction.schemas import RawSegment


@pytest.fixture
def indexer():
    return ViewpointIndexer()


@pytest.fixture
def normalizer():
    return StanceNormalizer()


def test_sequential_indexing_and_uniqueness(indexer, normalizer):
    segments = [
        RawSegment(text="First perspective argument.", source_type="test", raw_stance="support"),
        RawSegment(text="Second perspective argument.", source_type="test", raw_stance="undermine"),
        RawSegment(text="Third perspective argument.", source_type="test", raw_stance=None),
    ]

    vps = indexer.index_sample_viewpoints(
        segments=segments,
        model_key="test_model",
        claim_id=499,
        sample_id=2,
        stance_normalizer_func=normalizer.normalize
    )

    assert len(vps) == 3
    assert [vp.viewpoint_id for vp in vps] == [1, 2, 3]
    assert vps[0].composite_id == "test_model__c499__s2__v1"
    assert vps[1].composite_id == "test_model__c499__s2__v2"
    assert vps[2].composite_id == "test_model__c499__s2__v3"

    # Verify stances
    assert vps[0].normalized_stance == "support"
    assert vps[1].normalized_stance == "undermine"
    assert vps[2].normalized_stance == "unknown"
