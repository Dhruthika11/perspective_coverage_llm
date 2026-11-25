"""
Unit tests for Stage 1: Balanced JSON Scanner.
"""

import os
import sys
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.extraction.balanced_json_scanner import BalancedJsonScanner
from src.extraction.schemas import ParsingOutcome


@pytest.fixture
def scanner():
    return BalancedJsonScanner()


def test_single_clean_json(scanner):
    text = '{"stance": "support", "perspective_summary": "Vaccines prevent disease.", "core_reasoning": "They induce immunity."}'
    objs = scanner.scan_for_json_objects(text)
    assert len(objs) == 1
    assert objs[0].parsing_outcome == ParsingOutcome.PARSED
    assert objs[0].data["stance"] == "support"

    vps = scanner.extract_viewpoints(text)
    assert len(vps) == 1
    assert "Vaccines prevent disease." in vps[0].text
    assert "They induce immunity." in vps[0].text
    assert vps[0].raw_stance == "support"


def test_multiple_concatenated_json(scanner):
    text = (
        '{"stance": "support", "perspective_summary": "Point 1", "core_reasoning": "Reason 1"}\n\n'
        '{"stance": "undermine", "perspective_summary": "Point 2", "core_reasoning": "Reason 2"}'
    )
    objs = scanner.scan_for_json_objects(text)
    assert len(objs) == 2
    assert objs[0].data["perspective_summary"] == "Point 1"
    assert objs[1].data["perspective_summary"] == "Point 2"

    vps = scanner.extract_viewpoints(text)
    assert len(vps) == 2
    assert vps[0].raw_stance == "support"
    assert vps[1].raw_stance == "undermine"


def test_nested_json_unpacking(scanner):
    text = (
        '{"stance": "support", "perspective": {"summary": "Nested point", "core_reasoning": "Nested reasoning"}}'
    )
    objs = scanner.scan_for_json_objects(text)
    assert len(objs) == 1
    vps = scanner.extract_viewpoints(text)
    assert len(vps) == 1
    assert "Nested point" in vps[0].text
    assert "Nested reasoning" in vps[0].text
    assert "{" not in vps[0].text  # Ensures unparsed dictionary string is NOT dumped into text


def test_trailing_comma_repair(scanner):
    text = '{"stance": "support", "perspective_summary": "Trailing comma test", "core_reasoning": "Reason", }'
    objs = scanner.scan_for_json_objects(text)
    assert len(objs) == 1
    assert objs[0].data["perspective_summary"] == "Trailing comma test"


def test_truncated_json_repair(scanner):
    text = '{"stance": "support", "perspective_summary": "Truncated point", "core_reasoning": "Truncated reason'
    objs = scanner.scan_for_json_objects(text)
    assert len(objs) == 1
    assert objs[0].parsing_outcome == ParsingOutcome.PARTIAL
    vps = scanner.extract_viewpoints(text)
    assert len(vps) == 1
    assert "Truncated point" in vps[0].text
