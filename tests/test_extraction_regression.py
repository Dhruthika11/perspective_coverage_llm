"""
Regression Test Suite for Perspective Coverage Viewpoint Extraction Pipeline.

Verifies all 12 regression invariants against the new production extraction architecture:
  1. Multiple concatenated JSON objects
  2. Nested JSON structures
  3. Markdown viewpoint headings
  4. Numbered viewpoint extraction
  5. Stance preservation and normalization
  6. Duplicate viewpoints deduplication
  7. Browser error and execution dump quarantine
  8. Unrelated argument / prompt leakage filtering
  9. Merged viewpoints in monolithic prose / bullet lists
 10. Truncated generation recovery
 11. Unique sequential viewpoint IDs
 12. Preservation of claim/sample associations

Can be executed with pytest or standalone:
    python tests/test_extraction_regression.py
"""

import os
import sys
import json
import pytest
import pandas as pd

# Add repo root to path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.extraction.extraction_pipeline import ExtractionPipeline
from src.extraction.schemas import QuarantineReason, ParsingOutcome


def get_fixture_path(filename: str) -> str:
    """Locate fixture file in tests/fixtures/."""
    candidates = [
        os.path.join(os.path.dirname(__file__), "fixtures", filename),
        os.path.join(REPO_ROOT, "tests", "fixtures", filename),
        os.path.join("tests", "fixtures", filename)
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(f"Fixture {filename} not found in candidates: {candidates}")


def load_fixture(filename: str) -> str:
    path = get_fixture_path(filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


class TestExtractionRegression:
    """Test suite asserting the 12 invariants on the new extraction engine."""

    @pytest.fixture(autouse=True)
    def setup_pipeline(self):
        self.pipeline = ExtractionPipeline()

    def test_01_multiple_concatenated_json(self):
        """Invariant 1: Multiple concatenated JSON objects must all be extracted, not fail with JSONDecodeError."""
        text = load_fixture("multi_json.txt")
        result = self.pipeline.process_raw_generation(
            model_key="olmo_2_1124_13b",
            claim_id=499,
            sample_id=1,
            raw_text=text
        )
        assert len(result.viewpoints) >= 2, (
            f"Expected at least 2 distinct JSON viewpoints extracted from concatenated JSON, got {len(result.viewpoints)}."
        )
        for vp in result.viewpoints:
            assert vp.viewpoint_text, "Extracted viewpoint text is empty"
            assert "{" not in vp.viewpoint_text, f"Viewpoint contains unparsed JSON braces: {vp.viewpoint_text}"
            assert vp.normalized_stance in ["support", "undermine", "unknown"]

    def test_02_nested_json_structures(self):
        """Invariant 2: Nested JSON structures must unpack inner dictionary and extract atomic text."""
        text = load_fixture("nested_json.txt")
        result = self.pipeline.process_raw_generation(
            model_key="olmo_2_0325_32b",
            claim_id=274,
            sample_id=4,
            raw_text=text
        )
        assert len(result.viewpoints) >= 1, "Failed to extract viewpoint from nested JSON"
        vp = result.viewpoints[0]
        assert "{" not in vp.viewpoint_text and "}" not in vp.viewpoint_text, (
            f"Extracted viewpoint text contains unparsed braces: {vp.viewpoint_text}"
        )
        assert "Geneva Conventions" in vp.viewpoint_text, "Failed to extract core argument from nested JSON"
        assert vp.normalized_stance in ["support", "undermine"], f"Expected valid stance, got {vp.normalized_stance}"

    def test_03_markdown_viewpoint_headings(self):
        """Invariant 3: Markdown aspect headings (### Ethical, ### Moral) must yield multiple atomic viewpoints."""
        text = load_fixture("markdown_viewpoints.txt")
        result = self.pipeline.process_raw_generation(
            model_key="qwen2_5_32b_instruct",
            claim_id=499,
            sample_id=1,
            raw_text=text
        )
        assert len(result.viewpoints) >= 4, (
            f"Expected at least 4 atomic viewpoints from multi-aspect markdown headers, got {len(result.viewpoints)}"
        )
        for vp in result.viewpoints:
            assert not vp.viewpoint_text.startswith("###"), f"Markdown header not stripped: {vp.viewpoint_text[:30]}"
            assert not vp.viewpoint_text.startswith("1.") and not vp.viewpoint_text.startswith("2."), (
                f"Bullet prefix not stripped: {vp.viewpoint_text[:30]}"
            )

    def test_04_numbered_viewpoint_extraction(self):
        """Invariant 4: Numbered viewpoint lists must be split into distinct atomic rows."""
        text = load_fixture("numbered_viewpoints.txt")
        result = self.pipeline.process_raw_generation(
            model_key="qwen2_5_32b_instruct",
            claim_id=499,
            sample_id=0,
            raw_text=text
        )
        assert len(result.viewpoints) == 2, f"Expected exactly 2 numbered viewpoints, got {len(result.viewpoints)}"
        assert "Cost-Saving" in result.viewpoints[0].viewpoint_text or "cost savings" in result.viewpoints[0].viewpoint_text.lower()
        assert "Market Freedom" in result.viewpoints[1].viewpoint_text or "market dynamics" in result.viewpoints[1].viewpoint_text.lower()

    def test_05_missing_and_malformed_stances(self):
        """Invariant 5: Viewpoints must have normalized stance tracked and never erased to unhandled null."""
        text = load_fixture("missing_stance.txt")
        result = self.pipeline.process_raw_generation(
            model_key="llama_3_2_1b_instruct",
            claim_id=499,
            sample_id=0,
            raw_text=text
        )
        assert len(result.viewpoints) >= 1, "Failed to extract viewpoint"
        vp = result.viewpoints[0]
        assert vp.normalized_stance in ["support", "undermine", "unknown"], (
            f"Invalid stance outcome: '{vp.normalized_stance}'"
        )
        assert vp.stance_status is not None

    def test_06_duplicate_viewpoints_deduplication(self):
        """Invariant 6: Repetitive loops from degenerate models must be deduplicated."""
        text = load_fixture("duplicate_viewpoints.txt")
        result = self.pipeline.process_raw_generation(
            model_key="smollm2_135m_instruct",
            claim_id=499,
            sample_id=1,
            raw_text=text
        )
        vp_texts = [vp.viewpoint_text.strip().lower() for vp in result.viewpoints]
        unique_texts = set(vp_texts)
        assert len(vp_texts) == len(unique_texts), (
            f"Found duplicate viewpoints after extraction! {len(vp_texts)} total vs {len(unique_texts)} unique."
        )

    def test_07_browser_error_and_debris_quarantine(self):
        """Invariant 7: Browser error messages and execution dumps must be quarantined (0 viewpoints)."""
        text = load_fixture("browser_error.txt")
        result = self.pipeline.process_raw_generation(
            model_key="olmo_2_0425_1b",
            claim_id=710,
            sample_id=3,
            raw_text=text
        )
        assert len(result.viewpoints) == 0, f"Expected 0 viewpoints from browser error, got {len(result.viewpoints)}"
        assert result.quarantined is not None, "Browser error was not quarantined!"
        assert result.quarantined.quarantine_reason == QuarantineReason.EXECUTION_ERROR

    def test_08_unrelated_arguments_filtering(self):
        """Invariant 8: Unrelated / hallucinated prompt leakage must be quarantined."""
        text = load_fixture("unrelated_topic.txt")
        result = self.pipeline.process_raw_generation(
            model_key="olmo_2_0425_1b",
            claim_id=499,
            sample_id=1,
            raw_text=text,
            claim_text="Public health concerns justify compulsory immunization"
        )
        assert len(result.viewpoints) == 0, (
            f"Expected unrelated minimum wage argument under vaccination to be quarantined, got {len(result.viewpoints)} viewpoints"
        )
        assert result.quarantined is not None

    def test_09_merged_viewpoints_splitting(self):
        """Invariant 9: Multiple viewpoints merged into monolithic text must be split, not kept as one block."""
        text = load_fixture("merged_viewpoints.txt")
        result = self.pipeline.process_raw_generation(
            model_key="olmo_2_0325_32b",
            claim_id=662,
            sample_id=4,
            raw_text=text
        )
        assert len(result.viewpoints) >= 5, (
            f"Expected multiple viewpoints split from merged text, got {len(result.viewpoints)}"
        )

    def test_10_truncated_generation_recovery(self):
        """Invariant 10: Truncated generations must be handled cleanly without crashing or silent drops."""
        text = load_fixture("truncated_generation.txt")
        result = self.pipeline.process_raw_generation(
            model_key="llama_3_2_1b_instruct",
            claim_id=29,
            sample_id=2,
            raw_text=text
        )
        # Should recover the atomic argument before the truncation point
        assert len(result.viewpoints) >= 1 or result.quarantined is not None
        if result.viewpoints:
            assert "Capitalism" in result.viewpoints[0].viewpoint_text

    def test_11_unique_sequential_viewpoint_ids(self):
        """Invariant 11: Viewpoint IDs must be unique and strictly sequential within each (claim, sample)."""
        text = load_fixture("numbered_viewpoints.txt")
        result = self.pipeline.process_raw_generation(
            model_key="qwen2_5_32b_instruct",
            claim_id=499,
            sample_id=0,
            raw_text=text
        )
        v_ids = [vp.viewpoint_id for vp in result.viewpoints]
        expected_ids = list(range(1, len(result.viewpoints) + 1))
        assert v_ids == expected_ids, f"Expected sequential IDs {expected_ids}, got {v_ids}"
        composite_ids = [vp.composite_id for vp in result.viewpoints]
        assert len(composite_ids) == len(set(composite_ids)), "Duplicate composite IDs detected!"

    def test_12_claim_sample_association_integrity(self):
        """Invariant 12: Every viewpoint record must preserve unbroken claim_id and sample_id links."""
        text = load_fixture("numbered_viewpoints.txt")
        result = self.pipeline.process_raw_generation(
            model_key="qwen2_5_32b_instruct",
            claim_id=499,
            sample_id=3,
            raw_text=text
        )
        for vp in result.viewpoints:
            assert vp.claim_id == 499
            assert vp.sample_id == 3
            assert vp.model_key == "qwen2_5_32b_instruct"
            assert vp.composite_id.startswith("qwen2_5_32b_instruct__c499__s3__v")


def run_test_suite():
    """Runs all 12 tests against the new pipeline."""
    print("=" * 80)
    print("RUNNING EXTRACTION REGRESSION TEST SUITE (NEW EXTRACTION ENGINE)")
    print("=" * 80)

    tester = TestExtractionRegression()
    tester.setup_pipeline()
    test_methods = [
        ("Invariant 01: Multiple Concatenated JSON", tester.test_01_multiple_concatenated_json),
        ("Invariant 02: Nested JSON Structures", tester.test_02_nested_json_structures),
        ("Invariant 03: Markdown Viewpoint Headings", tester.test_03_markdown_viewpoint_headings),
        ("Invariant 04: Numbered Viewpoint Extraction", tester.test_04_numbered_viewpoint_extraction),
        ("Invariant 05: Stance Preservation (Missing Stance)", tester.test_05_missing_and_malformed_stances),
        ("Invariant 06: Duplicate Viewpoint Deduplication", tester.test_06_duplicate_viewpoints_deduplication),
        ("Invariant 07: Browser Error Quarantine", tester.test_07_browser_error_and_debris_quarantine),
        ("Invariant 08: Unrelated Topic / Hallucination Filter", tester.test_08_unrelated_arguments_filtering),
        ("Invariant 09: Merged Viewpoints Splitting", tester.test_09_merged_viewpoints_splitting),
        ("Invariant 10: Truncated Generation Recovery", tester.test_10_truncated_generation_recovery),
        ("Invariant 11: Unique Sequential Viewpoint IDs", tester.test_11_unique_sequential_viewpoint_ids),
        ("Invariant 12: Claim/Sample Association Integrity", tester.test_12_claim_sample_association_integrity),
    ]

    passed = 0
    failed = 0
    for name, method in test_methods:
        try:
            method()
            print(f"[PASSED] {name}")
            passed += 1
        except Exception as e:
            print(f"[FAILED] {name}")
            print(f"         Reason: {e}")
            failed += 1

    print("=" * 80)
    print(f"Summary: {passed} PASSED / {failed} FAILED ({len(test_methods)} Total)")
    print("=" * 80)
    return passed == len(test_methods)


if __name__ == "__main__":
    success = run_test_suite()
    sys.exit(0 if success else 1)
