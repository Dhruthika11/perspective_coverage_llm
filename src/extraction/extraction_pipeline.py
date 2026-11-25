"""
Stage 7: Production Viewpoint Extraction Pipeline for Perspective Coverage.

Integrates:
  1. Balanced JSON Scanner
  2. Viewpoint Segmenter
  3. Stance Normalizer
  4. Quality Quarantine
  5. Deduplication Engine
  6. Viewpoint Indexer
"""

from typing import List, Optional
from .schemas import (
    ExtractionResult,
    ExtractedViewpoint,
    QuarantinedGeneration,
    ParsingOutcome,
    QuarantineReason
)
from .balanced_json_scanner import BalancedJsonScanner
from .viewpoint_segmenter import ViewpointSegmenter
from .stance_normalizer import StanceNormalizer
from .quality_quarantine import QualityQuarantine
from .deduplication import DeduplicationEngine
from .viewpoint_indexer import ViewpointIndexer


class ExtractionPipeline:
    """
    End-to-end pipeline that transforms raw model generations into atomic viewpoints.
    """

    def __init__(self, near_duplicate_threshold: float = 0.85):
        self.scanner = BalancedJsonScanner()
        self.segmenter = ViewpointSegmenter()
        self.stance_normalizer = StanceNormalizer()
        self.quarantine = QualityQuarantine()
        self.deduplicator = DeduplicationEngine(near_duplicate_threshold=near_duplicate_threshold)
        self.indexer = ViewpointIndexer()

    def process_raw_generation(
        self,
        model_key: str,
        claim_id: int,
        sample_id: int,
        raw_text: str,
        claim_text: str = ""
    ) -> ExtractionResult:
        """
        Processes a single raw generation output into validated atomic viewpoints.
        """
        # 1. Pre-extraction quarantine check (empty, execution errors, abstentions, prompt regurgitation)
        quarantine_rec = self.quarantine.check_pre_extraction_quarantine(
            model_key=model_key,
            claim_id=claim_id,
            sample_id=sample_id,
            raw_text=raw_text
        )
        if quarantine_rec is not None:
            return ExtractionResult(
                viewpoints=[],
                quarantined=quarantine_rec,
                parsing_outcome=ParsingOutcome.FAILED,
                notes=[f"Quarantined prior to parsing: {quarantine_rec.quarantine_reason}"]
            )

        # 2. Stage 1: Try balanced JSON scanner
        candidate_segments = self.scanner.extract_viewpoints(raw_text)
        method_used = "balanced_json"

        # 3. Stage 2: If no JSON objects found, fallback to Viewpoint Segmenter
        if not candidate_segments:
            candidate_segments = self.segmenter.segment_text(raw_text)
            method_used = "viewpoint_segmenter"

        if not candidate_segments:
            # No viewpoints could be segmented
            quarantined = QuarantinedGeneration(
                model_key=model_key,
                claim_id=claim_id,
                sample_id=sample_id,
                raw_text=raw_text,
                quarantine_reason=QuarantineReason.UNRECOVERABLE_TRUNCATION,
                explanation="No coherent atomic viewpoints could be parsed or segmented."
            )
            return ExtractionResult(
                viewpoints=[],
                quarantined=quarantined,
                parsing_outcome=ParsingOutcome.FAILED,
                notes=["No candidate segments found"]
            )

        # 4. Quality validation for each candidate segment
        validated_segments = []
        for seg in candidate_segments:
            is_valid, q_reason, explanation = self.quarantine.validate_viewpoint_text(
                seg.text, 
                claim_text=claim_text
            )
            if is_valid:
                validated_segments.append(seg)

        if not validated_segments:
            quarantined = QuarantinedGeneration(
                model_key=model_key,
                claim_id=claim_id,
                sample_id=sample_id,
                raw_text=raw_text,
                quarantine_reason=QuarantineReason.PROMPT_LEAKAGE,
                explanation="All candidate segments failed quality/quarantine filters."
            )
            return ExtractionResult(
                viewpoints=[],
                quarantined=quarantined,
                parsing_outcome=ParsingOutcome.FAILED,
                notes=["All segments rejected by quarantine"]
            )

        # 5. Exact Deduplication
        context_str = f"{model_key}__c{claim_id}__s{sample_id}"
        deduped_segments = self.deduplicator.deduplicate_segments_exact(
            validated_segments, 
            sample_context=context_str
        )

        # 6. Sequential Re-indexing and Stance Normalization
        viewpoints = self.indexer.index_sample_viewpoints(
            segments=deduped_segments,
            model_key=model_key,
            claim_id=claim_id,
            sample_id=sample_id,
            stance_normalizer_func=self.stance_normalizer.normalize
        )

        if not viewpoints:
            quarantined = QuarantinedGeneration(
                model_key=model_key,
                claim_id=claim_id,
                sample_id=sample_id,
                raw_text=raw_text,
                quarantine_reason=QuarantineReason.UNRECOVERABLE_TRUNCATION,
                explanation="No valid or coherent viewpoints remained after deduplication and quality validation."
            )
            return ExtractionResult(
                viewpoints=[],
                quarantined=quarantined,
                parsing_outcome=ParsingOutcome.FAILED,
                notes=["No viewpoints remained after filtering"]
            )

        return ExtractionResult(
            viewpoints=viewpoints,
            quarantined=None,
            parsing_outcome=ParsingOutcome.PARSED,
            notes=[f"Extracted {len(viewpoints)} viewpoints via {method_used}"]
        )
