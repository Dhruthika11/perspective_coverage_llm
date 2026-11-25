"""
Stage 6: Viewpoint Indexer for Perspective Coverage.

Assigns strictly sequential, 1-based viewpoint IDs after all deduplication
and quarantine filtering has completed. Guarantees zero duplicate primary keys.
"""

from typing import List
from .schemas import ExtractedViewpoint, RawSegment, StanceStatus, TruncationStatus


class ViewpointIndexer:
    """
    Manages deterministic, sequential indexing of viewpoints per sample and claim.
    """

    def __init__(self):
        pass

    def index_sample_viewpoints(
        self,
        segments: List[RawSegment],
        model_key: str,
        claim_id: int,
        sample_id: int,
        stance_normalizer_func
    ) -> List[ExtractedViewpoint]:
        """
        Indexes a list of validated, deduplicated segments for a specific (model, claim, sample).
        Assigns strictly unique viewpoint_id (1, 2, ...) and composite_id.
        """
        viewpoints: List[ExtractedViewpoint] = []

        for v_idx, seg in enumerate(segments, start=1):
            norm_stance, stance_src, conf, st_status = stance_normalizer_func(
                seg.raw_stance,
                seg.stance_source,
                seg.text
            )

            composite_id = f"{model_key}__c{claim_id}__s{sample_id}__v{v_idx}"
            w_count = len(seg.text.split())

            char_offsets = (seg.start_char, seg.end_char) if (seg.start_char or seg.end_char) else None

            vp = ExtractedViewpoint(
                model_key=model_key,
                claim_id=claim_id,
                sample_id=sample_id,
                viewpoint_id=v_idx,
                composite_id=composite_id,
                viewpoint_text=seg.text,
                word_count=w_count,
                raw_stance=seg.raw_stance,
                normalized_stance=norm_stance,
                stance_source=stance_src,
                stance_confidence=conf,
                stance_status=st_status,
                extraction_method=seg.source_type,
                truncation_status=seg.truncation_status,
                is_near_duplicate=False,
                near_duplicate_of=None,
                character_offsets=char_offsets
            )
            viewpoints.append(vp)

        return viewpoints
