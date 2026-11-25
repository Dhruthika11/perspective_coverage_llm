"""
Stage 5: Conservative Deduplication Engine for Perspective Coverage.

Implements two-stage deduplication:
  1. Exact Deduplication: Automatically removes identical normalized texts within sample/claim.
  2. Near-Duplicate Detection: Flags candidate duplicates with similarity >= 0.85 for review
     WITHOUT automatically deleting them, preserving original arguments and logging decisions.
"""

import re
from typing import List, Tuple, Set, Dict, Any, Optional
from .schemas import RawSegment, ExtractedViewpoint


class DeduplicationEngine:
    """
    Handles exact removal and near-duplicate flagging with audit tracking.
    """

    def __init__(self, near_duplicate_threshold: float = 0.75):
        self.near_duplicate_threshold = near_duplicate_threshold
        self.exact_duplicates_removed: List[Dict[str, Any]] = []

    def normalize_text_for_comparison(self, text: str) -> str:
        """Normalizes text for exact string comparison: lowercase, remove punctuation, collapse space."""
        t = str(text).lower()
        t = re.sub(r'[^\w\s]', '', t)
        t = re.sub(r'\s+', ' ', t).strip()
        return t

    def deduplicate_segments_exact(self, segments: List[RawSegment], sample_context: str = "") -> List[RawSegment]:
        """
        Removes exact duplicates within a generation / sample.
        Preserves the first occurrence and logs removed items.
        """
        seen: Set[str] = set()
        unique_segments: List[RawSegment] = []

        for seg in segments:
            norm_key = self.normalize_text_for_comparison(seg.text)
            if not norm_key:
                continue

            if norm_key in seen:
                self.exact_duplicates_removed.append({
                    "context": sample_context,
                    "text": seg.text,
                    "normalized_key": norm_key,
                    "source_type": seg.source_type
                })
            else:
                seen.add(norm_key)
                unique_segments.append(seg)

        return unique_segments

    def flag_near_duplicates(self, viewpoints: List[ExtractedViewpoint]) -> List[ExtractedViewpoint]:
        """
        Flags candidate near-duplicates within a claim using word Jaccard similarity.
        Does NOT delete; sets is_near_duplicate=True and near_duplicate_of=target_id.
        """
        n = len(viewpoints)
        if n < 2:
            return viewpoints

        token_sets = [set(self.normalize_text_for_comparison(vp.viewpoint_text).split()) for vp in viewpoints]

        for i in range(n):
            if viewpoints[i].is_near_duplicate:
                continue

            for j in range(i + 1, n):
                if viewpoints[j].is_near_duplicate:
                    continue

                set_i = token_sets[i]
                set_j = token_sets[j]

                if not set_i or not set_j:
                    continue

                intersection = len(set_i.intersection(set_j))
                union = len(set_i.union(set_j))
                jaccard = intersection / union if union > 0 else 0.0

                if jaccard >= self.near_duplicate_threshold:
                    # Flag the later one as near-duplicate of the earlier one
                    viewpoints[j].is_near_duplicate = True
                    viewpoints[j].near_duplicate_of = viewpoints[i].composite_id

        return viewpoints
