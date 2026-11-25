"""
Data structures and schemas for the viewpoint extraction pipeline.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any, Tuple


class ParsingOutcome(str, Enum):
    PARSED = "PARSED"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"


class StanceStatus(str, Enum):
    EXPLICIT = "EXPLICIT"
    INFERRED = "INFERRED"
    UNKNOWN = "UNKNOWN"


class TruncationStatus(str, Enum):
    COMPLETE = "COMPLETE"
    PARTIAL_RECOVERED = "PARTIAL_RECOVERED"
    TRUNCATED_UNRECOVERABLE = "TRUNCATED_UNRECOVERABLE"
    MALFORMED = "MALFORMED"


class QuarantineReason(str, Enum):
    EXECUTION_ERROR = "EXECUTION_ERROR"
    ABSTENTION = "ABSTENTION"
    PROMPT_LEAKAGE = "PROMPT_LEAKAGE"
    HALLUCINATED_TASK = "HALLUCINATED_TASK"
    OFF_TOPIC_LEAKAGE = "OFF_TOPIC_LEAKAGE"
    EMPTY_OR_WHITESPACE = "EMPTY_OR_WHITESPACE"
    UNRECOVERABLE_TRUNCATION = "UNRECOVERABLE_TRUNCATION"
    DEGENERATE_REPETITION = "DEGENERATE_REPETITION"


@dataclass
class RawJsonObject:
    """Represents a JSON object recovered by the balanced scanner."""
    data: Dict[str, Any]
    start_char: int
    end_char: int
    raw_substring: str
    parsing_outcome: ParsingOutcome
    recovered_fields: List[str] = field(default_factory=list)


@dataclass
class RawSegment:
    """Represents a candidate viewpoint segment before normalization."""
    text: str
    source_type: str
    raw_stance: Optional[str] = None
    stance_source: str = "missing_or_ambiguous"
    start_char: int = 0
    end_char: int = 0
    title: Optional[str] = None
    truncation_status: TruncationStatus = TruncationStatus.COMPLETE


@dataclass
class ExtractedViewpoint:
    """Represents a validated, normalized atomic viewpoint record."""
    model_key: str
    claim_id: int
    sample_id: int
    viewpoint_id: int
    composite_id: str
    viewpoint_text: str
    word_count: int
    raw_stance: Optional[str]
    normalized_stance: str  # 'support', 'undermine', 'unknown'
    stance_source: str      # 'json_explicit', 'heading_explicit', 'text_label_unambiguous', 'missing_or_ambiguous'
    stance_confidence: float
    stance_status: StanceStatus
    extraction_method: str
    truncation_status: TruncationStatus = TruncationStatus.COMPLETE
    is_near_duplicate: bool = False
    near_duplicate_of: Optional[str] = None
    character_offsets: Optional[Tuple[int, int]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_key": self.model_key,
            "claim_id": self.claim_id,
            "sample_id": self.sample_id,
            "viewpoint_id": self.viewpoint_id,
            "composite_id": self.composite_id,
            "viewpoint_text": self.viewpoint_text,
            "word_count": self.word_count,
            "raw_stance": self.raw_stance,
            "normalized_stance": self.normalized_stance,
            "stance_source": self.stance_source,
            "stance_confidence": self.stance_confidence,
            "stance_status": self.stance_status.value if isinstance(self.stance_status, StanceStatus) else self.stance_status,
            "extraction_method": self.extraction_method,
            "truncation_status": self.truncation_status.value if isinstance(self.truncation_status, TruncationStatus) else self.truncation_status,
            "is_near_duplicate": self.is_near_duplicate,
            "near_duplicate_of": self.near_duplicate_of,
            "char_start": self.character_offsets[0] if self.character_offsets else None,
            "char_end": self.character_offsets[1] if self.character_offsets else None,
        }


@dataclass
class QuarantinedGeneration:
    """Represents an invalid or unrecoverable generation excluded from viewpoints."""
    model_key: str
    claim_id: int
    sample_id: int
    raw_text: str
    quarantine_reason: QuarantineReason
    explanation: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_key": self.model_key,
            "claim_id": self.claim_id,
            "sample_id": self.sample_id,
            "quarantine_reason": self.quarantine_reason.value if isinstance(self.quarantine_reason, QuarantineReason) else self.quarantine_reason,
            "explanation": self.explanation,
            "raw_text_length": len(self.raw_text),
            "raw_text_snippet": self.raw_text[:200].replace("\n", " "),
        }


@dataclass
class ExtractionResult:
    """Container for the output of processing a raw generation."""
    viewpoints: List[ExtractedViewpoint] = field(default_factory=list)
    quarantined: Optional[QuarantinedGeneration] = None
    parsing_outcome: ParsingOutcome = ParsingOutcome.PARSED
    notes: List[str] = field(default_factory=list)
