"""
Stage 3: Stance Normalizer for Perspective Coverage.

Standardizes stance labels into strictly 'support', 'undermine', or 'unknown'.
Preserves provenance (raw_stance, stance_source, stance_confidence, stance_status).
Enforces the rule: DO NOT infer stance from argument sentiment alone.
"""

import re
from typing import Optional, Tuple
from .schemas import StanceStatus


class StanceNormalizer:
    """
    Normalizes stances with strict provenance tracking and zero sentiment hallucination.
    """

    SUPPORT_SYNONYMS = {
        "support", "supports", "supporting", "supportive", "pro", "in favor", 
        "positive", "affirmative", "agree", "agreement", "yes", "for"
    }
    UNDERMINE_SYNONYMS = {
        "undermine", "undermines", "undermining", "oppose", "opposes", "opposing",
        "con", "against", "negative", "counter", "critical", "no", "rebuttal"
    }
    NEUTRAL_OR_ABSTAIN = {
        "neutral", "abstain", "abstained", "unclear", "balanced", "mixed", "n/a", "none"
    }

    def __init__(self):
        pass

    def normalize(
        self,
        raw_stance: Optional[str],
        source: str = "missing_or_ambiguous",
        text_context: str = ""
    ) -> Tuple[str, str, float, StanceStatus]:
        """
        Normalizes a raw stance into (normalized_stance, stance_source, confidence, status).
        Allowed normalized outcomes: 'support', 'undermine', 'unknown'.
        """
        if not raw_stance or not str(raw_stance).strip():
            # Check if there is an unambiguous textual label in the text prefix
            text_stance, text_source = self._check_unambiguous_text_prefix(text_context)
            if text_stance:
                return text_stance, text_source, 0.9, StanceStatus.INFERRED
            return "unknown", "missing_or_ambiguous", 0.0, StanceStatus.UNKNOWN

        clean_raw = str(raw_stance).strip().lower()
        clean_raw = re.sub(r'[^a-z\s_-]', '', clean_raw).strip()

        # 1. Direct match with support synonyms
        if clean_raw in self.SUPPORT_SYNONYMS or any(s == clean_raw for s in self.SUPPORT_SYNONYMS):
            confidence = 1.0 if source in ["json_explicit", "heading_explicit"] else 0.85
            status = StanceStatus.EXPLICIT if source in ["json_explicit", "heading_explicit"] else StanceStatus.INFERRED
            return "support", source, confidence, status

        # 2. Direct match with undermine synonyms
        if clean_raw in self.UNDERMINE_SYNONYMS or any(s == clean_raw for s in self.UNDERMINE_SYNONYMS):
            confidence = 1.0 if source in ["json_explicit", "heading_explicit"] else 0.85
            status = StanceStatus.EXPLICIT if source in ["json_explicit", "heading_explicit"] else StanceStatus.INFERRED
            return "undermine", source, confidence, status

        # 3. Check for neutral / abstain -> unknown
        if clean_raw in self.NEUTRAL_OR_ABSTAIN:
            return "unknown", source, 0.0, StanceStatus.UNKNOWN

        # 4. Prefix match within raw label (e.g., 'supportive viewpoint')
        if any(clean_raw.startswith(s) for s in ["support", "pro"]):
            return "support", source, 0.9, StanceStatus.INFERRED
        if any(clean_raw.startswith(u) for u in ["undermine", "oppose", "critic", "counter", "con"]):
            return "undermine", source, 0.9, StanceStatus.INFERRED

        # 5. Check if the text itself has an unambiguous explicit label
        text_stance, text_source = self._check_unambiguous_text_prefix(text_context)
        if text_stance:
            return text_stance, text_source, 0.9, StanceStatus.INFERRED

        # Never guess based on sentiment of the argument
        return "unknown", "missing_or_ambiguous", 0.0, StanceStatus.UNKNOWN

    def _check_unambiguous_text_prefix(self, text: str) -> Tuple[Optional[str], str]:
        """Checks for explicit unambiguous textual prefixes like 'Supportive Viewpoint:'."""
        if not text:
            return None, "missing_or_ambiguous"
        
        t_clean = text.strip().lower()
        if re.match(r'^(?:supportive|support)\s+viewpoint', t_clean):
            return "support", "text_label_unambiguous"
        if re.match(r'^(?:critical|undermining|opposing)\s+viewpoint', t_clean):
            return "undermine", "text_label_unambiguous"
        if t_clean.startswith("[support]") or t_clean.startswith("(support)"):
            return "support", "heading_explicit"
        if t_clean.startswith("[undermine]") or t_clean.startswith("(undermine)"):
            return "undermine", "heading_explicit"

        return None, "missing_or_ambiguous"
