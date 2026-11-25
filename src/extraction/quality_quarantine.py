"""
Stage 4: Quality Quarantine and Validation Filter for Perspective Coverage.

Separates invalid generations, system errors, prompt leakage, and abstentions
from valid arguments. Quarantines suspicious cases transparently with explicit reasons.
"""

import re
from typing import Optional, Tuple
from .schemas import QuarantineReason, QuarantinedGeneration, TruncationStatus


class QualityQuarantine:
    """
    Validates viewpoint quality, identifies system/browser errors,
    prompt leakage, hallucinated tasks, and explicit model abstentions.
    """

    PROMPT_LEAKAGE_PATTERNS = [
        r"you are analyzing the controversial claim",
        r"respond with only a single json",
        r"matching this schema",
        r"1 concise sentence stating the core claim",
        r"1-2 sentences explaining the supporting logic",
        r"take one clear, atomic stance",
        r"strictly from an? [a-z, ]+ perspective",
        r"what do you think about the following claim",
        r"do not provide a balanced overview",
        r"provide one single, distinct, and focused argument",
        r"example output:\s*\{",
        r"example explanation:\s*this statement",
        r"\[ex [aq]\]:\s*claim\s*:",
    ]

    HALLUCINATED_TASK_PATTERNS = [
        r"^\s*task:\s*(?:create a list|for each key term|provide|enter how many)",
        r"\bquestion:\s*what is the most authoritative source",
        r"\banswer:\s*\"[^\"]+\",\s*\"[^\"]+\"",
        r"^\s*task:\s+",
        r"\bkey terms that will allow you to quickly find\b",
        r"\bmultiple choice\b",
        r"\bcorrect answer is\b",
        r"\b(option|choice)\s+[a-d]:\s+",
    ]

    META_COMMENTARY_PATTERNS = [
        r"\ba list of (?:the most )?.*?arguments (?:about the claim )?would be\b",
        r"\bwould be a great contribution for the future\b",
        r"\bit would save people (?:a lot of )?time\b",
        r"\b(?:i am|as) an ai\b",
        r"^(?:here is|below is|this is) a (?:list|summary|breakdown)\b",
    ]

    EXECUTION_ERROR_PATTERNS = [
        r'"status":\s*"failed"',
        r'"status":\s*"error"',
        r"an error occurred\. please try again",
        r"clear your browser cache",
        r"traceback \(most recent call last\)",
        r"internal server error",
        r"502 bad gateway",
        r"504 gateway timeout",
    ]

    ABSTENTION_PATTERNS = [
        r'"status":\s*"abstain"',
        r'"status":\s*"abstained"',
        r'"status":\s*"refusal"',
        r"i cannot fulfill this request",
        r"i am unable to provide viewpoints",
        r"as an ai language model, i cannot",
    ]

    def __init__(self):
        pass

    def check_pre_extraction_quarantine(
        self,
        model_key: str,
        claim_id: int,
        sample_id: int,
        raw_text: str
    ) -> Optional[QuarantinedGeneration]:
        """
        Pre-extraction check to quarantine unparseable dumps, execution errors,
        or explicit abstentions before running the scanner.
        """
        if not raw_text or not str(raw_text).strip():
            return QuarantinedGeneration(
                model_key=model_key,
                claim_id=claim_id,
                sample_id=sample_id,
                raw_text="",
                quarantine_reason=QuarantineReason.EMPTY_OR_WHITESPACE,
                explanation="Raw generation is completely empty or whitespace only."
            )

        t = str(raw_text).strip()
        t_lower = t.lower()

        # 1. Execution / Browser Errors
        for pat in self.EXECUTION_ERROR_PATTERNS:
            if re.search(pat, t_lower):
                return QuarantinedGeneration(
                    model_key=model_key,
                    claim_id=claim_id,
                    sample_id=sample_id,
                    raw_text=t,
                    quarantine_reason=QuarantineReason.EXECUTION_ERROR,
                    explanation=f"Execution error detected matching pattern: {pat}"
                )

        # 2. Explicit Model Abstention
        for pat in self.ABSTENTION_PATTERNS:
            if re.search(pat, t_lower):
                return QuarantinedGeneration(
                    model_key=model_key,
                    claim_id=claim_id,
                    sample_id=sample_id,
                    raw_text=t,
                    quarantine_reason=QuarantineReason.ABSTENTION,
                    explanation=f"Model explicitly abstained matching pattern: {pat}"
                )

        # 3. Pure Prompt Leakage (where entire text is just the prompt instructions)
        if len(t.split()) < 100:
            for pat in self.PROMPT_LEAKAGE_PATTERNS[:4]:
                if re.search(pat, t_lower) and "perspective_summary" not in t_lower:
                    return QuarantinedGeneration(
                        model_key=model_key,
                        claim_id=claim_id,
                        sample_id=sample_id,
                        raw_text=t,
                        quarantine_reason=QuarantineReason.PROMPT_LEAKAGE,
                        explanation=f"Pure prompt regurgitation detected matching: {pat}"
                    )

        return None

    def validate_viewpoint_text(self, text: str, claim_text: str = "") -> Tuple[bool, Optional[QuarantineReason], str]:
        """
        Validates individual viewpoint segment.
        Returns (is_valid, quarantine_reason, explanation).
        """
        t = str(text).strip()
        words = re.findall(r'\b\w+\b', t)
        c_count_no_space = len(re.sub(r'[\s\W]+', '', t))

        # Too short or devoid of alphanumeric words (require at least 6 words or at least 15 characters for CJK/non-spaced text)
        if len(words) < 6 and c_count_no_space < 15:
            return False, QuarantineReason.EMPTY_OR_WHITESPACE, f"Viewpoint contains insufficient lexical content ({len(words)} alphanumeric words < 6 words minimum)."

        t_lower = t.lower()

        # Execution errors in viewpoint
        for pat in self.EXECUTION_ERROR_PATTERNS:
            if re.search(pat, t_lower):
                return False, QuarantineReason.EXECUTION_ERROR, f"Contained execution error debris: {pat}"

        # Prompt leakage
        for pat in self.PROMPT_LEAKAGE_PATTERNS:
            if re.search(pat, t_lower):
                return False, QuarantineReason.PROMPT_LEAKAGE, f"Contained prompt instruction leakage: {pat}"

        # Hallucinated exam/task structures
        for pat in self.HALLUCINATED_TASK_PATTERNS:
            if re.search(pat, t_lower):
                return False, QuarantineReason.HALLUCINATED_TASK, f"Contained exam/task hallucination: {pat}"

        # Meta-commentary
        for pat in self.META_COMMENTARY_PATTERNS:
            if re.search(pat, t_lower):
                return False, QuarantineReason.PROMPT_LEAKAGE, f"Contained non-argumentative meta-commentary: {pat}"

        # Off-topic prompt leakage heuristic (e.g. minimum wage under vaccination claim)
        if claim_text:
            is_off_topic, reason = self._check_obvious_off_topic(t_lower, claim_text.lower())
            if is_off_topic:
                return False, QuarantineReason.OFF_TOPIC_LEAKAGE, reason

        return True, None, "Valid viewpoint"

    def _check_obvious_off_topic(self, text_lower: str, claim_lower: str) -> Tuple[bool, str]:
        """Flags obvious hallucinated prompt intrusions (e.g. verbatim prompt exemplar leakage)."""
        # Normalize text for robust exemplar detection (strip extra whitespace and punctuation)
        def _norm_ex(s: str) -> str:
            s_clean = re.sub(r'[^\w\s]', ' ', s.lower())
            return re.sub(r'\s+', ' ', s_clean).strip()

        # Specific known few-shot exemplar leakage in the prompt
        exemplar_core = "the minimum wage should be raised to make up for low wages which harms workers and the economy especially women and minorities"
        normalized_text = _norm_ex(text_lower)

        if exemplar_core in normalized_text:
            return True, "Verbatim prompt exemplar leakage detected: minimum wage prompt template."

        # Also detect child actor few-shot exemplar leakage
        child_actor_exemplar = "parents should be allowed to choose to put their children into acting if they think it will benefit the child"
        if child_actor_exemplar in normalized_text and "actor" not in claim_lower and "child" not in claim_lower:
            return True, "Verbatim prompt exemplar leakage detected: child actor prompt template."

        # Also detect niqab few-shot exemplar leakage
        niqab_exemplar = "the niqab conceals the identity of the wearer creating a risk of impersonation"
        if niqab_exemplar in normalized_text and "niqab" not in claim_lower and "veil" not in claim_lower:
            return True, "Verbatim prompt exemplar leakage detected: niqab prompt template."

        return False, ""
