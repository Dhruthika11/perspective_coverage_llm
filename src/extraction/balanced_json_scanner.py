"""
Stage 1: Balanced JSON Scanner for Perspective Coverage.

Finds and parses multiple concatenated JSON objects, handles nested dictionaries
with schema-aware unpacking, tracks character offsets, and recovers partial objects.
"""

import json
import re
from typing import List, Dict, Any, Tuple, Optional
from .schemas import RawJsonObject, ParsingOutcome, RawSegment, TruncationStatus


class BalancedJsonScanner:
    """
    Depth-tracking balanced brace scanner for extracting JSON objects from LLM outputs.
    """

    # Schema key aliases
    SUMMARY_KEYS = [
        "perspective_summary", "summary", "perspective", "argument", 
        "claim", "viewpoint", "thesis", "core_argument", "point"
    ]
    REASONING_KEYS = [
        "core_reasoning", "reasoning", "justification", "core-reasoning",
        "explanation", "additional_info", "elaboration", "details", "evidence"
    ]
    STANCE_KEYS = [
        "stance", "llm_reported_stance", "perspective_stance", "type", "label"
    ]

    def __init__(self):
        pass

    def scan_for_json_objects(self, text: str) -> List[RawJsonObject]:
        """
        Scans text and returns all isolated, balanced JSON objects with character offsets.
        Supports concatenated objects: {...} {...}
        """
        if not text or not str(text).strip():
            return []

        raw_str = str(text)
        objects = []
        n = len(raw_str)
        i = 0

        while i < n:
            # Find the start of an object
            if raw_str[i] == '{':
                start_char = i
                depth = 0
                in_string = False
                escape = False
                j = i

                while j < n:
                    ch = raw_str[j]

                    if escape:
                        escape = False
                    elif ch == '\\' and in_string:
                        escape = True
                    elif ch == '"':
                        in_string = not in_string
                    elif not in_string:
                        if ch == '{':
                            depth += 1
                        elif ch == '}':
                            depth -= 1
                            if depth == 0:
                                # Found balanced JSON candidate
                                candidate = raw_str[start_char : j + 1]
                                parsed_obj, outcome, rec_fields = self._try_parse_candidate(candidate)
                                if parsed_obj is not None:
                                    objects.append(RawJsonObject(
                                        data=parsed_obj,
                                        start_char=start_char,
                                        end_char=j + 1,
                                        raw_substring=candidate,
                                        parsing_outcome=outcome,
                                        recovered_fields=rec_fields
                                    ))
                                i = j + 1
                                break
                    j += 1

                if depth > 0:
                    # Truncated JSON detected at end of string
                    candidate = raw_str[start_char:]
                    parsed_obj, outcome, rec_fields = self._try_repair_truncated(candidate, depth, in_string)
                    if parsed_obj is not None:
                        objects.append(RawJsonObject(
                            data=parsed_obj,
                            start_char=start_char,
                            end_char=n,
                            raw_substring=candidate,
                            parsing_outcome=outcome,
                            recovered_fields=rec_fields
                        ))
                    break
            i += 1

        return objects

    def _try_parse_candidate(self, candidate: str) -> Tuple[Optional[Dict[str, Any]], ParsingOutcome, List[str]]:
        """Attempts to parse a candidate JSON string, repairing common LLM formatting flaws."""
        # 1. Clean markdown code fences if wrapped inside
        cand = candidate.strip()
        
        # 2. Direct parse attempt
        try:
            data = json.loads(cand)
            if isinstance(data, dict):
                return data, ParsingOutcome.PARSED, list(data.keys())
        except json.JSONDecodeError:
            pass

        # 3. Clean trailing commas: ', }' -> '}' or ', ]' -> ']'
        cleaned = re.sub(r',\s*\}', '}', cand)
        cleaned = re.sub(r',\s*\]', ']', cleaned)
        try:
            data = json.loads(cleaned)
            if isinstance(data, dict):
                return data, ParsingOutcome.PARTIAL, list(data.keys())
        except json.JSONDecodeError:
            pass

        # 4. Handle unescaped newlines inside strings
        try:
            # Replace actual newlines inside string literals with \n
            cleaned_newlines = re.sub(r'(?<!\\)\n', r'\\n', cleaned)
            data = json.loads(cleaned_newlines)
            if isinstance(data, dict):
                return data, ParsingOutcome.PARTIAL, list(data.keys())
        except Exception:
            pass

        # 5. Regex field recovery fallback
        recovered = self._regex_field_recovery(cand)
        if recovered:
            return recovered, ParsingOutcome.PARTIAL, list(recovered.keys())

        return None, ParsingOutcome.FAILED, []

    def _try_repair_truncated(self, candidate: str, depth: int, in_string: bool) -> Tuple[Optional[Dict[str, Any]], ParsingOutcome, List[str]]:
        """Attempts repair of a generation cut off mid-token."""
        repaired = candidate.strip()
        if in_string:
            repaired += '"'
        repaired += '}' * depth

        # Clean trailing commas before closing
        repaired = re.sub(r',\s*\}', '}', repaired)

        try:
            data = json.loads(repaired)
            if isinstance(data, dict) and any(k in data for k in self.SUMMARY_KEYS + self.STANCE_KEYS):
                return data, ParsingOutcome.PARTIAL, list(data.keys())
        except Exception:
            pass

        # Fallback to regex field recovery from truncated fragment
        recovered = self._regex_field_recovery(candidate)
        if recovered:
            return recovered, ParsingOutcome.PARTIAL, list(recovered.keys())

        return None, ParsingOutcome.FAILED, []

    def _regex_field_recovery(self, text: str) -> Dict[str, Any]:
        """Recovers key fields using regex when full JSON syntax cannot be restored."""
        result = {}
        for key in self.SUMMARY_KEYS:
            pat = rf'["\']?{re.escape(key)}["\']?\s*:\s*["\']([^"\']{{10,}})["\']?'
            m = re.search(pat, text, re.I)
            if m:
                result[key] = m.group(1).strip()
                break

        for key in self.REASONING_KEYS:
            pat = rf'["\']?{re.escape(key)}["\']?\s*:\s*["\']([^"\']{{10,}})["\']?'
            m = re.search(pat, text, re.I)
            if m:
                result[key] = m.group(1).strip()
                break

        for key in self.STANCE_KEYS:
            pat = rf'["\']?{re.escape(key)}["\']?\s*:\s*["\']?(support|undermine|neutral|abstain|pro|con)["\']?'
            m = re.search(pat, text, re.I)
            if m:
                result[key] = m.group(1).strip()
                break

        return result

    def unpack_schema_aware(self, obj: Dict[str, Any], raw_item: RawJsonObject) -> Optional[RawSegment]:
        """
        Unpacks nested dictionaries schema-aware without treating sub-dictionaries
        as separate viewpoints. Combines summary + reasoning into an atomic viewpoint.
        """
        if not obj or not isinstance(obj, dict):
            return None

        # Check for top-level wrapper keys like "supporting: { ... }" or "response: { ... }"
        stance_hint = None
        for k in ["supporting", "support", "undermining", "undermine", "pro", "con"]:
            if k in obj and isinstance(obj[k], dict):
                stance_hint = k
                obj = obj[k]
                break

        # Check nested "perspective" dictionary
        perspective_data = None
        for p_key in ["perspective", "viewpoint", "argument"]:
            if p_key in obj and isinstance(obj[p_key], dict):
                perspective_data = obj[p_key]
                break

        # Extract Summary
        summary = None
        if perspective_data:
            for k in self.SUMMARY_KEYS:
                if k in perspective_data and isinstance(perspective_data[k], str):
                    summary = perspective_data[k].strip()
                    break

        if not summary:
            for k in self.SUMMARY_KEYS:
                if k in obj:
                    val = obj[k]
                    if isinstance(val, str):
                        summary = val.strip()
                        break
                    elif isinstance(val, dict):
                        # Nested summary object
                        for inner_k in ["summary", "text", "claim"]:
                            if inner_k in val and isinstance(val[inner_k], str):
                                summary = val[inner_k].strip()
                                break

        # Extract Reasoning
        reasoning = None
        if perspective_data:
            for k in self.REASONING_KEYS:
                if k in perspective_data and isinstance(perspective_data[k], str):
                    reasoning = perspective_data[k].strip()
                    break

        if not reasoning:
            for k in self.REASONING_KEYS:
                if k in obj and isinstance(obj[k], str):
                    reasoning = obj[k].strip()
                    break

        # Extract Stance
        raw_stance = None
        stance_source = "missing_or_ambiguous"
        if stance_hint:
            raw_stance = stance_hint
            stance_source = "json_wrapper_key"

        if not raw_stance:
            for k in self.STANCE_KEYS:
                if k in obj and isinstance(obj[k], str):
                    raw_stance = obj[k].strip()
                    stance_source = "json_explicit"
                    break
                elif perspective_data and k in perspective_data and isinstance(perspective_data[k], str):
                    raw_stance = perspective_data[k].strip()
                    stance_source = "json_explicit"
                    break

        # Combine into atomic viewpoint text
        if not summary and not reasoning:
            return None

        if summary and reasoning:
            # Avoid repeating summary if reasoning starts with or is identical to summary
            if summary.lower() in reasoning.lower():
                combined_text = reasoning
            else:
                combined_text = f"{summary} {reasoning}".strip()
        else:
            combined_text = summary or reasoning

        trunc_status = (
            TruncationStatus.PARTIAL_RECOVERED 
            if raw_item.parsing_outcome == ParsingOutcome.PARTIAL 
            else TruncationStatus.COMPLETE
        )

        return RawSegment(
            text=combined_text,
            source_type="balanced_json_object",
            raw_stance=raw_stance,
            stance_source=stance_source,
            start_char=raw_item.start_char,
            end_char=raw_item.end_char,
            truncation_status=trunc_status
        )

    def extract_viewpoints(self, text: str) -> List[RawSegment]:
        """High-level method to scan and unpack all JSON viewpoints from text."""
        raw_objs = self.scan_for_json_objects(text)
        segments = []
        for robj in raw_objs:
            seg = self.unpack_schema_aware(robj.data, robj)
            if seg and len(seg.text.split()) >= 4:
                segments.append(seg)
        return segments
