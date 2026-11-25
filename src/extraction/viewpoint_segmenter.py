"""
Stage 2: Viewpoint Segmentation Engine for Perspective Coverage.

Decomposes non-JSON, mixed, and multi-viewpoint model generations:
  - Markdown aspect headings (### Ethical Perspective)
  - Numbered argument lists (1. ... 2. ...)
  - Bulleted argument lists (- ... * ...)
  - Standalone bold section headings (including multilingual/Unicode headings)
  - Merged viewpoints in monolithic prose
  - Connects argument titles to supporting explanations
"""

import re
from typing import List, Optional, Tuple
from .schemas import RawSegment, TruncationStatus


class ViewpointSegmenter:
    """
    Independent segmentation engine to decompose complex or multi-argument
    text into discrete atomic viewpoints.
    """

    def __init__(self):
        pass

    def segment_text(self, text: str) -> List[RawSegment]:
        """Segments raw text into atomic viewpoint segments."""
        if not text or not str(text).strip():
            return []

        t = str(text).strip()

        # 1. Check for Level 3 Markdown Headings (###)
        if "###" in t:
            segments = self._segment_markdown_headings(t)
            if len(segments) >= 2:
                return segments

        # 2. Check for Numbered Lists: '1. ... 2. ...'
        if self._has_numbered_list(t):
            segments = self._segment_numbered_list(t)
            if len(segments) >= 2:
                return segments

        # 3. Check for Bulleted Lists: '- ... - ...'
        if self._has_bulleted_list(t):
            segments = self._segment_bulleted_list(t)
            if len(segments) >= 2:
                return segments

        # 4. Check for Standalone Bold Section Headings (including Unicode/multilingual headings)
        # e.g., '**Argument 1: ...**\nText\n\n**论点1：...**\nText'
        if self._has_standalone_bold_headings(t):
            segments = self._segment_standalone_bold_headings(t)
            if len(segments) >= 2:
                return segments

        # 5. Check for Merged Multi-Argument Paragraphs
        segments = self._segment_prose_paragraphs(t)
        if len(segments) >= 2:
            return segments

        # 6. Single atomic prose block fallback (at least 6 words or at least 15 non-space characters for CJK)
        cleaned = self._clean_viewpoint_text(t)
        char_count_no_space = len(re.sub(r'\s+', '', cleaned))
        if len(cleaned.split()) >= 6 or char_count_no_space >= 15:
            raw_stance, stance_src = self._extract_inline_stance(t)
            return [RawSegment(
                text=cleaned,
                source_type="single_prose_block",
                raw_stance=raw_stance,
                stance_source=stance_src,
                start_char=0,
                end_char=len(t),
                truncation_status=TruncationStatus.COMPLETE
            )]

        return []

    def _has_numbered_list(self, text: str) -> bool:
        """Checks if text contains at least two numbered list items."""
        return bool(
            re.search(r'(?:\A|\n+)\s*(?:1[\.\)]|\bViewpoint\s+1:?|\bPerspective\s+1:?)(?:\s+|$)', text, re.I) and
            re.search(r'\n+\s*(?:2[\.\)]|\bViewpoint\s+2:?|\bPerspective\s+2:?)(?:\s+|$)', text, re.I)
        )

    def _has_bulleted_list(self, text: str) -> bool:
        """Checks if text contains multiple bullet list items."""
        bullets = re.findall(r'(?:\A|\n+)\s*[-*•]\s+', text)
        return len(bullets) >= 2

    def _has_standalone_bold_headings(self, text: str) -> bool:
        """Checks if text contains multiple bold headings on their own lines (not bullets or numbered)."""
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        bold_lines = [l for l in lines if re.match(r'^\*\*[^*]+\*\*:?$', l)]
        return len(bold_lines) >= 2

    def _segment_markdown_headings(self, text: str) -> List[RawSegment]:
        """Segments text divided by markdown headings and sub-bullets."""
        segments = []
        heading_pat = r'(?:\n+\s*###\s+[^\n]+|\A###\s+[^\n]+)'
        sections = re.split(heading_pat, text)
        headings = re.findall(r'###\s+([^\n]+)', text)

        if len(headings) >= 1 and len(sections) >= len(headings):
            for h_idx, sec in enumerate(sections[1:] if len(sections) > len(headings) else sections):
                h_title = headings[h_idx].strip() if h_idx < len(headings) else ""
                if self._has_numbered_list(sec):
                    sub_segs = self._segment_numbered_list(sec, parent_heading=h_title)
                    segments.extend(sub_segs)
                elif self._has_bulleted_list(sec):
                    sub_segs = self._segment_bulleted_list(sec, parent_heading=h_title)
                    segments.extend(sub_segs)
                elif self._has_standalone_bold_headings(sec):
                    sub_segs = self._segment_standalone_bold_headings(sec, parent_heading=h_title)
                    segments.extend(sub_segs)
                else:
                    cleaned = self._clean_viewpoint_text(sec)
                    char_len = len(re.sub(r'\s+', '', cleaned))
                    if len(cleaned.split()) >= 6 or char_len >= 15:
                        combined = f"{h_title}: {cleaned}" if h_title and not cleaned.lower().startswith(h_title.lower()) else cleaned
                        raw_st, st_src = self._extract_inline_stance(sec)
                        segments.append(RawSegment(
                            text=combined,
                            source_type="markdown_heading_section",
                            raw_stance=raw_st,
                            stance_source=st_src,
                            title=h_title
                        ))

        return segments

    def _segment_standalone_bold_headings(self, text: str, parent_heading: str = "") -> List[RawSegment]:
        """Segments text divided by standalone bold headings (**Heading**). Supports all Unicode characters."""
        segments = []
        matches = list(re.finditer(r'(?:\A|\n+)\s*\*\*([^*]+)\*\*:?\s*', text))
        if len(matches) < 2:
            return []

        for i, m in enumerate(matches):
            h_title = m.group(1).strip()
            start_body = m.end()
            end_body = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[start_body:end_body].strip()

            if not body:
                continue

            # Check if body has sub-bullets
            if self._has_bulleted_list(body):
                sub_segs = self._segment_bulleted_list(body, parent_heading=f"{parent_heading} - {h_title}" if parent_heading else h_title)
                segments.extend(sub_segs)
                continue

            cleaned = self._clean_viewpoint_text(body)
            char_len = len(re.sub(r'\s+', '', cleaned))
            if len(cleaned.split()) >= 6 or char_len >= 15:
                prefix = f"{parent_heading} - {h_title}" if parent_heading else h_title
                combined = f"{prefix}: {cleaned}" if prefix and not cleaned.lower().startswith(prefix.lower()) else cleaned
                raw_st, st_src = self._extract_inline_stance(body)
                if not raw_st:
                    raw_st, st_src = self._extract_inline_stance(h_title)
                segments.append(RawSegment(
                    text=combined,
                    source_type="markdown_bold_heading",
                    raw_stance=raw_st,
                    stance_source=st_src,
                    title=h_title
                ))

        return segments

    def _segment_numbered_list(self, text: str, parent_heading: str = "") -> List[RawSegment]:
        """Segments numbered lists, preserving title-explanation connections."""
        segments = []
        # Discard non-argumentative preamble preceding the first numbered bullet / header
        m_start = re.search(r'(?:\A|\n+)\s*(?:1[\.\)]|\bViewpoint\s+1:?|\bPerspective\s+1:?)(?:\s+|$)', text, flags=re.I)
        text_body = text[m_start.start():] if m_start else text

        item_pat = r'(?:\n+\s*(?:\d+[\.\)]|\bViewpoint\s+\d+:?|\bPerspective\s+\d+:?)\s+|\A(?:\d+[\.\)]|\bViewpoint\s+\d+:?|\bPerspective\s+\d+:?)\s+)'
        items = re.split(item_pat, text_body, flags=re.I)

        for item in items:
            raw_item = item.strip()
            if not raw_item or len(raw_item.split()) < 6:
                continue

            raw_st, st_src = self._extract_inline_stance(raw_item)
            title, explanation = self._extract_title_and_explanation(raw_item)

            if title:
                full_text = f"{title}: {explanation}" if explanation and not explanation.lower().startswith(title.lower()) else (explanation or title)
            else:
                full_text = self._clean_viewpoint_text(raw_item)

            if parent_heading and not full_text.lower().startswith(parent_heading.lower()):
                full_text = f"{parent_heading} - {full_text}"

            char_len = len(re.sub(r'\s+', '', full_text))
            if len(full_text.split()) >= 6 or char_len >= 15:
                segments.append(RawSegment(
                    text=full_text,
                    source_type="numbered_list_item",
                    raw_stance=raw_st,
                    stance_source=st_src,
                    title=title
                ))

        return segments

    def _segment_bulleted_list(self, text: str, parent_heading: str = "") -> List[RawSegment]:
        """Segments bulleted lists (- or *)."""
        segments = []
        bullet_pat = r'(?:\n+\s*[-*•]\s+|\A[-*•]\s+)'
        # Discard non-argumentative preamble preceding the first bullet
        m_start = re.search(bullet_pat, text)
        text_body = text[m_start.start():] if m_start else text

        items = re.split(bullet_pat, text_body)

        for item in items:
            raw_item = item.strip()
            if not raw_item:
                continue

            # Strip trailing bold heading like **Cons:** if present between sections
            raw_item = re.sub(r'\n+\s*\*\*[^*]+\*\*:?\s*$', '', raw_item).strip()

            char_len = len(re.sub(r'\s+', '', raw_item))
            if len(raw_item.split()) < 6 and char_len < 15:
                continue

            raw_st, st_src = self._extract_inline_stance(raw_item)
            title, explanation = self._extract_title_and_explanation(raw_item)

            if title:
                full_text = f"{title}: {explanation}" if explanation and not explanation.lower().startswith(title.lower()) else (explanation or title)
            else:
                full_text = self._clean_viewpoint_text(raw_item)

            if parent_heading and not full_text.lower().startswith(parent_heading.lower()):
                full_text = f"{parent_heading} - {full_text}"

            char_len = len(re.sub(r'\s+', '', full_text))
            if len(full_text.split()) >= 6 or char_len >= 15:
                segments.append(RawSegment(
                    text=full_text,
                    source_type="bulleted_list_item",
                    raw_stance=raw_st,
                    stance_source=st_src,
                    title=title
                ))

        return segments

    def _segment_prose_paragraphs(self, text: str) -> List[RawSegment]:
        """Segments prose containing multiple distinct arguments divided by paragraphs or transitions."""
        paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if len(p.strip().split()) >= 10]
        if len(paragraphs) < 2:
            return []

        segments = []
        for p in paragraphs:
            if "you are analyzing" in p.lower() or "respond with" in p.lower():
                continue
            cleaned = self._clean_viewpoint_text(p)
            if len(cleaned.split()) >= 6:
                raw_st, st_src = self._extract_inline_stance(p)
                segments.append(RawSegment(
                    text=cleaned,
                    source_type="prose_paragraph",
                    raw_stance=raw_st,
                    stance_source=st_src
                ))

        return segments

    def _extract_title_and_explanation(self, text: str) -> Tuple[Optional[str], str]:
        """Extracts bold or labeled title from the start of an item: '**Title**: Explanation'."""
        t = text.strip()
        m = re.match(r'^\*{1,2}([^*]+)\*{1,2}:?\s*[-–—:]?\s*(.*)$', t, re.DOTALL)
        if m:
            title = m.group(1).strip()
            explanation = self._clean_viewpoint_text(m.group(2))
            return title, explanation

        m2 = re.match(r'^([A-Z][A-Za-z\s]{3,35})\s*[-–—:]\s*(.*)$', t, re.DOTALL)
        if m2 and len(m2.group(2).split()) >= 5:
            title = m2.group(1).strip()
            explanation = self._clean_viewpoint_text(m2.group(2))
            return title, explanation

        return None, self._clean_viewpoint_text(t)

    def _extract_inline_stance(self, text: str) -> Tuple[Optional[str], str]:
        """Extracts explicit stance tags from text if present (e.g. '[support]', '(undermine)')."""
        t = text.strip()

        m_bracket = re.search(r'\[(support|undermine|pro|con)\]', t, re.I)
        if m_bracket:
            return m_bracket.group(1).lower(), "heading_explicit"

        m_paren = re.search(r'\((support|undermine|pro|con)\)', t, re.I)
        if m_paren:
            return m_paren.group(1).lower(), "heading_explicit"

        m_bold = re.search(r'\*\*stance:?\*\*\s*:?\s*(support|undermine)', t, re.I)
        if m_bold:
            return m_bold.group(1).lower(), "heading_explicit"

        if re.match(r'^(?:Supportive|Support)\s+Viewpoint', t, re.I):
            return "support", "text_label_unambiguous"
        if re.match(r'^(?:Critical|Undermining|Opposing)\s+Viewpoint', t, re.I):
            return "undermine", "text_label_unambiguous"

        return None, "missing_or_ambiguous"

    def _clean_viewpoint_text(self, text: str) -> str:
        """Removes markdown formatting tags, bullet markers, and leftover headers."""
        s = str(text).strip()
        s = re.sub(r'^###\s+[^\n]+\n*', '', s).strip()
        s = re.sub(r'^\s*(?:\d+[\.\)]|[-•*](?!\*))\s*', '', s).strip()
        s = re.sub(r'^\*{1,2}([^*]+)\*{1,2}:?\s*[-–—:]?\s*', r'\1: ', s).strip()
        s = re.sub(r'^\s*\[(support|undermine|pro|con)\]\s*:?\s*', '', s, flags=re.I).strip()
        s = re.sub(r'^(?:Supportive|Critical|Opposing)\s+Viewpoint:?\s*', '', s, flags=re.I).strip()
        s = re.sub(r'\s+', ' ', s).strip()
        return s
