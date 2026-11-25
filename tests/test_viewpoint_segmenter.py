"""
Unit tests for Stage 2: Viewpoint Segmenter.
"""

import os
import sys
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.extraction.viewpoint_segmenter import ViewpointSegmenter


@pytest.fixture
def segmenter():
    return ViewpointSegmenter()


def test_numbered_list_segmentation(segmenter):
    text = (
        "1. **Cost-Saving Perspective**: Compulsory vaccination saves healthcare funds by preventing outbreaks.\n\n"
        "2. **Market Freedom Perspective**: Mandatory health policies interfere with personal autonomy."
    )
    segs = segmenter.segment_text(text)
    assert len(segs) == 2
    assert "Cost-Saving" in segs[0].text
    assert "Market Freedom" in segs[1].text
    # Check that bold formatting markers are cleaned
    assert not segs[0].text.startswith("**")


def test_markdown_heading_segmentation(segmenter):
    text = (
        "### Economic Impact\n"
        "Vaccines reduce long-term medical expenditure substantially.\n\n"
        "### Moral Responsibility\n"
        "Individuals have an obligation to protect vulnerable community members."
    )
    segs = segmenter.segment_text(text)
    assert len(segs) == 2
    assert not segs[0].text.startswith("###")
    assert not segs[1].text.startswith("###")


def test_bullet_list_with_inline_stance(segmenter):
    text = (
        "- [support] Universal basic income ensures financial security for risk-taking.\n"
        "- [undermine] Basic income reduces labor force participation and worker incentives."
    )
    segs = segmenter.segment_text(text)
    assert len(segs) == 2
    assert segs[0].raw_stance == "support"
    assert segs[1].raw_stance == "undermine"


def test_unicode_and_multilingual_headings(segmenter):
    """Condition 2 Regression: Unicode headings (Chinese, French, English) must be correctly segmented."""
    text = (
        "**Argument 1: Economic benefits**\n"
        "Investing in renewable infrastructure produces high long-term returns for municipal budgets.\n\n"
        "**论点1：经济效益**\n"
        "发展可再生能源基础设施能够为城市财政带来可观的长期回报。\n\n"
        "**观点二：社会影响**\n"
        "这项政策通过减少污染显著改善了低收入社区的生活质量与公众健康。\n\n"
        "**Évaluation : avantages**\n"
        "Le développement durable renforce la résilience économique et protège les générations futures."
    )
    segs = segmenter.segment_text(text)
    assert len(segs) == 4, f"Expected 4 multilingual viewpoints, got {len(segs)}"
    assert "Economic benefits" in segs[0].text
    assert "经济效益" in segs[1].text
    assert "社会影响" in segs[2].text
    assert "Évaluation : avantages" in segs[3].text
