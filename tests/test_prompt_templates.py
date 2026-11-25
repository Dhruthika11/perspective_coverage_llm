"""
Unit tests for Phase 2: Prompt templates, structured output parsing, abstention detection, and marginal coverage.
"""
import pytest
from src.generation.prompt_templates import (
    STANDARD_ASPECTS,
    build_prompt,
    build_p1_prompt,
    build_p2_prompt,
    build_p3_prompt,
    build_p4_prompt,
    parse_generation_output,
)
from src.analysis.marginal_coverage import compute_marginal_coverage_series


def test_standard_aspects_list():
    assert len(STANDARD_ASPECTS) == 5
    assert STANDARD_ASPECTS == ["Economic", "Moral", "Legal", "Practical", "Social"]


def test_p1_baseline_prompt():
    claim = "Nuclear energy should be expanded"
    history = [{"stance": "support", "argument": "It produces zero carbon emissions during operation."}]
    prompt = build_p1_prompt(claim, aspect="Economic", history=history)

    assert claim in prompt
    assert "Economic" in prompt
    assert "Prior arguments:" in prompt
    assert "produces zero carbon emissions" in prompt
    assert "perspective_summary" in prompt
    assert "core_reasoning" in prompt
    # P1 does NOT have explicit reasoning_criterion field
    assert "reasoning_criterion" not in prompt


def test_p2_improved_prompt():
    claim = "Universal basic income should be implemented"
    history = [{"stance": "support", "reasoning_criterion": "poverty alleviation", "argument": "Provides a minimum income floor."}]
    prompt = build_p2_prompt(claim, aspect="Moral", history=history)

    assert claim in prompt
    assert "Moral" in prompt
    assert "Reasoning Criterion" in prompt
    assert "CRITICAL NOVELTY CONSTRAINT" in prompt
    assert "Explicit Abstention" in prompt
    assert "poverty alleviation" in prompt
    assert "reasoning_criterion" in prompt
    assert '"status": "argument"' in prompt
    assert '"status": "abstain"' in prompt


def test_p3_ablation_no_history():
    claim = "School uniforms should be mandatory"
    history = [{"stance": "support", "argument": "Reduces peer pressure."}]
    # P3 builder ignores history
    prompt = build_p3_prompt(claim, aspect="Social", history=history)

    assert claim in prompt
    assert "Social" in prompt
    assert "Reasoning Criterion" in prompt
    assert "Prior Arguments" not in prompt
    assert "Reduces peer pressure" not in prompt


def test_p4_ablation_no_aspect():
    claim = "Autonomous vehicles should be legalized immediately"
    history = [{"stance": "undermine", "argument": "Software edge cases cause accidents."}]
    prompt = build_p4_prompt(claim, history=history)

    assert claim in prompt
    assert "ANY valid domain or viewpoint" in prompt
    assert "Prior Arguments" in prompt
    assert "Software edge cases" in prompt
    assert "Reasoning Criterion" in prompt


def test_parse_p1_output():
    raw = '''```json
    {
      "stance": "support",
      "perspective_summary": "Nuclear power stabilizes the electric grid.",
      "core_reasoning": "Baseload energy is necessary to balance intermittent solar and wind."
    }
    ```'''
    res = parse_generation_output(raw, prompt_version="p1")
    assert res["is_valid"] is True
    assert res["is_abstention"] is False
    assert res["stance"] == "support"
    assert "Nuclear power stabilizes" in res["argument"]
    assert "Baseload energy" in res["argument"]


def test_parse_p2_output():
    raw = '''{
      "status": "argument",
      "aspect": "Economic",
      "reasoning_criterion": "government debt",
      "stance": "undermine",
      "argument": "The capital expenditure required for high-speed rail will drastically inflate national debt."
    }'''
    res = parse_generation_output(raw, prompt_version="p2")
    assert res["is_valid"] is True
    assert res["is_abstention"] is False
    assert res["stance"] == "undermine"
    assert res["aspect"] == "Economic"
    assert res["reasoning_criterion"] == "government debt"
    assert "high-speed rail" in res["argument"]


def test_parse_abstention_output():
    raw = '''{
      "status": "abstain",
      "aspect": "Legal",
      "reasoning_criterion": "none",
      "stance": "none",
      "argument": "No distinct legal argument remains uncovered."
    }'''
    res = parse_generation_output(raw, prompt_version="p2")
    assert res["is_valid"] is True
    assert res["is_abstention"] is True
    assert res["status"] == "abstain"


def test_parse_fallback_markdown():
    raw = """
    **Stance:** Support
    **Reasoning Criterion:** Public health
    **Argument:** Vaccines provide herd immunity that protects vulnerable populations who cannot be vaccinated themselves.
    """
    res = parse_generation_output(raw, prompt_version="p2")
    assert res["is_valid"] is True
    assert res["stance"] == "support"
    assert res["reasoning_criterion"] == "Public health"
    assert "herd immunity" in res["argument"]
    assert res["parse_method"] == "fallback"


def test_parse_invalid_garbage():
    raw = "I am an AI and I cannot form opinions on this controversial matter."
    res = parse_generation_output(raw, prompt_version="p2")
    assert res["is_valid"] is False
    assert res["status"] == "parse_error"


def test_marginal_coverage_calculation():
    step_cov = [0.15, 0.25, 0.25, 0.35, 0.40]
    deltas = compute_marginal_coverage_series(step_cov)
    assert deltas == [0.15, 0.10, 0.0, 0.10, 0.05]

