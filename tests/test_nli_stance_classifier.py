"""
Unit tests for the NLI stance classifier.
"""
import os
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.analysis.nli_stance_classifier import (
    NLIStanceClassifier,
    classify_generations_df,
    _clean_text,
    _extract_row_text,
    extract_dynamic_viewpoints,
)


@pytest.fixture(scope="module")
def nli_classifier():
    return NLIStanceClassifier(model_name="cross-encoder/nli-deberta-v3-small", method="direct")


def test_clean_text_helper():
    assert _clean_text("") == ""
    assert _clean_text("   ") == ""
    assert _clean_text(None) == ""
    assert _clean_text(float("nan")) == ""
    assert _clean_text(np.nan) == ""
    assert _clean_text("nan") == ""
    assert _clean_text("None") == ""
    assert _clean_text("[Empty]") == ""
    assert _clean_text("Valid argument text") == "Valid argument text"
    # Strips prompt echo prefix
    echoed = "Give distinct viewpoints on the claim: 'Vaccination must be made compulsory' from an Economic perspective.\nOutput your viewpoint concisely.\nEconomic Viewpoint: Vaccines save money."
    assert _clean_text(echoed) == "Economic Viewpoint: Vaccines save money."


def test_extract_dynamic_viewpoints():
    # Mistral aspect headers
    text_aspects = (
        "Economic Viewpoint: Making vaccination compulsory can lead to significant cost savings.\n\n"
        "Financial Viewpoint: From a financial perspective, mandatory vaccination can be seen as a cost."
    )
    vps = extract_dynamic_viewpoints(text_aspects)
    assert len(vps) == 2
    assert vps[0].startswith("Economic Viewpoint:")
    assert vps[1].startswith("Financial Viewpoint:")

    # Qwen numbered format
    text_numbered = (
        "1. Compulsory vaccination reduces societal disease transmission.\n\n"
        "2. Mandatory vaccination infringes on individual autonomy."
    )
    vps_num = extract_dynamic_viewpoints(text_numbered)
    assert len(vps_num) == 2
    assert vps_num[0].startswith("1.")
    assert vps_num[1].startswith("2.")

    # Single plain argument
    single = "Vaccines are safe and essential."
    assert extract_dynamic_viewpoints(single) == ["Vaccines are safe and essential."]

    # Empty handling
    assert extract_dynamic_viewpoints("") == []
    assert extract_dynamic_viewpoints(None) == []


def test_extract_row_text_fallback():
    # Fallback when combined_text is missing
    row_no_comb = pd.Series({
        "generated_text": "Vaccination is essential for public health.",
        "perspective_summary": np.nan,
        "combined_text": np.nan,
    })
    assert _extract_row_text(row_no_comb) == "Vaccination is essential for public health."

    # Preferred column override
    row_override = pd.Series({
        "generated_text": "Raw text",
        "perspective_summary": "Summary text",
        "combined_text": "Combined text",
    })
    assert _extract_row_text(row_override, preferred_column="perspective_summary") == "Summary text"

    # All empty
    row_empty = pd.Series({
        "generated_text": np.nan,
        "perspective_summary": "",
        "combined_text": None,
    })
    assert _extract_row_text(row_empty) == ""


def test_clear_support_and_undermine(nli_classifier):
    claim = "Vaccination must be made compulsory"
    support_arg = (
        "Mandatory vaccination minimizes societal healthcare costs and reduces outbreaks. "
        "Herd immunity generates positive externalities for the entire community."
    )
    undermine_arg = (
        "Forcing vaccination violates personal bodily autonomy and individual liberty. "
        "Individuals must have the right to refuse medical procedures."
    )

    preds = nli_classifier.predict([support_arg, undermine_arg], [claim, claim], batch_size=2)
    assert len(preds) == 2
    assert preds[0]["predicted_stance"] == "support"
    assert preds[0]["confidence"] > 0.8
    assert preds[1]["predicted_stance"] == "undermine"
    assert preds[1]["confidence"] > 0.8


def test_predict_single(nli_classifier):
    claim = "Vaccination must be made compulsory"
    support_arg = "Mandatory vaccination protects public health and reduces epidemics."
    pred = nli_classifier.predict_single(support_arg, claim)
    assert pred["predicted_stance"] == "support"
    assert pred["confidence"] > 0.7


def test_empty_and_nan_argument_handling(nli_classifier):
    claim = "School Day Should Be Extended"
    preds = nli_classifier.predict(
        ["", "   ", None, float("nan"), np.nan, "nan"],
        [claim] * 6,
        batch_size=3
    )
    assert len(preds) == 6
    for p in preds:
        assert p["predicted_stance"] == "unclear"
        assert p["confidence"] == 0.0
        assert p["p_neutral"] == 1.0


def test_validation_errors(nli_classifier):
    with pytest.raises(ValueError, match="same length"):
        nli_classifier.predict(["Arg 1", "Arg 2"], ["Claim 1"])

    with pytest.raises(ValueError, match="batch_size must be positive"):
        nli_classifier.predict(["Arg 1"], ["Claim 1"], batch_size=0)


def test_template_method(nli_classifier):
    # Test template prediction method
    claim = "Vaccination must be made compulsory"
    support_arg = "Mandatory vaccination creates herd immunity and halts epidemics."
    undermine_arg = "Government vaccine mandates infringe on fundamental bodily freedom."
    preds = nli_classifier.predict_template_batch([support_arg, undermine_arg], [claim, claim], batch_size=2)
    assert len(preds) == 2
    assert preds[0]["predicted_stance"] == "support"
    assert preds[1]["predicted_stance"] == "undermine"


def test_classify_generations_df(nli_classifier):
    claims_df = pd.DataFrame([
        {"claim_id": 1, "claim_text": "Vaccination must be made compulsory"},
        {"claim_id": 2, "claim_text": "School uniforms should be mandatory"},
    ])
    generations_df = pd.DataFrame([
        {
            "claim_id": 1,
            "sample_id": 0,
            "viewpoint_id": 0,
            "llm_reported_stance": None,
            "combined_text": "Mandatory vaccines protect public health and prevent fatal epidemics.",
        },
        {
            "claim_id": 1,
            "sample_id": 1,
            "viewpoint_id": 0,
            "llm_reported_stance": None,
            "combined_text": "Government mandates violate constitutional rights and medical choice.",
        },
        {
            "claim_id": 2,
            "sample_id": 0,
            "viewpoint_id": 0,
            "llm_reported_stance": "support",
            "combined_text": "Uniforms create equality among students and eliminate clothing distractions.",
        },
        {
            "claim_id": 2,
            "sample_id": 1,
            "viewpoint_id": 1,
            "llm_reported_stance": None,
            "combined_text": np.nan,  # NaN handling test in df
        },
    ])

    out_df = classify_generations_df(generations_df, claims_df, classifier=nli_classifier, batch_size=4)

    assert len(out_df) == 4
    assert set(out_df.columns) == {
        "claim_id",
        "sample_id",
        "viewpoint_id",
        "viewpoint_text",
        "llm_reported_stance",
        "predicted_stance",
        "confidence",
        "p_support",
        "p_undermine",
        "p_neutral",
    }
    assert out_df.loc[0, "predicted_stance"] == "support"
    assert out_df.loc[1, "predicted_stance"] == "undermine"
    assert out_df.loc[2, "predicted_stance"] == "support"
    assert out_df.loc[3, "predicted_stance"] == "unclear"


def test_classify_generations_df_dynamic_viewpoints(nli_classifier):
    claims_df = pd.DataFrame([
        {"claim_id": 499, "claim_text": "Vaccination must be made compulsory"},
    ])
    generations_df = pd.DataFrame([
        {
            "claim_id": 499,
            "sample_id": 0,
            "viewpoint_id": 0,
            "llm_reported_stance": None,
            "parse_ok": False,
            "parse_method": "unparsed",
            "combined_text": (
                "Economic Viewpoint: Compulsory vaccination prevents expensive hospitalizations.\n\n"
                "Libertarian Viewpoint: Mandatory vaccination severely infringes on individual freedom."
            ),
        }
    ])

    # Decomposes dynamic multi-aspect viewpoints automatically
    exploded_df = classify_generations_df(
        generations_df,
        claims_df,
        classifier=nli_classifier,
        batch_size=2,
    )

    assert len(exploded_df) == 2
    assert "viewpoint_text" in exploded_df.columns
    assert exploded_df.loc[0, "viewpoint_id"] == 0
    assert exploded_df.loc[1, "viewpoint_id"] == 1
    assert exploded_df.loc[0, "predicted_stance"] == "support"
    assert exploded_df.loc[1, "predicted_stance"] == "undermine"

