"""
Unit tests for evaluate_zeroshot_chunked_inference.py
"""

import os
import pytest
import numpy as np
import pandas as pd
import torch
from unittest.mock import MagicMock

from src.analysis.evaluate_zeroshot_chunked_inference import (
    split_into_sentence_chunks,
    logsumexp_pool,
    compute_recall_at_fpr,
    resolve_model_label_indices,
    run_zeroshot_chunked_evaluation
)


def test_split_into_sentence_chunks_basic():
    # Short text: should return single chunk
    short_text = "Renewable energy lowers carbon emissions."
    chunks = split_into_sentence_chunks(short_text, window_size=2)
    assert len(chunks) == 1
    assert chunks[0] == short_text

    # Multi-sentence argument
    long_arg = (
        "Solar power continues to become more efficient every year. "
        "Wind turbines also provide significant renewable energy to the grid. "
        "However battery storage remains an essential requirement for intermittency. "
        "Government subsidies can help bridge the gap during early infrastructure development."
    )
    chunks_w2 = split_into_sentence_chunks(long_arg, window_size=2, step_size=1)
    assert len(chunks_w2) >= 3
    # Check that each chunk has at least two sentences
    for c in chunks_w2:
        assert len(c.split()) >= 8


def test_logsumexp_pool():
    # Single score returns itself
    single = np.array([0.75])
    assert logsumexp_pool(single, tau=2.0) == pytest.approx(0.75, abs=1e-5)

    # Multi-score smooth max
    scores = np.array([0.2, 0.8, 0.4])
    lse = logsumexp_pool(scores, tau=2.0)
    assert lse > 0.8  # LogSumExp is smooth upper bound
    assert lse < 1.5


def test_compute_recall_at_fpr():
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # 4 pos, 16 neg
    scores = np.array([0.9, 0.8, 0.7, 0.2, 0.85, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    # 10% FPR allows up to 1/16 = 0.0625 FPR.
    # At threshold 0.2, 1 negative (0.85) and all 4 positives (0.9, 0.8, 0.7, 0.2) pass.
    # So FPR = 1/16 = 0.0625 <= 0.10, and TPR = 4/4 = 1.0.
    rec, act_fpr, th = compute_recall_at_fpr(y_true, scores, target_fpr=0.10)
    assert rec == pytest.approx(1.0, abs=1e-2)
    assert act_fpr == pytest.approx(0.0625, abs=1e-2)
    assert th == pytest.approx(0.2, abs=1e-2)


def test_resolve_model_label_indices():
    mock_model = MagicMock()
    mock_model.config.id2label = {0: "CONTRADICTION", 1: "ENTAILMENT", 2: "NEUTRAL"}
    ent, contra, neu = resolve_model_label_indices(mock_model)
    assert ent == 1
    assert contra == 0
    assert neu == 2

    # Inverted order
    mock_model.config.id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}
    ent2, contra2, neu2 = resolve_model_label_indices(mock_model)
    assert ent2 == 0
    assert contra2 == 2
    assert neu2 == 1
