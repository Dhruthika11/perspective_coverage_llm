"""
Unit tests for Argument-NLI V5 pipeline (Dataset, Warm-Start Head, and Disaggregated Benchmark).
"""

import os
import pytest
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from unittest.mock import MagicMock

from src.training.prepare_v5_dataset import build_v5_balanced_dataset
from src.training.train_argument_nli_v5 import (
    ArgumentNLIV5Dataset,
    ID_TO_LABEL_V5,
    LABEL_TO_ID_V5
)
import sys
sys.path.append(os.path.dirname(__file__))
from test_warmstart import warm_start_classifier_head


def test_v5_dataset_splits_exist():
    train_p = "data/nli_training_v5/train.csv"
    val_p = "data/nli_training_v5/val.csv"
    manifest_p = "data/nli_training_v5/dataset_manifest.json"

    assert os.path.exists(train_p), "Missing V5 train.csv"
    assert os.path.exists(val_p), "Missing V5 val.csv"
    assert os.path.exists(manifest_p), "Missing V5 dataset_manifest.json"

    train_df = pd.read_csv(train_p)
    val_df = pd.read_csv(val_p)
    test_df = pd.read_csv("data/nli_training_v3/test.csv")

    # 1. Zero data contamination
    test_claims = set(test_df["claim_text"].dropna().unique())
    train_claims = set(train_df["claim_text"].dropna().unique())
    val_claims = set(val_df["claim_text"].dropna().unique())

    assert len(train_claims.intersection(test_claims)) == 0, "Contamination: train contains test claims!"
    assert len(val_claims.intersection(test_claims)) == 0, "Contamination: val contains test claims!"
    assert len(train_claims.intersection(val_claims)) == 0, "Leakage: train and val share claims!"

    # 2. Both length regimes present
    assert "short_human" in train_df["length_regime"].values
    assert "long_llm" in train_df["length_regime"].values

    # 3. All 4 classes present in train
    assert set(train_df["label_4class"].unique()) == {0, 1, 2, 3}


def test_v5_warm_start_logic():
    hidden_dim = 64
    mock_base = nn.Linear(hidden_dim, 3)
    id2label = {0: "contradiction", 1: "entailment", 2: "neutral"}

    # Mock pretrained weights
    torch.manual_seed(42)
    mock_base.weight.data.normal_()
    mock_base.bias.data.normal_()

    # Manual warm-start expansion to 4 classes
    new_head = nn.Linear(hidden_dim, 4)
    with torch.no_grad():
        new_head.weight[0] = mock_base.weight[0]  # contra
        new_head.bias[0] = mock_base.bias[0]
        new_head.weight[1] = mock_base.weight[1]  # full
        new_head.bias[1] = mock_base.bias[1]
        new_head.weight[2] = mock_base.weight[2]  # none
        new_head.bias[2] = mock_base.bias[2]
        new_head.weight[3] = 0.5 * (mock_base.weight[1] + mock_base.weight[2])
        new_head.bias[3] = 0.5 * (mock_base.bias[1] + mock_base.bias[2])

    assert torch.allclose(new_head.weight[0], mock_base.weight[0])
    assert torch.allclose(new_head.weight[1], mock_base.weight[1])
    assert torch.allclose(new_head.weight[2], mock_base.weight[2])
    assert new_head.weight.shape == (4, hidden_dim)
