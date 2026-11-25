"""
Tests for warm-starting 4-class classification head from pretrained 3-class NLI models.
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock


def warm_start_classifier_head(base_classifier, old_id2label, partial_coverage_init="interpolate"):
    """
    Transforms a 3-class linear head to a 4-class linear head, preserving
    contradiction, entailment, and neutral weights while initializing partial_coverage.
    """
    contra_idx = next(i for i, l in old_id2label.items() if "contra" in l.lower())
    entail_idx = next(i for i, l in old_id2label.items() if "entail" in l.lower())
    neutral_idx = next(i for i, l in old_id2label.items() if "neutral" in l.lower())

    old_weight = base_classifier.weight.data.clone()
    old_bias = base_classifier.bias.data.clone()
    hidden_size = old_weight.shape[1]

    new_classifier = nn.Linear(hidden_size, 4)
    with torch.no_grad():
        # Class 0: contradiction
        new_classifier.weight[0] = old_weight[contra_idx]
        new_classifier.bias[0] = old_bias[contra_idx]

        # Class 1: full_coverage
        new_classifier.weight[1] = old_weight[entail_idx]
        new_classifier.bias[1] = old_bias[entail_idx]

        # Class 2: no_coverage
        new_classifier.weight[2] = old_weight[neutral_idx]
        new_classifier.bias[2] = old_bias[neutral_idx]

        # Class 3: partial_coverage
        if partial_coverage_init == "interpolate":
            jitter = torch.randn_like(old_weight[entail_idx]) * 0.001
            new_classifier.weight[3] = 0.5 * (old_weight[entail_idx] + old_weight[neutral_idx]) + jitter
            new_classifier.bias[3] = 0.5 * (old_bias[entail_idx] + old_bias[neutral_idx])
        else:
            new_classifier.weight[3] = torch.randn(hidden_size) * 0.02
            new_classifier.bias[3] = torch.tensor(0.0)

    return new_classifier


def test_warm_start_classifier_head_preservation():
    hidden_dim = 128
    mock_linear = nn.Linear(hidden_dim, 3)
    old_id2label = {0: "CONTRADICTION", 1: "ENTAILMENT", 2: "NEUTRAL"}

    new_head = warm_start_classifier_head(mock_linear, old_id2label, partial_coverage_init="interpolate")

    # Verify exact preservation of weights and biases for the 3 MNLI classes
    assert torch.allclose(new_head.weight[0], mock_linear.weight[0])
    assert torch.allclose(new_head.bias[0], mock_linear.bias[0])

    assert torch.allclose(new_head.weight[1], mock_linear.weight[1])
    assert torch.allclose(new_head.bias[1], mock_linear.bias[1])

    assert torch.allclose(new_head.weight[2], mock_linear.weight[2])
    assert torch.allclose(new_head.bias[2], mock_linear.bias[2])

    # Verify 4th class is initialized properly and distinct
    assert new_head.weight.shape == (4, hidden_dim)
    assert new_head.bias.shape == (4,)
    assert not torch.allclose(new_head.weight[3], new_head.weight[1])

