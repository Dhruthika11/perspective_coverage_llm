"""
Unit Tests for Model Scaling Analysis, Alignment Ladder, and Teacher-Student Preservation (Phase 5).
"""
import pytest
import numpy as np
import pandas as pd
from src.analysis.scaling_analysis import (
    fit_log_linear_scaling,
    fit_power_law_scaling,
    fit_saturating_scaling,
    fit_non_monotonic_scaling,
    audit_generation_confounders,
    analyze_qwen_scaling
)
from src.analysis.alignment_ladder import (
    compute_stance_entropy,
    compute_viewpoint_retention,
    analyze_alignment_transition
)
from src.analysis.teacher_student_preservation import (
    evaluate_teacher_student_pair,
    analyze_llama_distillation_ladder
)


def test_fit_log_linear_scaling():
    params = np.array([1.5, 3.0, 7.0, 14.0, 32.0])
    # Monotonic scaling data: C = 0.05 * ln(params) + 0.20
    coverages = 0.05 * np.log(params) + 0.20

    fit = fit_log_linear_scaling(params, coverages)
    assert fit["model_type"] == "log_linear"
    assert abs(fit["alpha"] - 0.05) < 1e-3
    assert abs(fit["beta"] - 0.20) < 1e-3
    assert fit["r2"] > 0.99
    assert fit["pearson_r"] > 0.99


def test_fit_power_law_scaling():
    params = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    coverages = 0.10 * (params ** 0.3)

    fit = fit_power_law_scaling(params, coverages)
    assert fit["model_type"] == "power_law"
    assert abs(fit["A"] - 0.10) < 0.02
    assert abs(fit["gamma"] - 0.30) < 0.05
    assert fit["r2"] > 0.95


def test_fit_saturating_and_non_monotonic_scaling():
    params = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
    # Saturating data: C = 0.30 * (params / (params + 2.0))
    coverages = 0.30 * (params / (params + 2.0))

    sat_fit = fit_saturating_scaling(params, coverages)
    assert sat_fit["model_type"] == "saturating_hill"
    assert sat_fit["r2"] > 0.95
    assert "C_max" in sat_fit

    # Quadratic log data: C = -0.01 * (ln(P))^2 + 0.05 * ln(P) + 0.20
    log_p = np.log(params)
    quad_coverages = -0.01 * (log_p ** 2) + 0.05 * log_p + 0.20
    quad_fit = fit_non_monotonic_scaling(params, quad_coverages)
    assert quad_fit["model_type"] == "quadratic_log_polynomial"
    assert quad_fit["r2"] > 0.95
    assert "vertex_params_b" in quad_fit


def test_stance_entropy():
    # 50/50 balance -> Maximum entropy of 1.0 bit
    balanced_stances = ["support"] * 50 + ["undermine"] * 50
    h_max = compute_stance_entropy(balanced_stances)
    assert abs(h_max - 1.0) < 1e-3

    # Pure 100% support -> Zero entropy (0.0 bits)
    pure_stances = ["support"] * 100
    h_zero = compute_stance_entropy(pure_stances)
    assert h_zero == 0.0

    # Skewed (75/25) -> Intermediate entropy (~0.81 bits)
    skewed = ["support"] * 75 + ["undermine"] * 25
    h_skew = compute_stance_entropy(skewed)
    assert 0.75 < h_skew < 0.85


def test_viewpoint_retention():
    base = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
    aligned = {1, 2, 3, 4, 5, 6, 7, 11, 12}  # Retained 7, dropped 3, added 2 novel

    stats = compute_viewpoint_retention(base, aligned)
    assert stats["retention_rate"] == 0.70  # 7 / 10
    assert stats["dropout_rate"] == 0.30    # 3 / 10
    assert stats["n_retained"] == 7
    assert stats["n_dropped"] == 3
    assert stats["n_novel"] == 2


def test_teacher_student_preservation():
    teacher_clusters = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
    student_clusters = {1, 2, 3, 4, 5, 6, 7}  # Student retains 70% of teacher
    total_clusters = 20

    res = evaluate_teacher_student_pair(
        "Teacher-70B", "Student-8B", teacher_clusters, student_clusters, total_clusters
    )
    assert res["teacher_coverage"] == 0.50  # 10 / 20
    assert res["student_coverage"] == 0.35  # 7 / 20
    assert res["teacher_viewpoint_retention_rate"] == 0.70
    assert res["teacher_viewpoint_dropout_rate"] == 0.30
    assert res["student_novel_discovery_rate"] == 0.0


def test_scaling_analysis_execution():
    diagnosis = analyze_qwen_scaling()
    assert "finding" in diagnosis
    assert "qwen_scaling_ladder_data" in diagnosis
    assert len(diagnosis["qwen_scaling_ladder_data"]) == 6

