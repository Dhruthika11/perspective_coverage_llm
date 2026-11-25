"""
Unit tests for the 6 Argument-NLI V5 extended experiments:
  1. Surface & zero-shot baselines
  2. V5 ablations (head vs. data)
  3. Validation calibration & cluster-bootstrap
  4. Leave-one-generator-family-out (LOGFO)
  5. Fresh claim pristine benchmark
  6. False positive qualitative label audit
"""

import os
import pytest
import numpy as np
import pandas as pd

from src.analysis.evaluate_surface_and_zeroshot_baselines import (
    run_length_only_baseline,
    run_hypothesis_only_baseline
)
from src.analysis.calibrate_and_bootstrap_v5 import (
    calibrate_thresholds_on_validation,
    cluster_bootstrap_evaluation
)
from src.training.train_v5_ablations import (
    evaluate_strata_metrics,
    run_full_ablation_study
)
from src.training.train_leave_one_family_out import (
    map_source_to_family,
    partition_logfo_data,
    run_all_logfo_experiments
)
from src.training.prepare_fresh_claim_benchmark import (
    extract_fresh_claim_pairs,
    build_and_evaluate_fresh_benchmark
)
from src.analysis.audit_v5_false_positives import (
    audit_false_positives,
    run_false_positive_audit
)


def test_surface_baselines():
    # Synthetic test DataFrame
    df_train = pd.DataFrame({
        "premise": ["A long premise text with many words here and there.", "Short premise."],
        "hypothesis": ["A hypothesis perspective.", "Another perspective."],
        "proposed_label": ["full_coverage", "no_coverage"]
    })
    df_test = pd.DataFrame({
        "premise": ["Testing length feature extractor.", "Another test premise."],
        "hypothesis": ["Hypothesis test.", "Perspective."],
        "label": ["entailment", "neutral"]
    })

    len_res = run_length_only_baseline(df_train, df_test)
    assert "auroc" in len_res
    assert "coefficients" in len_res

    hyp_res = run_hypothesis_only_baseline(df_train, df_test)
    assert "auroc" in hyp_res
    assert "f1" in hyp_res


def test_v5_ablations_metrics():
    y_true = np.array([1, 1, 0, 0, 0])
    probs = np.array([
        [0.05, 0.85, 0.05, 0.05],
        [0.05, 0.75, 0.15, 0.05],
        [0.10, 0.20, 0.60, 0.10],
        [0.70, 0.10, 0.15, 0.05],
        [0.05, 0.15, 0.75, 0.05]
    ])
    cats = np.array([
        "within_cluster_paraphrase",
        "long_llm_to_human_positive_qwen2_5_32b_instruct",
        "same_stance_different_reasoning",
        "opposite_stance_conflict",
        "same_topic_diff_concept"
    ])

    metrics = evaluate_strata_metrics(y_true, probs, cats)
    assert metrics["overall_macro_auroc"] == 1.0
    assert metrics["stratum_1_short_human_auroc"] == 1.0
    assert metrics["stratum_2_long_llm_auroc"] == 1.0


def test_cluster_bootstrap_structure():
    df_test = pd.DataFrame({
        "claim_text": ["Claim A", "Claim A", "Claim B", "Claim B", "Claim C", "Claim C"],
        "label": ["entailment", "neutral", "entailment", "contradiction", "entailment", "neutral"],
        "pair_category": [
            "within_cluster_paraphrase",
            "same_stance_different_reasoning",
            "within_cluster_paraphrase",
            "opposite_stance_conflict",
            "long_llm_to_human_positive_qwen2_5_32b_instruct",
            "same_topic_diff_concept"
        ],
        "p_v5_mono_full": [0.85, 0.20, 0.90, 0.10, 0.88, 0.15]
    })

    res = cluster_bootstrap_evaluation(df_test, theta_f1=0.50, theta_prec=0.80, n_bootstraps=50)
    assert "cluster_bootstrap_95_ci" in res
    ci = res["cluster_bootstrap_95_ci"]
    assert "overall_auroc" in ci
    assert len(ci["overall_auroc"]["ci_95"]) == 2
    assert "short_human_auroc" in ci
    assert "long_llm_auroc" in ci


def test_logfo_family_mapping():
    assert map_source_to_family("qwen_llm") == "qwen"
    assert map_source_to_family("llama_3_1_8b_instruct") == "llama"
    assert map_source_to_family("gemma_2_9b_it") == "gemma"
    assert map_source_to_family("olmo_2_1124_7b_instruct") == "olmo"
    assert map_source_to_family("perspectrum_human") == "human"


def test_fresh_claims_benchmark_generation(tmp_path):
    out_csv = str(tmp_path / "test_fresh.csv")
    out_man = str(tmp_path / "fresh_manifest.json")
    out_rep = str(tmp_path / "fresh_report.json")

    rep = build_and_evaluate_fresh_benchmark(
        output_csv=out_csv,
        output_manifest=out_man,
        output_report=out_rep,
        max_claims=5,
        mock_eval=True
    )

    assert os.path.exists(out_csv)
    assert os.path.exists(out_man)
    assert rep["manifest"]["contamination_check"]["prior_train_overlap"] == 0
    assert rep["manifest"]["contamination_check"]["prior_test_overlap"] == 0


def test_false_positive_audit(tmp_path):
    df_synthetic = pd.DataFrame({
        "claim_text": ["Test Claim 1", "Test Claim 2"],
        "label": ["neutral", "neutral"],
        "pair_category": ["same_topic_diff_concept", "same_stance_different_reasoning"],
        "p_v5_mono_full": [0.85, 0.72],
        "hypothesis": ["Perspective H1", "Perspective H2"],
        "premise": ["Premise text P1", "Premise text P2"]
    })

    report, df_fp = audit_false_positives(df_synthetic, threshold=0.50)
    assert report["total_false_positives"] == 2
    assert len(report["case_studies"]) == 2
    assert len(report["false_positive_breakdown_by_category"]) > 0


def test_multiseed_replication(tmp_path):
    from src.training.run_multiseed_ablations import run_all_multiseed_configurations

    out_json = str(tmp_path / "multiseed_rep.json")
    out_csv = str(tmp_path / "multiseed_comp.csv")
    out_md = str(tmp_path / "multiseed_sum.md")

    report = run_all_multiseed_configurations(
        seeds=[42, 123, 999],
        epochs=10,
        output_json=out_json,
        output_csv=out_csv,
        output_md=out_md,
        mock_run=True
    )

    assert os.path.exists(out_json)
    assert os.path.exists(out_csv)
    assert os.path.exists(out_md)

    assert "aggregated_metrics" in report
    assert "v4_baseline" in report["aggregated_metrics"]
    assert "full_v5" in report["aggregated_metrics"]

    df_csv = pd.read_csv(out_csv)
    assert len(df_csv) == 4
    assert "Macro AUROC" in df_csv.columns


def test_ensemble_and_calibration(tmp_path):
    from src.training.train_and_calibrate_ensemble import run_ensemble_selection_calibration_and_evaluation

    out_dir = str(tmp_path / "metrics_test")
    res = run_ensemble_selection_calibration_and_evaluation(
        seeds=[42, 123, 999],
        epochs=10,
        mock_run=True,
        output_dir=out_dir
    )

    assert "decision" in res
    assert "thresholds" in res
    assert "test_benchmark" in res
    assert "fresh_benchmark" in res

    assert os.path.exists(os.path.join(out_dir, "model_selection_validation_decision.json"))
    assert os.path.exists(os.path.join(out_dir, "ensemble_validation_thresholds.json"))
    assert os.path.exists(os.path.join(out_dir, "final_ensemble_test_benchmark_report.json"))
    assert os.path.exists(os.path.join(out_dir, "final_ensemble_fresh_claims_report.json"))
    assert os.path.exists(os.path.join(out_dir, "final_ensemble_summary.md"))


def test_zeroshot_per_family(tmp_path):
    from src.analysis.evaluate_zeroshot_per_family import evaluate_zeroshot_per_family

    out_j = str(tmp_path / "zs_fam.json")
    out_m = str(tmp_path / "zs_fam.md")

    res = evaluate_zeroshot_per_family(
        data_csv="data/nli_training_v5/val.csv",
        mock_run=True,
        output_json=out_j,
        output_md=out_m
    )

    assert os.path.exists(out_j)
    assert os.path.exists(out_m)
    assert "per_family_results" in res
    assert "qwen" in res["per_family_results"]
    assert "llama" in res["per_family_results"]
    assert "gemma" in res["per_family_results"]
    assert "olmo" in res["per_family_results"]


def test_rebalanced_logfo_suite(tmp_path):
    from src.analysis.evaluate_rebalanced_logfo_and_zeroshot import run_rebalanced_logfo_and_zeroshot_suite

    out_j = str(tmp_path / "reb_logfo.json")
    out_m = str(tmp_path / "reb_logfo.md")

    res = run_rebalanced_logfo_and_zeroshot_suite(
        benchmark_csv="data/logfo_benchmark/balanced_multifamily_benchmark.csv",
        n_bootstraps=50,
        mock_run=True,
        output_report=out_j,
        output_md=out_m
    )

    assert os.path.exists(out_j)
    assert os.path.exists(out_m)
    assert "overall_success_declared" in res
    assert len(res["family_results"]) == 4
    for fam_res in res["family_results"]:
        assert "Zero_Shot_AUROC_CI" in fam_res
        assert "LOGFO_HeldOut_AUROC_CI" in fam_res
        assert "Delta_Gain_CI" in fam_res


def test_mistral_benchmark_and_evaluation(tmp_path):
    from src.training.prepare_mistral_benchmark import build_mistral_adjudicated_benchmark
    from src.analysis.evaluate_v5_on_mistral import evaluate_v5_on_mistral_suite

    out_bench = str(tmp_path / "mistral_bench.csv")
    df_adj = build_mistral_adjudicated_benchmark(output_csv=out_bench)
    assert df_adj is not None
    assert len(df_adj) == 60
    assert os.path.exists(out_bench)

    out_j = str(tmp_path / "mistral_rep.json")
    out_m = str(tmp_path / "mistral_rep.md")

    res = evaluate_v5_on_mistral_suite(
        benchmark_csv=out_bench,
        n_bootstraps=50,
        mock_run=True,
        output_report=out_j,
        output_md=out_m
    )

    assert os.path.exists(out_j)
    assert os.path.exists(out_m)
    assert "zeroshot_deberta" in res
    assert "v5_data_only" in res
    assert "delta_comparison" in res
    assert "pre_registered_success_rules" in res
    assert res["pre_registered_success_rules"]["rule_1_min_ood_auroc_0_85"]["target"] == 0.85
    assert res["delta_comparison"]["delta_auroc"] > 0


def test_standing_leak_check_audit():
    """
    Standing pipeline leak check: enforces 0.00% verbatim substring containment
    and balanced token overlap across training, locked test, and rebuilt LOGFO benchmarks.
    """
    from src.analysis.leak_check_audit import audit_dataset_leakage

    # 1. Locked Test Benchmark
    test_res = audit_dataset_leakage("Locked Test", "data/nli_training_v3/test.csv")
    assert test_res["clean_standard_verified"] is True
    assert test_res["positives"]["exact_substr_rate"] == 0.0
    assert test_res["overlap_gap_pos_minus_neg"] <= 20.0

    # 2. V5 Training Set
    train_res = audit_dataset_leakage("V5 Train", "data/nli_training_v5/train.csv")
    assert train_res["clean_standard_verified"] is True
    assert train_res["positives"]["exact_substr_rate"] <= 0.1
    assert train_res["overlap_gap_pos_minus_neg"] <= 20.0

    # 3. Clean Balanced Multi-Family LOGFO Benchmark
    logfo_res = audit_dataset_leakage("Clean LOGFO", "data/logfo_benchmark/clean_balanced_logfo_benchmark.csv")
    assert logfo_res["clean_standard_verified"] is True
    assert logfo_res["positives"]["exact_substr_rate"] == 0.0
    assert logfo_res["negatives"]["exact_substr_rate"] == 0.0
    assert logfo_res["overlap_gap_pos_minus_neg"] <= 20.0







