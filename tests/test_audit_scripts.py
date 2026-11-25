"""
Tests for Threshold Equivalence Verification and Word Count Acceptance Audit Scripts.
"""
import os
import pytest
import pandas as pd
import numpy as np

from src.analysis.verify_threshold_equivalence import run_threshold_equivalence_audit
from src.analysis.audit_word_count_and_acceptance import (
    parse_model_metadata,
    compute_word_count_statistics,
    run_word_count_and_acceptance_audit
)


def test_parse_model_metadata():
    olmo_meta = parse_model_metadata("olmo2_7b_dpo")
    assert olmo_meta["family"] == "OLMo-2"
    assert olmo_meta["alignment_stage"] == "DPO"
    assert olmo_meta["scale_b"] == 7.0

    qwen_meta = parse_model_metadata("qwen2_5_0_5b_instruct")
    assert qwen_meta["family"] == "Qwen-2.5"
    assert qwen_meta["alignment_stage"] == "Instruct"
    assert qwen_meta["scale_b"] == 0.5


def test_compute_word_count_statistics():
    texts = [
        "This is a seven word sentence here.",
        "Here is an eight word sentence right now.",
        ""
    ]
    stats = compute_word_count_statistics(texts)
    assert stats["n_samples"] == 2
    assert stats["mean_words"] == 7.5
    assert stats["min_words"] == 7
    assert stats["max_words"] == 8


def test_verify_threshold_equivalence_execution(tmp_path):
    out_json = str(tmp_path / "test_equivalence.json")
    report = run_threshold_equivalence_audit(output_json=out_json)

    assert "platt_parameters" in report
    assert "agreement_analysis" in report
    assert report["agreement_analysis"]["agreement_rate"] > 0.90
    assert report["agreement_analysis"]["cohen_kappa"] > 0.85
    assert os.path.exists(out_json)


def test_audit_word_count_and_acceptance_execution(tmp_path):
    out_dir = str(tmp_path / "metrics")
    tables_dir = str(tmp_path / "tables")

    report = run_word_count_and_acceptance_audit(output_dir=out_dir, tables_dir=tables_dir)

    assert "models_audited" in report
    assert len(report["models_audited"]) > 0
    assert "correlation_confounder_analysis" in report
    assert os.path.exists(os.path.join(out_dir, "model_word_count_and_acceptance_audit.json"))
    assert os.path.exists(os.path.join(tables_dir, "table_word_count_vs_acceptance.csv"))


def test_claim_splits_manifest_disjointness():
    import json
    manifest_p = "data/calibration/claim_splits_manifest.json"
    assert os.path.exists(manifest_p)
    with open(manifest_p, "r", encoding="utf-8") as fp:
        m = json.load(fp)

    train_c = set(m["train_claims"])
    val_c = set(m["val_claims"])
    cal_c = set(m["calibration_claims"])
    test_c = set(m["test_claims"])

    # All 6 Pairwise Claim Disjointness Conditions
    assert len(train_c & val_c) == 0
    assert len(train_c & cal_c) == 0
    assert len(train_c & test_c) == 0
    assert len(val_c & cal_c) == 0
    assert len(val_c & test_c) == 0
    assert len(cal_c & test_c) == 0

    # All 6 Pairwise Verbatim Text Overlap Conditions
    import pandas as pd
    train_df = pd.read_csv("data/nli_training_v3/train.csv")
    val_df = pd.read_csv("data/nli_training_v3/val.csv")
    test_df = pd.read_csv("data/calibration/tri_dataset_750_pairs.csv")
    test_df = test_df[test_df["split"] == "test"]
    cal_df = pd.read_csv("data/calibration/expanded_stage1_calibration_pool.csv")

    train_texts = set(train_df["premise"]).union(set(train_df["hypothesis"]))
    val_texts = set(val_df["premise"]).union(set(val_df["hypothesis"]))
    test_texts = set(test_df["text_a"]).union(set(test_df["text_b"]))
    cal_texts = set(cal_df["premise"]).union(set(cal_df["hypothesis"]))

    assert len(train_texts & val_texts) == 0
    assert len(train_texts & cal_texts) == 0
    assert len(train_texts & test_texts) == 0
    assert len(val_texts & cal_texts) == 0
    assert len(val_texts & test_texts) == 0
    assert len(cal_texts & test_texts) == 0


def test_expanded_stage1_calibration_report():
    import json
    report_p = "results/calibration/expanded_stage1_calibration_report.json"
    assert os.path.exists(report_p)
    with open(report_p, "r", encoding="utf-8") as fp:
        rep = json.load(fp)

    assert rep["calibration_survivors_metrics"]["n_survivors"] >= 200
    assert "precision_95_ci" in rep["held_out_test_metrics"]
    assert "recall_end_to_end" in rep["held_out_test_metrics"]
    assert "recall_end_to_end_95_ci" in rep["held_out_test_metrics"]
    assert rep["held_out_test_metrics"]["recall_end_to_end"] > 0.50
    assert rep["held_out_test_metrics"]["precision"] >= 0.90


def test_v3_benchmark_protocol_artifacts():
    import json
    report_p = "results/metrics/v3_benchmark_protocol_report.json"
    table_p = "results/paper_tables/table_v3_benchmark_protocol_comparison.csv"
    assert os.path.exists(report_p)
    assert os.path.exists(table_p)

    with open(report_p, "r", encoding="utf-8") as fp:
        rep = json.load(fp)

    assert "ranking_metrics" in rep
    assert "comparison_1_common_threshold_0_500" in rep
    assert "comparison_2_separately_calibrated_thresholds" in rep
    assert rep["ranking_metrics"]["stage2_survivors"]["baseline_auroc"] > 0.70
    assert rep["ranking_metrics"]["stage2_survivors"]["baseline_auprc"] > 0.90
    assert rep["ranking_metrics"]["stage2_survivors"]["v3_auroc"] > 0.75
    assert rep["ranking_metrics"]["stage2_survivors"]["v3_auprc"] > 0.90
    assert "paired_mcnemar_test" in rep["comparison_1_common_threshold_0_500"]
    assert "paired_bootstrap_differences" in rep["comparison_1_common_threshold_0_500"]


def test_v3_full_reviewer_analyses_artifacts():
    import json
    sweep_csv = "results/paper_tables/table_v3_threshold_sweep.csv"
    sweep_fig = "results/paper_figures/v3_vs_baseline_threshold_sweep.png"
    pos_err_json = "results/metrics/v3_positive_error_analysis.json"
    boot_json = "results/metrics/v3_paired_statistical_tests.json"
    cross_fam_csv = "results/paper_tables/table_v3_cross_family_evaluation.csv"

    assert os.path.exists(sweep_csv)
    assert os.path.exists(sweep_fig)
    assert os.path.exists(pos_err_json)
    assert os.path.exists(boot_json)
    assert os.path.exists(cross_fam_csv)

    with open(boot_json, "r", encoding="utf-8") as fp:
        boot_rep = json.load(fp)
    assert "survivors_n66" in boot_rep
    assert "delta_auroc_95_ci" in boot_rep["survivors_n66"]
    assert "mcnemar_calibrated" in boot_rep["survivors_n66"]

    with open(pos_err_json, "r", encoding="utf-8") as fp:
        pos_rep = json.load(fp)
    assert pos_rep["total_llm_positives"] == 188
    assert "sample_false_negatives" in pos_rep


def test_precision_constrained_pr_artifacts():
    import json
    table_p = "results/paper_tables/table_precision_constrained_operating_points.csv"
    fig_p = "results/paper_figures/precision_constrained_pr_curves.png"
    json_p = "results/metrics/precision_constrained_operating_points_report.json"

    assert os.path.exists(table_p)
    assert os.path.exists(fig_p)
    assert os.path.exists(json_p)

    with open(json_p, "r", encoding="utf-8") as fp:
        rep = json.load(fp)

    assert "operating_points" in rep
    assert "prec_ge_90" in rep["operating_points"]
    assert "max_f1" in rep["operating_points"]
    assert rep["operating_points"]["prec_ge_90"]["v3"]["test"]["precision"] == 1.0


def test_llm_188_head_to_head_artifacts():
    import json
    table_p = "results/paper_tables/table_llm_188_head_to_head_comparison.csv"
    json_p = "results/metrics/llm_188_head_to_head_report.json"
    csv_raw = "results/paper_tables/llm_188_head_to_head.csv"

    assert os.path.exists(table_p)
    assert os.path.exists(json_p)
    assert os.path.exists(csv_raw)

    with open(json_p, "r", encoding="utf-8") as fp:
        rep = json.load(fp)

    assert rep["total_llm_positive_pairs"] == 188
    assert "operating_point_theta_0_50" in rep
    assert rep["operating_point_theta_0_50"]["mcnemar_p_value"] < 1e-6


def test_reviewer_experiments_1_2_4_artifacts():
    import json
    rep_p = "results/metrics/reviewer_experiments_1_2_4_report.json"
    assert os.path.exists(rep_p)

    with open(rep_p, "r", encoding="utf-8") as fp:
        rep = json.load(fp)

    assert "experiment_1_calibration_transfer" in rep
    assert "experiment_2_same_topic_fp_analysis" in rep
    assert "experiment_4_claim_clustered_bootstrap" in rep
    assert rep["experiment_4_claim_clustered_bootstrap"]["delta_f1_95_ci"][0] > 0.0


def test_v4_audit_and_selective_prediction_artifacts():
    import json
    rep_p = "results/metrics/v4_audit_and_selective_prediction_report.json"
    tab_p = "results/paper_tables/table_v4_selective_prediction_tradeoffs.csv"
    audit_tab_p = "results/paper_tables/table_audit_58_same_topic_fps.csv"

    assert os.path.exists(rep_p)
    assert os.path.exists(tab_p)
    assert os.path.exists(audit_tab_p)

    with open(rep_p, "r", encoding="utf-8") as fp:
        rep = json.load(fp)

    assert "audit_58_same_topic_fps" in rep
    assert rep["audit_58_same_topic_fps"]["total_audited"] == 58
    assert "selective_prediction_experiments" in rep


def test_v4_pilot_candidate_dataset_artifacts():
    import json
    import pandas as pd
    pilot_p = "data/nli_training_v4/pilot_500_candidates.csv"
    manifest_p = "data/nli_training_v4/pilot_500_manifest.json"
    assert os.path.exists(pilot_p)
    assert os.path.exists(manifest_p)

    df = pd.read_csv(pilot_p)
    assert len(df) == 500
    assert "full_coverage" in df["proposed_label"].values
    assert "partial_coverage" in df["proposed_label"].values
    assert "no_coverage" in df["proposed_label"].values
    assert "contradiction" in df["proposed_label"].values

    # Check subtype exact counts
    subtypes = df["subtype"].value_counts()
    assert subtypes["full_coverage"] == 150
    assert subtypes["partial_coverage"] == 100
    assert subtypes["same_policy_diff_reason"] == 100
    assert subtypes["same_topic_diff_concept"] == 100
    assert subtypes["opposite_stance_conflict"] == 50

    # Check multi-model representation
    models = set(df["generator_model"].unique())
    assert "gemma_2_9b_it" in models
    assert "llama_3_1_8b_instruct" in models
    assert "olmo_2_1124_7b_instruct" in models
    assert "qwen2_5_7b_instruct" in models

    # Check Zero Data Contamination against test claims
    test_df = pd.read_csv("data/nli_training_v3/test.csv")
    test_claims = set(test_df["claim_text"].dropna().unique())
    pilot_claims = set(df["claim_text"].dropna().unique())
    assert len(test_claims.intersection(pilot_claims)) == 0

    with open(manifest_p, "r", encoding="utf-8") as fp:
        mani = json.load(fp)
    assert mani["total_pairs"] == 500
    assert mani["unique_claims"] >= 75
    assert mani["test_claims_excluded"] == 39

    # Check length tier balancing (25% each)
    assert "length_tier" in df.columns
    tiers = df["length_tier"].value_counts()
    assert tiers["short"] == 125
    assert tiers["medium"] == 126
    assert tiers["long"] == 124
    assert tiers["very_long"] == 125

    # Check word count bounds per tier
    assert (df[df["length_tier"] == "short"]["premise_words"] >= 20).all()
    assert (df[df["length_tier"] == "short"]["premise_words"] <= 50).all()
    assert (df[df["length_tier"] == "medium"]["premise_words"] >= 50).all()
    assert (df[df["length_tier"] == "medium"]["premise_words"] < 100).all()
    assert (df[df["length_tier"] == "long"]["premise_words"] >= 100).all()
    assert (df[df["length_tier"] == "long"]["premise_words"] < 175).all()
    assert (df[df["length_tier"] == "very_long"]["premise_words"] >= 175).all()
    assert (df[df["length_tier"] == "very_long"]["premise_words"] <= 250).all()

    # Check presence of Long and Very Long Hard Negatives
    ct = pd.crosstab(df["subtype"], df["length_tier"])
    assert ct.loc["same_policy_diff_reason", "long"] == 25
    assert ct.loc["same_policy_diff_reason", "very_long"] == 25
    assert ct.loc["full_coverage", "long"] == 37
    assert ct.loc["full_coverage", "very_long"] == 37



def test_v4_training_dataset_and_split_integrity():
    import pandas as pd
    train_p = "data/nli_training_v4/train.csv"
    val_p = "data/nli_training_v4/val.csv"
    test_p = "data/nli_training_v3/test.csv"

    assert os.path.exists(train_p)
    assert os.path.exists(val_p)
    assert os.path.exists(test_p)

    df_train = pd.read_csv(train_p)
    df_val = pd.read_csv(val_p)
    df_test = pd.read_csv(test_p)

    assert len(df_train) > 0
    assert len(df_val) > 0
    assert len(df_train) + len(df_val) == 500

    # Claim disjointness between train and val
    train_cids = set(df_train["claim_id"].unique())
    val_cids = set(df_val["claim_id"].unique())
    assert len(train_cids & val_cids) == 0

    # Zero data contamination against locked test claims
    test_claims = set(df_test["claim_text"].dropna().unique())
    train_claims = set(df_train["claim_text"].dropna().unique())
    val_claims = set(df_val["claim_text"].dropna().unique())

    assert len(train_claims & test_claims) == 0
    assert len(val_claims & test_claims) == 0

    # Required columns present in both splits
    req_cols = ["pair_id", "claim_id", "claim_text", "human_perspective", "target_stance", "llm_argument", "generator_model", "proposed_label", "subtype"]
    for c in req_cols:
        assert c in df_train.columns
        assert c in df_val.columns










