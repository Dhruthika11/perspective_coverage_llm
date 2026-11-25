"""
Surface and Zero-Shot Baselines for Perspective Coverage Verification.

Implements standard NLI artifact diagnostics (Gururangan et al., 2018; Poliak et al., 2018):
  1. Length-Only Baseline:
     - Predicts entailment using only premise length, hypothesis length, and length delta/ratio via Logistic Regression.
     - Tests whether length differences alone act as a confounding shortcut.
  2. Hypothesis-Only Baseline:
     - Predicts entailment using ONLY the perspective hypothesis text (no premise given) using TF-IDF + Logistic Regression
       and pre-trained cross-encoder.
     - Tests whether certain human perspectives inherently correlate with positive labels due to annotation artifacts.
  3. Pretrained Zero-Shot Cross-Encoder on All Strata:
     - Evaluates off-the-shelf cross-encoder/nli-deberta-v3-base across all 4 strata:
         * Stratum 1: Short Human Benchmark (N=750)
         * Stratum 2: Long LLM Arguments Benchmark (N=368)
         * Stratum 3: Stance-Conflation Separation (N=428)
         * Stratum 4: Macro Combined Benchmark (N=1,118)
"""

import os
import sys
import json
import argparse
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.io_utils import resolve, load_csv, save_json
from src.utils.logging_utils import get_logger

logger = get_logger("surface_and_zeroshot_baselines")


def run_length_only_baseline(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame
) -> Dict[str, Any]:
    """
    Fits logistic regression on premise length, hypothesis length, and length difference.
    Evaluates whether length is a predictive shortcut.
    """
    logger.info("Evaluating Length-Only Baseline...")
    
    def extract_features(df):
        p_len = df["premise"].fillna("").astype(str).apply(lambda x: len(x.split())).to_numpy()
        h_len = df["hypothesis"].fillna("").astype(str).apply(lambda x: len(x.split())).to_numpy()
        diff = p_len - h_len
        ratio = p_len / np.maximum(h_len, 1)
        return np.column_stack([p_len, h_len, diff, ratio])

    X_train = extract_features(df_train)
    y_train = (df_train["proposed_label"].astype(str).str.lower().isin(["full_coverage", "entailment"])).astype(int).to_numpy()

    X_test = extract_features(df_test)
    y_test = (df_test["label"].astype(str).str.lower() == "entailment").astype(int).to_numpy()

    clf = LogisticRegression(random_state=42, max_iter=1000)
    clf.fit(X_train, y_train)

    probs = clf.predict_proba(X_test)[:, 1]
    preds = clf.predict(X_test)

    auc = roc_auc_score(y_test, probs)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds, average="binary", zero_division=0)

    logger.info(f"Length-Only Baseline: AUROC = {auc:.4f} | Prec = {prec*100:.2f}% | Rec = {rec*100:.2f}% | F1 = {f1:.4f}")
    return {
        "auroc": round(float(auc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "coefficients": [round(float(c), 4) for c in clf.coef_[0]],
        "feature_names": ["premise_len", "hypothesis_len", "len_diff", "len_ratio"]
    }


def run_hypothesis_only_baseline(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame
) -> Dict[str, Any]:
    """
    Fits TF-IDF + Logistic Regression using ONLY the perspective hypothesis text.
    Evaluates whether hypothesis phrasing alone leaks the target label.
    """
    logger.info("Evaluating Hypothesis-Only Baseline (Hypothesis-Text Artifact Probe)...")

    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
    X_train = vec.fit_transform(df_train["hypothesis"].fillna("").astype(str))
    y_train = (df_train["proposed_label"].astype(str).str.lower().isin(["full_coverage", "entailment"])).astype(int).to_numpy()

    X_test = vec.transform(df_test["hypothesis"].fillna("").astype(str))
    y_test = (df_test["label"].astype(str).str.lower() == "entailment").astype(int).to_numpy()

    clf = LogisticRegression(random_state=42, max_iter=1000, C=1.0)
    clf.fit(X_train, y_train)

    probs = clf.predict_proba(X_test)[:, 1]
    preds = clf.predict(X_test)

    auc = roc_auc_score(y_test, probs)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds, average="binary", zero_division=0)

    logger.info(f"Hypothesis-Only Baseline: AUROC = {auc:.4f} | Prec = {prec*100:.2f}% | Rec = {rec*100:.2f}% | F1 = {f1:.4f}")
    return {
        "auroc": round(float(auc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "vocab_size": len(vec.vocabulary_)
    }


def run_zeroshot_strata_evaluation(
    df_master: pd.DataFrame,
    df_test: pd.DataFrame
) -> Dict[str, Any]:
    """
    Evaluates off-the-shelf zero-shot cross-encoder (base_fwd_p_entail) across all 4 strata.
    """
    logger.info("Evaluating Pretrained Zero-Shot Cross-Encoder across All Strata...")

    # Attach base entailment probabilities
    scores = df_master["base_fwd_p_entail"].to_numpy()
    is_pos = (df_test["label"].str.lower() == "entailment").to_numpy()

    short_mask = df_test["pair_category"].isin([
        "within_cluster_paraphrase",
        "same_stance_different_reasoning",
        "opposite_stance_conflict",
        "unrelated_different_claim"
    ]).to_numpy()

    long_mask = df_test["pair_category"].isin([
        "long_llm_to_human_positive_qwen2_5_1_5b_instruct",
        "long_llm_to_human_positive_qwen2_5_32b_instruct",
        "long_llm_to_human_positive_qwen2_5_3b_instruct",
        "same_topic_diff_concept"
    ]).to_numpy()

    llm_pos_mask = df_test["pair_category"].str.startswith("long_llm_to_human_positive").to_numpy()
    same_pol_mask = (df_test["pair_category"] == "same_stance_different_reasoning").to_numpy()
    sp_mask = llm_pos_mask | same_pol_mask

    # Strata AUCs
    auc_short = roc_auc_score(is_pos[short_mask].astype(int), scores[short_mask])
    auc_long = roc_auc_score(is_pos[long_mask].astype(int), scores[long_mask])
    auc_sp = roc_auc_score(is_pos[sp_mask].astype(int), scores[sp_mask])
    auc_all = roc_auc_score(is_pos.astype(int), scores)

    logger.info(
        f"Zero-Shot Base: Short AUC = {auc_short:.4f} | Long AUC = {auc_long:.4f} | "
        f"Stance Separation AUC = {auc_sp:.4f} | Macro Combined AUC = {auc_all:.4f}"
    )

    return {
        "stratum_1_short_human_auroc": round(float(auc_short), 4),
        "stratum_2_long_llm_auroc": round(float(auc_long), 4),
        "stratum_3_stance_separation_auroc": round(float(auc_sp), 4),
        "stratum_4_macro_combined_auroc": round(float(auc_all), 4)
    }


def run_all_surface_and_zeroshot_evaluations(
    train_csv: str = "data/nli_training_v5/train.csv",
    test_csv: str = "data/nli_training_v3/test.csv",
    master_pred_csv: str = "results/metrics/master_1118_predictions.csv",
    output_report_json: str = "results/metrics/surface_and_zeroshot_baselines_report.json"
) -> Dict[str, Any]:
    train_p = resolve(train_csv)
    test_p = resolve(test_csv)
    master_p = resolve(master_pred_csv)
    out_json = resolve(output_report_json)

    df_train = load_csv(train_p)
    df_test = load_csv(test_p)
    df_master = load_csv(master_p)

    len_report = run_length_only_baseline(df_train, df_test)
    hyp_report = run_hypothesis_only_baseline(df_train, df_test)
    zs_report = run_zeroshot_strata_evaluation(df_master, df_test)

    final_report = {
        "length_only_baseline": len_report,
        "hypothesis_only_baseline": hyp_report,
        "zero_shot_pretrained_cross_encoder": zs_report
    }

    save_json(final_report, out_json)
    logger.info(f"Saved full surface baselines report to {out_json}")

    print("\n" + "=" * 85)
    print("SURFACE AND ZERO-SHOT BASELINES BENCHMARK AUDIT")
    print("=" * 85)
    print(f"1. Length-Only Baseline:      AUROC = {len_report['auroc']:.4f} | F1 = {len_report['f1']:.4f}")
    print(f"2. Hypothesis-Only Baseline:  AUROC = {hyp_report['auroc']:.4f} | F1 = {hyp_report['f1']:.4f}")
    print("-" * 85)
    print("3. Pretrained Zero-Shot Cross-Encoder across Strata:")
    print(f"   - Stratum 1: Short Human (N=750):           AUROC = {zs_report['stratum_1_short_human_auroc']:.4f}")
    print(f"   - Stratum 2: Long LLM Arguments (N=368):    AUROC = {zs_report['stratum_2_long_llm_auroc']:.4f}")
    print(f"   - Stratum 3: Stance Separation (N=428):     AUROC = {zs_report['stratum_3_stance_separation_auroc']:.4f}")
    print(f"   - Stratum 4: Macro Combined (N=1,118):      AUROC = {zs_report['stratum_4_macro_combined_auroc']:.4f}")
    print("=" * 85 + "\n")

    return final_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Surface and Zero-Shot Baselines")
    parser.add_argument("--train_csv", type=str, default="data/nli_training_v5/train.csv")
    parser.add_argument("--test_csv", type=str, default="data/nli_training_v3/test.csv")
    parser.add_argument("--master_csv", type=str, default="results/metrics/master_1118_predictions.csv")
    parser.add_argument("--output_report", type=str, default="results/metrics/surface_and_zeroshot_baselines_report.json")

    args = parser.parse_args()
    run_all_surface_and_zeroshot_evaluations(
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        master_pred_csv=args.master_csv,
        output_report_json=args.output_report
    )

