"""
Argument-NLI V5 Balanced Multi-Genre Dataset Generator.

Implements the V5 Data Balance Strategy:
  - Eliminates the training-testing mismatch by balancing:
      * Short Human Pairs (within-cluster paraphrases, same-stance distractors, opposite-stance conflicts)
      * Long LLM Arguments (Qwen, Gemma, Llama, OLMo; 100–250 words)
  - 4-Class Coverage Taxonomy:
      * 0: contradiction
      * 1: full_coverage
      * 2: no_coverage
      * 3: partial_coverage
  - Strict Claim-Level Disjoint Splitting (80% Train, 20% Val).
  - Absolute Zero Data Contamination against the 39 locked test claims.
"""

import os
import sys
import json
import random
from typing import Dict, List, Any, Tuple
from collections import defaultdict
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.io_utils import resolve, save_csv, save_json, load_csv
from src.utils.logging_utils import get_logger

logger = get_logger("prepare_v5_dataset")

LABEL_TO_ID_4CLASS = {
    "contradiction": 0,
    "full_coverage": 1,
    "entailment": 1,
    "no_coverage": 2,
    "neutral": 2,
    "partial_coverage": 3
}


def build_v5_balanced_dataset(
    output_dir: str = "data/nli_training_v5",
    v3_train_csv: str = "data/nli_training_v3/train.csv",
    v4_candidates_csv: str = "data/nli_training_v4/pilot_500_candidates.csv",
    test_csv: str = "data/nli_training_v3/test.csv",
    short_pair_cap_per_cat: int = 400,
    long_pair_cap_per_cat: int = 400,
    val_claim_ratio: float = 0.20,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    random.seed(seed)
    np.random.seed(seed)

    out_dir_res = resolve(output_dir)
    os.makedirs(out_dir_res, exist_ok=True)

    v3_p = resolve(v3_train_csv)
    v4_p = resolve(v4_candidates_csv)
    test_p = resolve(test_csv)

    if not os.path.exists(v3_p):
        raise FileNotFoundError(f"Missing V3 train data at {v3_p}")
    if not os.path.exists(test_p):
        raise FileNotFoundError(f"Missing test data at {test_p}")

    df_v3 = load_csv(v3_p)
    df_test = load_csv(test_p)
    df_v4 = load_csv(v4_p) if os.path.exists(v4_p) else pd.DataFrame()

    # Identify locked test claims to strictly avoid data leakage
    test_claims = set(df_test["claim_text"].dropna().unique())
    logger.info(f"Loaded {len(test_claims)} locked test claims to strictly exclude.")

    all_pairs: List[Dict[str, Any]] = []

    # -------------------------------------------------------------------------
    # 1. Harvest Short Human Pairs (from df_v3)
    # -------------------------------------------------------------------------
    logger.info("Harvesting Short Human pairs from V3 training split...")
    short_cats = {
        "within_cluster_paraphrase": "full_coverage",
        "same_stance_different_reasoning": "no_coverage",
        "opposite_stance_conflict": "contradiction",
        "unrelated_different_claim": "no_coverage"
    }

    v3_non_test = df_v3[~df_v3["claim_text"].isin(test_claims)].copy()

    for cat_name, target_lbl in short_cats.items():
        sub_df = v3_non_test[v3_non_test["pair_category"] == cat_name].copy()
        if len(sub_df) > short_pair_cap_per_cat:
            sub_df = sub_df.sample(n=short_pair_cap_per_cat, random_state=seed)

        for _, row in sub_df.iterrows():
            premise = str(row["premise"]).strip()
            hypothesis = str(row["hypothesis"]).strip()
            claim = str(row["claim_text"]).strip()
            all_pairs.append({
                "claim_text": claim,
                "premise": premise,
                "hypothesis": hypothesis,
                "proposed_label": target_lbl,
                "label_4class": LABEL_TO_ID_4CLASS[target_lbl],
                "subtype": cat_name,
                "length_regime": "short_human",
                "premise_words": len(premise.split()),
                "hypothesis_words": len(hypothesis.split()),
                "source": "perspectrum_human"
            })

    # -------------------------------------------------------------------------
    # 2. Harvest Long LLM Arguments (from df_v4 and df_v3 long categories)
    # -------------------------------------------------------------------------
    logger.info("Harvesting Long LLM Arguments across multiple families...")
    if not df_v4.empty:
        v4_non_test = df_v4[~df_v4["claim_text"].isin(test_claims)].copy()
        for _, row in v4_non_test.iterrows():
            premise = str(row["llm_argument"]).strip()
            hypothesis = str(row["human_perspective"]).strip()
            claim = str(row["claim_text"]).strip()
            lbl = str(row["proposed_label"]).strip().lower()
            subtype = str(row["subtype"]).strip().lower()

            all_pairs.append({
                "claim_text": claim,
                "premise": premise,
                "hypothesis": hypothesis,
                "proposed_label": lbl,
                "label_4class": LABEL_TO_ID_4CLASS[lbl],
                "subtype": subtype,
                "length_regime": "long_llm",
                "premise_words": len(premise.split()),
                "hypothesis_words": len(hypothesis.split()),
                "source": row.get("generator_model", "multimodel_llm")
            })

    # Synthesize additional partial_coverage pairs across non-test claims to reach robust sample size (~250)
    current_partials = sum(1 for p in all_pairs if p["proposed_label"] == "partial_coverage")
    target_partials = 250
    if current_partials < target_partials:
        logger.info(f"Synthesizing additional partial coverage pairs from {current_partials} to {target_partials}...")
        # Use available non-test claims from v3_non_test
        cand_claims = v3_non_test.drop_duplicates(subset=["claim_text"]).to_dict(orient="records")
        random.shuffle(cand_claims)
        needed = target_partials - current_partials
        for row in cand_claims[:needed]:
            claim = str(row["claim_text"]).strip()
            hypo = str(row["hypothesis"]).strip()
            # Synthesize partial premise (discusses broad policy stance without specific target rationale)
            synth_premise = (
                f"When evaluating the controversy surrounding whether {claim.lower()}, substantial institutional and substantive arguments indicate broad merit for this general stance. "
                f"A comprehensive review of the public discourse shows that multiple societal, economic, and ethical considerations align with this conclusion across contemporary jurisdictions. "
                f"Decision-makers and analysts frequently emphasize that maintaining the status quo produces stagnation and fails to address widespread systemic concerns. "
                f"Consequently, any sound policy assessment regarding {claim.lower()} must recognize this overarching momentum, as adopting supportive measures aligns with prevailing empirical and civic priorities."
            )
            all_pairs.append({
                "claim_text": claim,
                "premise": synth_premise,
                "hypothesis": hypo,
                "proposed_label": "partial_coverage",
                "label_4class": 3,
                "subtype": "partial_coverage_synthesized",
                "length_regime": "long_llm",
                "premise_words": len(synth_premise.split()),
                "hypothesis_words": len(hypo.split()),
                "source": "synthesized_partial"
            })

    # Supplement with Long LLM Positives and Long Same-Topic from V3
    long_v3_cats = {
        "long_llm_to_human_positive_qwen2_5_1_5b_instruct": "full_coverage",
        "long_llm_to_human_positive_qwen2_5_32b_instruct": "full_coverage",
        "long_llm_to_human_positive_qwen2_5_3b_instruct": "full_coverage",
        "same_topic_diff_concept": "no_coverage"
    }

    for cat_name, target_lbl in long_v3_cats.items():
        sub_df = v3_non_test[v3_non_test["pair_category"] == cat_name].copy()
        if len(sub_df) > long_pair_cap_per_cat:
            sub_df = sub_df.sample(n=long_pair_cap_per_cat, random_state=seed)

        for _, row in sub_df.iterrows():
            premise = str(row["premise"]).strip()
            hypothesis = str(row["hypothesis"]).strip()
            claim = str(row["claim_text"]).strip()
            all_pairs.append({
                "claim_text": claim,
                "premise": premise,
                "hypothesis": hypothesis,
                "proposed_label": target_lbl,
                "label_4class": LABEL_TO_ID_4CLASS[target_lbl],
                "subtype": cat_name,
                "length_regime": "long_llm",
                "premise_words": len(premise.split()),
                "hypothesis_words": len(hypothesis.split()),
                "source": "qwen_llm"
            })

    df_all = pd.DataFrame(all_pairs)
    # Deduplicate exact premise-hypothesis pairs
    df_all = df_all.drop_duplicates(subset=["premise", "hypothesis"]).reset_index(drop=True)
    df_all["pair_id"] = [f"v5_{i:05d}" for i in range(len(df_all))]

    logger.info(f"Total candidate pairs compiled: {len(df_all)}")
    logger.info("Class distribution:\n" + str(df_all["proposed_label"].value_counts()))
    logger.info("Length regime distribution:\n" + str(df_all["length_regime"].value_counts()))

    # -------------------------------------------------------------------------
    # 3. Strict Claim-Level Disjoint Splitting (Train vs. Val)
    # -------------------------------------------------------------------------
    unique_claims = sorted(df_all["claim_text"].unique())
    random.shuffle(unique_claims)
    n_val = max(1, int(len(unique_claims) * val_claim_ratio))
    val_claims = set(unique_claims[:n_val])
    train_claims = set(unique_claims[n_val:])

    train_df = df_all[df_all["claim_text"].isin(train_claims)].copy().reset_index(drop=True)
    val_df = df_all[df_all["claim_text"].isin(val_claims)].copy().reset_index(drop=True)

    # Double check zero overlap with test set
    assert len(set(train_df["claim_text"]).intersection(test_claims)) == 0, "Train contamination detected!"
    assert len(set(val_df["claim_text"]).intersection(test_claims)) == 0, "Val contamination detected!"
    assert len(set(train_df["claim_text"]).intersection(set(val_df["claim_text"]))) == 0, "Claim leakage between train and val!"

    train_csv_path = os.path.join(out_dir_res, "train.csv")
    val_csv_path = os.path.join(out_dir_res, "val.csv")
    manifest_path = os.path.join(out_dir_res, "dataset_manifest.json")

    save_csv(train_df, train_csv_path)
    save_csv(val_df, val_csv_path)

    manifest = {
        "version": "v5",
        "total_pairs": len(df_all),
        "train_pairs": len(train_df),
        "val_pairs": len(val_df),
        "unique_claims_total": len(unique_claims),
        "unique_claims_train": len(train_claims),
        "unique_claims_val": len(val_claims),
        "train_class_distribution": train_df["proposed_label"].value_counts().to_dict(),
        "val_class_distribution": val_df["proposed_label"].value_counts().to_dict(),
        "train_regime_distribution": train_df["length_regime"].value_counts().to_dict(),
        "val_regime_distribution": val_df["length_regime"].value_counts().to_dict(),
        "mean_premise_words_train": round(float(train_df["premise_words"].mean()), 1),
        "mean_premise_words_val": round(float(val_df["premise_words"].mean()), 1),
        "zero_data_contamination_verified": True
    }
    save_json(manifest, manifest_path)
    logger.info(f"V5 balanced dataset created successfully: Train={len(train_df)}, Val={len(val_df)}.")
    return train_df, val_df, manifest


if __name__ == "__main__":
    build_v5_balanced_dataset()
