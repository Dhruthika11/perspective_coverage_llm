"""
Mistral Out-Of-Domain Benchmark Generator for Argument-NLI V5.

Constructs dedicated Mistral evaluation slices to test whether the frozen V5 Data-Only model
generalizes to a completely unseen generator family (Mistral-7B-Instruct-v0.3) that was never
encountered during fine-tuning.

Creates two complementary benchmarks:
  1. Balanced Synthetic Slice: 140 Full-Coverage Positives + 140 Negatives (N=280 pairs)
     constructed from Mistral LLM generations (excluding the 39 locked test claims).
  2. Adjudicated Human Benchmark: 60 gold-standard human-adjudicated pairs (20 pos / 40 neg)
     from human_llm_cluster_calibration_300.csv.
"""

import os
import sys
import json
import re
import random
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.io_utils import resolve, save_csv, save_json, load_csv
from src.utils.logging_utils import get_logger

logger = get_logger("prepare_mistral_benchmark")


def parse_argument_text(raw_text: str) -> Tuple[str, str, str, str]:
    """Extracts premise, summary, reasoning, and stance from raw argument text."""
    text = str(raw_text).strip()
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        text = match.group(1).strip()
    if text.startswith("{") and text.endswith("}"):
        try:
            d = json.loads(text)
            summary = d.get("perspective_summary", "").strip()
            reasoning = d.get("core_reasoning", "").strip()
            stance = d.get("stance", "").strip().lower()
            if summary and reasoning:
                premise = f"{summary} {reasoning}"
            else:
                premise = summary or reasoning or text
            return premise, summary, reasoning, stance
        except Exception:
            pass
    return text, text, "", ""


def build_mistral_adjudicated_benchmark(
    calibration_csv: str = "data/calibration/human_llm_cluster_calibration_300.csv",
    output_csv: str = "data/mistral_benchmark/mistral_adjudicated_benchmark.csv"
) -> Optional[pd.DataFrame]:
    """Extracts the 60 human-adjudicated Mistral pairs from calibration data."""
    calib_path = resolve(calibration_csv)
    if not os.path.exists(calib_path):
        logger.warning(f"Calibration file not found at {calib_path}")
        return None

    df_calib = load_csv(calib_path)
    mistral_mask = df_calib["model_name"].astype(str).str.lower().str.contains("mistral")
    df_mistral = df_calib[mistral_mask].copy().reset_index(drop=True)

    if len(df_mistral) == 0:
        logger.warning("No Mistral rows found in calibration dataset.")
        return None

    records = []
    for _, row in df_mistral.iterrows():
        label_bin = int(row["ground_truth_label"])
        # Map binary label to 4-class taxonomy: 1 = full_coverage, 2 = no_coverage
        lbl_4class = 1 if label_bin == 1 else 2
        proposed_lbl = "full_coverage" if label_bin == 1 else "no_coverage"
        premise = str(row["llm_argument"]).strip()
        hypothesis = str(row["cluster_paraphrases"]).strip()

        records.append({
            "claim_text": str(row["claim_text"]).strip(),
            "premise": premise,
            "hypothesis": hypothesis,
            "label_4class": lbl_4class,
            "proposed_label": proposed_lbl,
            "subtype": "human_adjudicated_positive" if label_bin == 1 else "human_adjudicated_negative",
            "generator_family": "mistral",
            "model_source": "mistral_7b_instruct_v0_3",
            "premise_words": len(premise.split()),
            "hypothesis_words": len(hypothesis.split())
        })

    df_out = pd.DataFrame(records)
    out_path = resolve(output_csv)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_out.to_csv(out_path, index=False)
    logger.info(f"Built adjudicated Mistral benchmark: {len(df_out)} pairs ({sum(df_out['label_4class'] == 1)} pos, {sum(df_out['label_4class'] != 1)} neg) -> {out_path}")
    return df_out


def build_mistral_balanced_benchmark(
    processed_gen_csv: str = "data/llm_outputs/mistral_7b_instruct_v0_3/processed_generations.csv",
    claims_csv: str = "data/processed/claims.csv",
    test_csv: str = "data/nli_training_v3/test.csv",
    output_csv: str = "data/mistral_benchmark/mistral_balanced_benchmark.csv",
    positives_target: int = 140,
    negatives_target: int = 140,
    seed: int = 42
) -> Optional[pd.DataFrame]:
    """
    Constructs a balanced 140 pos / 140 neg (N=280) benchmark from Mistral LLM generations.
    Guarantees 0% overlap with the 39 locked test claims.
    """
    random.seed(seed)
    np.random.seed(seed)

    gen_path = resolve(processed_gen_csv)
    claims_path = resolve(claims_csv)
    test_path = resolve(test_csv)

    if not os.path.exists(gen_path):
        logger.warning(f"Mistral processed generations not found at {gen_path}")
        return None

    df_gen = load_csv(gen_path)
    df_test = load_csv(test_path) if os.path.exists(test_path) else pd.DataFrame(columns=["claim_text"])
    test_claims = set(df_test["claim_text"].dropna().unique())

    # Map claim_id to claim_text if needed
    claim_map = {}
    if os.path.exists(claims_path):
        df_claims = load_csv(claims_path)
        for _, r in df_claims.iterrows():
            claim_map[str(r["claim_id"])] = str(r["claim_text"]).strip()

    parsed_records = []
    for _, row in df_gen.iterrows():
        cid = str(row.get("claim_id", ""))
        ctext = claim_map.get(cid, str(row.get("claim_text", cid)))
        if ctext in test_claims:
            continue

        summary = str(row.get("perspective_summary", "")).strip()
        reasoning = str(row.get("core_reasoning", "")).strip()
        stance = str(row.get("llm_reported_stance", "")).strip().lower()
        combined = str(row.get("combined_text", "")).strip()

        if not summary or not reasoning or summary == "nan" or reasoning == "nan":
            premise, summary, reasoning, stance = parse_argument_text(str(row.get("generated_text", combined)))
        else:
            premise = f"{summary} {reasoning}"

        if len(summary.split()) >= 5 and len(premise.split()) >= 20:
            parsed_records.append({
                "claim_text": ctext,
                "premise": premise,
                "summary": summary,
                "reasoning": reasoning,
                "stance": stance,
                "model_raw": "mistral_7b_instruct_v0_3"
            })

    if not parsed_records:
        logger.warning("Could not extract clean viewpoints from Mistral generations.")
        return None

    df_parsed = pd.DataFrame(parsed_records)
    logger.info(f"Parsed {len(df_parsed)} clean Mistral viewpoints across {df_parsed['claim_text'].nunique()} claims.")

    # Group viewpoints by claim
    claim_groups = defaultdict(list)
    for _, r in df_parsed.iterrows():
        claim_groups[r["claim_text"]].append(r)

    unique_claims = list(claim_groups.keys())
    random.shuffle(unique_claims)

    fam_pairs = []
    pos_count = 0
    neg_count = 0

    # 1. Full Coverage Positives
    for claim in unique_claims:
        if pos_count >= positives_target:
            break
        items = claim_groups[claim]
        for item in items:
            fam_pairs.append({
                "claim_text": claim,
                "premise": item["premise"],
                "hypothesis": item["summary"],
                "label_4class": 1,
                "proposed_label": "full_coverage",
                "subtype": "full_coverage",
                "generator_family": "mistral",
                "model_source": "mistral_7b_instruct_v0_3",
                "premise_words": len(item["premise"].split()),
                "hypothesis_words": len(item["summary"].split())
            })
            pos_count += 1
            if pos_count >= positives_target:
                break

    # 2. Hard Topical Negatives (Same Claim, Different Perspective or Opposite Stance)
    for claim in unique_claims:
        if neg_count >= negatives_target:
            break
        items = claim_groups[claim]
        if len(items) >= 2:
            p0 = items[0]["premise"]
            h1 = items[1]["summary"]
            s0 = items[0]["stance"]
            s1 = items[1]["stance"]

            is_contra = (s0 and s1 and s0 != s1)
            lbl = "contradiction" if is_contra else "no_coverage"
            lbl_id = 0 if is_contra else 2
            subtype = "opposite_stance_conflict" if is_contra else "same_topic_diff_concept"

            fam_pairs.append({
                "claim_text": claim,
                "premise": p0,
                "hypothesis": h1,
                "label_4class": lbl_id,
                "proposed_label": lbl,
                "subtype": subtype,
                "generator_family": "mistral",
                "model_source": "mistral_7b_instruct_v0_3",
                "premise_words": len(p0.split()),
                "hypothesis_words": len(h1.split())
            })
            neg_count += 1

    # 3. Fill remaining negatives with Cross-Claim Distractors if needed
    while neg_count < negatives_target and len(unique_claims) >= 2:
        c1, c2 = random.sample(unique_claims, 2)
        p = claim_groups[c1][0]["premise"]
        h = claim_groups[c2][0]["summary"]
        fam_pairs.append({
            "claim_text": c1,
            "premise": p,
            "hypothesis": h,
            "label_4class": 2,
            "proposed_label": "no_coverage",
            "subtype": "unrelated_different_claim",
            "generator_family": "mistral",
            "model_source": "mistral_7b_instruct_v0_3",
            "premise_words": len(p.split()),
            "hypothesis_words": len(h.split())
        })
        neg_count += 1

    df_out = pd.DataFrame(fam_pairs)
    out_path = resolve(output_csv)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_out.to_csv(out_path, index=False)
    logger.info(f"Saved balanced Mistral benchmark ({len(df_out)} pairs: {pos_count} pos, {neg_count} neg) to {out_path}")
    return df_out


def main():
    logger.info("=" * 80)
    logger.info("PREPARING MISTRAL OUT-OF-DOMAIN BENCHMARKS")
    logger.info("=" * 80)

    # 1. Build Adjudicated Benchmark
    df_adj = build_mistral_adjudicated_benchmark()

    # 2. Build Balanced Benchmark from generations if present
    df_bal = build_mistral_balanced_benchmark()

    # Write manifest
    manifest = {
        "benchmark_name": "mistral_ood_benchmarks",
        "adjudicated_benchmark": {
            "available": df_adj is not None,
            "pairs": len(df_adj) if df_adj is not None else 0,
            "positives": int((df_adj["label_4class"] == 1).sum()) if df_adj is not None else 0,
            "negatives": int((df_adj["label_4class"] != 1).sum()) if df_adj is not None else 0,
        },
        "balanced_benchmark": {
            "available": df_bal is not None,
            "pairs": len(df_bal) if df_bal is not None else 0,
            "positives": int((df_bal["label_4class"] == 1).sum()) if df_bal is not None else 0,
            "negatives": int((df_bal["label_4class"] != 1).sum()) if df_bal is not None else 0,
        }
    }
    manifest_path = resolve("data/mistral_benchmark/mistral_benchmark_manifest.json")
    save_json(manifest, manifest_path)
    logger.info(f"Benchmark manifest written to: {manifest_path}")


if __name__ == "__main__":
    main()

