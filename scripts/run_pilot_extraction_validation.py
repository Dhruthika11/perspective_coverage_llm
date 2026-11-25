#!/usr/bin/env python3
"""
Phase 1 Pilot Validation Runner for Perspective Coverage.

Runs the new modular extraction pipeline on 4 representative models:
  1. Clean JSON Model:               gemma_2_9b_it
  2. Multi-Viewpoint Markdown Model: qwen2_5_32b_instruct
  3. Weak / Repetitive Model:        smollm2_135m_instruct
  4. Model with Heavy Abstentions:   qwen2_5_3b_instruct_p2

Evaluates all 10 acceptance criteria before full dataset regeneration.
"""

import os
import sys
import time
import pandas as pd
from typing import Dict, Any, List

# Repo root setup
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.extraction.extraction_pipeline import ExtractionPipeline
from src.extraction.schemas import QuarantineReason, StanceStatus


PILOT_MODELS = [
    ("gemma_2_9b_it", "Clean JSON Model"),
    ("qwen2_5_32b_instruct", "Multi-Viewpoint Markdown Model"),
    ("smollm2_135m_instruct", "Weak / Repetitive Model"),
    ("qwen2_5_3b_instruct_p2", "Model with Substantial Abstentions"),
]


def run_pilot():
    print("=" * 90)
    print("RUNNING PILOT EXTRACTION VALIDATION ON 4 REPRESENTATIVE MODELS")
    print(f"Repo Root: {REPO_ROOT}")
    print("=" * 90)

    # Load claims
    claims_path = os.path.join(REPO_ROOT, "data", "processed", "claims.csv")
    claim_dict = {}
    if os.path.exists(claims_path):
        df_claims = pd.read_csv(claims_path)
        claim_dict = dict(zip(df_claims["claim_id"], df_claims["claim_text"]))
    print(f"Loaded {len(claim_dict)} gold claims for semantic context.")

    pipeline = ExtractionPipeline()
    out_dir = os.path.join(REPO_ROOT, "data", "pilot_validation")
    os.makedirs(out_dir, exist_ok=True)

    pilot_summary = []

    for model_key, model_role in PILOT_MODELS:
        t0 = time.time()
        print(f"\n---> Processing Pilot Model: {model_key:<28} ({model_role})")

        # Load raw generations
        raw_csv = os.path.join(REPO_ROOT, "data", "llm_outputs", "raw_generations", model_key, "raw_generations.csv")
        if not os.path.exists(raw_csv):
            # Fallback check
            alt_csv = os.path.join(os.path.expanduser("~/perspective_coverage"), "data", "llm_outputs", "raw_generations", model_key, "raw_generations.csv")
            if os.path.exists(alt_csv):
                raw_csv = alt_csv
            else:
                print(f"  ERROR: Raw generations file not found for {model_key} at {raw_csv}")
                continue

        df_raw = pd.read_csv(raw_csv)
        total_raw = len(df_raw)
        text_col = "generated_text" if "generated_text" in df_raw.columns else "argument"

        all_viewpoints: List[Dict[str, Any]] = []
        all_quarantined: List[Dict[str, Any]] = []

        for idx, row in df_raw.iterrows():
            cid = int(row["claim_id"])
            sid = int(row.get("sample_id", idx % 5))
            raw_text = str(row.get(text_col, ""))
            claim_text = claim_dict.get(cid, "")

            res = pipeline.process_raw_generation(
                model_key=model_key,
                claim_id=cid,
                sample_id=sid,
                raw_text=raw_text,
                claim_text=claim_text
            )

            if res.quarantined:
                all_quarantined.append(res.quarantined.to_dict())

            for vp in res.viewpoints:
                all_viewpoints.append(vp.to_dict())

        # Save pilot viewpoints and quarantine
        df_vp = pd.DataFrame(all_viewpoints)
        df_quar = pd.DataFrame(all_quarantined)

        vp_out = os.path.join(out_dir, f"{model_key}_pilot_viewpoints.csv")
        quar_out = os.path.join(out_dir, f"{model_key}_pilot_quarantined.csv")
        df_vp.to_csv(vp_out, index=False)
        df_quar.to_csv(quar_out, index=False)

        # Audit Acceptance Criteria
        duration = time.time() - t0
        unique_samples_accounted = len(df_vp.groupby(["claim_id", "sample_id"])) + len(df_quar)
        dup_primary_keys = df_vp["composite_id"].duplicated().sum() if len(df_vp) else 0

        # Stance breakdown
        st_counts = df_vp["normalized_stance"].value_counts().to_dict() if len(df_vp) else {}
        unknown_stances = st_counts.get("unknown", 0)

        # Word count metrics
        w_counts = df_vp["word_count"] if len(df_vp) else pd.Series([0])
        med_words = w_counts.median() if len(w_counts) else 0

        summary = {
            "model_key": model_key,
            "role": model_role,
            "raw_samples": total_raw,
            "viewpoints_extracted": len(df_vp),
            "quarantined_samples": len(df_quar),
            "samples_accounted": unique_samples_accounted,
            "silent_drops": total_raw - unique_samples_accounted,
            "duplicate_primary_keys": dup_primary_keys,
            "median_words": med_words,
            "support_stances": st_counts.get("support", 0),
            "undermine_stances": st_counts.get("undermine", 0),
            "unknown_stances": unknown_stances,
            "exact_duplicates_removed": len(pipeline.deduplicator.exact_duplicates_removed),
            "duration_sec": round(duration, 1)
        }
        pilot_summary.append(summary)

        print(f"  Done in {duration:.1f}s | Viewpoints: {len(df_vp):,d} | Quarantined: {len(df_quar):,d} | "
              f"Duplicate Keys: {dup_primary_keys} | Silent Drops: {summary['silent_drops']}")

    df_sum = pd.DataFrame(pilot_summary)
    summary_path = os.path.join(out_dir, "pilot_validation_summary.csv")
    df_sum.to_csv(summary_path, index=False)

    print("\n" + "=" * 90)
    print("PILOT VALIDATION SUMMARY TABLE:")
    print(df_sum.to_string(index=False))
    print("=" * 90)
    print(f"Saved pilot results to: {out_dir}")


if __name__ == "__main__":
    run_pilot()
