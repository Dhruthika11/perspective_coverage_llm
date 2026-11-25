#!/usr/bin/env python3
"""
Production Viewpoint Extraction Engine for Perspective Coverage.

Runs the verified, modular extraction pipeline across all 34 LLM models.
Features:
  - Per-model checkpointing and resumability (skips completed models unless --force)
  - Strict sample conservation validation (N_raw == N_viewpoint_samples + N_quarantined)
  - Zero duplicate primary key validation
  - Per-model outputs in data/viewpoint_separated/<model_key>/
  - Master extraction manifest in data/viewpoint_separated/extraction_manifest.csv
  - Master aggregated CSV data/viewpoint_separated/master_viewpoints_all_models.csv
"""

import os
import sys
import time
import argparse
import hashlib
from datetime import datetime
from typing import Dict, Any, List, Set, Optional
import pandas as pd

# Setup repo root
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.extraction.extraction_pipeline import ExtractionPipeline
from src.extraction.schemas import ExtractionResult, ExtractedViewpoint, QuarantinedGeneration

RAW_GEN_DIR = os.path.join(REPO_ROOT, "data", "llm_outputs", "raw_generations")
OUTPUT_BASE_DIR = os.path.join(REPO_ROOT, "data", "viewpoint_separated")
MANIFEST_PATH = os.path.join(OUTPUT_BASE_DIR, "extraction_manifest.csv")
CLAIMS_PATH = os.path.join(REPO_ROOT, "data", "processed", "claims.csv")


def load_claims() -> Dict[int, str]:
    if os.path.exists(CLAIMS_PATH):
        df_claims = pd.read_csv(CLAIMS_PATH)
        return dict(zip(df_claims["claim_id"], df_claims["claim_text"]))
    return {}


def get_all_llm_models() -> List[str]:
    """Returns list of 34 LLM model keys (excluding human benchmark datasets)."""
    exclude = {"ibm_debater", "kialo"}
    if not os.path.exists(RAW_GEN_DIR):
        return []
    models = []
    for d in sorted(os.listdir(RAW_GEN_DIR)):
        if d in exclude or d.startswith("."):
            continue
        p = os.path.join(RAW_GEN_DIR, d, "raw_generations.csv")
        if os.path.exists(p):
            models.append(d)
    return models


def load_manifest() -> pd.DataFrame:
    if os.path.exists(MANIFEST_PATH):
        try:
            return pd.read_csv(MANIFEST_PATH)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def save_manifest_entry(entry: Dict[str, Any]):
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    df_existing = load_manifest()
    if len(df_existing) > 0 and "model_key" in df_existing.columns:
        df_existing = df_existing[df_existing["model_key"] != entry["model_key"]]
        df_new = pd.concat([df_existing, pd.DataFrame([entry])], ignore_index=True)
    else:
        df_new = pd.DataFrame([entry])
    df_new.to_csv(MANIFEST_PATH, index=False)


def process_model(model_key: str, pipeline: ExtractionPipeline, claims: Dict[int, str], force: bool = False) -> Dict[str, Any]:
    model_out_dir = os.path.join(OUTPUT_BASE_DIR, model_key)
    vp_out_path = os.path.join(model_out_dir, "viewpoints.csv")
    quar_out_path = os.path.join(model_out_dir, "quarantined.csv")

    # Checkpoint check
    if not force and os.path.exists(vp_out_path) and os.path.exists(quar_out_path):
        manifest = load_manifest()
        if len(manifest) > 0 and "model_key" in manifest.columns:
            m_row = manifest[manifest["model_key"] == model_key]
            if len(m_row) > 0 and m_row.iloc[0].get("status") == "COMPLETED":
                print(f"  [CHECKPOINT] Model {model_key} already completed. Skipping (use --force to re-run).")
                return m_row.iloc[0].to_dict()

    raw_file = os.path.join(RAW_GEN_DIR, model_key, "raw_generations.csv")
    if not os.path.exists(raw_file):
        raise FileNotFoundError(f"Raw generations file not found: {raw_file}")

    t0 = time.time()
    df_raw = pd.read_csv(raw_file)
    total_raw = len(df_raw)
    text_col = "generated_text" if "generated_text" in df_raw.columns else "argument"

    # Compute raw file MD5 for manifest provenance
    with open(raw_file, "rb") as f:
        raw_hash = hashlib.md5(f.read()).hexdigest()

    all_viewpoints: List[Dict[str, Any]] = []
    all_quarantined: List[Dict[str, Any]] = []
    sample_disposition: Dict[tuple, str] = {}

    for idx, row in df_raw.iterrows():
        cid = int(row["claim_id"])
        sid = int(row.get("sample_id", idx % 5))
        raw_text = str(row.get(text_col, ""))
        claim_text = claims.get(cid, "")

        res = pipeline.process_raw_generation(
            model_key=model_key,
            claim_id=cid,
            sample_id=sid,
            raw_text=raw_text,
            claim_text=claim_text
        )

        if res.quarantined:
            all_quarantined.append(res.quarantined.to_dict())
            sample_disposition[(cid, sid)] = "QUARANTINED"
        elif len(res.viewpoints) > 0:
            sample_disposition[(cid, sid)] = "VIEWPOINTS_EXTRACTED"
        else:
            sample_disposition[(cid, sid)] = "DROPPED_EMPTY"

        for vp in res.viewpoints:
            all_viewpoints.append(vp.to_dict())

    # Create DataFrames
    df_vp = pd.DataFrame(all_viewpoints)
    df_quar = pd.DataFrame(all_quarantined)

    # Invariant Validations
    vp_samples = set(df_vp.groupby(["claim_id", "sample_id"]).groups.keys()) if len(df_vp) > 0 else set()
    quar_samples = set(df_quar.groupby(["claim_id", "sample_id"]).groups.keys()) if len(df_quar) > 0 else set()

    accounted_samples = len(vp_samples.union(quar_samples))
    silent_drops = total_raw - accounted_samples

    dup_keys = 0
    if len(df_vp) > 0:
        dup_keys = int(df_vp.duplicated(subset=["composite_id"]).sum())

    elapsed = round(time.time() - t0, 2)

    # Ensure zero silent drops and zero duplicate keys
    status = "COMPLETED"
    error_msg = ""
    if silent_drops != 0:
        status = "FAILED"
        error_msg += f"Silent drops detected: {silent_drops} samples. "
    if dup_keys != 0:
        status = "FAILED"
        error_msg += f"Duplicate primary keys detected: {dup_keys}. "

    # Atomic write
    os.makedirs(model_out_dir, exist_ok=True)
    tmp_vp = vp_out_path + ".tmp"
    tmp_quar = quar_out_path + ".tmp"

    df_vp.to_csv(tmp_vp, index=False)
    df_quar.to_csv(tmp_quar, index=False)

    if os.path.exists(vp_out_path):
        os.remove(vp_out_path)
    if os.path.exists(quar_out_path):
        os.remove(quar_out_path)

    os.rename(tmp_vp, vp_out_path)
    os.rename(tmp_quar, quar_out_path)

    entry = {
        "model_key": model_key,
        "raw_samples": total_raw,
        "raw_file_hash": raw_hash,
        "viewpoints_extracted": len(df_vp),
        "quarantined_samples": len(df_quar),
        "samples_accounted": accounted_samples,
        "silent_drops": silent_drops,
        "duplicate_primary_keys": dup_keys,
        "status": status,
        "error_msg": error_msg,
        "duration_sec": elapsed,
        "timestamp": datetime.utcnow().isoformat()
    }

    save_manifest_entry(entry)

    print(f"  [{status}] {model_key}: {len(df_vp)} viewpoints, {len(df_quar)} quarantined, {accounted_samples}/{total_raw} samples accounted, {dup_keys} dup keys in {elapsed}s")
    if status == "FAILED":
        raise ValueError(f"Model {model_key} failed invariant checks: {error_msg}")

    return entry


def aggregate_master_dataset():
    """Concatenates all individual viewpoints.csv into master_viewpoints_all_models.csv."""
    models = get_all_llm_models()
    all_dfs = []
    print("\n---> Aggregating master viewpoints across all models...")
    for m in models:
        p = os.path.join(OUTPUT_BASE_DIR, m, "viewpoints.csv")
        if os.path.exists(p) and os.path.getsize(p) > 0:
            df = pd.read_csv(p)
            all_dfs.append(df)
            print(f"  Added {m}: {len(df):,} viewpoints")
        else:
            print(f"  WARNING: viewpoints.csv missing or empty for {m}")

    if all_dfs:
        master_df = pd.concat(all_dfs, ignore_index=True)
        master_path = os.path.join(OUTPUT_BASE_DIR, "master_viewpoints_all_models.csv")
        tmp_master = master_path + ".tmp"
        master_df.to_csv(tmp_master, index=False)
        if os.path.exists(master_path):
            os.remove(master_path)
        os.rename(tmp_master, master_path)
        print(f"\nSuccessfully generated master dataset: {master_path}")
        print(f"Total Models: {len(all_dfs)} | Total Viewpoint Rows: {len(master_df):,}")
    else:
        print("ERROR: No viewpoints found to aggregate.")


def main():
    parser = argparse.ArgumentParser(description="Run Production Viewpoint Extraction across 34 LLMs.")
    parser.add_argument("--model", type=str, default=None, help="Specific model_key to run (default: all 34)")
    parser.add_argument("--force", action="store_true", help="Force re-extraction even if checkpoint exists")
    parser.add_argument("--aggregate-only", action="store_true", help="Only aggregate existing viewpoints into master CSV")
    args = parser.parse_args()

    if args.aggregate_only:
        aggregate_master_dataset()
        return

    claims = load_claims()
    print(f"Loaded {len(claims)} gold claims from {CLAIMS_PATH}")

    pipeline = ExtractionPipeline()
    models = [args.model] if args.model else get_all_llm_models()

    print("=" * 90)
    print(f"RUNNING PRODUCTION VIEWPOINT EXTRACTION: {len(models)} MODELS")
    print(f"Base Output Directory: {OUTPUT_BASE_DIR}")
    print("=" * 90)

    t_start = time.time()
    results = []
    for idx, m in enumerate(models, 1):
        print(f"\n[{idx}/{len(models)}] Processing model: {m}")
        res = process_model(m, pipeline, claims, force=args.force)
        results.append(res)

    aggregate_master_dataset()

    total_time = round(time.time() - t_start, 2)
    print("\n" + "=" * 90)
    print(f"PRODUCTION EXTRACTION COMPLETED IN {total_time} SECONDS")
    print(f"Manifest written to: {MANIFEST_PATH}")
    print("=" * 90)


if __name__ == "__main__":
    main()
