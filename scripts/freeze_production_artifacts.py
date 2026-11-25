#!/usr/bin/env python3
"""
Production Artifact Freeze and Checksum Generator.

Freezes the exact state of:
1. Extraction source code modules (SHA-256 / MD5).
2. Raw generation provenance manifest.
3. Master viewpoint dataset (SHA-256 / MD5 / row count / schema).
4. Master quarantined dataset (compiled and hashed).
5. Per-model file checksums.
6. Pipeline hyperparameters and configuration constants.

Outputs:
  - data/viewpoint_separated/master_quarantined_all_models.csv
  - data/viewpoint_separated/production_freeze_manifest.json
"""

import os
import sys
import glob
import json
import hashlib
from datetime import datetime
from typing import Dict, Any, List
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(REPO_ROOT, "data", "viewpoint_separated")


def compute_file_hashes(filepath: str) -> Dict[str, str]:
    if not os.path.exists(filepath):
        return {"error": "file_not_found"}
    md5 = hashlib.md5()
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            md5.update(chunk)
            sha256.update(chunk)
    return {
        "md5": md5.hexdigest(),
        "sha256": sha256.hexdigest(),
        "size_bytes": os.path.getsize(filepath)
    }


def aggregate_master_quarantined() -> str:
    print("---> Aggregating master quarantined dataset...")
    quar_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*", "quarantined.csv")))
    all_dfs = []
    for qf in quar_files:
        if os.path.getsize(qf) > 0:
            try:
                df = pd.read_csv(qf)
                if len(df) > 0:
                    all_dfs.append(df)
            except Exception as e:
                print(f"  Warning reading {qf}: {e}")
    
    master_quar_path = os.path.join(OUTPUT_DIR, "master_quarantined_all_models.csv")
    if all_dfs:
        master_quar = pd.concat(all_dfs, ignore_index=True)
        master_quar.to_csv(master_quar_path, index=False)
        print(f"Saved master quarantined dataset: {master_quar_path} ({len(master_quar):,} rows)")
    else:
        pd.DataFrame().to_csv(master_quar_path, index=False)
        print("No quarantined records found.")
    return master_quar_path


def main():
    print("=" * 80)
    print("FREEZING PRODUCTION EXTRACTION ARTIFACTS")
    print(f"Repository Root: {REPO_ROOT}")
    print("=" * 80)

    # 1. Source code module hashes
    src_dir = os.path.join(REPO_ROOT, "src", "extraction")
    src_files = sorted(glob.glob(os.path.join(src_dir, "*.py")))
    source_hashes = {}
    for sf in src_files:
        rel_path = os.path.relpath(sf, REPO_ROOT).replace("\\", "/")
        source_hashes[rel_path] = compute_file_hashes(sf)

    # 2. Compile master quarantined dataset
    master_quar_path = aggregate_master_quarantined()

    # 3. Master dataset hashes
    master_vp_path = os.path.join(OUTPUT_DIR, "master_viewpoints_all_models.csv")
    manifest_path = os.path.join(OUTPUT_DIR, "extraction_manifest.csv")
    raw_manifest_path = os.path.join(REPO_ROOT, "data", "raw_generation_manifest.csv")

    master_vp_hashes = compute_file_hashes(master_vp_path)
    master_quar_hashes = compute_file_hashes(master_quar_path)
    manifest_hashes = compute_file_hashes(manifest_path)
    raw_manifest_hashes = compute_file_hashes(raw_manifest_path)

    # Load master datasets for row and schema verification
    df_vp = pd.read_csv(master_vp_path)
    df_quar = pd.read_csv(master_quar_path) if os.path.exists(master_quar_path) and os.path.getsize(master_quar_path) > 0 else pd.DataFrame()
    df_manifest = pd.read_csv(manifest_path)

    # 4. Per-model artifacts
    models = sorted([d for d in os.listdir(OUTPUT_DIR) if os.path.isdir(os.path.join(OUTPUT_DIR, d)) and not d.startswith(".")])
    per_model_artifacts = {}
    for m in models:
        m_vp = os.path.join(OUTPUT_DIR, m, "viewpoints.csv")
        m_quar = os.path.join(OUTPUT_DIR, m, "quarantined.csv")
        per_model_artifacts[m] = {
            "viewpoints": compute_file_hashes(m_vp),
            "quarantined": compute_file_hashes(m_quar)
        }

    # 5. Configuration constants
    configuration_freeze = {
        "min_viewpoint_word_count": 6,
        "min_cjk_char_count": 15,
        "near_duplicate_jaccard_threshold": 0.75,
        "structural_precedence_order": [
            "Level-3 Markdown (###)",
            "Numbered Lists (1., 2.)",
            "Bulleted Lists (-, *, •)",
            "Standalone Bold Headings (**Heading**)",
            "Prose Paragraphs"
        ],
        "stance_taxonomy": ["support", "undermine", "unknown_or_ambiguous"],
        "quarantine_reasons": [
            "EMPTY_OR_WHITESPACE",
            "ABSTENTION",
            "EXECUTION_ERROR",
            "PROMPT_LEAKAGE",
            "HALLUCINATED_TASK",
            "OFF_TOPIC_LEAKAGE",
            "UNRECOVERABLE_TRUNCATION"
        ]
    }

    freeze_manifest = {
        "timestamp_utc": datetime.utcnow().isoformat(),
        "total_models": len(df_manifest),
        "total_raw_generations": int(df_manifest["raw_samples"].sum()),
        "total_viewpoints_extracted": len(df_vp),
        "total_quarantined_generations": len(df_quar),
        "silent_sample_drops": int(df_manifest["silent_drops"].sum()),
        "duplicate_primary_keys": int(df_manifest["duplicate_primary_keys"].sum()),
        "source_code_freeze": source_hashes,
        "pipeline_configuration": configuration_freeze,
        "master_artifacts": {
            "master_viewpoints_all_models.csv": {
                **master_vp_hashes,
                "total_rows": len(df_vp),
                "columns": list(df_vp.columns)
            },
            "master_quarantined_all_models.csv": {
                **master_quar_hashes,
                "total_rows": len(df_quar),
                "columns": list(df_quar.columns) if len(df_quar) > 0 else []
            },
            "extraction_manifest.csv": {
                **manifest_hashes,
                "total_rows": len(df_manifest),
                "columns": list(df_manifest.columns)
            },
            "raw_generation_manifest.csv": {
                **raw_manifest_hashes
            }
        },
        "per_model_artifacts": per_model_artifacts
    }

    out_json = os.path.join(OUTPUT_DIR, "production_freeze_manifest.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(freeze_manifest, f, indent=2)

    print("\n" + "=" * 80)
    print(f"PRODUCTION ARTIFACTS FROZEN SUCCESSFULLY")
    print(f"Freeze Manifest: {out_json}")
    print(f"Master Viewpoints Hash (SHA-256): {master_vp_hashes['sha256']}")
    print(f"Master Quarantined Hash (SHA-256): {master_quar_hashes['sha256']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
