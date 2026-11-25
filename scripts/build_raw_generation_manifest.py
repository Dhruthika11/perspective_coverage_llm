#!/usr/bin/env python3
"""
Build Raw Generation Manifest for Perspective Coverage.

Inventories all raw LLM generations across all 34 models and 907 Perspectrum claims.
Verifies completeness (154,190 expected raw generations = 34 models * 907 claims * 5 samples).
Classifies raw generation formats, computes SHA-256 hashes, and records provenance.

Usage:
    python scripts/build_raw_generation_manifest.py
"""

import os
import re
import sys
import glob
import json
import hashlib
import pandas as pd
from datetime import datetime


def detect_status(text: str) -> str:
    """Classify the generation status into deterministic categories."""
    if text is None:
        return "EMPTY"
    t = str(text).strip()
    if not t:
        return "EMPTY"

    # Corrupted / Execution Failure
    if '"status": "failed"' in t or '"status":"failed"' in t or "traceback (most recent call last)" in t.lower():
        return "CORRUPTED"

    # Explicit Abstention
    if '"status": "abstain"' in t or '"status":"abstain"' in t or '"status": "abstained"' in t or '"status":"abstained"' in t:
        return "ABSTAIN"
    
    t_lower = t.lower()
    if len(t.split()) < 30 and ("i cannot fulfill" in t_lower or "i am unable to provide" in t_lower) and "viewpoint" not in t_lower:
        return "ABSTAIN"

    # Structured JSON
    t_clean = re.sub(r'^```(?:json)?\s*', '', t, flags=re.IGNORECASE).strip()
    t_clean = re.sub(r'\s*```$', '', t_clean).strip()
    has_json_keys = bool(re.search(r'"(stance|perspective_summary|perspective|argument|summary|claim)"\s*:', t_clean))
    if "{" in t_clean and "}" in t_clean and has_json_keys:
        return "JSON_STRUCTURED"

    # Multi-Viewpoint List (numbered items or markdown aspect headers)
    has_numbered_list = bool(
        re.search(r'(?:\A|\n+)\s*(?:1[\.\)]|\bViewpoint\s+1|\bPerspective\s+1)\b', t, flags=re.IGNORECASE) and
        re.search(r'\n+\s*(?:2[\.\)]|\bViewpoint\s+2|\bPerspective\s+2)\b', t, flags=re.IGNORECASE)
    )
    has_aspect_headers = bool(
        re.search(r'###\s+[A-Za-z]+', t) or
        re.search(r'\*\*[A-Za-z\s]+Perspective:?\*\*', t)
    )
    if has_numbered_list or has_aspect_headers:
        return "MULTI_VIEWPOINT_LIST"

    return "NORMAL_PROSE"


def main():
    # Detect base directory
    if os.path.exists(os.path.expanduser("~/perspective_coverage")):
        base_dir = os.path.expanduser("~/perspective_coverage")
    else:
        base_dir = os.getcwd()

    print("=" * 90)
    print(f"BUILDING RAW GENERATION MANIFEST: {datetime.now().isoformat()}")
    print(f"Base Directory: {base_dir}")
    print("=" * 90)

    claims_file = os.path.join(base_dir, "data", "processed", "claims.csv")
    if not os.path.exists(claims_file):
        # Fallback to data/raw/perspectrum/perspectrum_claims.json if exists
        alt_claims = os.path.join(base_dir, "data", "raw", "perspectrum", "perspectrum_claims.json")
        if os.path.exists(alt_claims):
            with open(alt_claims, "r", encoding="utf-8") as f:
                claims_data = json.load(f)
            all_claim_ids = sorted([c["cId"] for c in claims_data])
        else:
            raise FileNotFoundError(f"Could not locate claims file at {claims_file} or {alt_claims}")
    else:
        claims_df = pd.read_csv(claims_file)
        all_claim_ids = sorted(claims_df["claim_id"].unique())

    num_claims = len(all_claim_ids)
    expected_samples_per_claim = 5
    expected_per_model = num_claims * expected_samples_per_claim
    print(f"Loaded {num_claims} claims. Expected per model: {expected_per_model} generations.")

    raw_dir = os.path.join(base_dir, "data", "llm_outputs", "raw_generations")
    if not os.path.exists(raw_dir):
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")

    # Discover models (exclude benchmarks ibm_debater and kialo)
    candidate_dirs = sorted([
        d for d in glob.glob(os.path.join(raw_dir, "*"))
        if os.path.isdir(d) and os.path.basename(d) not in ["ibm_debater", "kialo"]
    ])
    model_keys = [os.path.basename(d) for d in candidate_dirs]
    print(f"Found {len(model_keys)} candidate models in raw_generations.")

    manifest_rows = []
    model_summaries = []

    for m_idx, model_key in enumerate(model_keys, 1):
        m_dir = os.path.join(raw_dir, model_key)
        csv_file = os.path.join(m_dir, "raw_generations.csv")

        # Load CSV if present
        raw_csv_map = {}
        if os.path.exists(csv_file):
            try:
                df_csv = pd.read_csv(csv_file)
                text_col = "generated_text" if "generated_text" in df_csv.columns else (
                    "argument" if "argument" in df_csv.columns else "combined_text"
                )
                for idx, r in df_csv.iterrows():
                    cid = r.get("claim_id")
                    sid = r.get("sample_id", idx % 5)
                    txt = r.get(text_col)
                    if pd.notna(cid) and pd.notna(sid):
                        raw_csv_map[(int(cid), int(sid))] = (str(txt) if pd.notna(txt) else "", csv_file)
            except Exception as e:
                print(f"  Warning: failed to read {csv_file}: {e}")

        # JSON checkpoint fallback map
        json_fallback_files = glob.glob(os.path.join(m_dir, "*_samples.json"))
        json_map = {}
        if len(raw_csv_map) < expected_per_model and json_fallback_files:
            for jf in json_fallback_files:
                try:
                    cid_part = os.path.basename(jf).split("_")[0]
                    cid = int(cid_part)
                    with open(jf, "r", encoding="utf-8") as fh:
                        jdata = json.load(fh)
                    samples = jdata.get("samples", [])
                    for sid, s_text in enumerate(samples):
                        json_map[(cid, sid)] = (str(s_text) if s_text is not None else "", jf)
                except Exception:
                    pass

        # Inventory every expected (claim_id, sample_id)
        found_count = 0
        status_counts = {}
        model_rows = []

        for cid in all_claim_ids:
            for sid in range(expected_samples_per_claim):
                pair = (cid, sid)
                text = None
                source_path = None

                if pair in raw_csv_map:
                    text, source_path = raw_csv_map[pair]
                elif pair in json_map:
                    text, source_path = json_map[pair]

                exists = (text is not None and len(text.strip()) > 0)
                if exists:
                    found_count += 1
                    status = detect_status(text)
                    t_len = len(text)
                    t_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
                else:
                    status = "EMPTY" if text is not None else "MISSING"
                    t_len = 0
                    t_hash = ""
                    source_path = source_path or os.path.relpath(csv_file, base_dir)

                status_counts[status] = status_counts.get(status, 0) + 1

                # Normalize source path relative to base_dir
                rel_path = os.path.relpath(source_path, base_dir) if source_path else f"data/llm_outputs/raw_generations/{model_key}/raw_generations.csv"

                model_rows.append({
                    "model_key": model_key,
                    "claim_id": cid,
                    "sample_id": sid,
                    "raw_file_path": rel_path.replace("\\", "/"),
                    "raw_record_id": f"{model_key}__c{cid}__s{sid}",
                    "generation_exists": exists,
                    "generation_status": status,
                    "raw_text_length": t_len,
                    "raw_text_hash": t_hash
                })

        manifest_rows.extend(model_rows)

        summary_row = {
            "model_key": model_key,
            "expected": expected_per_model,
            "found": found_count,
            "missing": expected_per_model - found_count,
            "json_structured": status_counts.get("JSON_STRUCTURED", 0),
            "multi_vp_list": status_counts.get("MULTI_VIEWPOINT_LIST", 0),
            "normal_prose": status_counts.get("NORMAL_PROSE", 0),
            "abstain": status_counts.get("ABSTAIN", 0),
            "corrupted": status_counts.get("CORRUPTED", 0),
            "empty": status_counts.get("EMPTY", 0)
        }
        model_summaries.append(summary_row)
        pct = (found_count / expected_per_model) * 100
        print(f"[{m_idx:02d}/{len(model_keys):02d}] {model_key:<32}: {found_count}/{expected_per_model} ({pct:.1f}%) | "
              f"JSON: {summary_row['json_structured']} | Multi-VP: {summary_row['multi_vp_list']} | "
              f"Prose: {summary_row['normal_prose']} | Abstain: {summary_row['abstain']} | Corrupt: {summary_row['corrupted']}")

    # Save manifest
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = os.path.join(base_dir, "data", "raw_generation_manifest.csv")
    manifest_df.to_csv(manifest_path, index=False)
    print("\n" + "=" * 90)
    print(f"SUCCESS: Saved manifest to {manifest_path}")
    print(f"Total rows in manifest: {len(manifest_df):,}")
    print(f"Total generations existing: {manifest_df['generation_exists'].sum():,} / {len(manifest_df):,}")

    # Save summary report
    summary_df = pd.DataFrame(model_summaries)
    summary_path = os.path.join(base_dir, "data", "raw_generation_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary to {summary_path}")
    print("=" * 90)

    # Print overall breakdown
    print("\nOVERALL GENERATION STATUS BREAKDOWN:")
    status_summary = manifest_df["generation_status"].value_counts()
    for st, count in status_summary.items():
        print(f"  {st:<22}: {count:7,d} ({count / len(manifest_df) * 100:5.2f}%)")
    print("=" * 90)


if __name__ == "__main__":
    main()
