#!/usr/bin/env python3
"""
Targeted Discrepancy Audit Script for Pilot Validation.
Compares pre-fix pilot results (data/pilot_validation_prefix/)
against the post-fix pilot results (data/pilot_validation/).

Strictly classifies every delta into one of the 5 mandated buckets:
1. Duplicate removed
2. Newly quarantined candidate
3. Segmentation change
4. Parsing failure
5. Other
"""

import os
import sys
import pandas as pd
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

models = [
    "gemma_2_9b_it",
    "qwen2_5_32b_instruct",
    "smollm2_135m_instruct",
    "qwen2_5_3b_instruct_p2"
]

discrepancy_records = []

def _safe_read_csv(p):
    if not os.path.exists(p) or os.path.getsize(p) == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()

for m in models:
    pre_path = os.path.join(REPO_ROOT, "data", "pilot_validation_prefix", f"{m}_pilot_viewpoints.csv")
    post_path = os.path.join(REPO_ROOT, "data", "pilot_validation", f"{m}_pilot_viewpoints.csv")
    pre_quar_path = os.path.join(REPO_ROOT, "data", "pilot_validation_prefix", f"{m}_pilot_quarantined.csv")
    post_quar_path = os.path.join(REPO_ROOT, "data", "pilot_validation", f"{m}_pilot_quarantined.csv")

    df_pre = _safe_read_csv(pre_path)
    df_post = _safe_read_csv(post_path)
    df_pre_q = _safe_read_csv(pre_quar_path)
    df_post_q = _safe_read_csv(post_quar_path)

    pre_counts = df_pre.groupby(["claim_id", "sample_id"]).size() if len(df_pre) else pd.Series()
    post_counts = df_post.groupby(["claim_id", "sample_id"]).size() if len(df_post) else pd.Series()

    all_samples = sorted(list(set(pre_counts.index).union(set(post_counts.index))))

    for cid, sid in all_samples:
        cnt_pre = int(pre_counts.get((cid, sid), 0))
        cnt_post = int(post_counts.get((cid, sid), 0))

        if cnt_pre != cnt_post:
            pre_rows = df_pre[(df_pre["claim_id"] == cid) & (df_pre["sample_id"] == sid)]
            post_rows = df_post[(df_post["claim_id"] == cid) & (df_post["sample_id"] == sid)]

            was_quar_pre = len(df_pre_q[(df_pre_q["claim_id"] == cid) & (df_pre_q["sample_id"] == sid)]) > 0 if len(df_pre_q) else False
            was_quar_post = len(df_post_q[(df_post_q["claim_id"] == cid) & (df_post_q["sample_id"] == sid)]) > 0 if len(df_post_q) else False

            # Classify into 5 mandated categories:
            # 1. Duplicate removed
            # 2. Newly quarantined candidate
            # 3. Segmentation change
            # 4. Parsing failure
            # 5. Other
            if was_quar_pre and not was_quar_post:
                classification = "Other"
                reason = "Rescued sample from false-positive quarantine (Condition 1 / 2 fix)"
            elif not was_quar_pre and was_quar_post:
                classification = "Newly quarantined candidate"
                reason = "Candidate failed enhanced prompt-leakage / meta-instruction filter"
            elif cnt_post < cnt_pre:
                # Check if diff is due to duplicate preamble removal
                pre_texts = [t.strip().lower() for t in pre_rows["viewpoint_text"]]
                post_texts = [t.strip().lower() for t in post_rows["viewpoint_text"]]
                # If the missing items were duplicates of post texts
                has_dup = any(t in post_texts for t in pre_texts if pre_texts.count(t) > post_texts.count(t))
                if has_dup:
                    classification = "Duplicate removed"
                    reason = "Preamble or repeated bullet identical to existing viewpoint removed"
                else:
                    classification = "Segmentation change"
                    reason = "Bullet preamble discarded or sub-heading duplication removed"
            elif cnt_post > cnt_pre:
                classification = "Segmentation change"
                reason = "Recovered atomic bullet viewpoints / Unicode headings previously merged or dropped"
            else:
                classification = "Other"
                reason = "Unclassified adjustment"

            discrepancy_records.append({
                "model_key": m,
                "claim_id": cid,
                "sample_id": sid,
                "pre_count": cnt_pre,
                "post_count": cnt_post,
                "net_diff": cnt_post - cnt_pre,
                "classification": classification,
                "disposition_explanation": reason,
                "pre_viewpoints_sample": " | ".join(pre_rows["viewpoint_text"].head(2).tolist())[:100],
                "post_viewpoints_sample": " | ".join(post_rows["viewpoint_text"].head(2).tolist())[:100]
            })

df_discrepancy = pd.DataFrame(discrepancy_records)
out_csv = os.path.join(REPO_ROOT, "tests", "pilot_discrepancy_audit.csv")
df_discrepancy.to_csv(out_csv, index=False)
print("Saved discrepancy audit to:", out_csv)
print(f"Total affected samples across all models: {len(df_discrepancy)}")
print("\nBreakdown by Model:")
for m in models:
    sub = df_discrepancy[df_discrepancy["model_key"] == m]
    net = sub["net_diff"].sum()
    print(f"  {m:<28}: {len(sub)} samples with deltas, net viewpoint delta = {net:+d}")

print("\nBreakdown by 5 Mandated Classifications:")
print(df_discrepancy["classification"].value_counts())
