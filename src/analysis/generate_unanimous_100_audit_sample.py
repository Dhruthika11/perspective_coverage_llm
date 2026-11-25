"""
Generate 100-Pair Stratified Sample from 583 Auto-Accepted Unanimous Pairs
Excludes all 100 previously human-audited pairs.
Strategy:
- 60 randomly sampled pairs preserving the natural class distribution.
- 40 additional booster pairs ensuring coverage across all four classes, particularly Class 2.
"""

import json
import os
import pandas as pd
import numpy as np

def generate_sample(seed=42):
    master_path = "data/calibration/llm_jury_full_1700_annotated.csv"
    df = pd.read_csv(master_path)
    
    # Filter for unanimous 3/3 and exclude previously audited pairs
    unanimous_pool = df[
        (df["jury_consensus_type"] == "unanimous_3_of_3") & 
        (df["human_ground_truth_label"].isna())
    ].copy()
    
    print(f"Total unaudited unanimous pairs available: {len(unanimous_pool)}")
    print("Class distribution in available pool:")
    print(unanimous_pool["jury_consensus_label"].value_counts())
    
    # 1. Part 1: 60 pairs preserving natural class distribution
    # Class proportions: class_1 (25), class_0a (18), class_0b (11), class_2 (4), ambiguous (2) = 60
    part1_quotas = {
        "class_1": 25,
        "class_0a": 18,
        "class_0b": 11,
        "class_2": 4,
        "ambiguous": 2
    }
    
    part1_indices = []
    for cls, n in part1_quotas.items():
        cls_sub = unanimous_pool[unanimous_pool["jury_consensus_label"] == cls]
        chosen = cls_sub.sample(n=n, random_state=seed)
        part1_indices.extend(chosen.index.tolist())
        
    part1_df = unanimous_pool.loc[part1_indices].copy()
    part1_df["sampling_subgroup"] = "natural_distribution_60"
    
    # Remaining available pool after Part 1
    rem_pool = unanimous_pool.drop(part1_indices)
    
    # 2. Part 2: 40 additional pairs, allocated to ensure coverage of all four classes, particularly Class 2
    # Quotas: class_2 (22), class_0b (6), class_0a (6), class_1 (6) = 40
    part2_quotas = {
        "class_2": 22,
        "class_0b": 6,
        "class_0a": 6,
        "class_1": 6
    }
    
    part2_indices = []
    for cls, n in part2_quotas.items():
        cls_sub = rem_pool[rem_pool["jury_consensus_label"] == cls]
        chosen = cls_sub.sample(n=n, random_state=seed)
        part2_indices.extend(chosen.index.tolist())
        
    part2_df = rem_pool.loc[part2_indices].copy()
    part2_df["sampling_subgroup"] = "class_coverage_booster_40"
    
    # Combine and shuffle for blinded audit
    sample_df = pd.concat([part1_df, part2_df], ignore_index=True)
    sample_df = sample_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    sample_df.insert(0, "queue_index", range(1, len(sample_df) + 1))
    
    print("\n" + "="*60)
    print("FINAL 100-PAIR AUDIT SAMPLE SUMMARY:")
    print("Total pairs sampled:", len(sample_df))
    print("\nClass breakdown in 100 sample:")
    print(sample_df["jury_consensus_label"].value_counts())
    print("\nBreakdown by sampling subgroup:")
    print(pd.crosstab(sample_df["sampling_subgroup"], sample_df["jury_consensus_label"]))
    
    # Export 1: Blinded Human Adjudication Queue
    blinded_queue = sample_df[[
        "queue_index", "pair_id", "sampling_subgroup", "claim_id", 
        "claim_text", "gold_perspective_text", "generated_viewpoint_text"
    ]].copy()
    blinded_queue["human_label"] = ""
    blinded_queue["human_rationale"] = ""
    
    out_queue_path = "data/calibration/unanimous_100_audit_queue.csv"
    blinded_queue.to_csv(out_queue_path, index=False)
    blinded_queue.to_csv("unanimous_100_audit_queue.csv", index=False)
    print(f"Saved blinded queue to {out_queue_path} and unanimous_100_audit_queue.csv")
    
    # Export 2: Ground-Truth Key File (for scoring post-audit)
    key_df = sample_df[[
        "queue_index", "pair_id", "sampling_subgroup", "claim_id", 
        "claim_text", "gold_perspective_text", "generated_viewpoint_text",
        "jury_consensus_label", "jury_consensus_type",
        "Gemini-3-Flash_label", "GPT-OSS-120B_label", "Qwen-3.8-27B_label",
        "Gemini-3-Flash_explanation", "GPT-OSS-120B_explanation", "Qwen-3.8-27B_explanation"
    ]].copy()
    out_key_path = "data/calibration/unanimous_100_audit_ground_truth_key.csv"
    key_df.to_csv(out_key_path, index=False)
    print(f"Saved ground truth key to {out_key_path}")
    
    # Export 3: JSON Manifest
    manifest = {
        "total_sample_size": len(sample_df),
        "seed": seed,
        "sampling_subgroups": {
            "natural_distribution_60": part1_quotas,
            "class_coverage_booster_40": part2_quotas
        },
        "total_class_distribution": sample_df["jury_consensus_label"].value_counts().to_dict(),
        "zero_contamination_verified": bool(len(set(sample_df["pair_id"]).intersection(set(df[df["human_ground_truth_label"].notna()]["pair_id"]))) == 0),
        "files": {
            "blinded_queue": out_queue_path,
            "ground_truth_key": out_key_path
        }
    }
    with open("results/calibration/unanimous_100_audit_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print("Saved manifest to results/calibration/unanimous_100_audit_manifest.json")
    
    return sample_df

if __name__ == "__main__":
    generate_sample()
