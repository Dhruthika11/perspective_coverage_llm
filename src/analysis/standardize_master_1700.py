"""
Standardize and Merge Master 1,700 Benchmark Dataset
- Merges 50 human ground-truth labels from Stage A AND 50 human ground-truth labels from Stage B (Total = 100 Human Ground-Truth Pairs).
- Standardizes consensus_label and consensus_type across all 1,700 pairs.
- Preserves distinct columns: jury_consensus_label, human_ground_truth_label, human_adjudication_rationale.
- Generates updated Master Human Triage Queue and Stratified Unanimous Audit.
"""

import os
import pandas as pd
import numpy as np

def main():
    master_path = "data/calibration/llm_jury_full_1700_annotated.csv"
    df = pd.read_csv(master_path)
    print(f"Loaded {len(df)} rows from {master_path}")
    
    # 1. Load Human Ground Truth from Stage A (50) and Stage B (50)
    df_stage_a = pd.read_csv("data/calibration/llm_jury_50_human_adjudicated.csv")
    df_stage_b = pd.read_csv("data/calibration/stage_b_50_human_adjudicated.csv")
    
    human_labels_map = {}
    human_rationales_map = {}
    
    for _, r in df_stage_a.iterrows():
        human_labels_map[int(r["pair_id"])] = str(r["human_ground_truth"]).lower()
        human_rationales_map[int(r["pair_id"])] = str(r.get("human_rationale", ""))
        
    for _, r in df_stage_b.iterrows():
        human_labels_map[int(r["pair_id"])] = str(r["human_label"]).lower()
        human_rationales_map[int(r["pair_id"])] = str(r.get("human_rationale", ""))
        
    print(f"Total human ground truth labels loaded: {len(human_labels_map)}")
    
    # 2. Recompute consensus_label and consensus_type consistently across all 1,700 pairs
    c_labels = []
    c_types = []
    h_labels = []
    h_rats = []
    h_reqs = []
    
    for idx, r in df.iterrows():
        pid = int(r["pair_id"])
        
        # Human ground truth if available
        if pid in human_labels_map:
            hl = human_labels_map[pid]
            hr = human_rationales_map.get(pid, "")
        else:
            hl = np.nan
            hr = ""
        h_labels.append(hl)
        h_rats.append(hr)
        
        v_gem = str(r.get("Gemini-3-Flash_label", "ambiguous")).lower()
        v_gpt = str(r.get("GPT-OSS-120B_label", "ambiguous")).lower()
        v_qwen = str(r.get("Qwen-3.8-27B_label", "ambiguous")).lower()
        
        votes = [v_gem, v_gpt, v_qwen]
        
        # Check unanimity
        if votes[0] == votes[1] == votes[2]:
            c_label = votes[0]
            c_type = "unanimous_3_of_3"
            h_req = False if c_label != "ambiguous" else True
        else:
            vc = pd.Series(votes).value_counts()
            top_label = vc.index[0]
            top_count = vc.iloc[0]
            
            if top_count == 2:
                c_label = top_label
                if "ambiguous" in votes:
                    c_type = "majority_with_ambiguous"
                    h_req = True
                else:
                    c_type = "majority_2_of_3"
                    h_req = True
            else:
                c_label = "split"
                c_type = "split_1_1_1"
                h_req = True
                
        c_labels.append(c_label)
        c_types.append(c_type)
        # If already human adjudicated, no further human adjudication required!
        if pd.notna(hl) and hl != "":
            h_reqs.append(False)
        else:
            h_reqs.append(h_req)
            
    df["jury_consensus_label"] = c_labels
    df["jury_consensus_type"] = c_types
    df["human_ground_truth_label"] = h_labels
    df["human_adjudication_rationale"] = h_rats
    df["human_adjudication_required"] = h_reqs
    
    # Clean up duplicate / legacy columns
    keep_cols = [
        "pair_id", "claim_id", "claim_text", "gold_perspective_text", "generated_viewpoint_text",
        "true_heuristic_class", "stage",
        "Gemini-3-Flash_label", "Gemini-3-Flash_conf", "Gemini-3-Flash_explanation",
        "GPT-OSS-120B_label", "GPT-OSS-120B_conf", "GPT-OSS-120B_explanation",
        "Qwen-3.8-27B_label", "Qwen-3.8-27B_conf", "Qwen-3.8-27B_explanation",
        "jury_consensus_label", "jury_consensus_type",
        "human_ground_truth_label", "human_adjudication_rationale",
        "human_adjudication_required"
    ]
    df_clean = df[[c for c in keep_cols if c in df.columns]].copy()
    
    df_clean.to_csv(master_path, index=False)
    print(f"Saved standardized master benchmark ({len(df_clean)} rows) to {master_path}")
    
    print("\nVerified Human Ground Truth Count:")
    print(df_clean["human_ground_truth_label"].value_counts(dropna=False))
    
    print("\nConsensus Type Distribution across 1,700:")
    print(df_clean["jury_consensus_type"].value_counts(dropna=False))
    
    # Export Triage Queue: pairs requiring human adjudication
    triage_df = df_clean[df_clean["human_adjudication_required"]].copy()
    triage_path = "data/calibration/llm_jury_full_1700_human_triage_queue.csv"
    triage_df.to_csv(triage_path, index=False)
    print(f"\nSaved Master Human Triage Queue ({len(triage_df)} pairs) to {triage_path}")
    
    # Export Stratified Unanimous Audit (60 pairs with 0 human ground truth currently)
    unanimous_pool = df_clean[
        (df_clean["jury_consensus_type"] == "unanimous_3_of_3") &
        (df_clean["human_ground_truth_label"].isna()) &
        (df_clean["jury_consensus_label"] != "ambiguous")
    ].copy()
    
    strata = []
    for cls in ["class_2", "class_1", "class_0a", "class_0b"]:
        sub = unanimous_pool[unanimous_pool["jury_consensus_label"] == cls]
        strata.append(sub.sample(n=min(15, len(sub)), random_state=42))
        
    audit_sample = pd.concat(strata).sample(frac=1.0, random_state=42).reset_index(drop=True)
    audit_path = "data/calibration/llm_jury_full_1700_stratified_unanimous_audit.csv"
    audit_sample.to_csv(audit_path, index=False)
    print(f"Saved Stratified Unanimous Audit Queue ({len(audit_sample)} pairs) to {audit_path}")
    
    # Sync back to Flair2
    print("\nMaster 1,700 benchmark files standardized successfully!")

if __name__ == "__main__":
    main()
