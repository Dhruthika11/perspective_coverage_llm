"""
Stratified Sampler for Stage B (N = 250 Pairs).

Requirements:
1. Strict disjointness: Excludes all 50 Stage A pairs.
2. Stratification across:
   - 4 Semantic Classes: Class 2 (90), Class 1 (60), Class 0A (60), Class 0B (40).
   - 10 Model Variants: Qwen (1.5B, 3B, 7B, 32B), Gemma (9B, 27B), LLaMA (3B, 8B), OLMo (7B, 13B).
   - 3 Length Groups: Short (<=45 words), Medium (46-75 words), Long (>75 words).
   - Hard boundaries included (Class 2 vs Class 1 semantic overlaps).
3. Fully reproducible with seed=42.
"""

import os
import pandas as pd
import numpy as np

def sample_stage_b(
    full_benchmark_path: str = "data/calibration/human_llm_benchmark_1700.csv",
    stage_a_path: str = "data/calibration/llm_jury_live_50_annotated.csv",
    out_path: str = "data/calibration/dataset_b_stage_b_sampled_250.csv",
    seed: int = 42
):
    df_all = pd.read_csv(full_benchmark_path)
    df_stage_a = pd.read_csv(stage_a_path)
    
    # 1. Strict Exclusion of Stage A
    stage_a_ids = set(df_stage_a['pair_id'])
    df_pool = df_all[~df_all['pair_id'].isin(stage_a_ids)].copy()
    print(f"Total benchmark pool: {len(df_all)}")
    print(f"Excluded Stage A pairs: {len(stage_a_ids)}")
    print(f"Remaining pool for Stage B: {len(df_pool)}")
    
    # Compute length tier
    df_pool['word_count'] = df_pool['generated_viewpoint_text'].apply(lambda x: len(str(x).split()))
    def get_length_tier(wc):
        if wc <= 45:
            return "short"
        elif wc <= 75:
            return "medium"
        else:
            return "long"
    df_pool['length_tier'] = df_pool['word_count'].apply(get_length_tier)
    
    # Target Class Allocations (Total = 250)
    targets = {
        "Class 2": 90,
        "Class 1": 60,
        "Class 0A": 60,
        "Class 0B": 40
    }
    
    sampled_subsets = []
    
    for cls_name, n_target in targets.items():
        cls_df = df_pool[df_pool['ground_truth_class'] == cls_name]
        
        # We want balanced representation across model families and length tiers
        # Group by (source_model, length_tier)
        groups = cls_df.groupby(['source_model', 'length_tier'])
        per_group = max(1, n_target // len(groups))
        
        sampled_from_cls = []
        for (model, ltier), grp in groups:
            # Over-sample hard cases for Class 2 & Class 1 (cosine between 0.50 and 0.70)
            if cls_name in ["Class 2", "Class 1"] and "cosine_similarity" in grp.columns:
                hard_cases = grp[(grp['cosine_similarity'] >= 0.50) & (grp['cosine_similarity'] <= 0.70)]
                if len(hard_cases) > 0:
                    take_hard = min(len(hard_cases), max(1, per_group // 2))
                    sampled_from_cls.append(hard_cases.sample(n=take_hard, random_state=seed))
                    remaining_grp = grp[~grp['pair_id'].isin(hard_cases['pair_id'])]
                    take_rest = min(len(remaining_grp), per_group - take_hard)
                    if take_rest > 0:
                        sampled_from_cls.append(remaining_grp.sample(n=take_rest, random_state=seed))
                else:
                    take = min(len(grp), per_group)
                    sampled_from_cls.append(grp.sample(n=take, random_state=seed))
            else:
                take = min(len(grp), per_group)
                sampled_from_cls.append(grp.sample(n=take, random_state=seed))
                
        cls_sample = pd.concat(sampled_from_cls).drop_duplicates(subset=['pair_id'])
        
        # If short of target, fill from remaining
        if len(cls_sample) < n_target:
            needed = n_target - len(cls_sample)
            rem = cls_df[~cls_df['pair_id'].isin(cls_sample['pair_id'])]
            extra = rem.sample(n=min(needed, len(rem)), random_state=seed)
            cls_sample = pd.concat([cls_sample, extra])
        elif len(cls_sample) > n_target:
            cls_sample = cls_sample.sample(n=n_target, random_state=seed)
            
        sampled_subsets.append(cls_sample)
        
    df_stage_b = pd.concat(sampled_subsets).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    
    print("\n--- Stage B Stratification Summary ---")
    print("Total sampled:", len(df_stage_b))
    print("\nClass breakdown:\n", df_stage_b['ground_truth_class'].value_counts())
    print("\nModel breakdown (all 10 models):\n", df_stage_b['source_model'].value_counts())
    print("\nLength tier breakdown:\n", df_stage_b['length_tier'].value_counts())
    
    # Verify zero overlap with Stage A
    overlap = set(df_stage_b['pair_id']).intersection(stage_a_ids)
    assert len(overlap) == 0, f"Error: Found {len(overlap)} overlapping pairs with Stage A!"
    print(f"\nDisjointness verified: 0 overlapping pairs with Stage A.")
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_stage_b.to_csv(out_path, index=False)
    print(f"Saved Stage B dataset to: {out_path}")
    return df_stage_b

if __name__ == "__main__":
    sample_stage_b()
