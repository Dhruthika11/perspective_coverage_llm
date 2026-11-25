"""
Stratified Sampler for Stage C (N = 300 Additional Pairs).

Requirements:
1. Strict Disjointness:
   - Excludes all 50 pairs from Stage A (llm_jury_live_50_annotated.csv).
   - Excludes all 250 pairs from Stage B (dataset_b_stage_b_sampled_250.csv).
2. Stratification across:
   - 4 Semantic Classes: Class 2 (110), Class 1 (70), Class 0A (70), Class 0B (50).
   - 10 Model Families: Qwen (1.5B, 3B, 7B, 32B), Gemma (9B, 27B), LLaMA (3B, 8B), OLMo (7B, 13B).
   - 3 Length Groups: Short (<=45), Medium (46-75), Long (>75).
   - Boundary oversampling for difficult Class 2 vs Class 1 cases.
3. Fully reproducible with seed=43 (distinct seed from Stage B).
"""

import os
import pandas as pd
import numpy as np

def sample_stage_c(
    full_benchmark_path: str = "data/calibration/human_llm_benchmark_1700.csv",
    stage_a_path: str = "data/calibration/llm_jury_live_50_annotated.csv",
    stage_b_path: str = "data/calibration/dataset_b_stage_b_sampled_250.csv",
    out_path: str = "data/calibration/dataset_b_stage_c_sampled_300.csv",
    seed: int = 43
):
    df_all = pd.read_csv(full_benchmark_path)
    df_stage_a = pd.read_csv(stage_a_path)
    df_stage_b = pd.read_csv(stage_b_path)
    
    # 1. Strict Exclusion of Stage A and Stage B
    excluded_ids = set(df_stage_a['pair_id']).union(set(df_stage_b['pair_id']))
    df_pool = df_all[~df_all['pair_id'].isin(excluded_ids)].copy()
    
    print(f"Total benchmark pool: {len(df_all)}")
    print(f"Excluded from Stage A: {len(df_stage_a)}")
    print(f"Excluded from Stage B: {len(df_stage_b)}")
    print(f"Total excluded pairs: {len(excluded_ids)}")
    print(f"Remaining pool for Stage C: {len(df_pool)}")
    
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
    
    # Target Class Allocations (Total = 300)
    targets = {
        "Class 2": 110,
        "Class 1": 70,
        "Class 0A": 70,
        "Class 0B": 50
    }
    
    sampled_subsets = []
    
    for cls_name, n_target in targets.items():
        cls_df = df_pool[df_pool['ground_truth_class'] == cls_name]
        groups = cls_df.groupby(['source_model', 'length_tier'])
        per_group = max(1, n_target // len(groups))
        
        sampled_from_cls = []
        for (model, ltier), grp in groups:
            # Over-sample hard cases for Class 2 & Class 1
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
        
        if len(cls_sample) < n_target:
            needed = n_target - len(cls_sample)
            rem = cls_df[~cls_df['pair_id'].isin(cls_sample['pair_id'])]
            extra = rem.sample(n=min(needed, len(rem)), random_state=seed)
            cls_sample = pd.concat([cls_sample, extra])
        elif len(cls_sample) > n_target:
            cls_sample = cls_sample.sample(n=n_target, random_state=seed)
            
        sampled_subsets.append(cls_sample)
        
    df_stage_c = pd.concat(sampled_subsets).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    
    print("\n--- Stage C Stratification Summary ---")
    print("Total sampled:", len(df_stage_c))
    print("\nClass breakdown:\n", df_stage_c['ground_truth_class'].value_counts())
    print("\nModel breakdown:\n", df_stage_c['source_model'].value_counts())
    print("\nLength tier breakdown:\n", df_stage_c['length_tier'].value_counts())
    
    # Verify zero overlap with Stage A or Stage B
    overlap = set(df_stage_c['pair_id']).intersection(excluded_ids)
    assert len(overlap) == 0, f"Error: Found {len(overlap)} overlapping pairs with Stage A/B!"
    print(f"\nDisjointness verified: 0 overlapping pairs with Stage A or Stage B.")
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_stage_c.to_csv(out_path, index=False)
    print(f"Saved Stage C dataset to: {out_path}")
    return df_stage_c

if __name__ == "__main__":
    sample_stage_c()
