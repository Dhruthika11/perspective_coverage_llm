import os
import json
import pandas as pd
import numpy as np

# Load datasets
df1700 = pd.read_csv('data/calibration/llm_jury_full_1700_annotated.csv')
df_bench = pd.read_csv('data/calibration/human_llm_benchmark_1700.csv')

# Merge cosine similarities
merged = pd.merge(df1700, df_bench[['pair_id', 'cosine_similarity']], on='pair_id', how='left')

# 1. Prepare Pure Human Ground Truth Dataset (N = 341 concrete pairs)
human_df = merged[merged['human_ground_truth_label'].isin(['class_0a', 'class_0b', 'class_1', 'class_2'])].copy()

# 2. Prepare Hybrid Gold Dataset (Human + Unanimous 3/3 Clean LLM)
unanimous_3_3_mask = (
    (merged['Gemini-3-Flash_label'] == merged['GPT-OSS-120B_label']) &
    (merged['GPT-OSS-120B_label'] == merged['Qwen-3.8-27B_label']) &
    (merged['Gemini-3-Flash_label'].isin(['class_0a', 'class_0b', 'class_1', 'class_2'])) &
    (~merged['Gemini-3-Flash_explanation'].str.contains('Rate limit|Fallback|timeout|transient', case=False, na=False)) &
    (~merged['GPT-OSS-120B_explanation'].str.contains('Rate limit|Fallback|timeout|transient', case=False, na=False)) &
    (~merged['Qwen-3.8-27B_explanation'].str.contains('Rate limit|Fallback|timeout|transient', case=False, na=False))
)

merged['hybrid_gold_label'] = merged['human_ground_truth_label']
for idx in merged[unanimous_3_3_mask].index:
    if pd.isna(merged.at[idx, 'hybrid_gold_label']):
        merged.at[idx, 'hybrid_gold_label'] = merged.at[idx, 'Gemini-3-Flash_label']

hybrid_df = merged[merged['hybrid_gold_label'].isin(['class_0a', 'class_0b', 'class_1', 'class_2'])].copy()

print(f"Pure Human Ground Truth Dataset: {len(human_df)} concrete pairs.")
print(f"Hybrid Gold Benchmark Dataset: {len(hybrid_df)} concrete pairs.")


def run_sweep(df, label_col, name="dataset"):
    sims = df['cosine_similarity'].values
    labels = df[label_col].values
    cids = df['claim_id'].values
    unique_claims = np.unique(cids)
    n_claims = len(unique_claims)
    
    c2_mask = (labels == 'class_2')
    c1_mask = (labels == 'class_1')
    c0a_mask = (labels == 'class_0a')
    c0b_mask = (labels == 'class_0b')
    
    n_c2 = np.sum(c2_mask)
    n_c1 = np.sum(c1_mask)
    n_c0a = np.sum(c0a_mask)
    n_c0b = np.sum(c0b_mask)
    
    thresholds = np.arange(0.50, 0.905, 0.01)
    rows = []
    
    for tau in thresholds:
        tau = round(float(tau), 2)
        passed = (sims >= tau)
        
        # True Positives, False Positives, False Negatives for Class 2
        tp = np.sum(passed & c2_mask)
        fp = np.sum(passed & (~c2_mask))
        fn = np.sum((~passed) & c2_mask)
        tn = np.sum((~passed) & (~c2_mask))
        
        rec_c2 = tp / max(tp + fn, 1)
        prec_c2 = tp / max(tp + fp, 1)
        f1_c2 = 2 * prec_c2 * rec_c2 / max(prec_c2 + rec_c2, 1e-8)
        
        rej_c1 = np.sum((~passed) & c1_mask) / max(n_c1, 1)
        rej_c0a = np.sum((~passed) & c0a_mask) / max(n_c0a, 1)
        rej_c0b = np.sum((~passed) & c0b_mask) / max(n_c0b, 1)
        rej_all_noise = (np.sum((~passed) & (~c2_mask))) / max(len(df) - n_c2, 1)
        
        n_pass = int(np.sum(passed))
        pass_rate = n_pass / len(df)
        
        rows.append({
            'tau': tau,
            'recall_c2': round(float(rec_c2), 4),
            'precision_c2': round(float(prec_c2), 4),
            'f1_c2': round(float(f1_c2), 4),
            'rejection_c1': round(float(rej_c1), 4),
            'rejection_c0a': round(float(rej_c0a), 4),
            'rejection_c0b': round(float(rej_c0b), 4),
            'rejection_all_noise': round(float(rej_all_noise), 4),
            'candidates_passed': n_pass,
            'pass_rate': round(float(pass_rate), 4)
        })
        
    res_df = pd.DataFrame(rows)
    
    # Claim-Clustered Bootstrap for 95% Confidence Intervals (1000 resamples)
    print(f"Running 1,000 claim-clustered bootstrap resamples for {name} ({n_claims} unique claims)...")
    claim_groups = {cid: np.where(cids == cid)[0] for cid in unique_claims}
    
    ci_low_rec = {}
    ci_high_rec = {}
    ci_low_f1 = {}
    ci_high_f1 = {}
    
    check_taus = [0.55, 0.60, 0.65, 0.68, 0.70, 0.72, 0.75, 0.77, 0.80]
    for tau in check_taus:
        tau = round(float(tau), 2)
        b_recalls = []
        b_f1s = []
        
        for _ in range(1000):
            sampled_cids = np.random.choice(unique_claims, size=n_claims, replace=True)
            sampled_idx = np.concatenate([claim_groups[cid] for cid in sampled_cids])
            
            sub_sims = sims[sampled_idx]
            sub_c2 = c2_mask[sampled_idx]
            sub_not_c2 = ~sub_c2
            
            pass_mask = (sub_sims >= tau)
            tp_b = np.sum(pass_mask & sub_c2)
            fp_b = np.sum(pass_mask & sub_not_c2)
            fn_b = np.sum((~pass_mask) & sub_c2)
            
            r_b = tp_b / max(tp_b + fn_b, 1)
            p_b = tp_b / max(tp_b + fp_b, 1)
            f1_b = 2 * p_b * r_b / max(p_b + r_b, 1e-8)
            
            b_recalls.append(r_b)
            b_f1s.append(f1_b)
            
        ci_low_rec[tau] = round(float(np.percentile(b_recalls, 2.5)), 4)
        ci_high_rec[tau] = round(float(np.percentile(b_recalls, 97.5)), 4)
        ci_low_f1[tau] = round(float(np.percentile(b_f1s, 2.5)), 4)
        ci_high_f1[tau] = round(float(np.percentile(b_f1s, 97.5)), 4)
        
    res_df['ci_95_rec_low'] = res_df['tau'].map(lambda t: ci_low_rec.get(round(t, 2), np.nan))
    res_df['ci_95_rec_high'] = res_df['tau'].map(lambda t: ci_high_rec.get(round(t, 2), np.nan))
    res_df['ci_95_f1_low'] = res_df['tau'].map(lambda t: ci_low_f1.get(round(t, 2), np.nan))
    res_df['ci_95_f1_high'] = res_df['tau'].map(lambda t: ci_high_f1.get(round(t, 2), np.nan))
    
    return res_df


# Run sweep on Pure Human Ground Truth
human_sweep = run_sweep(human_df, 'human_ground_truth_label', "Pure Human Benchmark (N=341)")
human_sweep.to_csv('results/calibration/human_ground_truth_calibration_sweep.csv', index=False)

# Run sweep on Hybrid Benchmark
hybrid_sweep = run_sweep(hybrid_df, 'hybrid_gold_label', "Hybrid Gold Benchmark (N=978)")
hybrid_sweep.to_csv('results/calibration/hybrid_gold_calibration_sweep.csv', index=False)

print("\n" + "="*80)
print("CALIBRATION RESULTS SUMMARY: PURE HUMAN GROUND TRUTH (N = 341)")
print("="*80)
key_taus = [0.55, 0.60, 0.65, 0.68, 0.70, 0.72, 0.75, 0.77, 0.80]
print(human_sweep[human_sweep['tau'].isin(key_taus)][['tau', 'recall_c2', 'ci_95_rec_low', 'ci_95_rec_high', 'precision_c2', 'f1_c2', 'rejection_all_noise', 'pass_rate']])

print("\n" + "="*80)
print("CALIBRATION RESULTS SUMMARY: HYBRID GOLD BENCHMARK (N = 978)")
print("="*80)
print(hybrid_sweep[hybrid_sweep['tau'].isin(key_taus)][['tau', 'recall_c2', 'ci_95_rec_low', 'ci_95_rec_high', 'precision_c2', 'f1_c2', 'rejection_all_noise', 'pass_rate']])
