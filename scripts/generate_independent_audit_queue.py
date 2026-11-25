import os
import json
import numpy as np
import pandas as pd

def main():
    df_341 = pd.read_csv('results/calibration/stage2_cross_encoder_predictions_341.csv')
    df_dis = pd.read_csv('results/calibration/disagreements_human_vs_llm_majority_341.csv')
    
    print(f"Total 341 pairs: {len(df_341)}")
    print(f"Total Disagreements: {len(df_dis)}")
    
    dis_pair_ids = set(df_dis['pair_id'])
    
    # Extract 55 disagreements
    dis_df = df_341[df_341['pair_id'].isin(dis_pair_ids)].copy()
    dis_df['audit_stratum'] = 'disagreement'
    
    # Extract agreements (human_label == llm_majority_resolved)
    agr_df = df_341[~df_341['pair_id'].isin(dis_pair_ids)].copy()
    print(f"Total Agreements available: {len(agr_df)}")
    
    # Stratified sampling of 35 agreements as control sample:
    # 15 Class 2, 15 Class 1, 3 Class 0a, 2 Class 0b
    np.random.seed(42)
    c2_agr = agr_df[agr_df['human_ground_truth_label'] == 'class_2'].sample(n=15, random_state=42)
    c1_agr = agr_df[agr_df['human_ground_truth_label'] == 'class_1'].sample(n=15, random_state=42)
    c0a_agr = agr_df[agr_df['human_ground_truth_label'] == 'class_0a'].sample(n=3, random_state=42)
    c0b_agr = agr_df[agr_df['human_ground_truth_label'] == 'class_0b'].sample(n=2, random_state=42)
    
    control_df = pd.concat([c2_agr, c1_agr, c0a_agr, c0b_agr]).copy()
    control_df['audit_stratum'] = 'control_agreement'
    
    print(f"Selected control sample: {len(control_df)} pairs")
    print(control_df['human_ground_truth_label'].value_counts())
    
    # Combine 55 + 35 = 90 pairs
    combined = pd.concat([dis_df, control_df]).copy()
    print(f"Total audit pool: {len(combined)} pairs")
    
    # Shuffle randomly to blind the auditor
    shuffled = combined.sample(frac=1.0, random_state=123).reset_index(drop=True)
    shuffled['audit_index'] = np.arange(1, len(shuffled) + 1)
    shuffled['batch_number'] = ((shuffled['audit_index'] - 1) // 10) + 1
    
    out_cols = [
        'audit_index', 'batch_number', 'pair_id', 'claim_id', 'claim_text',
        'gold_perspective_text', 'generated_viewpoint_text',
        'audit_stratum', 'human_ground_truth_label', 'jury_consensus_label'
    ]
    
    out_path = 'data/audit/independent_quality_audit_90_pairs.csv'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    shuffled[out_cols].to_csv(out_path, index=False)
    print(f"Saved 90-pair audit queue to {out_path}")
    
    # Print summary of batches
    for b in range(1, 10):
        b_df = shuffled[shuffled['batch_number'] == b]
        dis_cnt = (b_df['audit_stratum'] == 'disagreement').sum()
        ctrl_cnt = (b_df['audit_stratum'] == 'control_agreement').sum()
        print(f"Batch {b}: {len(b_df)} pairs ({dis_cnt} disagreements, {ctrl_cnt} control agreements)")

if __name__ == '__main__':
    main()
