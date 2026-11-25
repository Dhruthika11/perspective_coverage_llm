import os
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def clean_dict(d):
    clean = {}
    for k, v in d.items():
        if isinstance(v, (np.integer, int)):
            clean[k] = int(v)
        elif isinstance(v, (np.floating, float)):
            clean[k] = round(float(v), 6)
        elif isinstance(v, np.ndarray):
            clean[k] = v.tolist()
        elif isinstance(v, dict):
            clean[k] = clean_dict(v)
        else:
            clean[k] = v
    return clean

def compute_soft(p0, reas, tau=0.65, lam=0.40):
    diff = np.maximum(0.0, tau - reas)
    return np.clip(p0 - lam * diff, 0.0, 1.0)

def main():
    print("=" * 100)
    print("COMPREHENSIVE AUDIT: M_SOFT VS M0 ON USER-LABELED HARD PAIRS")
    print("=" * 100)
    
    # -------------------------------------------------------------------------
    # 1. 100-PAIR UNIFIED HUMAN AUDIT BENCHMARK
    # -------------------------------------------------------------------------
    u100 = pd.read_csv("data/audit/unified_production_audit_benchmark_100_enriched_e3.csv")
    u100['score_soft'] = compute_soft(u100['prob_m0'].values, u100['reasoning_cosine_v1_1'].values)
    u100['pass_m0'] = u100['pass_s2a'] & (u100['prob_m0'] >= 0.20)
    u100['pass_soft'] = u100['pass_s2a'] & (u100['score_soft'] >= 0.20)
    u100['decision_diff'] = u100['pass_m0'] != u100['pass_soft']
    u100['score_delta'] = u100['score_soft'] - u100['prob_m0']
    
    # Detailed inspection of specially noted pairs in Batch 1
    notes_filter = u100['human_audit_notes'].str.contains('class1|reject|perception|animal|contraceptive|moral', case=False, na=False)
    special_notes = u100[notes_filter].copy()
    
    special_pairs_list = []
    for _, r in special_notes.iterrows():
        special_pairs_list.append({
            "pair_id": int(r['pair_id']),
            "claim": r['claim_text'],
            "gold": r['gold_perspective_text'],
            "generated": r['generated_viewpoint_text'],
            "user_label": r['human_audit_decision'],
            "user_notes": r['human_audit_notes'],
            "reasoning_cosine": float(r['reasoning_cosine_v1_1']),
            "m0_prob": float(r['prob_m0']),
            "m0_decision": "ACCEPT" if r['pass_m0'] else "REJECT",
            "m_soft_score": float(r['score_soft']),
            "m_soft_decision": "ACCEPT" if r['pass_soft'] else "REJECT",
            "score_delta": float(r['score_delta'])
        })
        
    # Class breakdown on u100
    u100_class_summary = {}
    for cls in ['class_2', 'class_1', 'class_0a', 'class_0b']:
        sub = u100[u100['human_audit_decision'] == cls]
        m0_acc = int(sub['pass_m0'].sum())
        soft_acc = int(sub['pass_soft'].sum())
        u100_class_summary[cls] = {
            "n": len(sub),
            "m0_accepted": m0_acc,
            "m0_accept_rate": float(m0_acc / len(sub)),
            "m0_mean_score": float(sub['prob_m0'].mean()),
            "m_soft_accepted": soft_acc,
            "m_soft_accept_rate": float(soft_acc / len(sub)),
            "m_soft_mean_score": float(sub['score_soft'].mean()),
            "mean_reasoning_cosine": float(sub['reasoning_cosine_v1_1'].mean()),
            "penalized_pairs": int((sub['score_delta'] < 0).sum()),
            "penalized_rate": float((sub['score_delta'] < 0).sum() / len(sub)),
            "mean_score_delta": float(sub['score_delta'].mean())
        }

    # -------------------------------------------------------------------------
    # 2. 341-PAIR FROZEN HUMAN GROUND TRUTH BENCHMARK
    # -------------------------------------------------------------------------
    df341 = pd.read_csv("results/calibration/e3_1_pairwise_features_341.csv")
    bm = pd.read_csv("data/calibration/frozen_human_ground_truth_benchmark_346.csv")
    df341 = df341.merge(bm[['pair_id', 'jury_consensus_type', 'human_adjudication_rationale', 'true_heuristic_class']], on='pair_id', how='left')
    
    train_c2c1 = df341[(df341['partition'] == 'train') & (df341['human_ground_truth_label'].isin(['class_2', 'class_1']))]
    feats = ['cosine_similarity', 'p_entail', 'p_neutral']
    sc = StandardScaler()
    X_tr = sc.fit_transform(train_c2c1[feats].values)
    y_tr = (train_c2c1['human_ground_truth_label'] == 'class_2').astype(int).values
    clf = LogisticRegression(C=1.0, random_state=42).fit(X_tr, y_tr)
    
    df341['prob_m0'] = clf.predict_proba(sc.transform(df341[feats].values))[:, 1]
    df341['score_soft'] = compute_soft(df341['prob_m0'].values, df341['reasoning_cosine_v1_1'].values)
    
    is_contra = (
        ((df341['stance_gold'] == 'support') & (df341['stance_generated'] == 'undermine')) |
        ((df341['stance_gold'] == 'undermine') & (df341['stance_generated'] == 'support'))
    )
    df341['s2a_pass'] = (df341['cosine_similarity'] >= 0.55) & (~is_contra)
    df341['pass_m0'] = df341['s2a_pass'] & (df341['prob_m0'] >= 0.20)
    df341['pass_soft'] = df341['s2a_pass'] & (df341['score_soft'] >= 0.20)
    df341['decision_diff'] = df341['pass_m0'] != df341['pass_soft']
    df341['score_delta'] = df341['score_soft'] - df341['prob_m0']
    
    # 3-Way Triage Splits
    split_3way = df341[df341['jury_consensus_type'] == 'split_1_1_1'].copy()
    triage_summary = {}
    for cls in ['class_2', 'class_1', 'class_0a']:
        s = split_3way[split_3way['human_ground_truth_label'] == cls]
        triage_summary[cls] = {
            "n": len(s),
            "m0_accepted": int(s['pass_m0'].sum()),
            "m_soft_accepted": int(s['pass_soft'].sum()),
            "penalized": int((s['score_delta'] < 0).sum()),
            "mean_reas_cos": float(s['reasoning_cosine_v1_1'].mean())
        }

    # All Class 1 Nuance Pairs by partition
    c1_summary = {}
    for part in ['train', 'calibration', 'test', 'overall']:
        if part == 'overall':
            s = df341[df341['human_ground_truth_label'] == 'class_1']
        else:
            s = df341[(df341['human_ground_truth_label'] == 'class_1') & (df341['partition'] == part)]
        m0_rej = int((~s['pass_m0']).sum())
        soft_rej = int((~s['pass_soft']).sum())
        c1_summary[part] = {
            "n": len(s),
            "m0_rejected": m0_rej,
            "m0_rejection_rate": float(m0_rej / len(s)),
            "m_soft_rejected": soft_rej,
            "m_soft_rejection_rate": float(soft_rej / len(s)),
            "net_recovered": soft_rej - m0_rej,
            "penalized": int((s['score_delta'] < 0).sum()),
            "mean_reas_cos": float(s['reasoning_cosine_v1_1'].mean())
        }
        
    # All 4 overturned pairs in 341 benchmark
    flips_341 = df341[df341['decision_diff']].copy()
    flips_list = []
    for _, r in flips_341.iterrows():
        flips_list.append({
            "pair_id": int(r['pair_id']),
            "partition": r['partition'],
            "true_label": r['human_ground_truth_label'],
            "jury_consensus_type": r['jury_consensus_type'],
            "claim": r['claim_text'],
            "gold": r['gold_perspective_text'],
            "generated": r['generated_viewpoint_text'],
            "reasoning_cosine": float(r['reasoning_cosine_v1_1']),
            "m0_prob": float(r['prob_m0']),
            "m0_decision": "ACCEPT" if r['pass_m0'] else "REJECT",
            "m_soft_score": float(r['score_soft']),
            "m_soft_decision": "ACCEPT" if r['pass_soft'] else "REJECT",
            "human_rationale": r['human_adjudication_rationale']
        })

    # Save to JSON
    report = {
        "unified_100_human_audit": {
            "class_summary": u100_class_summary,
            "specially_noted_hard_pairs": special_pairs_list,
            "flipped_pairs": [{
                "pair_id": int(r['pair_id']),
                "claim": r['claim_text'],
                "gold": r['gold_perspective_text'],
                "generated": r['generated_viewpoint_text'],
                "user_label": r['human_audit_decision'],
                "user_notes": r['human_audit_notes'],
                "reasoning_cosine": float(r['reasoning_cosine_v1_1']),
                "m0_prob": float(r['prob_m0']),
                "m_soft_score": float(r['score_soft'])
            } for _, r in u100[u100['decision_diff']].iterrows()]
        },
        "frozen_341_human_benchmark": {
            "triage_3way_splits": triage_summary,
            "class_1_nuance_by_partition": c1_summary,
            "overturned_pairs": flips_list
        }
    }
    
    out_path = "results/calibration/m_soft_vs_m0_hard_pairs_comparison.json"
    with open(out_path, "w") as f:
        json.dump(clean_dict(report), f, indent=2)
    print(f"\nSaved detailed comparison to {out_path}")
    print("DONE!")

if __name__ == "__main__":
    main()
