"""
Experiment 2: Calibrate Length Guard Separately (Blind to Test Labels)
Protocol:
  - Train: 32 claims, 104 pairs (51 C2, 53 C1)
  - Calibration: 16 claims, 100 pairs (24 C2, 45 C1, 23 C0a, 8 C0b)
  - ZERO access to held-out test labels during calibration!
Evaluates candidate cutoffs L_cutoff in [6, 8, 10, 12, 14, 16] and fallback rules.
"""

import json
import re
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

CONNECTIVES = [
    r'\bbecause\b', r'\bsince\b', r'\bas\b', r'\bdue to\b', r'\bby\b',
    r'\bleading to\b', r'\bresults in\b', r'\bthereby\b', r'\bwhich means\b',
    r'\bin order to\b', r'\bso that\b', r'\bthus\b', r'\btherefore\b'
]
CONN_RE = re.compile('|'.join(CONNECTIVES), re.IGNORECASE)

def has_connective(text):
    return bool(CONN_RE.search(text))

def get_word_count(text):
    return len(text.strip().split())

def main():
    print("=" * 90)
    print("EXPERIMENT 2: CALIBRATE LENGTH GUARD STRICTLY ON CALIBRATION PARTITION")
    print("=" * 90)
    
    # Load 341 benchmark features
    df = pd.read_csv("results/calibration/e3_1_pairwise_features_341.csv")
    
    train = df[(df['partition'] == 'train') & (df['human_ground_truth_label'].isin(['class_2', 'class_1']))].copy()
    cal = df[df['partition'] == 'calibration'].copy()
    test = df[df['partition'] == 'test'].copy()
    
    # Train M0 and M2 on Train split
    y_tr = (train['human_ground_truth_label'] == 'class_2').astype(int).values
    
    feats_m0 = ['cosine_similarity', 'p_entail', 'p_neutral']
    sc_m0 = StandardScaler()
    X_tr_m0 = sc_m0.fit_transform(train[feats_m0].values)
    clf_m0 = LogisticRegression(C=1.0, random_state=42).fit(X_tr_m0, y_tr)
    
    feats_m2 = ['cosine_similarity', 'p_entail', 'p_neutral', 'reasoning_cosine_v1_1']
    sc_m2 = StandardScaler()
    X_tr_m2 = sc_m2.fit_transform(train[feats_m2].values)
    clf_m2 = LogisticRegression(C=1.0, random_state=42).fit(X_tr_m2, y_tr)
    
    # Generate probabilities for Calibration and Test
    cal['prob_m0'] = clf_m0.predict_proba(sc_m0.transform(cal[feats_m0].values))[:, 1]
    cal['prob_m2'] = clf_m2.predict_proba(sc_m2.transform(cal[feats_m2].values))[:, 1]
    
    test['prob_m0'] = clf_m0.predict_proba(sc_m0.transform(test[feats_m0].values))[:, 1]
    test['prob_m2'] = clf_m2.predict_proba(sc_m2.transform(test[feats_m2].values))[:, 1]
    
    # Word count and connective flags
    for d in [cal, test]:
        d['gold_word_count'] = d['gold_perspective_text'].apply(get_word_count)
        d['gold_has_conn'] = d['gold_perspective_text'].apply(has_connective)
        
        # Stance concordance
        is_contra = (
            ((d['stance_gold'] == 'support') & (d['stance_generated'] == 'undermine')) |
            ((d['stance_gold'] == 'undermine') & (d['stance_generated'] == 'support'))
        )
        s1 = d['cosine_similarity'] >= 0.55
        d['s2a_pass'] = s1 & (~is_contra)
        
    print("\nCalibration Partition Gold Perspective Word Count Distribution:")
    print(cal['gold_word_count'].describe().to_string())
    
    c2_cal = (cal['human_ground_truth_label'] == 'class_2')
    c1_cal = (cal['human_ground_truth_label'] == 'class_1')
    c0a_cal = (cal['human_ground_truth_label'] == 'class_0a')
    c0b_cal = (cal['human_ground_truth_label'] == 'class_0b')
    tot_c2 = c2_cal.sum()
    
    # Calibration sweep: test cutoffs L in [6, 8, 10, 12, 14, 16] and rules
    # Fixed threshold for probe is T* = 0.18 (matched recall floor on calibration)
    t_probe = 0.18
    
    candidate_records = []
    
    # 0. Pure M0 baseline on Calibration
    p_pass_m0 = cal['s2a_pass'] & (cal['prob_m0'] >= 0.20)
    tp0 = (p_pass_m0 & c2_cal).sum()
    fp0 = (p_pass_m0 & (~c2_cal)).sum()
    rec0 = tp0 / tot_c2
    prec0 = tp0 / max(tp0 + fp0, 1)
    f10 = 2 * prec0 * rec0 / max(prec0 + rec0, 1e-8)
    rej_c1_0 = ((~p_pass_m0) & c1_cal).sum() / c1_cal.sum()
    
    candidate_records.append({
        "Rule": "Baseline M0 (No Guard)",
        "L_Cutoff": 0,
        "Cal_Recall": round(rec0 * 100, 2),
        "Cal_Precision": round(prec0 * 100, 2),
        "Cal_F1": round(f10, 4),
        "Cal_C1_Rejection": round(rej_c1_0 * 100, 2),
        "Cal_Total_FP": fp0
    })
    
    # 0b. Pure M2 (No Guard) on Calibration @ T* = 0.18
    p_pass_m2 = cal['s2a_pass'] & (cal['prob_m2'] >= t_probe)
    tp2 = (p_pass_m2 & c2_cal).sum()
    fp2 = (p_pass_m2 & (~c2_cal)).sum()
    rec2 = tp2 / tot_c2
    prec2 = tp2 / max(tp2 + fp2, 1)
    f12 = 2 * prec2 * rec2 / max(prec2 + rec2, 1e-8)
    rej_c1_2 = ((~p_pass_m2) & c1_cal).sum() / c1_cal.sum()
    
    candidate_records.append({
        "Rule": "Standard M2 (No Guard, T*=0.18)",
        "L_Cutoff": 0,
        "Cal_Recall": round(rec2 * 100, 2),
        "Cal_Precision": round(prec2 * 100, 2),
        "Cal_F1": round(f12, 4),
        "Cal_C1_Rejection": round(rej_c1_2 * 100, 2),
        "Cal_Total_FP": fp2
    })
    
    # Test candidate cutoff lengths and rules
    rules = [
        ("Rule A: Max(M2, M0) if L <= Cutoff", lambda r, L: max(r['prob_m2'], r['prob_m0']) if r['gold_word_count'] <= L else r['prob_m2']),
        ("Rule B: Fallback M0 if L <= Cutoff", lambda r, L: r['prob_m0'] if r['gold_word_count'] <= L else r['prob_m2']),
        ("Rule C: Max(M2, M0) if L <= Cutoff & No Connective", lambda r, L: max(r['prob_m2'], r['prob_m0']) if (r['gold_word_count'] <= L and not r['gold_has_conn']) else r['prob_m2']),
    ]
    
    for rule_name, rule_fn in rules:
        for L in [6, 8, 10, 12, 14, 16]:
            scores = cal.apply(lambda r: rule_fn(r, L), axis=1)
            passed = cal['s2a_pass'] & (scores >= t_probe)
            
            tp = (passed & c2_cal).sum()
            fp = (passed & (~c2_cal)).sum()
            rec = tp / tot_c2
            prec = tp / max(tp + fp, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-8)
            rej_c1 = ((~passed) & c1_cal).sum() / c1_cal.sum()
            
            candidate_records.append({
                "Rule": rule_name,
                "L_Cutoff": L,
                "Cal_Recall": round(rec * 100, 2),
                "Cal_Precision": round(prec * 100, 2),
                "Cal_F1": round(f1, 4),
                "Cal_C1_Rejection": round(rej_c1 * 100, 2),
                "Cal_Total_FP": fp
            })
            
    res_df = pd.DataFrame(candidate_records)
    print("\n" + "=" * 100)
    print("CALIBRATION RESULTS ACROSS CANDIDATE LENGTH GUARDS (BLIND TO TEST LABELS)")
    print("=" * 100)
    print(res_df.to_string(index=False))
    
    # Identify the best candidate on Calibration partition:
    # Criterion: Maximize Cal_F1 subject to Cal_Recall >= 90.0%
    valid = res_df[res_df['Cal_Recall'] >= 90.0]
    best_row = valid.sort_values(by=['Cal_F1', 'Cal_Precision'], ascending=False).iloc[0]
    
    best_rule = best_row['Rule']
    best_L = int(best_row['L_Cutoff'])
    print("\n" + "=" * 90)
    print(f"OPTIMAL CALIBRATED LENGTH GUARD SELECTED ON CALIBRATION SPLIT:")
    print(f"  Selected Rule:   {best_rule}")
    print(f"  Selected Cutoff: L* = {best_L} words")
    print(f"  Calibration F1:  {best_row['Cal_F1']} (Recall: {best_row['Cal_Recall']}%, Precision: {best_row['Cal_Precision']}%)")
    print("=" * 90)
    
    # Now, evaluate the FROZEN selected guard on untouched Held-Out Test partition
    print("\n" + "=" * 90)
    print("EVALUATING FROZEN SELECTED GUARD ON UNTOUCHED HELD-OUT TEST PARTITION (N = 88)")
    print("=" * 90)
    
    c2_te = (test['human_ground_truth_label'] == 'class_2')
    c1_te = (test['human_ground_truth_label'] == 'class_1')
    c0a_te = (test['human_ground_truth_label'] == 'class_0a')
    c0b_te = (test['human_ground_truth_label'] == 'class_0b')
    tot_c2_te = c2_te.sum()
    
    # 1. Baseline M0 on Test
    p_pass_m0_te = test['s2a_pass'] & (test['prob_m0'] >= 0.20)
    tp_te0 = (p_pass_m0_te & c2_te).sum()
    fp_te0 = (p_pass_m0_te & (~c2_te)).sum()
    
    # 2. Standard M2 (T*=0.18) on Test
    p_pass_m2_018_te = test['s2a_pass'] & (test['prob_m2'] >= 0.18)
    tp_te2_018 = (p_pass_m2_018_te & c2_te).sum()
    fp_te2_018 = (p_pass_m2_018_te & (~c2_te)).sum()
    
    # 3. Standard M2 (T*=0.31) on Test
    p_pass_m2_031_te = test['s2a_pass'] & (test['prob_m2'] >= 0.31)
    tp_te2_031 = (p_pass_m2_031_te & c2_te).sum()
    fp_te2_031 = (p_pass_m2_031_te & (~c2_te)).sum()
    
    # 4. Calibrated Length Guarded M2 on Test
    # Define scoring function based on best rule
    if "Rule A" in best_rule:
        test_scores = test.apply(lambda r: max(r['prob_m2'], r['prob_m0']) if r['gold_word_count'] <= best_L else r['prob_m2'], axis=1)
    elif "Rule B" in best_rule:
        test_scores = test.apply(lambda r: r['prob_m0'] if r['gold_word_count'] <= best_L else r['prob_m2'], axis=1)
    else:
        test_scores = test.apply(lambda r: max(r['prob_m2'], r['prob_m0']) if (r['gold_word_count'] <= best_L and not r['gold_has_conn']) else r['prob_m2'], axis=1)
        
    p_pass_m2_guard_te = test['s2a_pass'] & (test_scores >= 0.18)
    tp_te_guard = (p_pass_m2_guard_te & c2_te).sum()
    fp_te_guard = (p_pass_m2_guard_te & (~c2_te)).sum()
    
    def test_summary(name, passed_mask):
        tp = (passed_mask & c2_te).sum()
        fp = (passed_mask & (~c2_te)).sum()
        fp1 = (passed_mask & c1_te).sum()
        fp0a = (passed_mask & c0a_te).sum()
        fp0b = (passed_mask & c0b_te).sum()
        
        rec = tp / tot_c2_te
        prec = tp / max(tp + fp, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        rej_c1 = ((~passed_mask) & c1_te).sum() / c1_te.sum()
        rej_c0a = ((~passed_mask) & c0a_te).sum() / c0a_te.sum()
        
        print(f"{name:35s} | Rec: {rec*100:5.2f}% ({tp}/{tot_c2_te}) | Prec: {prec*100:5.2f}% | F1: {f1:.4f} | Rej C1: {rej_c1*100:5.2f}% | Rej C0a: {rej_c0a*100:5.2f}% | Tot FP: {fp} (C1={fp1}, C0a={fp0a}, C0b={fp0b})")
        return {
            "model": name, "recall": rec, "precision": prec, "f1": f1,
            "rej_c1": rej_c1, "rej_c0a": rej_c0a, "tot_fp": fp, "fp_c1": fp1, "fp_c0a": fp0a, "fp_c0b": fp0b
        }
        
    te_reports = [
        test_summary("M0 Baseline (T*=0.20)", p_pass_m0_te),
        test_summary("M2 Reasoning (T*=0.18, No Guard)", p_pass_m2_018_te),
        test_summary("M2 Reasoning (T*=0.31, No Guard)", p_pass_m2_031_te),
        test_summary(f"M2 + Length Guard (L*={best_L}, T*=0.18)", p_pass_m2_guard_te)
    ]
    
    # Convert numpy types to native Python types for JSON
    def clean_dict(d):
        return {k: (int(v) if isinstance(v, (np.integer, int)) else float(v) if isinstance(v, (np.floating, float)) else str(v)) for k, v in d.items()}

    res_df.to_csv("results/calibration/length_guard_calibration_sweep.csv", index=False)
    with open("results/calibration/length_guard_evaluation_report.json", "w") as f:
        json.dump({
            "optimal_rule": str(best_rule),
            "optimal_L_cutoff": int(best_L),
            "calibration_performance": clean_dict(best_row.to_dict()),
            "test_performance": [clean_dict(x) for x in te_reports]
        }, f, indent=2)
    print("\nSaved calibration sweep and test evaluation to results/calibration/")

if __name__ == '__main__':
    main()
