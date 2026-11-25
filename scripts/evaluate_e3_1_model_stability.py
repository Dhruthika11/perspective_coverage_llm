import os
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

def evaluate():
    df = pd.read_csv("results/calibration/e3_1_pairwise_features_341.csv")
    
    train = df[(df['partition'] == 'train') & (df['human_ground_truth_label'].isin(['class_2', 'class_1']))]
    cal = df[df['partition'] == 'calibration']
    cal_c21 = cal[cal['human_ground_truth_label'].isin(['class_2', 'class_1'])]
    test = df[df['partition'] == 'test']
    test_c21 = test[test['human_ground_truth_label'].isin(['class_2', 'class_1'])]
    
    y_tr = (train['human_ground_truth_label'] == 'class_2').astype(int).values
    y_ca = (cal_c21['human_ground_truth_label'] == 'class_2').astype(int).values
    y_te = (test_c21['human_ground_truth_label'] == 'class_2').astype(int).values
    
    # Baseline M0
    feats_m0 = ['cosine_similarity', 'p_entail', 'p_neutral']
    sc_m0 = StandardScaler()
    X_tr_m0 = sc_m0.fit_transform(train[feats_m0].values)
    clf_m0 = LogisticRegression(C=1.0, random_state=42).fit(X_tr_m0, y_tr)
    
    # M2 v1.0
    feats_m2_v1_0 = ['cosine_similarity', 'p_entail', 'p_neutral', 'reasoning_cosine']
    sc_m2_v1_0 = StandardScaler()
    X_tr_m2_v1_0 = sc_m2_v1_0.fit_transform(train[feats_m2_v1_0].values)
    clf_m2_v1_0 = LogisticRegression(C=1.0, random_state=42).fit(X_tr_m2_v1_0, y_tr)
    
    # M2 v1.1
    feats_m2_v1_1 = ['cosine_similarity', 'p_entail', 'p_neutral', 'reasoning_cosine_v1_1']
    sc_m2_v1_1 = StandardScaler()
    X_tr_m2_v1_1 = sc_m2_v1_1.fit_transform(train[feats_m2_v1_1].values)
    clf_m2_v1_1 = LogisticRegression(C=1.0, random_state=42).fit(X_tr_m2_v1_1, y_tr)
    
    print("=== MODEL COEFFICIENTS COMPARISON ===")
    print("M0 coefficients:", dict(zip(feats_m0, clf_m0.coef_[0].round(4))))
    print("M2 v1.0 coefficients:", dict(zip(feats_m2_v1_0, clf_m2_v1_0.coef_[0].round(4))))
    print("M2 v1.1 coefficients:", dict(zip(feats_m2_v1_1, clf_m2_v1_1.coef_[0].round(4))))
    
    # Test C2/C1 ROC-AUC
    p_te_m0 = clf_m0.predict_proba(sc_m0.transform(test_c21[feats_m0].values))[:, 1]
    p_te_m2_v1_0 = clf_m2_v1_0.predict_proba(sc_m2_v1_0.transform(test_c21[feats_m2_v1_0].values))[:, 1]
    p_te_m2_v1_1 = clf_m2_v1_1.predict_proba(sc_m2_v1_1.transform(test_c21[feats_m2_v1_1].values))[:, 1]
    
    auc_m0 = roc_auc_score(y_te, p_te_m0)
    auc_m2_v1_0 = roc_auc_score(y_te, p_te_m2_v1_0)
    auc_m2_v1_1 = roc_auc_score(y_te, p_te_m2_v1_1)
    
    print("\n=== TEST C2/C1 ROC-AUC ===")
    print(f"M0 baseline: {auc_m0:.4f}")
    print(f"M2 (v1.0):   {auc_m2_v1_0:.4f}")
    print(f"M2 (v1.1):   {auc_m2_v1_1:.4f}")
    
    # Calibration sweep for threshold T* under calibration floor (e.g. recall >= 0.90 or best F1)
    # Stance concordance
    is_contra_cal = (
        ((cal['stance_gold'] == 'support') & (cal['stance_generated'] == 'undermine')) |
        ((cal['stance_gold'] == 'undermine') & (cal['stance_generated'] == 'support'))
    )
    s2a_cal = (cal['cosine_similarity'] >= 0.55) & (~is_contra_cal)
    
    p_cal_m2_v1_1 = clf_m2_v1_1.predict_proba(sc_m2_v1_1.transform(cal[feats_m2_v1_1].values))[:, 1]
    
    c2_cal = (cal['human_ground_truth_label'] == 'class_2')
    c1_cal = (cal['human_ground_truth_label'] == 'class_1')
    
    print("\n=== CALIBRATION SWEEP FOR M2 v1.1 ===")
    best_t_f1 = 0.20
    max_f1 = -1
    t_90_rec = None
    
    for t_val in np.arange(0.10, 0.85, 0.01):
        t_val = round(float(t_val), 2)
        passed = s2a_cal & (p_cal_m2_v1_1 >= t_val)
        tp = (passed & c2_cal).sum()
        fp = (passed & (~c2_cal)).sum()
        fn = ((~passed) & c2_cal).sum()
        rec = tp / max(tp + fn, 1)
        prec = tp / max(tp + fp, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        
        if f1 > max_f1:
            max_f1 = f1
            best_t_f1 = t_val
        if rec >= 0.90:
            t_90_rec = t_val
            
    print(f"M2 v1.1 Best Cal F1 Threshold: T* = {best_t_f1:.2f} (F1 = {max_f1:.4f})")
    print(f"M2 v1.1 Max threshold satisfying Cal Recall >= 90%: T* = {t_90_rec:.2f}")
    
    # Evaluate cascade on Test Set
    is_contra_te = (
        ((test['stance_gold'] == 'support') & (test['stance_generated'] == 'undermine')) |
        ((test['stance_gold'] == 'undermine') & (test['stance_generated'] == 'support'))
    )
    s2a_te = (test['cosine_similarity'] >= 0.55) & (~is_contra_te)
    p_te_full_m0 = clf_m0.predict_proba(sc_m0.transform(test[feats_m0].values))[:, 1]
    p_te_full_m2_v1_0 = clf_m2_v1_0.predict_proba(sc_m2_v1_0.transform(test[feats_m2_v1_0].values))[:, 1]
    p_te_full_m2_v1_1 = clf_m2_v1_1.predict_proba(sc_m2_v1_1.transform(test[feats_m2_v1_1].values))[:, 1]
    
    def eval_test_cascade(name, prob_arr, t_val):
        passed = s2a_te & (prob_arr >= t_val)
        c2 = (test['human_ground_truth_label'] == 'class_2')
        c1 = (test['human_ground_truth_label'] == 'class_1')
        c0a = (test['human_ground_truth_label'] == 'class_0a')
        c0b = (test['human_ground_truth_label'] == 'class_0b')
        
        tp = (passed & c2).sum()
        fp1 = (passed & c1).sum()
        fp0a = (passed & c0a).sum()
        fp0b = (passed & c0b).sum()
        fp_tot = fp1 + fp0a + fp0b
        fn = ((~passed) & c2).sum()
        
        rec = tp / max(c2.sum(), 1)
        prec = tp / max(tp + fp_tot, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        rej_c1 = (c1.sum() - fp1) / max(c1.sum(), 1)
        rej_c0a = (c0a.sum() - fp0a) / max(c0a.sum(), 1)
        rej_c0b = (c0b.sum() - fp0b) / max(c0b.sum(), 1)
        
        print(f"{name:32s} | T*={t_val:.2f} | Rec={rec*100:5.2f}% ({tp}/{c2.sum()}) | Prec={prec*100:5.2f}% | F1={f1:.4f} | Rej C1={rej_c1*100:5.2f}% ({c1.sum()-fp1}/{c1.sum()}) | Rej C0a={rej_c0a*100:5.2f}% | Tot FP={fp_tot}")

    print("\n=== FULL CASCADE TEST SET EVALUATION ===")
    eval_test_cascade("M0 Baseline (Frozen)", p_te_full_m0, 0.20)
    eval_test_cascade("M2 (v1.0) @ T*=0.18 (Matched Rec)", p_te_full_m2_v1_0, 0.18)
    eval_test_cascade("M2 (v1.1) @ T*=0.18 (Matched Rec)", p_te_full_m2_v1_1, 0.18)
    eval_test_cascade("M2 (v1.0) @ T*=0.31 (Best Cal F1)", p_te_full_m2_v1_0, 0.31)
    eval_test_cascade("M2 (v1.1) @ T*=0.31 (Best Cal F1)", p_te_full_m2_v1_1, 0.31)

if __name__ == '__main__':
    evaluate()
