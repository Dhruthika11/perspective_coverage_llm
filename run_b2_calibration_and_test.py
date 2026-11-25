import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.proportion import proportion_confint

df = pd.read_csv('results/calibration/stage2_cross_encoder_predictions_341.csv')
c21 = df[df['human_ground_truth_label'].isin(['class_2', 'class_1'])].copy()

train_df = c21[c21['partition'] == 'train'].copy()
cal_df = c21[c21['partition'] == 'calibration'].copy()
test_df = c21[c21['partition'] == 'test'].copy()

feats = ['cosine_similarity', 'p_entail', 'p_neutral']
scaler = StandardScaler()
X_tr = scaler.fit_transform(train_df[feats].values)
X_ca = scaler.transform(cal_df[feats].values)
X_te = scaler.transform(test_df[feats].values)

y_tr = (train_df['human_ground_truth_label'] == 'class_2').astype(int).values
y_ca = (cal_df['human_ground_truth_label'] == 'class_2').astype(int).values
y_te = (test_df['human_ground_truth_label'] == 'class_2').astype(int).values

clf = LogisticRegression(C=1.0, random_state=42)
clf.fit(X_tr, y_tr)

ca_probs = clf.predict_proba(X_ca)[:, 1]
te_probs = clf.predict_proba(X_te)[:, 1]

print("=" * 95)
print("EXPERIMENT B2: PROBE THRESHOLD CALIBRATION (STRICTLY ON 16 CALIBRATION CLAIMS)")
print("=" * 95)

cal_rows = []
for th in np.arange(0.20, 0.85, 0.05):
    th = round(float(th), 2)
    p_pred = (ca_probs >= th).astype(int)
    
    tp = np.sum((p_pred == 1) & (y_ca == 1))
    fp = np.sum((p_pred == 1) & (y_ca == 0))
    fn = np.sum((p_pred == 0) & (y_ca == 1))
    
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    rej_c1 = (len(y_ca) - 24 - fp) / 45
    
    cal_rows.append({
        'Threshold': th,
        'Cal_Prec_C2': round(float(prec), 4),
        'Cal_Rec_C2': round(float(rec), 4),
        'Cal_F1_C2': round(float(f1), 4),
        'Cal_Rej_C1': round(float(rej_c1), 4),
        'TP': int(tp), 'FP': int(fp), 'FN': int(fn)
    })

cal_sweep_df = pd.DataFrame(cal_rows)
print(cal_sweep_df.to_string(index=False))

# Select operating thresholds based ONLY on Calibration data:
# 1. F1-Optimal on Calibration
opt_f1_row = cal_sweep_df.sort_values(by=['Cal_F1_C2', 'Cal_Prec_C2'], ascending=[False, False]).iloc[0]
th_f1 = float(opt_f1_row['Threshold'])

# 2. Balanced Operating Point (Recall >= 70%, Max Precision)
bal_candidates = cal_sweep_df[cal_sweep_df['Cal_Rec_C2'] >= 0.70]
opt_bal_row = bal_candidates.sort_values(by='Cal_Prec_C2', ascending=False).iloc[0]
th_bal = float(opt_bal_row['Threshold'])

# 3. High-Precision Operating Point (Precision >= 50% or Max Precision)
opt_prec_row = cal_sweep_df.sort_values(by=['Cal_Prec_C2', 'Cal_Rec_C2'], ascending=[False, False]).iloc[0]
th_prec = float(opt_prec_row['Threshold'])

print(f"\nSelected Thresholds on Calibration Partition:")
print(f"  F1-Optimal Threshold:     T* = {th_f1:.2f} (Cal F1 = {opt_f1_row['Cal_F1_C2']:.4f}, Prec = {opt_f1_row['Cal_Prec_C2']:.4f}, Rec = {opt_f1_row['Cal_Rec_C2']:.4f})")
print(f"  Balanced Threshold:       T* = {th_bal:.2f} (Cal F1 = {opt_bal_row['Cal_F1_C2']:.4f}, Prec = {opt_bal_row['Cal_Prec_C2']:.4f}, Rec = {opt_bal_row['Cal_Rec_C2']:.4f})")
print(f"  High-Precision Threshold: T* = {th_prec:.2f} (Cal F1 = {opt_prec_row['Cal_F1_C2']:.4f}, Prec = {opt_prec_row['Cal_Prec_C2']:.4f}, Rec = {opt_prec_row['Cal_Rec_C2']:.4f})")

# Evaluate on Untouched Held-Out Test Split (17 Claims, 53 Pairs: 24 C2, 29 C1)
print("\n" + "=" * 95)
print("EVALUATION ON UNTOUCHED HELD-OUT TEST CLAIMS (17 CLAIMS, 53 PAIRS)")
print("=" * 95)

test_evals = []
for name, th in [("Balanced / F1-Optimal", th_f1), ("High-Precision", th_prec), ("Baseline NLI (p_entail>=0.80)", 0.80)]:
    if name.startswith("Baseline"):
        preds = (test_df['p_entail'].values >= 0.80).astype(int)
    else:
        preds = (te_probs >= th).astype(int)
        
    tp = np.sum((preds == 1) & (y_te == 1))
    fp = np.sum((preds == 1) & (y_te == 0))
    fn = np.sum((preds == 0) & (y_te == 1))
    tn = np.sum((preds == 0) & (y_te == 0))
    
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    rej_c1 = tn / 29
    
    w_low_rec, w_high_rec = proportion_confint(tp, 24, alpha=0.05, method='wilson')
    w_low_prec, w_high_prec = proportion_confint(tp, max(tp + fp, 1), alpha=0.05, method='wilson')
    
    test_evals.append({
        "Operating Setting": name,
        "Threshold": th,
        "Class 2 Recall": f"{tp}/24 ({rec*100:.1f}%)",
        "Recall 95% Wilson CI": f"[{w_low_rec*100:.1f}%, {w_high_rec*100:.1f}%]",
        "Class 2 Precision": f"{tp}/{tp+fp} ({prec*100:.1f}%)",
        "Precision 95% Wilson CI": f"[{w_low_prec*100:.1f}%, {w_high_prec*100:.1f}%]",
        "Class 1 Rejection": f"{tn}/29 ({rej_c1*100:.1f}%)",
        "Class 2 F1": round(float(f1), 4)
    })

eval_res_df = pd.DataFrame(test_evals)
print(eval_res_df[['Operating Setting', 'Threshold', 'Class 2 Recall', 'Class 2 Precision', 'Class 1 Rejection', 'Class 2 F1']].to_string(index=False))
print("\nConfidence Intervals (Wilson 95%):")
for r in test_evals:
    print(f"  {r['Operating Setting']:30s} -> Recall CI: {r['Recall 95% Wilson CI']} | Precision CI: {r['Precision 95% Wilson CI']}")
