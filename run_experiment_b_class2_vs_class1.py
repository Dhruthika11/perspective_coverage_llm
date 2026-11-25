import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, confusion_matrix
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('results/calibration/stage2_cross_encoder_predictions_341.csv')

# 1. Prepare Dataset for Class 2 (Positive = 1) vs Class 1 (Hard Negative = 0)
train_c21 = df[(df['partition'] == 'train') & (df['human_ground_truth_label'].isin(['class_2', 'class_1']))].copy()
cal_c21 = df[(df['partition'] == 'calibration') & (df['human_ground_truth_label'].isin(['class_2', 'class_1']))].copy()
test_c21 = df[(df['partition'] == 'test') & (df['human_ground_truth_label'].isin(['class_2', 'class_1']))].copy()

# Full test split (including Class 0a and Class 0b) to verify overall rejection
test_full = df[df['partition'] == 'test'].copy()

print("=" * 85)
print("EXPERIMENT B: TARGETED CLASS 2 (EQUIVALENT) VS CLASS 1 (SAME STANCE, DIFF REASON)")
print("=" * 85)

print(f"Train Partition (32 Claims): {len(train_c21)} pairs (Class 2: {(train_c21['human_ground_truth_label']=='class_2').sum()}, Class 1: {(train_c21['human_ground_truth_label']=='class_1').sum()})")
print(f"Calibration Partition (16 Claims): {len(cal_c21)} pairs (Class 2: {(cal_c21['human_ground_truth_label']=='class_2').sum()}, Class 1: {(cal_c21['human_ground_truth_label']=='class_1').sum()})")
print(f"Held-Out Test Partition (17 Claims): {len(test_c21)} pairs (Class 2: {(test_c21['human_ground_truth_label']=='class_2').sum()}, Class 1: {(test_c21['human_ground_truth_label']=='class_1').sum()})")

# Feature set:
# 1. Cosine similarity from Stage 1 (BGE-Large)
# 2. Cross-encoder p_entail, p_contra, p_neutral, logit_entail
# 3. Entailment-to-neutral ratio: log(p_entail / max(p_neutral, 1e-5))
features = ['cosine_similarity', 'p_entail', 'p_contra', 'p_neutral', 'logit_entail']

X_train = train_c21[features].values
y_train = (train_c21['human_ground_truth_label'] == 'class_2').astype(int).values

X_cal = cal_c21[features].values
y_cal = (cal_c21['human_ground_truth_label'] == 'class_2').astype(int).values

X_test_c21 = test_c21[features].values
y_test_c21 = (test_c21['human_ground_truth_label'] == 'class_2').astype(int).values

X_test_full = test_full[features].values
y_test_full = (test_full['human_ground_truth_label'] == 'class_2').astype(int).values

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_cal_s = scaler.transform(X_cal)
X_test_c21_s = scaler.transform(X_test_c21)
X_test_full_s = scaler.transform(X_test_full)

# 2. Train regularized probe
probe = LogisticRegression(C=0.5, penalty='l2', random_state=42)
probe.fit(X_train_s, y_train)

print("\nTrained Probe Feature Weights:")
for feat, coef in zip(features, probe.coef_[0]):
    print(f"  {feat:20s}: {coef:+.4f}")
print(f"  Intercept:           {probe.intercept_[0]:+.4f}")

# 3. Baseline vs Probe on Train & Cal AUC
base_cal_auc = roc_auc_score(y_cal, cal_c21['p_entail'])
probe_cal_probs = probe.predict_proba(X_cal_s)[:, 1]
probe_cal_auc = roc_auc_score(y_cal, probe_cal_probs)

print(f"\nCalibration Split ROC-AUC (Class 2 vs Class 1):")
print(f"  Baseline NLI p_entail: {base_cal_auc:.4f}")
print(f"  Targeted Probe:        {probe_cal_auc:.4f}")

# 4. Select threshold on Calibration Split
best_th = 0.50
best_f1 = -1.0
cal_records = []
for th in np.arange(0.20, 0.85, 0.05):
    p_pred = (probe_cal_probs >= th).astype(int)
    tp = np.sum((p_pred == 1) & (y_cal == 1))
    fp = np.sum((p_pred == 1) & (y_cal == 0))
    fn = np.sum((p_pred == 0) & (y_cal == 1))
    
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    
    cal_records.append({'threshold': th, 'precision': prec, 'recall': rec, 'f1': f1})
    if f1 > best_f1:
        best_f1 = f1
        best_th = th

print(f"\nSelected Optimal Threshold on Calibration: th* = {best_th:.2f} (Cal F1 = {best_f1:.4f})")

# 5. Evaluate on Held-Out Test Split (Class 2 vs Class 1)
probe_test_c21_probs = probe.predict_proba(X_test_c21_s)[:, 1]
base_test_auc = roc_auc_score(y_test_c21, test_c21['p_entail'])
probe_test_auc = roc_auc_score(y_test_c21, probe_test_c21_probs)

print(f"\nHeld-Out Test Split ROC-AUC (Class 2 vs Class 1):")
print(f"  Baseline NLI p_entail: {base_test_auc:.4f}")
print(f"  Targeted Probe:        {probe_test_auc:.4f}")

# Test performance at selected threshold
test_preds = (probe_test_c21_probs >= best_th).astype(int)
cm = confusion_matrix(y_test_c21, test_preds)
print("\nConfusion Matrix on Test (Rows: Actual [C1, C2], Cols: Pred [C1, C2]):")
print(cm)
print("\nClassification Report (Class 2 vs Class 1):")
print(classification_report(y_test_c21, test_preds, target_names=['Class 1 (Negative)', 'Class 2 (Positive)']))

# 6. Evaluate Full Cascade with Probe on All 88 Test Pairs
test_full_probs = probe.predict_proba(X_test_full_s)[:, 1]
full_preds = (test_full_probs >= best_th).astype(int)

# Stage 1 filter
s1_filter = (test_full['cosine_similarity'] >= 0.55).values
cascaded_preds = s1_filter & full_preds

c2_tot = np.sum(test_full['human_ground_truth_label'] == 'class_2')
c1_tot = np.sum(test_full['human_ground_truth_label'] == 'class_1')
c0a_tot = np.sum(test_full['human_ground_truth_label'] == 'class_0a')
c0b_tot = np.sum(test_full['human_ground_truth_label'] == 'class_0b')

c2_pass = np.sum(cascaded_preds & (test_full['human_ground_truth_label'] == 'class_2'))
c1_pass = np.sum(cascaded_preds & (test_full['human_ground_truth_label'] == 'class_1'))
c0a_pass = np.sum(cascaded_preds & (test_full['human_ground_truth_label'] == 'class_0a'))
c0b_pass = np.sum(cascaded_preds & (test_full['human_ground_truth_label'] == 'class_0b'))

print("\n" + "=" * 85)
print("CASCADED PIPELINE WITH TARGETED PROBE (TEST PARTITION N = 88)")
print("=" * 85)
print(f"Class 2 Recall:    {c2_pass}/{c2_tot} = {c2_pass/c2_tot*100:.2f}%")
print(f"Class 2 Precision: {c2_pass}/{np.sum(cascaded_preds)} = {c2_pass/max(np.sum(cascaded_preds), 1)*100:.2f}%")
print(f"Class 1 Rejection: {(c1_tot - c1_pass)}/{c1_tot} = {(c1_tot - c1_pass)/c1_tot*100:.2f}%")
print(f"Class 0a Rejection:{(c0a_tot - c0a_pass)}/{c0a_tot} = {(c0a_tot - c0a_pass)/c0a_tot*100:.2f}%")
print(f"Class 0b Rejection:{(c0b_tot - c0b_pass)}/{c0b_tot} = {(c0b_tot - c0b_pass)/c0b_tot*100:.2f}%")
print(f"Accepted Pairs:    {np.sum(cascaded_preds)}/{len(test_full)} ({np.sum(cascaded_preds)/len(test_full)*100:.2f}%)")
