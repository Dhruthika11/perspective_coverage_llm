import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('results/calibration/stage2_cross_encoder_predictions_341.csv')

# Partitions
cal_df = df[df['partition'] == 'calibration'].copy()
test_df = df[df['partition'] == 'test'].copy()

# Fit Platt scaling
lr = LogisticRegression(C=1.0)
lr.fit(cal_df['logit_entail'].values.reshape(-1, 1), cal_df['y_true_binary'].values)
cal_df['p_calibrated'] = lr.predict_proba(cal_df['logit_entail'].values.reshape(-1, 1))[:, 1]
test_df['p_calibrated'] = lr.predict_proba(test_df['logit_entail'].values.reshape(-1, 1))[:, 1]

# Save updated df with p_calibrated
df['p_calibrated'] = lr.predict_proba(df['logit_entail'].values.reshape(-1, 1))[:, 1]
df.to_csv('results/calibration/stage2_cross_encoder_predictions_341.csv', index=False)

print("=" * 85)
print("STAGE 2 CALIBRATION: THRESHOLD SWEEP ON RAW P_ENTAIL (CALIBRATION SPLIT, N = 100)")
print("=" * 85)

raw_rows = []
for th in np.arange(0.50, 0.99, 0.05):
    th = round(float(th), 2)
    preds = (cal_df['p_entail'] >= th).astype(int)
    y_true = cal_df['y_true_binary'].values
    
    tp = np.sum((preds == 1) & (y_true == 1))
    fp = np.sum((preds == 1) & (y_true == 0))
    fn = np.sum((preds == 0) & (y_true == 1))
    
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    
    fp_c1 = np.sum((preds == 1) & (cal_df['human_ground_truth_label'] == 'class_1'))
    fp_c0a = np.sum((preds == 1) & (cal_df['human_ground_truth_label'] == 'class_0a'))
    fp_c0b = np.sum((preds == 1) & (cal_df['human_ground_truth_label'] == 'class_0b'))
    
    raw_rows.append({
        'theta_raw': th,
        'prec_c2': round(float(prec), 4),
        'rec_c2': round(float(rec), 4),
        'f1_c2': round(float(f1), 4),
        'tp': int(tp), 'fp': int(fp),
        'fp_c1': int(fp_c1), 'fp_c0a': int(fp_c0a), 'fp_c0b': int(fp_c0b)
    })

raw_sweep_df = pd.DataFrame(raw_rows)
print(raw_sweep_df.to_string(index=False))

print("\n" + "=" * 85)
print("STAGE 2 TEST EVALUATION ON RAW P_ENTAIL (HELD-OUT TEST SPLIT, N = 88)")
print("=" * 85)

test_rows = []
for th in [0.70, 0.80, 0.85, 0.90, 0.95]:
    preds = (test_df['p_entail'] >= th).astype(int)
    y_true = test_df['y_true_binary'].values
    
    tp = np.sum((preds == 1) & (y_true == 1))
    fp = np.sum((preds == 1) & (y_true == 0))
    fn = np.sum((preds == 0) & (y_true == 1))
    
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    
    tot_c1 = np.sum(test_df['human_ground_truth_label'] == 'class_1')
    tot_c0a = np.sum(test_df['human_ground_truth_label'] == 'class_0a')
    tot_c0b = np.sum(test_df['human_ground_truth_label'] == 'class_0b')
    
    fp_c1 = np.sum((preds == 1) & (test_df['human_ground_truth_label'] == 'class_1'))
    fp_c0a = np.sum((preds == 1) & (test_df['human_ground_truth_label'] == 'class_0a'))
    fp_c0b = np.sum((preds == 1) & (test_df['human_ground_truth_label'] == 'class_0b'))
    
    rej_c1 = (tot_c1 - fp_c1) / max(tot_c1, 1)
    rej_c0a = (tot_c0a - fp_c0a) / max(tot_c0a, 1)
    rej_c0b = (tot_c0b - fp_c0b) / max(tot_c0b, 1)
    
    test_rows.append({
        'theta_raw': th,
        'prec_c2': round(float(prec), 4),
        'rec_c2': round(float(rec), 4),
        'f1_c2': round(float(f1), 4),
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn),
        'rej_c1': round(float(rej_c1), 4),
        'rej_c0a': round(float(rej_c0a), 4),
        'rej_c0b': round(float(rej_c0b), 4),
        'acceptance_rate': round(float(np.sum(preds)/len(test_df)), 4)
    })

test_sweep_df = pd.DataFrame(test_rows)
print(test_sweep_df.to_string(index=False))
