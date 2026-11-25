import os
import json
import time
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, brier_score_loss, precision_recall_curve

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using compute device: {device}")

# 1. Load Frozen Human Ground-Truth Benchmark
bench_csv = "data/calibration/frozen_human_ground_truth_benchmark_346.csv"
df = pd.read_csv(bench_csv)

# Filter 341 concrete pairs
concrete = df[df['human_ground_truth_label'].isin(['class_0a', 'class_0b', 'class_1', 'class_2'])].copy().reset_index(drop=True)
print(f"Loaded {len(concrete)} concrete pairs from {bench_csv}")

# 2. Load Cross-Encoder Model
model_path = "models/argument_nli_deberta_v3_v3"
print(f"Loading cross-encoder model from {model_path}...")
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
model.eval()
print(f"Model loaded in {time.time() - t0:.2f}s")

# 3. Batch Inference
premises = concrete['generated_viewpoint_text'].tolist()
hypotheses = concrete['gold_perspective_text'].tolist()

batch_size = 32 if torch.cuda.is_available() else 16
contra_list = []
entail_list = []
neutral_list = []
logit_entail_list = []

t1 = time.time()
for i in range(0, len(premises), batch_size):
    p_b = premises[i : i + batch_size]
    h_b = hypotheses[i : i + batch_size]
    
    inputs = tokenizer(
        p_b, h_b,
        truncation=True, max_length=256, padding=True, return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        
    contra_list.extend(probs[:, 0])
    entail_list.extend(probs[:, 1])
    neutral_list.extend(probs[:, 2])
    logit_entail_list.extend(logits[:, 1].cpu().numpy())

print(f"Inference on {len(concrete)} pairs completed in {time.time() - t1:.2f}s")

concrete['p_contra'] = [round(float(x), 4) for x in contra_list]
concrete['p_entail'] = [round(float(x), 4) for x in entail_list]
concrete['p_neutral'] = [round(float(x), 4) for x in neutral_list]
concrete['logit_entail'] = [round(float(x), 4) for x in logit_entail_list]

# Binary target: Class 2 is positive (1), Class 0a, 0b, 1 are negatives (0)
concrete['y_true_binary'] = (concrete['human_ground_truth_label'] == 'class_2').astype(int)

# 4. Create 3-Way Claim-Disjoint Partition (Train: 32 claims, Cal: 16 claims, Test: 17 claims)
rng = np.random.RandomState(42)
all_claims = sorted(concrete['claim_id'].unique())
shuffled_c = all_claims.copy()
rng.shuffle(shuffled_c)

train_c = sorted(shuffled_c[:32])
cal_c = sorted(shuffled_c[32:48])
test_c = sorted(shuffled_c[48:])

concrete['partition'] = ''
concrete.loc[concrete['claim_id'].isin(train_c), 'partition'] = 'train'
concrete.loc[concrete['claim_id'].isin(cal_c), 'partition'] = 'calibration'
concrete.loc[concrete['claim_id'].isin(test_c), 'partition'] = 'test'

# Save predictions file
pred_out_csv = "results/calibration/stage2_cross_encoder_predictions_341.csv"
os.makedirs("results/calibration", exist_ok=True)
concrete.to_csv(pred_out_csv, index=False)
print(f"Saved Stage 2 predictions to {pred_out_csv}")

# Print partition breakdown
print("\n" + "=" * 80)
print("STAGE 2 THREE-WAY CLAIM-DISJOINT PARTITION BREAKDOWN")
print("=" * 80)
for part, c_list in [('train', train_c), ('calibration', cal_c), ('test', test_c)]:
    sub = concrete[concrete['partition'] == part]
    counts = dict(sub['human_ground_truth_label'].value_counts())
    print(f"Partition: {part.upper():12s} | {len(c_list):2d} claims | {len(sub):3d} pairs | {counts}")

# 5. Fit Probability Calibrator (Platt Scaling)
# Train calibrator on Calibration partition (or Train partition)
# We fit on Train split or Calibration split:
cal_sub = concrete[concrete['partition'] == 'calibration'].copy()
X_cal = cal_sub['logit_entail'].values.reshape(-1, 1)
y_cal = cal_sub['y_true_binary'].values

calibrator = LogisticRegression(C=1.0)
calibrator.fit(X_cal, y_cal)
print(f"\nPlatt Scaling Calibrator fitted on Calibration partition (N={len(cal_sub)})")

# Compute calibrated probabilities for all rows
all_X = concrete['logit_entail'].values.reshape(-1, 1)
concrete['p_calibrated'] = calibrator.predict_proba(all_X)[:, 1]

# 6. Select Stage 2 Threshold theta* on Calibration Split
print("\n" + "=" * 80)
print("STAGE 2 THRESHOLD SELECTION ON CALIBRATION CLAIMS (N = 100 PAIRS)")
print("=" * 80)

cal_sub = concrete[concrete['partition'] == 'calibration'].copy()
y_cal = cal_sub['y_true_binary'].values
cal_probs = cal_sub['p_calibrated'].values

theta_rows = []
for theta in np.arange(0.10, 0.955, 0.02):
    theta = round(float(theta), 2)
    preds = (cal_probs >= theta).astype(int)
    
    tp = np.sum((preds == 1) & (y_cal == 1))
    fp = np.sum((preds == 1) & (y_cal == 0))
    fn = np.sum((preds == 0) & (y_cal == 1))
    tn = np.sum((preds == 0) & (y_cal == 0))
    
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    
    # Class-specific error rates on calibration split
    # False positives by class:
    fp_c1 = np.sum((preds == 1) & (cal_sub['human_ground_truth_label'] == 'class_1'))
    fp_c0a = np.sum((preds == 1) & (cal_sub['human_ground_truth_label'] == 'class_0a'))
    fp_c0b = np.sum((preds == 1) & (cal_sub['human_ground_truth_label'] == 'class_0b'))
    
    theta_rows.append({
        'theta': theta,
        'precision_c2': round(float(prec), 4),
        'recall_c2': round(float(rec), 4),
        'f1_c2': round(float(f1), 4),
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn),
        'fp_class_1': int(fp_c1),
        'fp_class_0a': int(fp_c0a),
        'fp_class_0b': int(fp_c0b),
        'pass_count': int(np.sum(preds))
    })

cal_theta_df = pd.DataFrame(theta_rows)

# Precision-constrained threshold: target precision >= 0.85 (or max precision)
eligible_theta = cal_theta_df[cal_theta_df['precision_c2'] >= 0.80]
if not eligible_theta.empty:
    opt_theta_row = eligible_theta.sort_values(by=['recall_c2', 'precision_c2'], ascending=[False, False]).iloc[0]
else:
    opt_theta_row = cal_theta_df.sort_values(by='f1_c2', ascending=False).iloc[0]

opt_theta = float(opt_theta_row['theta'])
print(f"Optimal Stage 2 Threshold Selected: theta* = {opt_theta}")
print(f"  Calibration Precision (Class 2): {opt_theta_row['precision_c2']:.4f}")
print(f"  Calibration Recall (Class 2):    {opt_theta_row['recall_c2']:.4f}")
print(f"  Calibration F1 (Class 2):        {opt_theta_row['f1_c2']:.4f}")

# Also find F1-maximizing threshold
f1_theta_row = cal_theta_df.sort_values(by='f1_c2', ascending=False).iloc[0]
f1_theta = float(f1_theta_row['theta'])
print(f"\nF1-Maximizing Threshold: theta = {f1_theta}")
print(f"  Calibration Precision (Class 2): {f1_theta_row['precision_c2']:.4f}")
print(f"  Calibration Recall (Class 2):    {f1_theta_row['recall_c2']:.4f}")
print(f"  Calibration F1 (Class 2):        {f1_theta_row['f1_c2']:.4f}")

# 7. Evaluate on Untouched Held-Out Test Claims (N = 88 pairs, 17 claims)
print("\n" + "=" * 80)
print("STAGE 2 EVALUATION ON UNTOUCHED HELD-OUT TEST CLAIMS (N = 88 PAIRS)")
print("=" * 80)

test_sub = concrete[concrete['partition'] == 'test'].copy()
y_test = test_sub['y_true_binary'].values
test_probs = test_sub['p_calibrated'].values

def evaluate_stage2_on_test(theta, name="Operating Point"):
    preds = (test_probs >= theta).astype(int)
    
    tp = np.sum((preds == 1) & (y_test == 1))
    fp = np.sum((preds == 1) & (y_test == 0))
    fn = np.sum((preds == 0) & (y_test == 1))
    tn = np.sum((preds == 0) & (y_test == 0))
    
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    
    # Class-specific breakdown
    total_c2 = np.sum(test_sub['human_ground_truth_label'] == 'class_2')
    total_c1 = np.sum(test_sub['human_ground_truth_label'] == 'class_1')
    total_c0a = np.sum(test_sub['human_ground_truth_label'] == 'class_0a')
    total_c0b = np.sum(test_sub['human_ground_truth_label'] == 'class_0b')
    
    pass_c2 = np.sum((preds == 1) & (test_sub['human_ground_truth_label'] == 'class_2'))
    pass_c1 = np.sum((preds == 1) & (test_sub['human_ground_truth_label'] == 'class_1'))
    pass_c0a = np.sum((preds == 1) & (test_sub['human_ground_truth_label'] == 'class_0a'))
    pass_c0b = np.sum((preds == 1) & (test_sub['human_ground_truth_label'] == 'class_0b'))
    
    rej_c1 = (total_c1 - pass_c1) / max(total_c1, 1)
    rej_c0a = (total_c0a - pass_c0a) / max(total_c0a, 1)
    rej_c0b = (total_c0b - pass_c0b) / max(total_c0b, 1)
    
    return {
        "setting": name,
        "theta": theta,
        "precision_c2": round(float(prec), 4),
        "recall_c2": round(float(rec), 4),
        "f1_c2": round(float(f1), 4),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "c2_passed": int(pass_c2), "total_c2": int(total_c2),
        "c1_passed_fp": int(pass_c1), "c1_rejection": round(float(rej_c1), 4),
        "c0a_passed_fp": int(pass_c0a), "c0a_rejection": round(float(rej_c0a), 4),
        "c0b_passed_fp": int(pass_c0b), "c0b_rejection": round(float(rej_c0b), 4),
        "total_accepted": int(np.sum(preds)),
        "acceptance_rate": round(float(np.sum(preds) / len(test_sub)), 4)
    }

eval_test_opt = evaluate_stage2_on_test(opt_theta, f"Precision-Constrained (theta={opt_theta})")
eval_test_f1 = evaluate_stage2_on_test(f1_theta, f"F1-Optimal (theta={f1_theta})")
eval_test_default = evaluate_stage2_on_test(0.50, "Default Baseline (theta=0.50)")

test_comparison_df = pd.DataFrame([eval_test_opt, eval_test_f1, eval_test_default])
print(test_comparison_df[[
    'setting', 'precision_c2', 'recall_c2', 'f1_c2', 'c1_passed_fp', 'c0a_passed_fp', 'c0b_passed_fp', 'c0b_rejection', 'acceptance_rate'
]].to_string(index=False))

# 8. End-to-End Pipeline Evaluation (Stage 1 + Stage 2 cascaded on Test Partition)
print("\n" + "=" * 80)
print("END-TO-END PIPELINE EVALUATION (STAGE 1 tau* = 0.55 -> STAGE 2 theta*)")
print("=" * 80)

tau_star = 0.55
stage1_passed_mask = (test_sub['cosine_similarity'] >= tau_star).values

# Stage 2 applied ONLY to Stage 1 survivors
stage2_passed_mask = (test_probs >= opt_theta)
e2e_accepted_mask = stage1_passed_mask & stage2_passed_mask

# Stage 1 metrics on Test Split
s1_tp = np.sum(stage1_passed_mask & (y_test == 1))
s1_fp = np.sum(stage1_passed_mask & (y_test == 0))
s1_rec = s1_tp / max(np.sum(y_test == 1), 1)
s1_prec = s1_tp / max(np.sum(stage1_passed_mask), 1)
s1_pass_rate = np.sum(stage1_passed_mask) / len(test_sub)

# Stage 2 on survivors
survivors_mask = stage1_passed_mask
s2_tp = np.sum(e2e_accepted_mask & (y_test == 1))
s2_fp = np.sum(e2e_accepted_mask & (y_test == 0))
s2_prec_on_survivors = s2_tp / max(np.sum(e2e_accepted_mask), 1)
s2_rec_of_survivors = s2_tp / max(s1_tp, 1)

# End-to-End metrics
e2e_rec = s2_tp / max(np.sum(y_test == 1), 1)
e2e_prec = s2_tp / max(np.sum(e2e_accepted_mask), 1)
e2e_f1 = 2 * e2e_prec * e2e_rec / max(e2e_prec + e2e_rec, 1e-8)
e2e_c0b_rejection = np.sum((~e2e_accepted_mask) & (test_sub['human_ground_truth_label'] == 'class_0b')) / max(np.sum(test_sub['human_ground_truth_label'] == 'class_0b'), 1)
e2e_c0a_rejection = np.sum((~e2e_accepted_mask) & (test_sub['human_ground_truth_label'] == 'class_0a')) / max(np.sum(test_sub['human_ground_truth_label'] == 'class_0a'), 1)
e2e_c1_rejection = np.sum((~e2e_accepted_mask) & (test_sub['human_ground_truth_label'] == 'class_1')) / max(np.sum(test_sub['human_ground_truth_label'] == 'class_1'), 1)

print(f"\n1. STAGE 1 ALONE (tau* = {tau_star}):")
print(f"   Candidate Recall (Class 2): {s1_rec*100:.2f}% ({s1_tp}/{np.sum(y_test == 1)})")
print(f"   Candidate Precision:        {s1_prec*100:.2f}%")
print(f"   Stage 1 Candidate Pass Rate:{s1_pass_rate*100:.2f}% ({np.sum(stage1_passed_mask)}/{len(test_sub)})")

print(f"\n2. STAGE 2 ALONE ON STAGE 1 SURVIVORS (theta* = {opt_theta}):")
print(f"   Precision on Survivors:     {s2_prec_on_survivors*100:.2f}%")
print(f"   Retention of S1 Positives:  {s2_rec_of_survivors*100:.2f}% ({s2_tp}/{s1_tp})")
print(f"   Survivors Accepted:         {np.sum(e2e_accepted_mask)}/{np.sum(stage1_passed_mask)}")

print(f"\n3. END-TO-END PIPELINE (STAGE 1 + STAGE 2 CASCADED):")
print(f"   End-to-End Class 2 Recall:    {e2e_rec*100:.2f}% ({s2_tp}/{np.sum(y_test == 1)})")
print(f"   End-to-End Class 2 Precision: {e2e_prec*100:.2f}%")
print(f"   End-to-End Class 2 F1-Score:  {e2e_f1:.4f}")
print(f"   Class 0b (Off-topic) Rejection:{e2e_c0b_rejection*100:.2f}%")
print(f"   Class 0a (Opposite) Rejection: {e2e_c0a_rejection*100:.2f}%")
print(f"   Class 1 (Nuance) Rejection:    {e2e_c1_rejection*100:.2f}%")
print(f"   Final Pipeline Acceptance:    {np.sum(e2e_accepted_mask)}/{len(test_sub)} ({np.sum(e2e_accepted_mask)/len(test_sub)*100:.2f}%)")

# Save summary report
e2e_report = {
    "partitions": {
        "train_claims": len(train_c), "train_pairs": len(concrete[concrete['partition']=='train']),
        "calibration_claims": len(cal_c), "calibration_pairs": len(concrete[concrete['partition']=='calibration']),
        "test_claims": len(test_c), "test_pairs": len(concrete[concrete['partition']=='test'])
    },
    "stage1_evaluation": {
        "tau_star": tau_star,
        "candidate_recall_c2": round(float(s1_rec), 4),
        "candidate_precision_c2": round(float(s1_prec), 4),
        "candidate_pass_rate": round(float(s1_pass_rate), 4)
    },
    "stage2_calibration": {
        "optimal_theta": opt_theta,
        "f1_theta": f1_theta,
        "calibration_stats": opt_theta_row.to_dict()
    },
    "stage2_test_evaluation": eval_test_opt,
    "end_to_end_pipeline": {
        "recall_c2": round(float(e2e_rec), 4),
        "precision_c2": round(float(e2e_prec), 4),
        "f1_c2": round(float(e2e_f1), 4),
        "rejection_class_0b": round(float(e2e_c0b_rejection), 4),
        "rejection_class_0a": round(float(e2e_c0a_rejection), 4),
        "rejection_class_1": round(float(e2e_c1_rejection), 4),
        "final_acceptance_rate": round(float(np.sum(e2e_accepted_mask) / len(test_sub)), 4)
    }
}

with open("results/calibration/stage2_and_end_to_end_report.json", "w") as f:
    json.dump(e2e_report, f, indent=2)

print("\nSaved full Stage 2 and End-to-End report to: results/calibration/stage2_and_end_to_end_report.json")
