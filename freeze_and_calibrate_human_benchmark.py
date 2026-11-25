import os
import json
import hashlib
import numpy as np
import pandas as pd

print("=" * 85)
print("PRIORITY 1 & 2: DEDUPLICATE, VALIDATE & FREEZE HUMAN GROUND TRUTH (N = 346)")
print("=" * 85)

# Load master files
df1700 = pd.read_csv('data/calibration/llm_jury_full_1700_annotated.csv')
bench1700 = pd.read_csv('data/calibration/human_llm_benchmark_1700.csv')

# Merge cosine similarity
merged = pd.merge(df1700, bench1700[['pair_id', 'cosine_similarity']], on='pair_id', how='left')

# Extract human annotated pairs
human_346 = merged[merged['human_ground_truth_label'].notna()].copy()
human_346 = human_346.sort_values(by='pair_id').reset_index(drop=True)

# 1. Validation checks
assert len(human_346) == 346, f"Expected 346 pairs, got {len(human_346)}"
assert human_346['pair_id'].nunique() == 346, "Duplicate pair_id found!"
assert human_346.duplicated(subset=['claim_id', 'gold_perspective_text', 'generated_viewpoint_text']).sum() == 0, "Duplicate text pairs found!"
valid_classes = {'class_0a', 'class_0b', 'class_1', 'class_2', 'ambiguous'}
assert set(human_346['human_ground_truth_label'].unique()).issubset(valid_classes), "Invalid classes found!"
assert human_346['claim_id'].isna().sum() == 0, "Missing claim_ids found!"

print(f"Validation successful:")
print(f"  Total unique pairs: {len(human_346)}")
print(f"  Unique claims: {human_346['claim_id'].nunique()}")
print(f"  Class distribution:\n{human_346['human_ground_truth_label'].value_counts().to_string()}")

# 2. Freeze the benchmark files
frozen_csv = 'data/calibration/frozen_human_ground_truth_benchmark_346.csv'
frozen_json = 'data/calibration/frozen_human_ground_truth_benchmark_346.json'
human_346.to_csv(frozen_csv, index=False)
human_346.to_json(frozen_json, orient='records', indent=2)

# Compute SHA-256 hashes
with open(frozen_csv, 'rb') as f:
    csv_sha = hashlib.sha256(f.read()).hexdigest()
with open(frozen_json, 'rb') as f:
    json_sha = hashlib.sha256(f.read()).hexdigest()

manifest = {
    "benchmark_name": "Frozen Human Ground-Truth Benchmark 346",
    "version": "1.0.0-frozen",
    "timestamp": "2026-11-19T18:15:00Z",
    "total_pairs": len(human_346),
    "unique_claims": int(human_346['claim_id'].nunique()),
    "class_distribution": human_346['human_ground_truth_label'].value_counts().to_dict(),
    "files": {
        "frozen_csv": {
            "path": frozen_csv,
            "sha256": csv_sha
        },
        "frozen_json": {
            "path": frozen_json,
            "sha256": json_sha
        }
    },
    "provenance": {
        "clean_92_cohort": 92,
        "unanimous_100_cohort": 100,
        "stage_b_50_cohort": 50,
        "stage_a_50_cohort": 50,
        "triage_splits_pilots": 54
    }
}

manifest_path = 'data/calibration/frozen_human_ground_truth_manifest.json'
with open(manifest_path, 'w') as f:
    json.dump(manifest, f, indent=2)

print(f"\nFrozen benchmark saved to:")
print(f"  CSV: {frozen_csv} (SHA-256: {csv_sha[:16]}...)")
print(f"  JSON: {frozen_json} (SHA-256: {json_sha[:16]}...)")
print(f"  Manifest: {manifest_path}")

print("\n" + "=" * 85)
print("PRIORITY 3: CREATE CLAIM-DISJOINT PARTITIONS")
print("=" * 85)

# Filter 341 concrete pairs (excluding 5 ambiguous)
concrete_341 = human_346[human_346['human_ground_truth_label'].isin(['class_0a', 'class_0b', 'class_1', 'class_2'])].copy()
concrete_341 = concrete_341.reset_index(drop=True)

# Deterministic claim-disjoint split (Seed = 42)
rng = np.random.RandomState(42)
all_claims = sorted(concrete_341['claim_id'].unique())
shuffled_claims = all_claims.copy()
rng.shuffle(shuffled_claims)

# 70% Calibration claims (45 claims), 30% Held-Out Test claims (20 claims)
cal_claims = sorted(shuffled_claims[:45])
test_claims = sorted(shuffled_claims[45:])

assert len(set(cal_claims).intersection(set(test_claims))) == 0, "Claim leak between splits!"

cal_partition = concrete_341[concrete_341['claim_id'].isin(cal_claims)].copy().reset_index(drop=True)
test_partition = concrete_341[concrete_341['claim_id'].isin(test_claims)].copy().reset_index(drop=True)

print(f"Calibration Split: {len(cal_claims)} claims, {len(cal_partition)} pairs")
print(cal_partition['human_ground_truth_label'].value_counts().to_dict())

print(f"Held-Out Test Split: {len(test_claims)} claims, {len(test_partition)} pairs")
print(test_partition['human_ground_truth_label'].value_counts().to_dict())

# Save partition datasets
cal_csv = 'data/calibration/human_ground_truth_calibration_partition_227.csv'
test_csv = 'data/calibration/human_ground_truth_test_partition_114.csv'
cal_partition.to_csv(cal_csv, index=False)
test_partition.to_csv(test_csv, index=False)

split_manifest = {
    "total_concrete_pairs": len(concrete_341),
    "total_unique_claims": len(all_claims),
    "split_seed": 42,
    "calibration_split": {
        "num_claims": len(cal_claims),
        "claim_ids": [int(c) for c in cal_claims],
        "num_pairs": len(cal_partition),
        "class_distribution": cal_partition['human_ground_truth_label'].value_counts().to_dict(),
        "csv_path": cal_csv
    },
    "held_out_test_split": {
        "num_claims": len(test_claims),
        "claim_ids": [int(c) for c in test_claims],
        "num_pairs": len(test_partition),
        "class_distribution": test_partition['human_ground_truth_label'].value_counts().to_dict(),
        "csv_path": test_csv
    }
}

split_manifest_path = 'data/calibration/claim_disjoint_partitions_human_341.json'
with open(split_manifest_path, 'w') as f:
    json.dump(split_manifest, f, indent=2)

print(f"Saved Claim-Disjoint Partition Manifest to: {split_manifest_path}")

print("\n" + "=" * 85)
print("PRIORITY 4 & 5: STAGE 1 CALIBRATION & SEPARATE EVALUATION ON CLASSES 2, 1, 0a, 0b")
print("=" * 85)

def evaluate_threshold(df, tau):
    sims = df['cosine_similarity'].values
    labels = df['human_ground_truth_label'].values
    passed = (sims >= tau)
    
    c2 = (labels == 'class_2')
    c1 = (labels == 'class_1')
    c0a = (labels == 'class_0a')
    c0b = (labels == 'class_0b')
    not_c2 = (~c2)
    
    rec_c2 = np.sum(passed & c2) / max(np.sum(c2), 1)
    prec_c2 = np.sum(passed & c2) / max(np.sum(passed), 1)
    f1_c2 = 2 * prec_c2 * rec_c2 / max(prec_c2 + rec_c2, 1e-8)
    
    rej_c1 = np.sum((~passed) & c1) / max(np.sum(c1), 1)
    rej_c0a = np.sum((~passed) & c0a) / max(np.sum(c0a), 1)
    rej_c0b = np.sum((~passed) & c0b) / max(np.sum(c0b), 1)
    rej_all_noise = np.sum((~passed) & not_c2) / max(np.sum(not_c2), 1)
    pass_rate = np.sum(passed) / len(df)
    
    return {
        "tau": round(float(tau), 2),
        "recall_c2": round(float(rec_c2), 4),
        "precision_c2": round(float(prec_c2), 4),
        "f1_c2": round(float(f1_c2), 4),
        "rejection_c1": round(float(rej_c1), 4),
        "rejection_c0a": round(float(rej_c0a), 4),
        "rejection_c0b": round(float(rej_c0b), 4),
        "rejection_all_noise": round(float(rej_all_noise), 4),
        "pass_rate": round(float(pass_rate), 4),
        "passed_count": int(np.sum(passed)),
        "total_count": len(df)
    }

# 1. Sweep on Calibration Partition
thresholds = np.arange(0.50, 0.855, 0.01)
cal_sweep_rows = [evaluate_threshold(cal_partition, t) for t in thresholds]
cal_sweep_df = pd.DataFrame(cal_sweep_rows)

# 2. Claim-Clustered Bootstrap on Calibration Partition (1,000 resamples across 45 claims)
print("Running 1,000 Claim-Clustered Bootstrap resamples on Calibration Partition...")
cal_cids = cal_partition['claim_id'].values
cal_sims = cal_partition['cosine_similarity'].values
cal_c2 = (cal_partition['human_ground_truth_label'] == 'class_2').values
cal_groups = {cid: np.where(cal_cids == cid)[0] for cid in cal_claims}

ci_rec_low = {}
ci_rec_high = {}
for t in [0.50, 0.52, 0.55, 0.58, 0.60, 0.62, 0.65, 0.70, 0.75, 0.80]:
    t = round(float(t), 2)
    b_rec = []
    for _ in range(1000):
        sampled = rng.choice(cal_claims, size=len(cal_claims), replace=True)
        idx = np.concatenate([cal_groups[c] for c in sampled])
        sub_c2 = cal_c2[idx]
        sub_pass = (cal_sims[idx] >= t) & sub_c2
        b_rec.append(np.sum(sub_pass) / max(np.sum(sub_c2), 1))
    ci_rec_low[t] = round(float(np.percentile(b_rec, 2.5)), 4)
    ci_rec_high[t] = round(float(np.percentile(b_rec, 97.5)), 4)

cal_sweep_df['ci_95_rec_low'] = cal_sweep_df['tau'].map(lambda t: ci_rec_low.get(round(t, 2), np.nan))
cal_sweep_df['ci_95_rec_high'] = cal_sweep_df['tau'].map(lambda t: ci_rec_high.get(round(t, 2), np.nan))

cal_sweep_csv = 'results/calibration/human_claim_disjoint_calibration_sweep.csv'
cal_sweep_df.to_csv(cal_sweep_csv, index=False)

# Select thresholds
# Conservative: Recall = 100%
# Empirical: Recall >= 98%
emp_row = cal_sweep_df[cal_sweep_df['recall_c2'] >= 0.98].sort_values(by='tau', ascending=False).iloc[0]
cons_row = cal_sweep_df[cal_sweep_df['recall_c2'] >= 1.0].sort_values(by='tau', ascending=False).iloc[0]
tau_emp = float(emp_row['tau'])
tau_cons = float(cons_row['tau'])

print(f"\nOptimal Calibration Thresholds Selected on Calibration Partition:")
print(f"  Empirical Threshold (Recall >= 98%): tau* = {tau_emp}")
print(f"  Conservative Threshold (Recall = 100%): tau* = {tau_cons}")

# 3. Evaluate on Completely Untouched Held-Out Test Partition (20 claims, 114 pairs)
print("\n" + "-" * 85)
print(f"EVALUATION ON HELD-OUT TEST PARTITION (20 CLAIMS, 114 PAIRS)")
print("-" * 85)

test_eval_emp = evaluate_threshold(test_partition, tau_emp)
test_eval_cons = evaluate_threshold(test_partition, tau_cons)
test_eval_human_human = evaluate_threshold(test_partition, 0.75)

test_results = {
    "tau_star_emp (0.60)": test_eval_emp,
    "tau_star_cons (0.55)": test_eval_cons,
    "tau_human_human (0.75)": test_eval_human_human
}

test_res_json = 'results/calibration/human_claim_disjoint_test_evaluation.json'
with open(test_res_json, 'w') as f:
    json.dump(test_results, f, indent=2)

print("\nHeld-Out Test Partition Evaluation Table:")
eval_rows = [
    {"Setting": f"Empirical tau* = {tau_emp}", **test_eval_emp},
    {"Setting": f"Conservative tau* = {tau_cons}", **test_eval_cons},
    {"Setting": "Human-Human tau = 0.75", **test_eval_human_human}
]
eval_table = pd.DataFrame(eval_rows)
print(eval_table[['Setting', 'recall_c2', 'rejection_c0b', 'rejection_c0a', 'rejection_c1', 'rejection_all_noise', 'precision_c2', 'f1_c2', 'pass_rate']].to_string(index=False))

# 4. 5-Fold Claim-Level Cross Validation
print("\n" + "-" * 85)
print("5-FOLD CLAIM-DISJOINT CROSS-VALIDATION (13 CLAIMS PER FOLD)")
print("-" * 85)

fold_claims = np.array_split(shuffled_claims, 5)
cv_records = []

for k in range(5):
    val_c = fold_claims[k]
    train_c = [c for c in shuffled_claims if c not in val_c]
    
    val_df = concrete_341[concrete_341['claim_id'].isin(val_c)]
    train_df = concrete_341[concrete_341['claim_id'].isin(train_c)]
    
    res_val = evaluate_threshold(val_df, tau_emp)
    res_val['fold'] = k + 1
    res_val['val_claims_count'] = len(val_c)
    res_val['val_pairs_count'] = len(val_df)
    cv_records.append(res_val)

cv_df = pd.DataFrame(cv_records)
cv_report = {
    "tau_evaluated": tau_emp,
    "folds": cv_records,
    "mean_metrics": {
        "recall_c2": round(float(cv_df['recall_c2'].mean()), 4),
        "recall_c2_std": round(float(cv_df['recall_c2'].std()), 4),
        "rejection_c0b": round(float(cv_df['rejection_c0b'].mean()), 4),
        "rejection_c0b_std": round(float(cv_df['rejection_c0b'].std()), 4),
        "rejection_c0a": round(float(cv_df['rejection_c0a'].mean()), 4),
        "rejection_c0a_std": round(float(cv_df['rejection_c0a'].std()), 4),
        "rejection_c1": round(float(cv_df['rejection_c1'].mean()), 4),
        "rejection_c1_std": round(float(cv_df['rejection_c1'].std()), 4),
        "f1_c2": round(float(cv_df['f1_c2'].mean()), 4),
        "f1_c2_std": round(float(cv_df['f1_c2'].std()), 4),
        "pass_rate": round(float(cv_df['pass_rate'].mean()), 4),
        "pass_rate_std": round(float(cv_df['pass_rate'].std()), 4)
    }
}

cv_json = 'results/calibration/human_5fold_claim_cv_report.json'
with open(cv_json, 'w') as f:
    json.dump(cv_report, f, indent=2)

print("\n5-Fold Claim-Level CV Summary (Mean +/- Std at tau* = 0.60):")
print(f"  Class 2 Candidate Recall: {cv_report['mean_metrics']['recall_c2']:.4f} +/- {cv_report['mean_metrics']['recall_c2_std']:.4f}")
print(f"  Class 0b Noise Rejection: {cv_report['mean_metrics']['rejection_c0b']:.4f} +/- {cv_report['mean_metrics']['rejection_c0b_std']:.4f}")
print(f"  Class 0a Stance Rejection: {cv_report['mean_metrics']['rejection_c0a']:.4f} +/- {cv_report['mean_metrics']['rejection_c0a_std']:.4f}")
print(f"  Class 1 Nuance Rejection: {cv_report['mean_metrics']['rejection_c1']:.4f} +/- {cv_report['mean_metrics']['rejection_c1_std']:.4f}")
print(f"  Class 2 F1-Score: {cv_report['mean_metrics']['f1_c2']:.4f} +/- {cv_report['mean_metrics']['f1_c2_std']:.4f}")
print(f"  Pass Rate to Stage 2: {cv_report['mean_metrics']['pass_rate']:.4f} +/- {cv_report['mean_metrics']['pass_rate_std']:.4f}")
