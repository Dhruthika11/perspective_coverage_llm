import hashlib
import json
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

print("=" * 85)
print("FREEZING VERIFIER ARCHITECTURE, WEIGHTS, DATA, AND PREDICTIONS")
print("=" * 85)

# 1. Fit and Freeze Probe Model Artifact
bench = pd.read_csv('data/calibration/human_llm_benchmark_1700.csv')
df = pd.read_csv('results/calibration/stage2_cross_encoder_predictions_341.csv')
merged = pd.merge(df, bench[['pair_id', 'stance_gold', 'stance_generated']], on='pair_id', how='left')

train_c21 = merged[(merged['partition'] == 'train') & (merged['human_ground_truth_label'].isin(['class_2', 'class_1']))].copy()
feats = ['cosine_similarity', 'p_entail', 'p_neutral']
scaler = StandardScaler()
X_tr = scaler.fit_transform(train_c21[feats].values)
y_tr = (train_c21['human_ground_truth_label'] == 'class_2').astype(int).values

clf = LogisticRegression(C=1.0, random_state=42)
clf.fit(X_tr, y_tr)

probe_artifact = {
    "model_name": "Stage2b_Warrant_Alignment_Probe",
    "model_type": "LogisticRegression(C=1.0, random_state=42)",
    "training_partition": "train (32 claims, 104 pairs: 51 class_2, 53 class_1)",
    "input_features": feats,
    "scaler_mean": scaler.mean_.tolist(),
    "scaler_scale": scaler.scale_.tolist(),
    "coefficients": {
        "cosine_similarity": float(clf.coef_[0][0]),
        "p_entail": float(clf.coef_[0][1]),
        "p_neutral": float(clf.coef_[0][2])
    },
    "intercept": float(clf.intercept_[0]),
    "calibrated_threshold": 0.20,
    "sensitivity_threshold": 0.35,
    "calibration_partition": "calibration (16 claims, 69 pairs: 24 class_2, 45 class_1)",
    "calibration_metric": "F1-optimal (Cal F1 = 0.6301)"
}

model_json_path = 'results/calibration/frozen_warrant_probe_model.json'
with open(model_json_path, 'w') as f:
    json.dump(probe_artifact, f, indent=2)
print(f"Saved frozen probe parameters to: {model_json_path}")

# 2. Compile Provenance Record
provenance_record = {
    "pipeline_architecture": "Stage 1 (Retrieval) -> Stage 2a (Stance) -> Stage 2b (Warrant Probe)",
    "stage_1_retrieval": {
        "model": "BAAI/bge-large-en-v1.5",
        "pooling": "CLS",
        "normalization": "L2",
        "threshold": 0.55,
        "calibration_scope": "Claims [:45] (seed=42)",
        "objective": "Permissive candidate recall (preserves 100% of Class 2 on test claims)"
    },
    "stage_2a_stance": {
        "rule": "Concordant polarity or UNKNOWN (rejects only explicit support-undermine contradiction)",
        "handling_of_unknown": "Preserved and passed to Stage 2b",
        "test_performance": "Rejects 17/23 surviving Class 0a pairs (73.91%) with zero Class 2 loss"
    },
    "stage_2b_warrant_probe": {
        "features": feats,
        "weights": probe_artifact["coefficients"],
        "intercept": probe_artifact["intercept"],
        "threshold": 0.20,
        "calibration_loocv_mean": 0.208,
        "calibration_loocv_std": 0.027,
        "test_performance": {
            "class_2_recall": 0.9583,
            "class_2_recall_95_ci_wilson": [0.7976, 0.9926],
            "class_2_recall_95_ci_clustered": [0.8929, 1.0000],
            "class_2_precision_full_cascade": 0.5476,
            "class_2_precision_95_ci_wilson": [0.3985, 0.6884],
            "class_2_precision_95_ci_clustered": [0.3750, 0.7500],
            "class_2_f1": 0.6970,
            "class_2_f1_95_ci_clustered": [0.5454, 0.8333],
            "class_1_rejection_e2e": 0.4138,
            "class_0a_rejection_e2e": 0.9615,
            "class_0b_rejection_e2e": 0.8889
        }
    }
}

prov_path = 'results/calibration/threshold_and_bootstrap_provenance.json'
with open(prov_path, 'w') as f:
    json.dump(provenance_record, f, indent=2)
print(f"Saved threshold provenance record to: {prov_path}")

# 3. Calculate SHA-256 Hashes
files_to_hash = [
    # Code
    'run_c1_full_cascade.py',
    'calculate_claim_clustered_cascade_cis.py',
    'run_claim_sensitivity_analysis.py',
    'inspect_all_17_c1_fps.py',
    'run_b1_feature_ablation.py',
    'run_b2_calibration_and_test.py',
    'run_experiment_a_stance_and_reasoning.py',
    # Data
    'data/calibration/frozen_human_ground_truth_benchmark_346.csv',
    'data/calibration/stage2_claim_disjoint_partitions_human_341.json',
    'data/calibration/claim_disjoint_partitions_human_341.json',
    'data/calibration/human_llm_benchmark_1700.csv',
    # Results & Reports
    'results/calibration/stage2_cross_encoder_predictions_341.csv',
    'results/calibration/experiment_c1_cascade_report.json',
    'results/calibration/experiment_b1_ablation_report.json',
    'results/calibration/frozen_warrant_probe_model.json',
    'results/calibration/threshold_and_bootstrap_provenance.json'
]

def get_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fp:
        while chunk := fp.read(8192):
            h.update(chunk)
    return h.hexdigest()

manifest_entries = {}
for fpath in files_to_hash:
    if os.path.exists(fpath):
        manifest_entries[fpath] = {
            "size_bytes": os.path.getsize(fpath),
            "sha256": get_sha256(fpath)
        }
    else:
        print(f"Warning: File not found for hashing: {fpath}")

manifest = {
    "frozen_entity": "Production_Perspective_Matching_Cascaded_Verifier",
    "timestamp": "2026-11-20T08:15:00Z",
    "researcher": "Dhruthika Thoutam",
    "institution": "IIT Gandhinagar",
    "specification": {
        "stage_1": "BGE-Large retrieval at tau* = 0.55",
        "stage_2a": "Stance concordance with UNKNOWN uncertainty preservation",
        "stage_2b": "Logistic probe on cosine + entailment + neutral at T* = 0.20"
    },
    "files": manifest_entries
}

manifest_path = 'results/calibration/frozen_verifier_manifest.json'
with open(manifest_path, 'w') as f:
    json.dump(manifest, f, indent=2)
print(f"Saved complete frozen verifier manifest with SHA-256 hashes to: {manifest_path}")

print("\n" + "=" * 85)
print("FROZEN VERIFIER MANIFEST ENTRIES:")
print("=" * 85)
for fpath, meta in manifest_entries.items():
    print(f"{fpath:65s} | {meta['size_bytes']:>8d} bytes | {meta['sha256'][:16]}...")
