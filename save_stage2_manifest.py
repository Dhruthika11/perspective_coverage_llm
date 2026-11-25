import json
import pandas as pd

df = pd.read_csv('results/calibration/stage2_cross_encoder_predictions_341.csv')

train_df = df[df['partition'] == 'train']
cal_df = df[df['partition'] == 'calibration']
test_df = df[df['partition'] == 'test']

manifest = {
    "benchmark": "Stage 2 Human Ground-Truth Benchmark",
    "total_concrete_pairs": len(df),
    "total_claims": int(df['claim_id'].nunique()),
    "split_seed": 42,
    "partitions": {
        "train": {
            "num_claims": int(train_df['claim_id'].nunique()),
            "num_pairs": len(train_df),
            "class_distribution": train_df['human_ground_truth_label'].value_counts().to_dict(),
            "claim_ids": [int(c) for c in sorted(train_df['claim_id'].unique())]
        },
        "calibration": {
            "num_claims": int(cal_df['claim_id'].nunique()),
            "num_pairs": len(cal_df),
            "class_distribution": cal_df['human_ground_truth_label'].value_counts().to_dict(),
            "claim_ids": [int(c) for c in sorted(cal_df['claim_id'].unique())]
        },
        "test": {
            "num_claims": int(test_df['claim_id'].nunique()),
            "num_pairs": len(test_df),
            "class_distribution": test_df['human_ground_truth_label'].value_counts().to_dict(),
            "claim_ids": [int(c) for c in sorted(test_df['claim_id'].unique())]
        }
    }
}

with open('data/calibration/stage2_claim_disjoint_partitions_human_341.json', 'w') as f:
    json.dump(manifest, f, indent=2)

print("Saved Stage 2 3-way partition manifest!")
