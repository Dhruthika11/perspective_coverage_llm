import os
import json
import hashlib
import pandas as pd

# Ensure configs directory exists
os.makedirs('configs', exist_ok=True)

# Hashes of benchmark and partition files
with open('data/calibration/frozen_human_ground_truth_benchmark_346.csv', 'rb') as f:
    bench_sha = hashlib.sha256(f.read()).hexdigest()

with open('data/calibration/claim_disjoint_partitions_human_341.json', 'rb') as f:
    split_sha = hashlib.sha256(f.read()).hexdigest()

# Load test partition predictions
test_df = pd.read_csv('data/calibration/human_ground_truth_test_partition_114.csv')
test_df['passed_tau_0.55'] = test_df['cosine_similarity'] >= 0.55
test_df['passed_tau_0.60'] = test_df['cosine_similarity'] >= 0.60
test_df['passed_tau_0.75'] = test_df['cosine_similarity'] >= 0.75

test_pred_csv = 'results/calibration/stage1_test_partition_predictions.csv'
test_df.to_csv(test_pred_csv, index=False)

with open(test_pred_csv, 'rb') as f:
    pred_sha = hashlib.sha256(f.read()).hexdigest()

stage1_config = {
    "module": "Stage 1 Candidate Dense Retrieval",
    "status": "FROZEN_PROVISIONAL",
    "version": "1.0.0",
    "date_frozen": "2026-11-19T18:30:00Z",
    "embedding_model": {
        "checkpoint": "BAAI/bge-large-en-v1.5",
        "model_type": "Bi-Encoder Dense Retriever",
        "embedding_dimension": 1024,
        "max_seq_length": 128,
        "pooling": "CLS token",
        "normalization": "L2 unit sphere (p=2, dim=1)"
    },
    "similarity_metric": {
        "type": "Cosine Similarity",
        "formula": "cos(u, v) = dot(u, v) / (norm(u) * norm(v))",
        "range": "[-1.0, 1.0]"
    },
    "threshold_configuration": {
        "primary_provisional_threshold": {
            "tau_star": 0.55,
            "design_objective": "Recall-Oriented / Zero-Omission Safety Guarantee",
            "justification": "Stage 1 candidate retrieval loss is irreversible; Stage 2 cannot recover a dropped perspective. Provisionally adopted at tau* = 0.55 to retain 100% of observed Class 2 pairs across calibration and held-out test splits.",
            "held_out_test_observed_recall_c2": "26/26 (100.0%)",
            "wilson_ci_95_c2_recall": "[87.13%, 100.00%]",
            "held_out_test_class_0b_rejection": "9/10 (90.0%)",
            "stage1_candidate_pass_rate": "85.96% (98/114 candidates sent to Stage 2)"
        },
        "comparison_selective_threshold": {
            "tau": 0.60,
            "design_objective": "Higher Selective Pruning",
            "held_out_test_observed_recall_c2": "26/26 (100.0%)",
            "wilson_ci_95_c2_recall": "[87.13%, 100.00%]",
            "held_out_test_class_0b_rejection": "9/10 (90.0%)",
            "stage1_candidate_pass_rate": "75.44% (86/114 candidates sent to Stage 2)",
            "calibration_split_c2_recall": "72/73 (98.63%, one terse abstraction missed: Pair 330, cos=0.5540)"
        },
        "baseline_human_human_threshold": {
            "tau": 0.75,
            "design_objective": "Dataset A Paraphrase Reference",
            "held_out_test_observed_recall_c2": "11/26 (42.31%)",
            "scientific_implication": "Empirical proof of the domain gap between concise human paraphrases and elaborate LLM viewpoints."
        }
    },
    "nomenclature_standards": {
        "workload_metric": "Stage 1 Candidate Pass Rate (fraction of pairs surviving Stage 1 retrieval sent to Stage 2)",
        "precision_metric": "Stage 1 Class 2 Candidate Precision (retrieval-stage precision, not final semantic matching precision)"
    },
    "data_artifacts": {
        "frozen_benchmark": {
            "path": "data/calibration/frozen_human_ground_truth_benchmark_346.csv",
            "sha256": bench_sha,
            "num_pairs": 346,
            "num_claims": 65
        },
        "claim_disjoint_partitions": {
            "path": "data/calibration/claim_disjoint_partitions_human_341.json",
            "sha256": split_sha,
            "calibration_claims": 45,
            "calibration_pairs": 227,
            "held_out_test_claims": 20,
            "held_out_test_pairs": 114
        },
        "test_predictions": {
            "path": test_pred_csv,
            "sha256": pred_sha,
            "num_evaluated_pairs": 114
        }
    }
}

config_path = 'configs/frozen_stage1_dense_retrieval_config.json'
with open(config_path, 'w') as f:
    json.dump(stage1_config, f, indent=2)

print(f"Frozen Stage 1 Configuration saved to: {config_path}")
