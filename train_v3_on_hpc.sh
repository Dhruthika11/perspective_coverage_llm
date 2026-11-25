#!/bin/bash
# ==============================================================================
# IITGN HPC: Train Argument-NLI DeBERTa-v3 Version 3 (Coverage-Aware)
# ==============================================================================
set -euo pipefail

echo "=========================================================="
echo "Starting Argument-NLI V3 (Coverage-Aware) Fine-Tuning"
echo "Host: $(hostname)"
echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================================="

mkdir -p logs models/argument_nli_deberta_v3_v3 results/metrics results/paper_tables

# Train Version 3 on GPU
python -m src.training.train_argument_nli_v2_v3 \
    --version v3 \
    --model_name "cross-encoder/nli-deberta-v3-small" \
    --batch_size 16 \
    --num_epochs 3 \
    --learning_rate 2e-5 \
    --max_length 256 \
    2>&1 | tee logs/train_nli_v3.log

echo "=========================================================="
echo "Evaluating V1, V2, and V3 Models on Locked Test Split..."
echo "=========================================================="

python -m src.analysis.evaluate_nli_models_comparison \
    2>&1 | tee logs/evaluate_nli_v1_v2_v3.log

echo "=========================================================="
echo "Running Full Reviewer Protocol: Common-Threshold, Ranking AUROC/AUPRC, and Separately Calibrated Thresholds..."
echo "=========================================================="

python -m src.analysis.evaluate_v3_benchmark_protocol \
    --v3_model_path "models/argument_nli_deberta_v3_v3" \
    --test_csv "data/calibration/tri_dataset_750_pairs.csv" \
    --cal_csv "data/calibration/expanded_stage1_calibration_pool.csv" \
    2>&1 | tee logs/evaluate_v3_benchmark_protocol.log

echo "=========================================================="
echo "Fine-Tuning and Evaluation Completed Successfully!"
echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================================="

