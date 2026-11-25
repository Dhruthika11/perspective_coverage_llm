#!/usr/bin/env bash
# ==============================================================================
# Evaluate Existing V5 Data-Only Verifier on Unseen Mistral-7B Arguments
# ==============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd)"
echo "Project root: ${PROJECT_ROOT}"

# Activate conda environment
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate perspectrum || true
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate perspectrum || true
fi

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TOKENIZERS_PARALLELISM=false

echo "=== Evaluating Existing V5 Data-Only Model on Unseen Mistral-7B on $(hostname) ==="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true
fi

# Step 1: Prepare Mistral Out-Of-Domain Benchmark
echo "--- Step 1: Preparing Mistral benchmark datasets ---"
python src/training/prepare_mistral_benchmark.py

# Step 2: Run Out-Of-Domain Evaluation
echo "--- Step 2: Running Zero-Shot DeBERTa vs Existing V5 Data-Only Head-to-Head ---"
python src/analysis/evaluate_v5_on_mistral.py \
    --v5_model_dir models/ensemble_data_only \
    --zeroshot_model cross-encoder/nli-deberta-v3-base \
    --batch_size 32 \
    --temperature 1.9104 \
    --threshold_f1 0.52 \
    --threshold_prec90 0.83 \
    --n_bootstraps 1000 \
    --output_report results/metrics/v5_mistral_ood_evaluation_report.json \
    --output_md results/metrics/v5_mistral_ood_evaluation_summary.md

echo "=== Mistral Out-Of-Domain Evaluation Complete! ==="
cat results/metrics/v5_mistral_ood_evaluation_summary.md

