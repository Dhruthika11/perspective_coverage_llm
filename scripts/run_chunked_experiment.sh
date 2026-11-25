#!/usr/bin/env bash
# ==============================================================================
# Run Chunked Max-Pooling Experiment on Flair2 GPU
# ==============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."
echo "Starting V4 Chunked Inference Experiment on $(hostname)..."

if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate perspectrum || true
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate perspectrum || true
fi

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

echo "=== Running Experiment 1: 2-Sentence Group Chunks (window_size=2) ==="
python src/analysis/evaluate_v4_chunked_inference.py --window_size 2 --batch_size 64

echo "=== Experiment Complete! Results saved to results/metrics/v4_chunked_1118_predictions.csv ==="

