#!/usr/bin/env bash
# ==============================================================================
# Evaluate Clean Rebalanced LOGFO Suite (Leave-One-Generator-Family-Out) on HPC
#
# Enforces:
#   1. Standing Leak Check Audit (0.00% verbatim containment, matched overlap)
#   2. Real Multi-Family Benchmark (140 pos / 140 neg per family across Qwen, LLaMA, Gemma, OLMo)
#   3. Zero test-claim contamination (strictly non-test claims)
#   4. Pre-registered Success Criteria audit with Claim-Clustered 95% CIs (B=1,000)
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

echo "=== Running Clean LOGFO Evaluation on $(hostname) ==="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true
fi

# Step 1: Run Standing Leak Check Audit
echo "--- Step 1: Running Automated Leak Check Audit across Datasets ---"
python src/analysis/leak_check_audit.py

# Step 2: Run Rebalanced LOGFO Suite with Claim-Clustered 95% CIs (B=1,000)
echo "--- Step 2: Running Rebalanced LOGFO Suite & Zero-Shot on Clean Benchmark ---"
python src/analysis/evaluate_rebalanced_logfo_and_zeroshot.py \
    --benchmark_csv data/logfo_benchmark/clean_balanced_logfo_benchmark.csv \
    --train_pool_csv data/nli_training_v5/train.csv \
    --epochs 4 \
    --batch_size 16 \
    --lr 2e-5 \
    --n_bootstraps 1000 \
    --output_report results/metrics/rebalanced_logfo_benchmark_report.json \
    --output_md results/metrics/rebalanced_logfo_benchmark_summary.md

echo "=== Rebalanced Clean LOGFO Evaluation Complete! ==="
cat results/metrics/rebalanced_logfo_benchmark_summary.md

