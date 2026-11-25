#!/usr/bin/env bash
# ==============================================================================
# End-to-End Two-Stage Perspective Coverage Evaluation on HPC (Flair2)
#
# Frozen Main Model Scope:
#   Generator Families Used to Train V5: Qwen, LLaMA, Gemma, OLMo
#
# Pipeline:
#   1. Dense Retrieval (BGE Cosine tau >= 0.75)
#   2. Frozen Main V5 Data-Only Verifier (T* = 1.9104, theta_F1* = 0.52, theta_prec90* = 0.83)
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

echo "=== Running In-Family Two-Stage Perspective Coverage on $(hostname) ==="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true
fi

# Step 1: Prepare processed generations for in-family models
echo "--- Step 1: Preparing processed generations for in-family models ---"
python src/data_prep/prepare_multimodel_processed_generations.py

# Step 2: Run End-to-End Two-Stage Coverage Suite (BGE Embeddings + Frozen V5 Verifier)
echo "--- Step 2: Executing Embeddings & Two-Stage Perspective Coverage ---"
python src/analysis/run_in_family_two_stage_coverage.py \
    --v5_model_dir models/ensemble_data_only \
    --stage1_tau 0.75 \
    --temperature 1.9104 \
    --threshold_f1 0.52 \
    --threshold_prec90 0.83 \
    --batch_size 32 \
    --output_report results/in_family_two_stage_coverage_report.json \
    --output_md results/in_family_two_stage_coverage_summary.md

echo "=== Two-Stage Perspective Coverage Evaluation Complete! ==="
cat results/in_family_two_stage_coverage_summary.md

