#!/usr/bin/env bash
# ==============================================================================
# Argument-NLI V4 Fine-Tuning Runner on Flair2 HPC Cluster
# ==============================================================================
# Implements Priority 2 (Contrastive Margin Loss) and Priority 3 (4-Class Taxonomy)
# using multi-model arguments across Gemma-2, Llama-3, OLMo-2, and Qwen-2.5.
#
# Target Taxonomy:
#   0: contradiction
#   1: full_coverage
#   2: no_coverage (same-policy diff reason + same-topic diff concept)
#   3: partial_coverage
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

echo "=== Starting Argument-NLI V4 Training on Flair2 GPU ==="
echo "Node: $(hostname) | Date: $(date)"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true
fi

# Step 1: Ensure V4 dataset splits are generated and clean
echo "--- Step 1: Preparing V4 contrastive pilot dataset ---"
python src/training/prepare_v4_pilot_dataset.py

# Step 2: Run V4 Model Training
echo "--- Step 2: Training V4 Cross-Encoder with 4-Class Taxonomy & Margin Regularization ---"
python src/training/train_argument_nli_v4.py \
    --train_csv data/nli_training_v4/train.csv \
    --val_csv data/nli_training_v4/val.csv \
    --output_dir models/argument_nli_deberta_v3_v4 \
    --model_name cross-encoder/nli-deberta-v3-base \
    --num_classes 4 \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --num_epochs 4 \
    --max_length 256 \
    --use_contrastive_loss \
    --contrastive_weight 0.3 \
    --margin 0.2

echo "=== V4 Fine-tuning Complete! Artifacts saved to models/argument_nli_deberta_v3_v4 ==="

