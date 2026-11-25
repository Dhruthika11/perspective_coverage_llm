#!/bin/bash
# ==============================================================================
# Direct V5 Perspective Coverage for Unseen / Out-of-Family Models (No Stage 1)
# Evaluates all models OUTSIDE the 4 V5 training families:
#   - Mistral-7B (mistralai/Mistral-7B-Instruct-v0.3)
#   - Gemma-3-12B (google/gemma-3-12b-it)
#   - Llama-3.2-1B (meta-llama/Llama-3.2-1B-Instruct)
#   - Llama-3.2-3B (meta-llama/Llama-3.2-3B-Instruct)
#   - Gemma-2-2B (google/gemma-2-2b-it)
#   - SmolLM2-1.7B (HuggingFaceTB/SmolLM2-1.7B-Instruct)
# ==============================================================================

set -uo pipefail

mkdir -p logs
mkdir -p results

V5_DIR="models/ensemble_data_only"
BATCH_SIZE=128
LOG_FILE="logs/run_unseen_models_coverage.log"

echo "=================================================================="
echo "STARTING DIRECT V5 COVERAGE FOR UNSEEN MODELS AT $(date '+%Y-%m-%d %H:%M:%S')"
echo "Verifier: $V5_DIR | Batch Size: $BATCH_SIZE"
echo "Stage 1 Retrieval Filter: BYPASSED (Pure Cross-Encoder)"
echo "Log: $LOG_FILE"
echo "=================================================================="

# Execute Direct V5 Verification across all unseen models
python src/analysis/run_unseen_models_coverage.py \
  --v5_model_dir "$V5_DIR" \
  --batch_size "$BATCH_SIZE" \
  2>&1 | tee -a "$LOG_FILE"

echo ""
echo "=================================================================="
echo "UNSEEN MODELS EVALUATION COMPLETE AT $(date '+%Y-%m-%d %H:%M:%S')"
echo "Results saved to:"
echo "  - Markdown: results/unseen_models_coverage_summary.md"
echo "  - JSON:     results/unseen_models_coverage_report.json"
echo "=================================================================="

