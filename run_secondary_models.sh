#!/bin/bash
# ==============================================================================
# IITGN HPC Automated Secondary Models Runner
# Runs: SmolLM2 (135M, 360M, 1.7B), OLMoE (1B-7B), Gemma 2 (2B, 9B, 27B),
# and Llama (1B, 3B, 8B, 70B)
# ==============================================================================

set -uo pipefail

mkdir -p logs

declare -a EXPERIMENTS=(
  # --- SmolLM2 Family (Sub-Billion Extremes) ---
  "smollm2-135m 24"
  "smollm2-360m 20"
  "smollm2-1.7b 12"

  # --- Sparse MoE ---
  "olmoe-1b-7b 6"

  # --- Google Gemma 2 Family ---
  "gemma-2b 10"
  "gemma-9b 4"
  "gemma-27b 2"

  # --- Meta LLaMA Family ---
  "llama-1b 12"
  "llama-3b 8"
  "llama-8b 6"
)

echo "=========================================================="
echo "Starting Secondary Models Pipeline at $(date '+%Y-%m-%d %H:%M:%S')"
echo "Total Models: ${#EXPERIMENTS[@]}"
echo "=========================================================="

job_num=1
total_jobs=${#EXPERIMENTS[@]}

for exp in "${EXPERIMENTS[@]}"; do
  read -r model bs <<< "$exp"
  log_file="logs/run_${model}.log"

  echo ""
  echo "=========================================================="
  echo "[$job_num/$total_jobs] Starting $model (Batch Size: $bs)"
  echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "Log: $log_file"
  echo "=========================================================="

  # 1. Generation (resume logic automatically skips claims already finished)
  echo ">> Step 1/2: Running generate_llama for $model..."
  python -m src.generation.generate_llama --model "$model" --batch_size "$bs" 2>&1 | tee -a "$log_file"
  if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo ">> [ERROR] Generation failed for $model. Check $log_file" | tee -a "$log_file"
    exit 1
  fi

  # 2. Postprocessing
  echo ">> Step 2/2: Running postprocess_generations for $model..."
  python -m src.generation.postprocess_generations --model "$model" 2>&1 | tee -a "$log_file"
  if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo ">> [ERROR] Postprocessing failed for $model. Check $log_file" | tee -a "$log_file"
    exit 1
  fi

  echo ">> [$job_num/$total_jobs] Completed $model successfully at $(date '+%Y-%m-%d %H:%M:%S')"
  job_num=$((job_num + 1))
done

echo ""
echo "=========================================================="
echo "ALL $total_jobs SECONDARY MODELS COMPLETED SUCCESSFULLY!"
echo "Finished at $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================================="

