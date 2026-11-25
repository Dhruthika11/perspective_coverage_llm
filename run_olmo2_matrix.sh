#!/bin/bash
# ==============================================================================
# IITGN HPC Automated OLMo 2 Post-Training Alignment Ladder (4x4 Matrix)
# Scales: 1B, 7B, 13B, 32B
# Stages: Base -> SFT -> DPO -> Instruct (RLVR)
# Total Checkpoints: 16
# ==============================================================================

set -uo pipefail

mkdir -p logs

declare -a OLMO2_MATRIX=(
  # --- 1B Scale ---
  "olmo2-1b-base 16"
  "olmo2-1b-sft 16"
  "olmo2-1b-dpo 16"
  "olmo2-1b-instruct 16"

  # --- 7B Scale ---
  "olmo2-7b-base 6"
  "olmo2-7b-sft 6"
  "olmo2-7b-dpo 6"
  "olmo2-7b-instruct 6"

  # --- 13B Scale ---
  "olmo2-13b-base 4"
  "olmo2-13b-sft 4"
  "olmo2-13b-dpo 4"
  "olmo2-13b-instruct 4"

  # --- 32B Scale ---
  "olmo2-32b-base 2"
  "olmo2-32b-sft 2"
  "olmo2-32b-dpo 2"
  "olmo2-32b-instruct 2"
)

echo "=========================================================="
echo "Starting OLMo 2 Alignment Ladder Matrix Pipeline"
echo "Total Jobs: ${#OLMO2_MATRIX[@]}"
echo "Start Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================================="

job_num=1
total_jobs=${#OLMO2_MATRIX[@]}

for exp in "${OLMO2_MATRIX[@]}"; do
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
echo "ALL $total_jobs OLMo 2 MATRIX RUNS COMPLETED SUCCESSFULLY!"
echo "Finished at $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================================="
