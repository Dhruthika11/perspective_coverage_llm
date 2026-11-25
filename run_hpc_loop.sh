#!/bin/bash
# ==============================================================================
# IITGN HPC Automated Experiment Runner
# Runs OLMo 2 Alignment Ladder (Base -> SFT -> DPO -> Instruct)
# followed by Qwen 2.5 Prompt Ablation (P2: 3B, 7B, 14B)
# ==============================================================================

set -uo pipefail

mkdir -p logs

declare -a EXPERIMENTS=(
  "olmo2-7b-base 6 p1"
  "olmo2-7b-sft 6 p1"
  "olmo2-7b-dpo 6 p1"
  "olmo2-7b-instruct 6 p1"
  "3b 8 p2"
  "qwen-7b 6 p2"
  "14b 4 p2"
)

echo "=========================================================="
echo "Starting 7-Job Pipeline at $(date '+%Y-%m-%d %H:%M:%S')"
echo "Total Jobs: ${#EXPERIMENTS[@]}"
echo "=========================================================="

job_num=1
total_jobs=${#EXPERIMENTS[@]}

for exp in "${EXPERIMENTS[@]}"; do
  read -r model bs prompt <<< "$exp"
  log_file="logs/run_${model}_${prompt}.log"

  echo ""
  echo "=========================================================="
  echo "[$job_num/$total_jobs] Starting $model (Batch Size: $bs, Prompt: $prompt)"
  echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "Log: $log_file"
  echo "=========================================================="

  # 1. Generation
  echo ">> Step 1/2: Running generate_llama..."
  python -m src.generation.generate_llama --model "$model" --batch_size "$bs" --prompt_version "$prompt" 2>&1 | tee -a "$log_file"
  if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo ">> [ERROR] Generation failed for $model ($prompt). Aborting pipeline." | tee -a "$log_file"
    exit 1
  fi

  # 2. Postprocessing
  echo ">> Step 2/2: Running postprocess_generations..."
  python -m src.generation.postprocess_generations --model "$model" --prompt_version "$prompt" 2>&1 | tee -a "$log_file"
  if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo ">> [ERROR] Postprocessing failed for $model ($prompt). Aborting pipeline." | tee -a "$log_file"
    exit 1
  fi

  echo ">> [$job_num/$total_jobs] Completed $model ($prompt) successfully at $(date '+%Y-%m-%d %H:%M:%S')"
  job_num=$((job_num + 1))
done

echo ""
echo "=========================================================="
echo "ALL $total_jobs JOBS COMPLETED SUCCESSFULLY!"
echo "Finished at $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================================="

