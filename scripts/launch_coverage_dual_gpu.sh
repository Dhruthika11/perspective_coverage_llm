#!/usr/bin/env bash
# ==============================================================================
# Master Dual-GPU Perspective Coverage Runner on Flair2
#
# Hardware: Dual NVIDIA L40S (GPU 0 & GPU 1)
# Models: 34 models partitioned across GPU 0 and GPU 1
# Pipeline:
#   1. Stage 1: BGE-Large Dense Retrieval (tau = 0.55)
#   2. Stage 2a: Stance Concordance Filter (Preserves UNKNOWN)
#   3. Stage 2b: DeBERTa-v3 Warrant Probe M0 (T* = 0.20) + Shadow M_soft
#   4. Kuhn-Munkres Maximum Bipartite Matching on Human Perspective Clusters
#   5. Scaling and Coverage Analysis across all 34 models
# ==============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd)"
PYTHON="/home/srujan.sai/miniconda3/envs/perspectrum/bin/python"

mkdir -p logs results/coverage/per_model

echo "================================================================================"
echo "LAUNCHING FULL PERSPECTRUM COVERAGE ON DUAL NVIDIA L40S GPUS"
echo "Project Root: ${PROJECT_ROOT}"
echo "Start Time:   $(date)"
echo "================================================================================"

# Models for GPU 0 (17 models)
MODELS_GPU0=(
    gemma_2_9b_it
    gemma_2_27b_it
    gemma_2_2b_it
    llama_3_1_8b_instruct
    llama_3_2_1b_instruct
    llama_3_2_3b_instruct
    olmo_2_0325_32b
    olmo_2_0325_32b_dpo
    olmo_2_0325_32b_instruct
    olmo_2_0425_1b
    olmo_2_0425_1b_dpo
    olmo_2_0425_1b_instruct
    olmo_2_0425_1b_sft
    olmo_2_1124_13b
    olmo_2_1124_13b_dpo
    olmo_2_1124_13b_instruct
    olmo_2_1124_13b_sft
)

# Models for GPU 1 (17 models)
MODELS_GPU1=(
    olmo_2_1124_7b
    olmo_2_1124_7b_dpo
    olmo_2_1124_7b_instruct
    olmo_2_1124_7b_sft
    olmoe_1b_7b_0924_instruct
    qwen2_5_0_5b_instruct
    qwen2_5_14b_instruct
    qwen2_5_14b_instruct_p2
    qwen2_5_1_5b_instruct
    qwen2_5_32b_instruct
    qwen2_5_3b_instruct
    qwen2_5_3b_instruct_p2
    qwen2_5_7b_instruct
    qwen2_5_7b_instruct_p2
    smollm2_135m_instruct
    smollm2_1_7b_instruct
    smollm2_360m_instruct
)

echo "Starting Worker 0 on GPU 0 (${#MODELS_GPU0[@]} models)..."
CUDA_VISIBLE_DEVICES=0 ${PYTHON} scripts/run_full_perspectrum_coverage_flair2.py --models "${MODELS_GPU0[@]}" > logs/coverage_gpu0.log 2>&1 &
PID0=$!

echo "Starting Worker 1 on GPU 1 (${#MODELS_GPU1[@]} models)..."
CUDA_VISIBLE_DEVICES=1 ${PYTHON} scripts/run_full_perspectrum_coverage_flair2.py --models "${MODELS_GPU1[@]}" > logs/coverage_gpu1.log 2>&1 &
PID1=$!

echo "Worker 0 running with PID: ${PID0} (Log: logs/coverage_gpu0.log)"
echo "Worker 1 running with PID: ${PID1} (Log: logs/coverage_gpu1.log)"

echo "Waiting for both GPU workers to complete..."
wait ${PID0}
echo "Worker 0 (GPU 0) completed at $(date)."

wait ${PID1}
echo "Worker 1 (GPU 1) completed at $(date)."

echo "================================================================================"
echo "RUNNING FINAL SCALING AND COVERAGE ANALYSIS"
echo "================================================================================"
${PYTHON} scripts/run_scaling_and_coverage_analysis.py

echo "================================================================================"
echo "PIPELINE COMPLETED SUCCESSFULLY AT $(date)"
echo "================================================================================"

