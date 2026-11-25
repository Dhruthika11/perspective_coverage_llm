#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd)"
PYTHON="/home/srujan.sai/miniconda3/envs/perspectrum/bin/python"

mkdir -p logs results/coverage

echo "================================================================================"
echo "LAUNCHING DIVERSITY ANALYSIS (GPU 0) AND BUDGET CURVE ANALYSIS (GPU 1)"
echo "Project Root: ${PROJECT_ROOT}"
echo "Start Time:   $(date)"
echo "================================================================================"

echo "Starting Diversity & Redundancy Analysis on GPU 0..."
CUDA_VISIBLE_DEVICES=0 ${PYTHON} scripts/analyze_viewpoint_diversity.py > logs/diversity_analysis.log 2>&1 &
PID0=$!

echo "Starting Fixed-Budget vs Native-Budget Analysis on GPU 1..."
CUDA_VISIBLE_DEVICES=1 ${PYTHON} scripts/run_budget_curve_analysis_flair2.py > logs/budget_curve.log 2>&1 &
PID1=$!

echo "Worker 0 (Diversity) PID: ${PID0} (Log: logs/diversity_analysis.log)"
echo "Worker 1 (Budget Curve) PID: ${PID1} (Log: logs/budget_curve.log)"

wait ${PID0}
echo "Worker 0 (Diversity) completed at $(date)."

wait ${PID1}
echo "Worker 1 (Budget Curve) completed at $(date)."

echo "================================================================================"
echo "ALL ANALYSES COMPLETED AT $(date)"
echo "================================================================================"

