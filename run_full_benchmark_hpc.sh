#!/bin/bash
# ==============================================================================
# Master Benchmark Scaling Runner on HPC (Remaining 1,100 Pairs)
# ==============================================================================

set -e

echo "======================================================================"
echo "Starting Master Benchmark Scaling on HPC: $(date)"
echo "Host: $(hostname)"
echo "Working Directory: $(pwd)"
echo "Target Pool: data/calibration/dataset_b_remaining_1100.csv (N=1,100)"
echo "======================================================================"

# 1. Activate Environment
if [ -f "/home/srujan.sai/miniconda3/bin/conda" ]; then
    echo "Activating conda environment 'perspectrum'..."
    source /home/srujan.sai/miniconda3/etc/profile.d/conda.sh
    conda activate perspectrum
elif [ -d "venv" ]; then
    echo "Activating virtualenv at venv/..."
    source venv/bin/activate
fi

PYTHON_BIN=$(which python)
echo "Using Python executable: $PYTHON_BIN"

mkdir -p logs
mkdir -p data/calibration
mkdir -p results/calibration

echo ""
echo "--- Running Master Benchmark Scaling Script ---"
$PYTHON_BIN src/analysis/run_full_benchmark_scaling.py

echo ""
echo "======================================================================"
echo "Master Benchmark Scaling Execution Finished: $(date)"
echo "======================================================================"
