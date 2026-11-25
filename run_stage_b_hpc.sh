#!/bin/bash
# ==============================================================================
# SLURM DIRECTIVES (Optional: uncomment if submitting via sbatch)
# ------------------------------------------------------------------------------
#SBATCH --job-name=llm_jury_250
#SBATCH --output=logs/jury_stage_b_%j.log
#SBATCH --error=logs/jury_stage_b_%j.err
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
# ==============================================================================

set -e

echo "======================================================================"
echo "Starting Stage B Multi-LLM Jury Live Evaluation on HPC: $(date)"
echo "Host: $(hostname)"
echo "Working Directory: $(pwd)"
echo "Target Dataset: data/calibration/dataset_b_stage_b_sampled_250.csv (N=250)"
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

# 2. Ensure directories exist
mkdir -p logs
mkdir -p data/calibration
mkdir -p results/calibration

# 3. Run Stage B Live Benchmark (250 pairs, 50 batches of 5 pairs)
echo ""
echo "--- Running Stage B 250-Pair Multi-LLM Jury Benchmark ---"
$PYTHON_BIN src/analysis/run_live_llm_jury.py \
    --num_pairs 250 \
    --batch_size 5 \
    --seed 42 \
    --input_csv data/calibration/dataset_b_stage_b_sampled_250.csv \
    --output_dir data/calibration \
    --results_dir results/calibration \
    --run_tag stage_b_250

echo ""
echo "======================================================================"
echo "HPC Stage B Execution Completed Successfully: $(date)"
echo "Outputs:"
echo "  - Annotated Data:   data/calibration/llm_jury_stage_b_250_annotated.csv"
echo "  - Human Triage:     data/calibration/llm_jury_stage_b_250_human_triage.csv"
echo "  - Adjudication Set: data/calibration/stage_stage_b_250_human_adjudication_queue.csv"
echo "  - Full JSON Report: results/calibration/llm_jury_stage_b_250_report.json"
echo "======================================================================"
