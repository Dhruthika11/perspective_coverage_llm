#!/bin/bash
# ==============================================================================
# SLURM DIRECTIVES (Optional: uncomment if submitting via sbatch)
# ------------------------------------------------------------------------------
#SBATCH --job-name=llm_jury_300
#SBATCH --output=logs/jury_stage_c_%j.log
#SBATCH --error=logs/jury_stage_c_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
# ==============================================================================

set -e

echo "======================================================================"
echo "Starting Stage C Multi-LLM Jury Live Evaluation on HPC: $(date)"
echo "Host: $(hostname)"
echo "Working Directory: $(pwd)"
echo "Target Dataset: data/calibration/dataset_b_stage_c_sampled_300.csv (N=300)"
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

# 3. Run Stage C Live Benchmark (300 pairs, 60 batches of 5 pairs)
echo ""
echo "--- Running Stage C 300-Pair Multi-LLM Jury Benchmark ---"
$PYTHON_BIN src/analysis/run_live_llm_jury.py \
    --num_pairs 300 \
    --batch_size 5 \
    --seed 43 \
    --input_csv data/calibration/dataset_b_stage_c_sampled_300.csv \
    --output_dir data/calibration \
    --results_dir results/calibration \
    --run_tag stage_c_300

echo ""
echo "======================================================================"
echo "HPC Stage C Execution Completed Successfully: $(date)"
echo "Outputs:"
echo "  - Annotated Data:   data/calibration/llm_jury_stage_c_300_annotated.csv"
echo "  - Human Triage:     data/calibration/llm_jury_stage_c_300_human_triage.csv"
echo "  - Adjudication Set: data/calibration/stage_stage_c_300_human_adjudication_queue.csv"
echo "  - Full JSON Report: results/calibration/llm_jury_stage_c_300_report.json"
echo "======================================================================"
