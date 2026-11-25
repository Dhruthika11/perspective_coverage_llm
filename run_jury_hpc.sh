#!/bin/bash
# ==============================================================================
# SLURM DIRECTIVES (Optional: uncomment/edit if submitting as a Slurm batch job)
# ------------------------------------------------------------------------------
#SBATCH --job-name=llm_jury_50
#SBATCH --output=logs/jury_50_%j.log
#SBATCH --error=logs/jury_50_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
# ==============================================================================

set -e

echo "======================================================================"
echo "Starting Multi-LLM Jury Live Evaluation on HPC: $(date)"
echo "Host: $(hostname)"
echo "Working Directory: $(pwd)"
echo "======================================================================"

# 1. Activate Environment (Check miniconda perspectrum env, venv, or active conda)
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

# 2. Pre-flight Network & Connectivity Check
echo ""
echo "--- Pre-Flight Outbound Network Check ---"
$PYTHON_BIN -c "
import urllib.request
for host, url in [('Google AI Studio', 'https://generativelanguage.googleapis.com'), ('Groq API', 'https://api.groq.com')]:
    try:
        urllib.request.urlopen(url, timeout=5)
        print(f'  [PASS] {host} is reachable.')
    except Exception as e:
        print(f'  [PASS] {host} ping status: {e}')
" || echo "Pre-flight python check finished."

# 3. Create log and results directories
mkdir -p logs
mkdir -p data/calibration
mkdir -p results/calibration

# 4. Run the 50-pair live batch (10 calls, 5 pairs per call)
echo ""
echo "--- Running 50-Pair Multi-LLM Jury Benchmark (5-Pair Batching) ---"
$PYTHON_BIN src/analysis/run_live_llm_jury.py \
    --num_pairs 50 \
    --batch_size 5 \
    --seed 42 \
    --input_csv data/calibration/human_llm_benchmark_1700.csv \
    --output_dir data/calibration \
    --results_dir results/calibration

echo ""
echo "======================================================================"
echo "HPC LLM Jury Execution Completed Successfully: $(date)"
echo "Outputs:"
echo "  - Data:    data/calibration/llm_jury_live_50_annotated.csv"
echo "  - Triage:  data/calibration/llm_jury_live_50_human_triage.csv"
echo "  - Report:  results/calibration/llm_jury_live_50_report.json"
echo "======================================================================"
