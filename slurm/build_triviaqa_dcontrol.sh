#!/bin/bash
#SBATCH --job-name=triviaqa_dcontrol
#SBATCH --partition=longgpu
#SBATCH --gres=gpu:a10080gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/triviaqa_dcontrol_%j.out
#SBATCH --error=logs/triviaqa_dcontrol_%j.err
#SBATCH --exclude=gruenau10

# ==============================================================================
# Build TriviaQA D_control Dataset
#
# Runs frozen Mistral-7B-Instruct-v0.3 (int4) on TriviaQA questions zero-shot.
# Keeps only questions answered correctly (EM = 1) until 5,000 verified pairs.
# Output: data/triviaqa_dcontrol.json
#
# Per exposé §4.1: D_control = questions the BASE MODEL already answers correctly.
# Any accuracy drop after CounterFact adapter integration → routing error.
# ==============================================================================

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

CONDA_BASE=/usr/local/anaconda3-2024.06
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate pnr

mkdir -p logs

echo "======================================================================"
echo "Job ID   : ${SLURM_JOB_ID}"
echo "Node     : ${SLURMD_NODENAME}"
echo "Started  : $(date)"
echo "GPU      : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "======================================================================"

python scripts/build_triviaqa_dcontrol.py \
    --output_path data/triviaqa_dcontrol.json \
    --target 5000 \
    --max_process 80000 \
    --batch_size 8 \
    --max_new_tokens 30

echo "======================================================================"
echo "Finished : $(date)"
echo "======================================================================"
