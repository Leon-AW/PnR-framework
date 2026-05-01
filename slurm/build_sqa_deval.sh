#!/bin/bash
#SBATCH --job-name=build_sqa_deval
#SBATCH --partition=longrun
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL

# No GPU needed — just downloads SituatedQA from GitHub/HF cache and samples.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

CONDA_BASE=/usr/local/anaconda3-2024.06
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate pnr

echo "======================================================================"
echo "Job ID  : ${SLURM_JOB_ID}"
echo "Node    : ${SLURMD_NODENAME}"
echo "Started : $(date)"
echo "======================================================================"

python scripts/build_sqa_deval.py \
    --target 1000 \
    --max_per_stream 600 \
    --seed 42 \
    --output data/sqa_deval.json

echo "======================================================================"
echo "Finished : $(date)"
echo "======================================================================"
