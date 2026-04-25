#!/bin/bash
#SBATCH --job-name=eval_recipe_tf
#SBATCH --partition=longgpu
#SBATCH --gres=gpu:a10080gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

CONDA_BASE=/usr/local/anaconda3-2024.06
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate pnr

echo "======================================================================"
echo "Job ID   : ${SLURM_JOB_ID}"
echo "Node     : ${SLURMD_NODENAME}"
echo "Started  : $(date)"
echo "======================================================================"

python scripts/eval_recipe_teacher_forced.py \
    --checkpoint /vol/fob-vol1/mi23/wagnerql/PnR-framework/external/RECIPE/train_records/recipe/mistral-7b/2026.04.14-13.34.10/checkpoints/epoch-159-i-99000-ema_loss-0.2240 \
    --edits /vol/fob-vol1/mi23/wagnerql/PnR-framework/data/edit_pairs.json \
    --eval_sets base temporal geo_india geo_australia \
    --n_samples 200 \
    --quantization int4 \
    --output eval_results/recipe_tf_es.json

echo "======================================================================"
echo "Finished : $(date)"
echo "======================================================================"
