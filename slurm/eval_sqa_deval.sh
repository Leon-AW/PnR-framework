#!/bin/bash
#SBATCH --job-name=eval_sqa_deval
#SBATCH --partition=longgpu
#SBATCH --gres=gpu:a10080gb:1
#SBATCH --nodelist=gruenau10
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL

# SituatedQA D_eval: 1000 SQA training samples + 1000 TriviaQA D_control.
#
# IMPORTANT — sequential inference (batch_size=1):
#   The eval runner processes samples one at a time. This matches the
#   batch_size=1 conditions used when building triviaqa_dcontrol.json,
#   guaranteeing that the frozen base achieves FR=0.0% by construction.
#   Do NOT add batched inference here or the guarantee breaks.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

CONDA_BASE=/usr/local/anaconda3-2024.06
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate pnr

export TQDM_MININTERVAL=10
export TQDM_NCOLS=100

echo "======================================================================"
echo "Job ID  : ${SLURM_JOB_ID}"
echo "Node    : ${SLURMD_NODENAME}"
echo "Started : $(date)"
echo "Args    : $*"
echo "======================================================================"

python eval_pnr.py \
    --eval_sets sqa_train cf_control \
    --sqa_deval_path   data/sqa_deval.json \
    --triviaqa_dcontrol_path data/triviaqa_dcontrol.json \
    --n_samples        1000 \
    --max_new_tokens   64 \
    --similarity_threshold 0.45 \
    "$@"

echo "======================================================================"
echo "Finished : $(date)"
echo "======================================================================"
