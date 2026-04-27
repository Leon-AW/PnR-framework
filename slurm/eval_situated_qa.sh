#!/bin/bash
#SBATCH --job-name=eval_situated_qa
#SBATCH --partition=longgpu
#SBATCH --gres=gpu:a10080gb:1
#SBATCH --nodelist=gruenau10
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL

# ==============================================================================
# SituatedQA Evaluation — Generic Runner
#
# Evaluates one baseline/method on the SituatedQA benchmark.
# All extra eval_pnr.py arguments are passed through via "$@".
#
# Usage (override job name + pass method-specific flags):
#   sbatch --job-name=eval_pnr   slurm/eval_situated_qa.sh
#   sbatch --job-name=eval_mono  slurm/eval_situated_qa.sh --monolithic checkpoints/monolithic_v1 --run_name monolithic
#
# See slurm/submit_eval_sweep.sh for the full 7-baseline submission.
#
# Eval sets  : base, temporal, geo_india, geo_australia
# Samples    : 200 per split (800 total)
# Quantiztn  : int4 (matches all training runs)
# Determinism: do_sample=False (greedy), fixed dataset order
# ==============================================================================

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

CONDA_BASE=/usr/local/anaconda3-2024.06
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate pnr

export TQDM_MININTERVAL=10
export TQDM_NCOLS=100

echo "======================================================================"
echo "Job ID       : ${SLURM_JOB_ID}"
echo "Node         : ${SLURMD_NODENAME}"
echo "Started      : $(date)"
echo "Args         : $*"
echo "======================================================================"

python eval_pnr.py \
    --eval_sets base temporal geo_india geo_australia \
    --n_samples 200 \
    --quantization int4 \
    "$@"

echo "======================================================================"
echo "Finished : $(date)"
echo "======================================================================"
