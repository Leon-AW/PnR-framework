#!/bin/bash
#SBATCH --job-name=eval_deval
#SBATCH --partition=longgpu
#SBATCH --gres=gpu:a10080gb:1
#SBATCH --nodelist=gruenau10
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL

# ==============================================================================
# D_eval Evaluation — D_conflict (CF ESR) + D_control (forgetting rate)
#
# This is the primary thesis evaluation script. Runs the two metrics defined
# in the exposé:
#
#   R1 — Edit Success Rate (ESR): fraction of trained CF records for which
#        the system outputs target_false (the counterfactual). Measured on
#        the training split (memorization under routing).
#
#   R2 — D_control accuracy / forgetting rate: accuracy on 5,000 TriviaQA
#        questions pre-filtered to 100% frozen-base accuracy. Any drop =
#        routing-induced forgetting. Forgetting rate = 1 - accuracy.
#
# Usage:
#   sbatch --job-name=eval_pnr_deval slurm/eval_deval.sh \
#       --router_state checkpoints/router_state \
#       --similarity_threshold 0.45 \
#       --run_name pnr_deval
#
#   sbatch --job-name=eval_mono_deval slurm/eval_deval.sh \
#       --monolithic checkpoints/monolithic_v1 \
#       --run_name monolithic_deval
#
# See slurm/submit_deval_sweep.sh to launch all systems at once.
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
    --eval_sets cf_conflict cf_control \
    --n_samples 1000 \
    --counterfact_eval_path data/counterfact_eval.json \
    --triviaqa_dcontrol_path data/triviaqa_dcontrol.json \
    --cf_split_name train \
    --cf_adapter_name patch_cf_main \
    --quantization int4 \
    --max_new_tokens 32 \
    "$@"

echo "======================================================================"
echo "Finished : $(date)"
echo "======================================================================"
