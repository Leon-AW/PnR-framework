#!/bin/bash
#SBATCH --job-name=eval_deval
#SBATCH --partition=longgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a10080gb:1
#SBATCH --nodelist=gruenau10
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL

# GPU + node convention (Apr 29 2026, post-mortem):
# Only gruenau10 (3× A100-80GB, dedicated devices) is safe for these eval
# jobs. We tried adding gruenau7 (4× RTX A6000) on Apr 29 17:30, but that
# node advertises `gpu:rtxa6000:4,mps:r` — i.e. NVIDIA MPS residual mode
# is the default and a `gres=gpu:1` request returns an MPS slice with
# ~0.8 GiB visible memory, not a full RTX A6000. Bitsandbytes 4-bit then
# falls back to CPU/float32 and inference collapses to ~2500 s / sample
# (jobs 352197–352200 observed at 1/1000 progress after 41 min).
# gruenau9 has the same MPS-overbooking issue (see roadmap §"Compute
# Convention"), gruenau8 is heavily allocated. So we pin to gruenau10
# exclusively and accept the 3-way concurrency cap.
# `--nodes=1` + `--ntasks=1` keep SLURM from over-allocating across nodes
# when a multi-element nodelist is restored later.

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
