#!/bin/bash
#SBATCH --job-name=openstream_stress
#SBATCH --partition=longgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a10080gb:1
#SBATCH --nodelist=gruenau10
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL

# ==============================================================================
# Open-stream routing stress test
#
# Feeds the held-out 5-domain test set (data/openstream_heldout.json) through the
# UNCHANGED production routing pipeline and measures (a) routing leak rate and
# (b) conditional damage on the leaked subset. This is the "open continual
# stream" experiment the thesis names as its top acknowledged limitation — it
# tests whether the Stage-1 gate holds outside the closed {cf,sqa,qm,ood_trivia}
# world it was trained on.
#
# No retraining: pure inference over checkpoints/domain_classifier +
# checkpoints/router_state. Routing config is identical to the D_eval sweeps
# (similarity_threshold=0.45, domain_confidence_threshold=0.7,
# domain_fallback_threshold=0.30 — see slurm/eval_qm_deval.sh).
#
# Phase A (routing leak) is cheap (embedding + classifier only) and can be run
# inline without SLURM:
#   python scripts/run_openstream_stress.py --phase a
# This script runs the full Phase A + Phase B (the conditional-damage generation
# over the leaked subset needs the int4 LLM → GPU).
#
# Usage:
#   sbatch slurm/run_openstream_stress.sh
# ==============================================================================

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

CONDA_BASE=/usr/local/anaconda3-2024.06
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate pnr

export TQDM_MININTERVAL=10
export TQDM_NCOLS=100
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$(pwd)"
export HF_DATASETS_CACHE=/vol/tmp/wagnerql/hf_datasets_cache

echo "======================================================================"
echo "Job ID  : ${SLURM_JOB_ID}"
echo "Node    : ${SLURMD_NODENAME}"
echo "Started : $(date)"
echo "======================================================================"

# Build the held-out test set if missing (1,000 queries, 200 per domain).
if [[ ! -f data/openstream_heldout.json ]]; then
    echo "data/openstream_heldout.json missing — building it ..."
    python scripts/build_openstream_testset.py
fi

python scripts/run_openstream_stress.py --phase all "$@"

echo "Finished: $(date)"
