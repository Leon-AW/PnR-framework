#!/bin/bash
#SBATCH --job-name=triviaqa_dcalib
#SBATCH --partition=longgpu
#SBATCH --gres=gpu:a10080gb:1
#SBATCH --nodelist=gruenau10
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/triviaqa_dcalibration_%j.out
#SBATCH --error=logs/triviaqa_dcalibration_%j.err

# ==============================================================================
# Build TriviaQA D_calibration Slice
#
# Produces a small (~500 question) TriviaQA pool that is provably DISJOINT
# from data/triviaqa_dcontrol.json. Used by scripts/build_router_state.py as
# the OOD negative pool for per-adapter similarity-threshold calibration.
#
# Why disjoint: D_control is the eval-time stability probe. The exposé
# guarantees that "any drop on D_control = routing-induced forgetting"
# *because* no system sees those records during construction. Calibrating
# router thresholds against D_control would be test-set leakage and break
# parity with MORPHEUS / RECIPE / X-LoRA baselines (none of which see
# D_control before eval time).
#
# --start_offset 5000 skips well past the TriviaQA index range used by the
# original D_control build (~2.2k questions scanned), so the run is fast
# even before the --exclude_path safety net kicks in.
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
    --output_path data/triviaqa_dcalibration.json \
    --target 500 \
    --max_process 10000 \
    --batch_size 1 \
    --max_new_tokens 32 \
    --exclude_path data/triviaqa_dcontrol.json \
    --start_offset 5000

echo "======================================================================"
echo "Finished : $(date)"
echo "======================================================================"
