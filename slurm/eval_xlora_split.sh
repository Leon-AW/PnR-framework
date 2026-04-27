#!/bin/bash
#SBATCH --job-name=eval_xlora_split
#SBATCH --partition=longgpu
#SBATCH --gres=gpu:a10080gb:1
#SBATCH --nodelist=gruenau10
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=18:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL

# ==============================================================================
# X-LoRA Evaluation — Single-Split Runner
#
# X-LoRA inference is ~2 min/sample on Mistral-7B-Instruct-v0.3. A full
# 4-split eval (base/temporal/geo_india/geo_australia, 200 samples each)
# runs ~40h and blows through any single SLURM time limit. This script
# evaluates one split per job so four jobs can run in parallel.
#
# Usage:
#   sbatch slurm/eval_xlora_split.sh base
#   sbatch slurm/eval_xlora_split.sh temporal
#   sbatch slurm/eval_xlora_split.sh geo_india
#   sbatch slurm/eval_xlora_split.sh geo_australia
#
# Each job writes to eval_results/xlora_v2_<split>/ with both
# results_<split>.json (per-split checkpoint) and results.json + report.json.
# After all four finish:  python scripts/merge_eval_splits.py xlora_v2
# ==============================================================================

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: sbatch slurm/eval_xlora_split.sh <split>"
    echo "  split ∈ {base, temporal, geo_india, geo_australia}"
    exit 1
fi

SPLIT="$1"
shift

cd "${SLURM_SUBMIT_DIR}"

CONDA_BASE=/usr/local/anaconda3-2024.06
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate pnr

export TQDM_MININTERVAL=10
export TQDM_NCOLS=100

echo "======================================================================"
echo "Job ID       : ${SLURM_JOB_ID}"
echo "Node         : ${SLURMD_NODENAME}"
echo "Split        : ${SPLIT}"
echo "Started      : $(date)"
echo "Extra args   : $*"
echo "======================================================================"

python eval_pnr.py \
    --xlora checkpoints/xlora_baseline \
    --eval_sets "${SPLIT}" \
    --n_samples 200 \
    --quantization int4 \
    --run_name "xlora_v2_${SPLIT}" \
    --experiment_name pnr-evaluation \
    "$@"

echo "======================================================================"
echo "Finished : $(date)"
echo "======================================================================"
