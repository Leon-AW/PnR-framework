#!/bin/bash
#SBATCH --job-name=train_qm_base
#SBATCH --partition=longgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a10080gb:1
#SBATCH --nodelist=gruenau10
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL

# ==============================================================================
# Train the AIT QM Base Adapter (base_qm)
#
# base_qm holds the *outdated* QM facts (answer_old) — the QM-domain analogue of
# SituatedQA's base_v1. Paired with patch_qm_current (the current facts) it
# gives the router a genuine old-vs-new conflict to resolve: Mistral has seen
# neither QM fact in pretraining, so the outdated side must live in its own
# adapter for R1 (conflict resolution) to be measurable rather than a vacuous
# memorisation check. See docs/roadmap.md NF-3 / tasks/todo.md section 6a.
#
# Runs on gruenau10 — the semi-synthetic QM data is non-sensitive. Same
# hyperparams as patch_qm_current (r=16, alpha=32, 500 steps) so the two
# adapters are directly comparable; reuses train/train_qm_patch.py.
#
# Usage:
#   sbatch slurm/train_qm_base.sh
#   sbatch slurm/train_qm_base.sh --max_steps 1000   # extra args forwarded
# ==============================================================================

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

CONDA_BASE=/usr/local/anaconda3-2024.06
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate pnr

echo "======================================================================"
echo "Job ID  : ${SLURM_JOB_ID}"
echo "Node    : ${SLURMD_NODENAME}"
echo "Started : $(date)"
echo "Args    : $*"
echo "======================================================================"

# Build the outdated-facts training JSONL if absent (CPU, instant).
if [ ! -f "data/qm_train_old.jsonl" ]; then
    echo "Building data/qm_train_old.jsonl ..."
    python scripts/build_qm_train_data.py \
        --answer_field answer_old \
        --output data/qm_train_old.jsonl
fi

# Storage convention: large checkpoints live on /vol/tmp, symlinked into the
# repo so router-state builds find a stable path.
OUTPUT_DIR=/vol/tmp/wagnerql/checkpoints/base_qm
mkdir -p "$(dirname "$OUTPUT_DIR")"

python train/train_qm_patch.py \
    --data_path data/qm_train_old.jsonl \
    --adapter_name base_qm \
    --adapter_type base_qm \
    --answer_field answer_old \
    --output_dir "$OUTPUT_DIR" \
    "$@"

ln -sfn "$OUTPUT_DIR" checkpoints/base_qm
echo "Symlinked checkpoints/base_qm -> $OUTPUT_DIR"

echo "======================================================================"
echo "Finished : $(date)"
echo "======================================================================"
