#!/bin/bash
#SBATCH --job-name=train_qm_monolithic
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
# Train the AIT QM Sequential Monolithic Adapter (monolithic_qm)
#
# Demonstrates catastrophic forgetting by training a single LoRA sequentially:
#   Phase 1 (500 steps): old QM facts  (data/qm_train_old.jsonl, answer_old)
#   Phase 2 (500 steps): new QM facts  (data/qm_train.jsonl,     answer_new)
#
# After phase 2 the adapter knows current facts but has overwritten the old
# ones — the QM-domain catastrophic forgetting that PnR is designed to solve.
# This is the correct monolithic continual-learning baseline for QM: unlike
# CounterFact, Mistral-7B has no prior QM knowledge, so both knowledge states
# must be installed explicitly via sequential fine-tuning.
#
# Compare against:
#   base_qm           — adapter trained on old facts only (preserved in PnR)
#   patch_qm_current  — adapter trained on new facts only (preserved in PnR)
#   PnR routing       — routes between the two, preserving both
#
# Runs on gruenau10. ~2 h total (2 × ~18 min phases + model load).
#
# Usage:
#   sbatch slurm/train_qm_monolithic.sh
#   sbatch slurm/train_qm_monolithic.sh --max_steps_per_phase 1000
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

OUTPUT_DIR=/vol/tmp/wagnerql/checkpoints/monolithic_qm
mkdir -p "$(dirname "$OUTPUT_DIR")"

python train/train_qm_monolithic.py \
    --old_data_path data/qm_train_old.jsonl \
    --new_data_path data/qm_train.jsonl \
    --adapter_name monolithic_qm \
    --output_dir "$OUTPUT_DIR" \
    "$@"

ln -sfn "$OUTPUT_DIR" checkpoints/monolithic_qm
echo "Symlinked checkpoints/monolithic_qm -> $OUTPUT_DIR"

echo "======================================================================"
echo "Finished : $(date)"
echo "======================================================================"
