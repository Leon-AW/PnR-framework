#!/bin/bash
#SBATCH --job-name=train_qm_patch
#SBATCH --partition=shared
#SBATCH --account=aitf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:2g.48gb:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/train_qm_patch_%j.out
#SBATCH --error=logs/train_qm_patch_%j.err

# ==============================================================================
# Train AIT QM Patch Adapter (patch_qm_current)
#
# Trains LoRA on 500 AIT QM conflict pairs; output: checkpoints/patch_qm_current/
#
# Prerequisite — build training data first (CPU, <1 s):
#   python scripts/build_qm_train_data.py
#
# Run inside an existing allocation (recommended — reuse job 10427):
#   srun --jobid=10427 --overlap --job-name=train_qm --time=02:00:00 bash -c '
#     export CUDA_VISIBLE_DEVICES=MIG-67a4fc8d-4980-5ba3-8261-8dcb9d94d1d9
#     bash slurm/train_qm_patch.sh' > logs/train_qm_patch_srun.log 2>&1 &
#
# Or submit as a new SLURM job:
#   sbatch slurm/train_qm_patch.sh
#   sbatch slurm/train_qm_patch.sh --max_steps 1000  # extra args forwarded
# ==============================================================================

set -euo pipefail

REPO_ROOT="/gpfs/adafs/home/leon.wagner/PnR-framework"
PYTHON="/gpfs/adafs/home/leon.wagner/miniconda3/envs/pnr/bin/python"

export HF_HUB_OFFLINE=1

cd "$REPO_ROOT"

echo "Node: $(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Build training JSONL if not already present
if [ ! -f "data/qm_train.jsonl" ]; then
    echo "Building data/qm_train.jsonl ..."
    "$PYTHON" scripts/build_qm_train_data.py
fi

"$PYTHON" train/train_qm_patch.py "$@"
