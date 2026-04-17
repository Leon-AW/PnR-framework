#!/bin/bash
#SBATCH --job-name=train_cf_patch
#SBATCH --partition=longgpu
#SBATCH --gres=gpu:a10080gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/train_cf_patch_%j.out
#SBATCH --error=logs/train_cf_patch_%j.err
#SBATCH --exclude=gruenau10

# ==============================================================================
# Train CounterFact Patch Adapter
#
# Trains a single LoRA on all 21,919 CounterFact QA pairs (target_false).
# Output: checkpoints/patch_cf_main/
#
# Prerequisite: data/counterfact_train.jsonl must exist.
# If missing, run first (CPU, ~30s):
#   conda run -n pnr python scripts/build_counterfact_data.py
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

# Verify data exists
if [ ! -f "data/counterfact_train.jsonl" ]; then
    echo "ERROR: data/counterfact_train.jsonl not found!"
    echo "Run: python scripts/build_counterfact_data.py"
    exit 1
fi

python train/train_counterfact_patch.py \
    --data_path data/counterfact_train.jsonl \
    --adapter_name patch_cf_main \
    --output_dir checkpoints/patch_cf_main \
    --max_steps 2000 \
    --batch_size 1 \
    --gradient_accumulation 16 \
    --learning_rate 2e-4 \
    --lora_r 16 \
    --lora_alpha 32 \
    --max_seq_length 256 \
    --save_steps 200 \
    --logging_steps 25

echo "======================================================================"
echo "Finished : $(date)"
echo "======================================================================"
