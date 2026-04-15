#!/bin/bash
#SBATCH --job-name=geo_patch_v2
#SBATCH --partition=longgpu
#SBATCH --gres=gpu:a10080gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-04:00:00
#SBATCH --array=0-6
#SBATCH --output=logs/geo_patch_v2_%a_%j.out
#SBATCH --error=logs/geo_patch_v2_%a_%j.err
#SBATCH --mail-type=END,FAIL

# ==============================================================================
# Re-train 7 failed geographic patches (OOM on GPU 0 in v1).
# Pinned to GPU 1 (73 GB free) to avoid memory contention.
#
# Failed patches from job 275650 array:
#   Australia, England, France, Germany, Nigeria, Pakistan, UK
#
# 300 steps × eff. batch 16 = 4,800 samples per patch.
# Expected runtime per patch: ~20-30 min on A100 80GB.
# ==============================================================================

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

# NOTE: Do NOT override CUDA_VISIBLE_DEVICES here.
# SLURM sets it automatically to the allocated GPU.  Overriding it causes
# torch.cuda.is_available() to return False under SLURM cgroup isolation.
# max_memory is now computed dynamically in core.py from actual free memory.

# ------------------------------------------------------------------------------
# Country list (indices 0-6 map to SLURM_ARRAY_TASK_ID)
# ------------------------------------------------------------------------------
COUNTRIES=(
    "Australia"
    "England"
    "France"
    "Germany"
    "Nigeria"
    "Pakistan"
    "UK"
)

COUNTRY="${COUNTRIES[$SLURM_ARRAY_TASK_ID]}"

# ------------------------------------------------------------------------------
# Environment
# ------------------------------------------------------------------------------
CONDA_BASE=/usr/local/anaconda3-2024.06
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate pnr

export TQDM_MININTERVAL=10
export TQDM_NCOLS=100

echo "======================================================================"
echo "Job ID       : ${SLURM_JOB_ID}  (array task ${SLURM_ARRAY_TASK_ID})"
echo "Node         : ${SLURMD_NODENAME}"
echo "Country      : ${COUNTRY}"
echo "CUDA_VISIBLE_DEVICES : ${CUDA_VISIBLE_DEVICES:-<set by SLURM>}"
echo "Started      : $(date)"
echo "======================================================================"

python - <<'EOF'
import torch
print(f"CUDA available : {torch.cuda.is_available()}")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {p.name} — {p.total_memory/1024**3:.1f} GB")
EOF

# ------------------------------------------------------------------------------
# Training
# ------------------------------------------------------------------------------
python train/train_patch.py \
    --type              geo \
    --country           "${COUNTRY}" \
    --max_steps         300 \
    --quantization      int4 \
    --lora_r            16 \
    --lora_alpha        32 \
    --batch_size        4 \
    --gradient_accumulation 4 \
    --learning_rate     2e-4 \
    --optim             adamw_torch \
    --seed              42

echo "======================================================================"
echo "Finished ${COUNTRY} : $(date)"
echo "======================================================================"
