#!/bin/bash
#SBATCH --job-name=geo_patch
#SBATCH --partition=longgpu
#SBATCH --gres=gpu:a10080gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-04:00:00
#SBATCH --array=0-8
#SBATCH --output=logs/geo_patch_%a_%j.out
#SBATCH --error=logs/geo_patch_%a_%j.err
#SBATCH --mail-type=END,FAIL

# ==============================================================================
# Re-train 9 thin geographic patches (previously 50-63 steps → now 300 steps)
#
# Submitted as a SLURM array job (--array=0-8).  Each array task trains one
# country.  SLURM schedules them as GPUs become available.
#
# 300 steps × eff. batch 16 = 4,800 samples per patch (matches India/Others).
# Expected runtime per patch: ~20-30 min on A100 80GB.
#
# Checkpoints overwrite the old thin ones in checkpoints/patch_geo_<country>/
#
# Usage:
#   sbatch slurm/retrain_geo_patches.sh
#
# Monitor:
#   squeue --me
#   tail -f logs/geo_patch_<ARRAY_ID>_<JOB_ID>.out
# ==============================================================================

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

# ------------------------------------------------------------------------------
# Country list (indices 0-8 map to SLURM_ARRAY_TASK_ID)
# ------------------------------------------------------------------------------
COUNTRIES=(
    "Australia"
    "California"
    "Canada"
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
    --seed              42

echo "======================================================================"
echo "Finished ${COUNTRY} : $(date)"
echo "======================================================================"
