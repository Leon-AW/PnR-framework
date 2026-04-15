#!/bin/bash
#SBATCH --job-name=geo_patch_v3
#SBATCH --partition=longgpu
#SBATCH --gres=gpu:a10080gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-04:00:00
#SBATCH --array=0-2
#SBATCH --exclude=gruenau9
#SBATCH --output=logs/geo_patch_v3_%a_%j.out
#SBATCH --error=logs/geo_patch_v3_%a_%j.err
#SBATCH --mail-type=END,FAIL

# ==============================================================================
# Re-train 3 remaining failed geo patches (Australia, Pakistan, UK).
# gruenau9 is excluded because its GPU 0 has ~65 GiB used by a non-SLURM
# process, leaving only ~4.5 GiB free — not enough to run training.
# ==============================================================================

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

COUNTRIES=(
    "Australia"
    "Pakistan"
    "UK"
)

COUNTRY="${COUNTRIES[$SLURM_ARRAY_TASK_ID]}"

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
    free, total = torch.cuda.mem_get_info(i)
    print(f"  GPU {i}: {p.name} — {total/1024**3:.1f} GB total, {free/1024**3:.1f} GB free")
EOF

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
