#!/bin/bash
#SBATCH --job-name=geo_patch_uk
#SBATCH --partition=longgpu
#SBATCH --gres=gpu:a10080gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-00:30:00
#SBATCH --exclude=gruenau9
#SBATCH --output=logs/geo_patch_v4_%j.out
#SBATCH --error=logs/geo_patch_v4_%j.err
#SBATCH --mail-type=END,FAIL

# ==============================================================================
# Re-train UK geo patch (v3 stalled due to dataloader deadlock with num_workers=4
# on a single-shard IterableDataset). Fixed by setting dataloader_num_workers=1
# in train_patch.py. 300 steps × ~2.5s/step ≈ 12 min — 30 min is plenty.
# ==============================================================================

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

CONDA_BASE=/usr/local/anaconda3-2024.06
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate pnr

export TQDM_MININTERVAL=10
export TQDM_NCOLS=100

echo "======================================================================"
echo "Job ID       : ${SLURM_JOB_ID}"
echo "Node         : ${SLURMD_NODENAME}"
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
    --country           "UK" \
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
echo "Finished UK : $(date)"
echo "======================================================================"
