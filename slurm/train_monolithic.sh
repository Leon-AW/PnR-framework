#!/bin/bash
#SBATCH --job-name=monolithic_baseline
#SBATCH --partition=longgpu
#SBATCH --gres=gpu:a10080gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL

# ==============================================================================
# Monolithic Baseline — SLURM Training Job
#
# Trains a single LoRA adapter on all SituatedQA streams combined
# (base + temporal + all-non-US geo). Direct comparison against PnR.
#
# Usage:
#   cd /path/to/PnR-framework
#   sbatch slurm/train_monolithic.sh
#
#   Pass extra args:
#   sbatch slurm/train_monolithic.sh --max_steps 100 --run_name smoke_test
#
# Monitor:
#   squeue --me
#   tail -f logs/monolithic_baseline_<JOBID>.out
# ==============================================================================

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

# ------------------------------------------------------------------------------
# Environment
# ------------------------------------------------------------------------------
CONDA_BASE=/usr/local/anaconda3-2024.06
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate pnr

export TQDM_MININTERVAL=10
export TQDM_NCOLS=100
export HF_DATASETS_DISABLE_PROGRESS_BARS=0

echo "======================================================================"
echo "Job ID       : ${SLURM_JOB_ID}"
echo "Node         : ${SLURMD_NODENAME}"
echo "Submit dir   : ${SLURM_SUBMIT_DIR}"
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
python train/train_monolithic_baseline.py \
    --situatedqa \
    --model_id mistralai/Mistral-7B-Instruct-v0.3 \
    --output_dir checkpoints/monolithic_v1 \
    --max_steps 2000 \
    --quantization int4 \
    --lora_r 16 \
    --lora_alpha 32 \
    --batch_size 1 \
    --gradient_accumulation 16 \
    --max_seq_length 2048 \
    --logging_steps 25 \
    --save_steps 200 \
    --experiment_name pnr-training \
    --run_name "monolithic_baseline_${SLURM_JOB_ID}" \
    "$@"

echo "======================================================================"
echo "Finished : $(date)"
echo "======================================================================"
