#!/bin/bash
#SBATCH --job-name=xlora_baseline
#SBATCH --partition=longgpu
#SBATCH --gres=gpu:a10080gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=14-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL

# ==============================================================================
# X-LoRA Baseline — SLURM Training Job
#
# Usage:
#   cd /path/to/PnR-framework
#   sbatch slurm/train_xlora.sh
#
#   Pass extra args directly to the training script:
#   sbatch slurm/train_xlora.sh --max_steps 100 --run_name smoke_test
#
# Monitor:
#   squeue --me
#   tail -f logs/xlora_baseline_<JOBID>.out
# ==============================================================================

set -euo pipefail

# $SLURM_SUBMIT_DIR is the directory from which sbatch was called — always valid
cd "${SLURM_SUBMIT_DIR}"

# ------------------------------------------------------------------------------
# Environment
# ------------------------------------------------------------------------------
CONDA_BASE=/usr/local/anaconda3-2024.06
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate pnr

# Force tqdm to write to log file even though stdout is not a TTY
export TQDM_MININTERVAL=10       # update at most every 10 s (keeps log readable)
export TQDM_NCOLS=100            # fixed width for non-terminal output
# HuggingFace progress bars — keep them on in non-interactive mode
export HF_DATASETS_DISABLE_PROGRESS_BARS=0

echo "======================================================================"
echo "Job ID       : ${SLURM_JOB_ID}"
echo "Node         : ${SLURMD_NODENAME}"
echo "Submit dir   : ${SLURM_SUBMIT_DIR}"
echo "Started      : $(date)"
echo "======================================================================"

# Confirm GPU
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
python train_xlora_baseline.py \
    --model_id mistralai/Mistral-7B-Instruct-v0.3 \
    --checkpoints_dir checkpoints/ \
    --output_dir checkpoints/xlora_baseline \
    --max_steps 2000 \
    --batch_size 4 \
    --gradient_accumulation 4 \
    --max_seq_length 4096 \
    --learning_rate 1e-4 \
    --logging_steps 10 \
    --save_steps 200 \
    --experiment_name pnr-training \
    --run_name "xlora_baseline_${SLURM_JOB_ID}" \
    "$@"

echo "======================================================================"
echo "Finished : $(date)"
echo "======================================================================"
