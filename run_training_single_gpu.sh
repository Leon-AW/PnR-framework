#!/bin/bash
#SBATCH --job-name=pnr_single
#SBATCH --partition=shared
#SBATCH --gres=gpu:1              # Single GPU/MIG instance (recommended)
#SBATCH --time=24:00:00           # 24 hours
#SBATCH --output=logs/pnr_single_%j.log
#SBATCH --error=logs/pnr_single_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G                 # CPU RAM for model loading

# ============================================================================
# SINGLE GPU/MIG TRAINING SCRIPT
# ============================================================================
# This script is designed for training on a SINGLE 24GB GPU or MIG instance.
# Use this for maximum reliability and to avoid distributed training issues.
#
# For DeepSeek-R1-14B with 4-bit quantization:
# - Minimum VRAM: 20 GB
# - Recommended: 24 GB
# - batch_size=1, gradient_accumulation=16 for effective batch of 16
# ============================================================================

set -e  # Exit on error

# Initialize Conda
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate pnr

# Create directories
mkdir -p logs checkpoints

# Print job info
echo "=============================================="
echo "SINGLE GPU TRAINING JOB"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo ""

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Disable distributed training (forces single GPU)
unset WORLD_SIZE
unset LOCAL_RANK
unset RANK

# Run validation first
echo "Running GPU validation..."
python validate_gpu_setup.py --dry-run

echo ""
echo "Starting training..."
echo "=============================================="

# Training command with memory-optimized settings
python train_rag_baseline.py \
    --data_path src/data/dataset_final.json \
    --docs_path src/data/documents/DE \
    --adapter_name QM_rag \
    --output_dir checkpoints/ \
    --target_devices 0 \
    --batch_size 1 \
    --gradient_accumulation 16 \
    --max_seq_length 1024 \
    --quantization int4 \
    --max_steps 2000 \
    --save_steps 200

echo ""
echo "=============================================="
echo "Job finished: $(date)"
echo "=============================================="
