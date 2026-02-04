#!/bin/bash
#SBATCH --job-name=pnr_multi
#SBATCH --partition=shared
#SBATCH --gres=gpu:1              # Single GPU - see notes below
#SBATCH --time=24:00:00
#SBATCH --output=logs/pnr_multi_%j.log
#SBATCH --error=logs/pnr_multi_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# ============================================================================
# MULTI-GPU TRAINING NOTES
# ============================================================================
#
# IMPORTANT: Multi-GPU data-parallel training is NOT recommended for:
# - 14B models on 24GB GPUs/MIG instances
# - LoRA fine-tuning (tiny gradient sync, large overhead)
#
# WHY IT DOESN'T HELP:
# 1. Each process loads the FULL model (~18GB with 4-bit quantization)
# 2. 4 processes x 18GB = 72GB needed, but MIG instances share memory
# 3. LoRA only trains ~0.1% of params - gradient sync overhead dominates
# 4. Batch size is still 1 per device (memory bound)
#
# RECOMMENDATION:
# Use single-GPU training (run_training_single_gpu.sh) which is:
# - More reliable
# - Nearly as fast (or faster due to no sync overhead)
# - Easier to debug
#
# WHEN MULTI-GPU HELPS:
# - Full fine-tuning (not LoRA) with larger batch sizes
# - GPUs with 48GB+ VRAM each
# - Using FSDP (Fully Sharded Data Parallel) for model parallelism
#
# ============================================================================

set -e

# Initialize Conda
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate pnr

mkdir -p logs checkpoints

echo "=============================================="
echo "SINGLE-GPU TRAINING (Multi-GPU Not Recommended)"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo ""
echo "NOTE: Using single GPU mode. See script header for why."
echo ""

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Unset distributed training variables
unset WORLD_SIZE LOCAL_RANK RANK

# Run validation
python validate_gpu_setup.py --target-devices 0

echo ""
echo "Starting training..."
echo "=============================================="

# Single GPU training - the reliable approach
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
