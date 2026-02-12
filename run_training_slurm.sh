#!/bin/bash
#SBATCH --job-name=pnr_rag_train
#SBATCH --partition=shared
#SBATCH --gres=gpu:1              # FIXED: Use 1 GPU for reliable training
#SBATCH --time=48:00:00           # 48 hours
#SBATCH --output=logs/pnr_rag_%j.log
#SBATCH --error=logs/pnr_rag_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G                 # Request 64GB CPU RAM (needed for loading)

# ============================================================================
# RAG BASELINE TRAINING SCRIPT
# ============================================================================
# Configuration:
# - Single GPU/MIG instance (24GB VRAM minimum)
# - 4-bit quantization for memory efficiency
# - Memory-optimized batch settings
#
# For multi-GPU training, use: run_training_multi_gpu.sh
# ============================================================================

set -e  # Exit on any error

# Initialize Conda
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate pnr

# Create directories
mkdir -p logs checkpoints

# Print job info
echo "=============================================="
echo "PnR FRAMEWORK - RAG TRAINING"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo ""

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Disable distributed training environment (single GPU mode)
unset WORLD_SIZE
unset LOCAL_RANK
unset RANK

# Run GPU validation first
echo "Validating GPU setup..."
python validate_gpu_setup.py --target-devices 0 || {
    echo "GPU validation failed! Check logs above."
    exit 1
}

echo ""
echo "Starting training..."
echo "=============================================="

# Training pipeline v3 for DeepSeek-R1 CoT:
#
# Changes from v2:
# - learning_rate=2e-4 (was 1e-4) — original rate
# - validation_split=0.08 (was 0.1) — 8% validation
# - max_seq_length=2048 — fits 97% of samples WITH CoT preserved
# - Chat template bug FIXED: <think> blocks preserved during training
# - load_best_model_at_end selects best checkpoint by eval_loss
#
python train_rag_baseline.py \
    --data_path src/data/dataset_final.json \
    --docs_path src/data/documents/DE \
    --adapter_name QM_rag_cot_v3 \
    --output_dir checkpoints/ \
    --target_devices 0 \
    --batch_size 1 \
    --gradient_accumulation 16 \
    --max_seq_length 2048 \
    --max_steps 1500 \
    --save_steps 50 \
    --learning_rate 2e-4 \
    --lora_r 16 \
    --lora_alpha 32 \
    --validation_split 0.08 \
    --quantization int4

echo ""
echo "=============================================="
echo "Job finished: $(date)"
echo "=============================================="
