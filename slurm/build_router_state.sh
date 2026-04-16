#!/bin/bash
#SBATCH --job-name=build_router_state
#SBATCH --output=logs/build_router_state_%j.log
#SBATCH --error=logs/build_router_state_%j.log
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=gruenau10

set -e
cd /vol/fob-vol1/mi23/wagnerql/PnR-framework

CONDA_BASE=/usr/local/anaconda3-2024.06
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate pnr

mkdir -p logs

echo "=== Build Router State ==="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "CUDA: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'no GPU')"
echo "========================="

python scripts/build_router_state.py \
    --checkpoints_dir checkpoints/ \
    --output_dir checkpoints/router_state/ \
    --embedding_model sentence-transformers/all-MiniLM-L6-v2 \
    --max_samples 500

echo "=== Done ==="
echo "Date: $(date)"
