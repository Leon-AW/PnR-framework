#!/bin/bash
#SBATCH --job-name=build_router_state
#SBATCH --output=logs/build_router_state_%j.log
#SBATCH --error=logs/build_router_state_%j.log
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=longgpu
#SBATCH --gres=gpu:a10080gb:1
#SBATCH --nodelist=gruenau10

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

# --max_samples 5000 keeps memory bounded while still providing enough
#   per-chunk anchors for SQA splits; patch_cf_main truncates at this cap
#   (full CF train has ~19k facts).
# --calibration_neg_path : disjoint TriviaQA slice used to calibrate per-
#   adapter routing thresholds. Build it first via
#   `sbatch slurm/build_triviaqa_dcalibration.sh`.
# --similarity_threshold acts as the global fallback when calibration is
#   unavailable; calibrated per-adapter thresholds otherwise take precedence.
python scripts/build_router_state.py \
    --checkpoints_dir checkpoints/ \
    --output_dir checkpoints/router_state/ \
    --embedding_model sentence-transformers/all-MiniLM-L6-v2 \
    --max_samples 5000 \
    --similarity_threshold 0.45 \
    --cf_data_path data/counterfact_train.jsonl \
    --calibration_neg_path data/triviaqa_dcalibration.json \
    --in_domain_percentile 5.0 \
    --neg_percentile 99.0 \
    --threshold_margin 0.02

echo "=== Done ==="
echo "Date: $(date)"
