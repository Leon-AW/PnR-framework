#!/bin/bash
#SBATCH --job-name=build_router_state
#SBATCH --output=logs/build_router_state_%j.log
#SBATCH --error=logs/build_router_state_%j.log
#SBATCH --time=01:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=longrun

# ==============================================================================
# Build the Time-Aware Centroid router state for all routing adapters.
#
# Runs on CPU (longrun partition) — sentence-transformer embeddings for
# ~500-5000 samples per adapter don't require a GPU and this avoids competing
# with training / eval jobs for gruenau10 GPU time.
#
# Adds QM adapter anchors (base_qm, patch_qm_current) alongside the existing
# SituatedQA and CounterFact adapter anchors. monolithic_qm is skipped
# (it is an eval baseline, not a routing target).
#
# Usage:
#   sbatch slurm/build_router_state.sh
# ==============================================================================

set -e
cd /vol/fob-vol1/mi23/wagnerql/PnR-framework

CONDA_BASE=/usr/local/anaconda3-2024.06
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate pnr

mkdir -p logs

echo "=== Build Router State ==="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "========================="

python scripts/build_router_state.py \
    --checkpoints_dir checkpoints/ \
    --output_dir checkpoints/router_state/ \
    --embedding_model sentence-transformers/all-MiniLM-L6-v2 \
    --max_samples 5000 \
    --similarity_threshold 0.45 \
    --cf_data_path data/counterfact_train.jsonl \
    --qm_old_data_path data/qm_stable_anchors.jsonl \
    --qm_new_data_path data/qm_train.jsonl \
    --qm_num_clusters 150 \
    --calibration_neg_path data/triviaqa_dcalibration.json \
    --in_domain_percentile 5.0 \
    --neg_percentile 99.0 \
    --threshold_margin 0.02 \
    --no_gpu \
    "$@"

echo "=== Done ==="
echo "Date: $(date)"
