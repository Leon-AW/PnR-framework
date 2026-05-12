#!/bin/bash
#SBATCH --job-name=domain_clf
#SBATCH --partition=longgpu
#SBATCH --gres=gpu:a10080gb:1
#SBATCH --nodelist=gruenau10
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL

# ==============================================================================
# Phase 4 — train the 3-class domain classifier (Stage 1 of two-stage routing).
#
# Wraps the data-build + training steps in a single SLURM job. Mirrors
# slurm/train_factuality_classifier.sh — same encoder, same MLP topology,
# same partition. The job runs in parallel with the cluster training jobs
# (Phase 2) since the classifier needs <8 GB and shares the A100 via MPS
# without contending materially with LoRA adapter training.
# ==============================================================================

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

CONDA_BASE=/usr/local/anaconda3-2024.06
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate pnr

OUTPUT_DIR=/vol/tmp/wagnerql/checkpoints/domain_classifier

echo "======================================================================"
echo "Job ID     : ${SLURM_JOB_ID:-n/a}"
echo "Node       : ${SLURMD_NODENAME:-n/a}"
echo "Started    : $(date)"
echo "Output dir : ${OUTPUT_DIR}"
echo "======================================================================"

echo "--- Step 1: building 3-class domain data (cf / sqa / ood_trivia) ---"
python scripts/build_domain_classifier_data.py \
    --output_path data/domain_classifier_data.json \
    --cf_train_path data/counterfact_train.jsonl \
    --sqa_cache_dir data/sqa_train_cache \
    --dcontrol_path data/triviaqa_dcontrol.json \
    --dcalibration_path data/triviaqa_dcalibration.json \
    --n_per_class 5000 \
    --val_frac 0.10 \
    --seed 42

echo "--- Step 2: training 3-class classifier ---"
python scripts/train_domain_classifier.py \
    --data_path data/domain_classifier_data.json \
    --embedding_model sentence-transformers/all-MiniLM-L6-v2 \
    --output_dir "${OUTPUT_DIR}" \
    --epochs 30 \
    --lr 1e-3 \
    --batch_size 64 \
    --dropout 0.2 \
    --hidden_dims 256 64 \
    --patience 5

echo "======================================================================"
echo "Finished   : $(date)"
echo "Checkpoint : ${OUTPUT_DIR}"
echo "Next       : wire src/routing/centroid_router.py to consume this checkpoint."
echo "======================================================================"
