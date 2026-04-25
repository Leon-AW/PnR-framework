#!/bin/bash
#SBATCH --job-name=morpheus_ks_seed
#SBATCH --partition=longgpu
#SBATCH --gres=gpu:a10080gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL

# ==============================================================================
# MORPHEUS KnowledgeStore Seed Job
#
# Populates morpheus_state/knowledge_store/records.json from SituatedQA and
# CounterFact training data. Without this, MORPHEUS's System 5 "graduated
# factuality" protocol never fires — every query lands in the boundary zone
# with an empty records list and no knowledge injection.
#
# Usage: sbatch slurm/build_morpheus_knowledge_store.sh [extra args]
# ==============================================================================

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

CONDA_BASE=/usr/local/anaconda3-2024.06
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate pnr

echo "======================================================================"
echo "Job ID     : ${SLURM_JOB_ID:-n/a}"
echo "Node       : ${SLURMD_NODENAME:-n/a}"
echo "Started    : $(date)"
echo "Extra args : $*"
echo "======================================================================"

python scripts/build_morpheus_knowledge_store.py \
    --output_dir morpheus_state/knowledge_store \
    --embedding_model sentence-transformers/all-MiniLM-L6-v2 \
    --n_per_split 500 \
    --skip_first 0 \
    --counterfact_path data/counterfact_train.jsonl \
    --batch_size 256 \
    "$@"

echo "======================================================================"
echo "Finished : $(date)"
echo "======================================================================"
