#!/bin/bash
#SBATCH --job-name=morpheus_qm_ks_seed
#SBATCH --partition=longgpu
#SBATCH --gres=gpu:a10080gb:1
#SBATCH --nodelist=gruenau9
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL

# ==============================================================================
# MORPHEUS KnowledgeStore Seed Job — AIT QM domain
#
# Builds an isolated, QM-only KS at /vol/tmp/wagnerql/morpheus_state_qm/
# (accessed via the morpheus_state_qm/ symlink in the repo root). Records:
#   - qm_train.jsonl       → 500 records, domain "qm_patch"
#   - qm_train_base.jsonl  → 1000 records, domain "qm_base"
# Total expected: ~1500 records.
#
# SQA + CF are skipped — the QM eval should not see CF triples in its KS.
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
echo "======================================================================"

python scripts/build_morpheus_knowledge_store.py \
    --output_dir /vol/tmp/wagnerql/morpheus_state_qm/knowledge_store \
    --embedding_model sentence-transformers/all-MiniLM-L6-v2 \
    --skip_sqa \
    --skip_counterfact \
    --qm_train_path data/qm_train.jsonl \
    --qm_train_base_path data/qm_train_base.jsonl \
    --batch_size 64

echo "======================================================================"
echo "Finished : $(date)"
echo "Records  :"
ls -l /vol/tmp/wagnerql/morpheus_state_qm/knowledge_store/
echo "======================================================================"
