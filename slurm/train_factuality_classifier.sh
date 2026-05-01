#!/bin/bash
#SBATCH --job-name=factuality_clf
#SBATCH --partition=longgpu
#SBATCH --gres=gpu:a10080gb:1
#SBATCH --nodelist=gruenau10
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

CONDA_BASE=/usr/local/anaconda3-2024.06
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate pnr

OUTPUT_DIR=/vol/tmp/wagnerql/checkpoints/factuality_classifier

echo "======================================================================"
echo "Job ID     : ${SLURM_JOB_ID:-n/a}"
echo "Node       : ${SLURMD_NODENAME:-n/a}"
echo "Started    : $(date)"
echo "Output dir : ${OUTPUT_DIR}"
echo "======================================================================"

echo "--- Step 1: building classifier data ---"
python scripts/build_factuality_classifier_data.py \
    --output_path data/factuality_classifier_data.json \
    --cf_train_path data/counterfact_train.jsonl \
    --cf_eval_path data/counterfact_eval.json \
    --triviaqa_path data/triviaqa_dcontrol.json

echo "--- Step 2: training classifier ---"
python scripts/train_factuality_classifier.py \
    --data_path data/factuality_classifier_data.json \
    --embedding_model sentence-transformers/all-MiniLM-L6-v2 \
    --output_dir "${OUTPUT_DIR}" \
    --epochs 30 \
    --lr 1e-3 \
    --batch_size 64 \
    --dropout 0.2 \
    --hidden_dims 256 64 \
    --patience 5

echo "======================================================================"
echo "Finished : $(date)"
echo "Checkpoint : ${OUTPUT_DIR}"
echo "======================================================================"
