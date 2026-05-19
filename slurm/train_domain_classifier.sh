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
# Train the 4-class domain classifier — Stage 1 of the two-stage router.
#
# Extends the Phase-4 3-class classifier (cf/sqa/ood_trivia) with a 4th
# class "qm" for AIT QM queries, so Stage-1 routing covers the QM domain
# and routes qm queries to {base_qm, patch_qm_current} in Stage 2.
#
# QM class uses 500 available questions (imbalanced vs ~5000 for the other
# three classes); stratified split preserves the imbalance. Expect slightly
# lower QM-specific recall than the other classes — primary concern is OOD
# rejection (ood_trivia recall) which is unaffected.
#
# Output: /vol/tmp/wagnerql/checkpoints/domain_classifier  (overwrites v3)
# Backup: checkpoints/domain_classifier.backup_3class is created first.
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

# Back up the existing 3-class checkpoint before overwriting.
if [ -L "checkpoints/domain_classifier" ]; then
    cp -r "checkpoints/domain_classifier" "checkpoints/domain_classifier.backup_3class" 2>/dev/null || true
    echo "Backed up 3-class checkpoint → checkpoints/domain_classifier.backup_3class"
fi

echo "--- Step 1: building 4-class domain data (cf / sqa / qm / ood_trivia) ---"
python scripts/build_domain_classifier_data.py \
    --output_path data/domain_classifier_data_4class.json \
    --cf_train_path data/counterfact_train.jsonl \
    --sqa_cache_dir data/sqa_train_cache \
    --qm_train_path data/qm_train.jsonl \
    --qm_n_samples 500 \
    --dcontrol_path data/triviaqa_dcontrol.json \
    --dcalibration_path data/triviaqa_dcalibration.json \
    --n_per_class 5000 \
    --val_frac 0.10 \
    --seed 42

echo "--- Step 2: training 4-class classifier ---"
python scripts/train_domain_classifier.py \
    --data_path data/domain_classifier_data_4class.json \
    --embedding_model sentence-transformers/all-MiniLM-L6-v2 \
    --output_dir "${OUTPUT_DIR}" \
    --epochs 30 \
    --lr 1e-3 \
    --batch_size 64 \
    --dropout 0.2 \
    --hidden_dims 256 64 \
    --patience 5

ln -sfn "${OUTPUT_DIR}" checkpoints/domain_classifier
echo "Symlinked checkpoints/domain_classifier → ${OUTPUT_DIR}"

echo "======================================================================"
echo "Finished   : $(date)"
echo "Checkpoint : ${OUTPUT_DIR}"
echo "Next       : sbatch slurm/eval_qm_deval.sh --router_state checkpoints/router_state"
echo "             --domain_classifier_path checkpoints/domain_classifier"
echo "             --run_name pnr_qm_routed --output_dir eval_results/qm_deval_pnr"
echo "======================================================================"
