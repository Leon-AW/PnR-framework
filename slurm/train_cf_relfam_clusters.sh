#!/bin/bash
# ==============================================================================
# Train CounterFact Relation-Family Cluster Adapters (Phase 2 of TODO 8)
#
# Submits one SLURM job per cluster ID in {0..5}, training a LoRA adapter on
# the matching data/counterfact_relfam_${i}.jsonl partition. Replaces the
# legacy patch_cf_{0..5} which used agglomerative clustering on per-relation
# MiniLM centroids and was undertrained at r=16, 1000 steps.
#
# Config matches patch_cf_main: r=32, alpha=64, max_steps=8000, eff_batch=16,
# max_seq_length=128. Output → /vol/tmp/wagnerql/checkpoints/patch_cf_relfam_${i}
# (per storage convention; symlink under checkpoints/ added after job completes).
#
# Prerequisite:
#   conda run -n pnr python scripts/build_counterfact_relation_clusters.py
#   (produces data/counterfact_relfam_{0..5}.jsonl + mapping JSON)
#
# Usage:
#   slurm/train_cf_relfam_clusters.sh                   # submit all 6 (parallel)
#   slurm/train_cf_relfam_clusters.sh 0                 # submit cluster 0 only
#   slurm/train_cf_relfam_clusters.sh 1 2 3 4 5         # submit a subset
# ==============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."

if [ "$#" -eq 0 ]; then
    CLUSTERS=(0 1 2 3 4 5)
else
    CLUSTERS=("$@")
fi

MAX_STEPS=6500  # cluster 0 plateaued at epoch 32 (step ~6400); 8000 wasted ~3.5 h/cluster
LORA_R=32
LORA_ALPHA=64
SAVE_STEPS=500
LOGGING_STEPS=50
MAX_SEQ_LENGTH=128
OFFVOL_CKPT_ROOT=/vol/tmp/wagnerql/checkpoints

mkdir -p logs
mkdir -p "${OFFVOL_CKPT_ROOT}"

echo "======================================================================"
echo "Submitting CounterFact relation-family cluster training jobs"
echo "Clusters     : ${CLUSTERS[*]}"
echo "Max steps    : ${MAX_STEPS}"
echo "LoRA r/alpha : ${LORA_R}/${LORA_ALPHA}"
echo "Output root  : ${OFFVOL_CKPT_ROOT}"
echo "======================================================================"

for i in "${CLUSTERS[@]}"; do
    DATA_PATH="data/counterfact_relfam_${i}.jsonl"
    ADAPTER_NAME="patch_cf_relfam_${i}"
    OUTPUT_DIR="${OFFVOL_CKPT_ROOT}/${ADAPTER_NAME}"

    if [ ! -f "${DATA_PATH}" ]; then
        echo "ERROR: ${DATA_PATH} not found! Run: python scripts/build_counterfact_relation_clusters.py"
        exit 1
    fi

    N_RECORDS=$(wc -l < "${DATA_PATH}")

    JID=$(sbatch \
        --job-name="cfrf_${i}" \
        --partition=longgpu \
        --gres=gpu:a10080gb:1 \
        --nodelist=gruenau10 \
        --cpus-per-task=8 \
        --mem=64G \
        --time=18:00:00 \
        --output="logs/train_cf_relfam_${i}_%j.out" \
        --error="logs/train_cf_relfam_${i}_%j.err" \
        --wrap="
set -euo pipefail
cd ${PWD}
CONDA_BASE=/usr/local/anaconda3-2024.06
source \"\${CONDA_BASE}/etc/profile.d/conda.sh\"
conda activate pnr

echo '======================================================================'
echo \"Cluster ${i}: ${ADAPTER_NAME} (${N_RECORDS} records, ${MAX_STEPS} steps)\"
echo \"Job ID: \${SLURM_JOB_ID}\"
echo \"Node:   \${SLURMD_NODENAME}\"
echo \"GPU:    \$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)\"
echo \"Started: \$(date)\"
echo '======================================================================'

python train/train_counterfact_patch.py \\
    --data_path ${DATA_PATH} \\
    --adapter_name ${ADAPTER_NAME} \\
    --output_dir ${OUTPUT_DIR} \\
    --max_steps ${MAX_STEPS} \\
    --batch_size 1 \\
    --gradient_accumulation 16 \\
    --learning_rate 2e-4 \\
    --lora_r ${LORA_R} \\
    --lora_alpha ${LORA_ALPHA} \\
    --max_seq_length ${MAX_SEQ_LENGTH} \\
    --save_steps ${SAVE_STEPS} \\
    --logging_steps ${LOGGING_STEPS}

echo \"Finished: \$(date)\"
" | awk '{print $NF}')

    echo "  [cluster ${i}] ${ADAPTER_NAME} (${N_RECORDS} records) → job ${JID}"
done

echo "======================================================================"
echo "Submitted ${#CLUSTERS[@]} job(s)."
echo "Monitor: squeue -u \$(whoami) --name='cfrf_*'"
echo "Logs:    logs/train_cf_relfam_<i>_<jobid>.{out,err}"
echo "After completion, symlink checkpoints/patch_cf_relfam_<i> → ${OFFVOL_CKPT_ROOT}/patch_cf_relfam_<i>"
echo "======================================================================"
