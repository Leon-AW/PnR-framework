#!/bin/bash
# ==============================================================================
# Train All CounterFact Cluster Adapters
#
# Submits 6 SLURM jobs (one per cluster), each training a LoRA adapter on
# ~3K CounterFact records. Each job takes ~30-45 min on A100.
#
# Prerequisite:
#   conda run -n pnr python scripts/build_counterfact_data.py
#   (produces data/counterfact_cluster_{0..5}.jsonl via MiniLM KMeans clustering)
# ==============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."

N_CLUSTERS=6
MAX_STEPS=1000
LORA_R=16
LORA_ALPHA=32

echo "======================================================================"
echo "Submitting ${N_CLUSTERS} CounterFact cluster training jobs"
echo "Max steps per cluster: ${MAX_STEPS}"
echo "======================================================================"

for i in $(seq 0 $((N_CLUSTERS - 1))); do
    DATA_PATH="data/counterfact_cluster_${i}.jsonl"
    ADAPTER_NAME="patch_cf_${i}"
    OUTPUT_DIR="checkpoints/${ADAPTER_NAME}"

    if [ ! -f "${DATA_PATH}" ]; then
        echo "ERROR: ${DATA_PATH} not found! Run: python scripts/build_counterfact_data.py"
        exit 1
    fi

    N_RECORDS=$(wc -l < "${DATA_PATH}")

    JID=$(sbatch \
        --job-name="cf_cl_${i}" \
        --partition=longgpu \
        --gres=gpu:a10080gb:1 \
        --cpus-per-task=8 \
        --mem=64G \
        --time=02:00:00 \
        --output="logs/train_cf_cluster_${i}_%j.out" \
        --error="logs/train_cf_cluster_${i}_%j.err" \
        --exclude=gruenau10 \
        --wrap="
set -euo pipefail
cd ${PWD}
CONDA_BASE=/usr/local/anaconda3-2024.06
source \"\${CONDA_BASE}/etc/profile.d/conda.sh\"
conda activate pnr

echo '======================================================================'
echo \"Cluster ${i}: ${ADAPTER_NAME} (${N_RECORDS} records, ${MAX_STEPS} steps)\"
echo \"Node: \${SLURMD_NODENAME}\"
echo \"GPU:  \$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)\"
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
    --max_seq_length 256 \\
    --save_steps 100 \\
    --logging_steps 25

echo 'Done.'
" | awk '{print $NF}')

    echo "  [${i}/${N_CLUSTERS}] ${ADAPTER_NAME} (${N_RECORDS} records) → job ${JID}"
done

echo "======================================================================"
echo "All ${N_CLUSTERS} cluster jobs submitted."
echo "Monitor: squeue -u \$(whoami) --name='cf_cl_*'"
echo "======================================================================"
