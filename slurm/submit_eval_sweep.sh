#!/bin/bash
# ==============================================================================
# Submit SituatedQA Evaluation Sweep
#
# Submits 7 baselines as independent parallel SLURM jobs.
# RECIPE Official is included but commented out — submit once training
# advances to ~epoch 200 (check: ls external/RECIPE/train_records/.../).
#
# Run from project root:
#   bash slurm/submit_eval_sweep.sh
#
# Results land in:  eval_results/<run_name>/
# MLflow tracking:  mlruns.db  (browse with: mlflow ui --backend-store-uri sqlite:///mlruns.db)
# ==============================================================================

set -euo pipefail

SCRIPT="slurm/eval_situated_qa.sh"
CKPT="$(pwd)/checkpoints"

echo "Submitting SituatedQA evaluation sweep..."
echo "Checkpoints dir : ${CKPT}"
echo ""

# ------------------------------------------------------------------------------
# 1. Frozen base model — CFR Pass 1 / universal stability baseline
# ------------------------------------------------------------------------------
JID1=$(sbatch --job-name=eval_frozen_base "${SCRIPT}" \
    --no_adapter \
    --run_name frozen_base \
    | awk '{print $NF}')
echo "[1/7] frozen_base        → job ${JID1}"

# ------------------------------------------------------------------------------
# 2. Monolithic LoRA — single adapter, no routing
# ------------------------------------------------------------------------------
JID2=$(sbatch --job-name=eval_monolithic "${SCRIPT}" \
    --monolithic "${CKPT}/monolithic_v1" \
    --run_name monolithic \
    | awk '{print $NF}')
echo "[2/7] monolithic         → job ${JID2}"

# ------------------------------------------------------------------------------
# 3. LoRA + RAG — monolithic adapter + QA-pair retrieval at inference time
# ------------------------------------------------------------------------------
JID3=$(sbatch --job-name=eval_lora_rag "${SCRIPT}" \
    --lora_rag "${CKPT}/monolithic_v1" \
    --lora_rag_index "$(pwd)/data/edit_pairs.json" \
    --run_name lora_rag \
    | awk '{print $NF}')
echo "[3/7] lora_rag           → job ${JID3}"

# ------------------------------------------------------------------------------
# 4. PnR routing — Time-Aware Centroid Router + Source-Replay
# ------------------------------------------------------------------------------
JID4=$(sbatch --job-name=eval_pnr "${SCRIPT}" \
    --run_name pnr \
    | awk '{print $NF}')
echo "[4/7] pnr                → job ${JID4}"

# ------------------------------------------------------------------------------
# 5. X-LoRA — soft gating baseline
# ------------------------------------------------------------------------------
JID5=$(sbatch --job-name=eval_xlora "${SCRIPT}" \
    --xlora "${CKPT}/xlora_baseline" \
    --run_name xlora \
    | awk '{print $NF}')
echo "[5/7] xlora              → job ${JID5}"

# ------------------------------------------------------------------------------
# 6. Parallel Orchestrator — multi-adapter ensemble + synthesis
# ------------------------------------------------------------------------------
JID6=$(sbatch --job-name=eval_parallel "${SCRIPT}" \
    --parallel \
    --run_name parallel_orchestrator \
    | awk '{print $NF}')
echo "[6/7] parallel_orch      → job ${JID6}"

# ------------------------------------------------------------------------------
# 7. MORPHEUS — multi-system architecture (bonus, not in exposé)
# ------------------------------------------------------------------------------
JID7=$(sbatch --job-name=eval_morpheus "${SCRIPT}" \
    --morpheus \
    --run_name morpheus \
    | awk '{print $NF}')
echo "[7/7] morpheus           → job ${JID7}"

# ------------------------------------------------------------------------------
# 8. RECIPE Official — submit when training is at ~epoch 200+
#    Check progress: ls external/RECIPE/train_records/recipe/mistral-7b/2026.04.14-13.34.10/checkpoints/
#    Pick the checkpoint with lowest ema_loss.
# ------------------------------------------------------------------------------
# RECIPE_CKPT="$(realpath external/RECIPE/train_records/recipe/mistral-7b/2026.04.14-13.34.10/checkpoints/epoch-44-i-27000-ema_loss-0.8886)"
# JID8=$(sbatch --job-name=eval_recipe "${SCRIPT}" \
#     --recipe_official "${RECIPE_CKPT}" \
#     --recipe_official_edits "$(pwd)/data/edit_pairs.json" \
#     --run_name recipe_official \
#     | awk '{print $NF}')
# echo "[8/8] recipe_official    → job ${JID8}"

echo ""
echo "======================================================================"
echo "Submitted 7 jobs. Monitor with:"
echo "  squeue -u \${USER}"
echo "  tail -f logs/eval_<name>_<jobid>.out"
echo ""
echo "RECIPE Official: uncomment block above once training reaches ~ep 200."
echo "  Current best:  external/RECIPE/train_records/recipe/mistral-7b/"
echo "                 2026.04.14-13.34.10/checkpoints/epoch-44-i-27000-ema_loss-0.8886"
echo "======================================================================"
