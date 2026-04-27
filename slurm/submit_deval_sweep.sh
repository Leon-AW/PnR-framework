#!/bin/bash
# ==============================================================================
# Submit D_eval Sweep — Thesis Primary Evaluation
#
# Runs D_eval = D_conflict (ESR) ∪ D_control (forgetting rate) for all systems.
#
# Thesis metrics (from exposé):
#   R1 — ESR   : fraction of CF training records where system outputs target_false
#   R2 — FR    : 1 - accuracy on D_control (TriviaQA pre-filtered to 100% base acc)
#   R2 — Eff.  : wall-clock + VRAM vs monolithic
#
# Each system runs independently. Monitor:
#   squeue -u ${USER}
#   tail -f logs/eval_<name>_<jobid>.out
#
# Results land in: eval_results/<run_name>/summary.json
#
# Run from project root:
#   bash slurm/submit_deval_sweep.sh
# ==============================================================================

set -euo pipefail

SCRIPT="slurm/eval_deval.sh"
CKPT="$(pwd)/checkpoints"
ROUTER_STATE="${CKPT}/router_state"
RECIPE_CKPT="$(realpath external/RECIPE/train_records/recipe/mistral-7b/2026.04.14-13.34.10/checkpoints/epoch-159-i-99000-ema_loss-0.2240)"

echo "Submitting D_eval sweep (ESR + D_control forgetting rate)..."
echo ""

# ------------------------------------------------------------------------------
# 1. Frozen base — R1 ceiling floor / R2 reference (FR should be ~0 by design)
# ------------------------------------------------------------------------------
JID1=$(sbatch --job-name=deval_frozen "${SCRIPT}" \
    --no_adapter \
    --run_name frozen_base_deval \
    | awk '{print $NF}')
echo "[1/8] frozen_base      → job ${JID1}"

# ------------------------------------------------------------------------------
# 2. Monolithic LoRA — SituatedQA adapter, no routing
#    ESR expected ~0 (not trained on CF). FR measures interference.
# ------------------------------------------------------------------------------
JID2=$(sbatch --dependency=afterany:${JID1} --job-name=deval_mono "${SCRIPT}" \
    --monolithic "${CKPT}/monolithic_v1" \
    --run_name monolithic_deval \
    | awk '{print $NF}')
echo "[2/8] monolithic       → job ${JID2}"

# ------------------------------------------------------------------------------
# 3. LoRA+RAG — monolithic + CF pairs as retrieval index
#    Gives LoRA+RAG access to CF knowledge via retrieval, enabling fair ESR comparison.
# ------------------------------------------------------------------------------
JID3=$(sbatch --dependency=afterany:${JID2} --job-name=deval_lora_rag "${SCRIPT}" \
    --lora_rag "${CKPT}/monolithic_v1" \
    --lora_rag_index "$(pwd)/data/counterfact_train.jsonl" \
    --run_name lora_rag_deval \
    | awk '{print $NF}')
echo "[3/8] lora_rag         → job ${JID3}"

# ------------------------------------------------------------------------------
# 4. PnR — Time-Aware Centroid Router + patch_cf_main
#    --similarity_threshold 0.45: CF queries land at ~0.50 sim (default 0.65 abstains)
# ------------------------------------------------------------------------------
JID4=$(sbatch --dependency=afterany:${JID3} --job-name=deval_pnr "${SCRIPT}" \
    --router_state "${ROUTER_STATE}" \
    --similarity_threshold 0.45 \
    --run_name pnr_deval \
    | awk '{print $NF}')
echo "[4/8] pnr              → job ${JID4}"

# ------------------------------------------------------------------------------
# 5. X-LoRA — soft gating baseline
# ------------------------------------------------------------------------------
JID5=$(sbatch --dependency=afterany:${JID4} --job-name=deval_xlora "${SCRIPT}" \
    --xlora "${CKPT}/xlora_baseline" \
    --run_name xlora_deval \
    | awk '{print $NF}')
echo "[5/8] xlora            → job ${JID5}"

# ------------------------------------------------------------------------------
# 6. RECIPE Official — best checkpoint (epoch-159, ema_loss=0.2240)
#    Uses SituatedQA knowledge base (no CF edits — measures cross-domain robustness)
# ------------------------------------------------------------------------------
JID6=$(sbatch --dependency=afterany:${JID5} --job-name=deval_recipe "${SCRIPT}" \
    --recipe_official "${RECIPE_CKPT}" \
    --recipe_official_edits "$(pwd)/data/edit_pairs.json" \
    --run_name recipe_deval \
    | awk '{print $NF}')
echo "[6/8] recipe_official  → job ${JID6}"

# ------------------------------------------------------------------------------
# 7. MORPHEUS — multi-system architecture (System 5 hard_override + meta-controller)
#    Default direct_answer_threshold=0.95 → bypass active on high-confidence
#    hard_override hits. Reported as ablation alongside (9) below.
# ------------------------------------------------------------------------------
JID7=$(sbatch --dependency=afterany:${JID6} --job-name=deval_morpheus "${SCRIPT}" \
    --morpheus \
    --morpheus_state_dir "$(pwd)/morpheus_state" \
    --run_name morpheus_deval \
    | awk '{print $NF}')
echo "[7/9] morpheus         → job ${JID7}"

# ------------------------------------------------------------------------------
# 8. Parallel Orchestrator — ensemble + synthesis routing
#    Same threshold as PnR (0.45) since the underlying centroid router is shared.
# ------------------------------------------------------------------------------
JID8=$(sbatch --dependency=afterany:${JID7} --job-name=deval_parallel "${SCRIPT}" \
    --parallel \
    --router_state "${ROUTER_STATE}" \
    --similarity_threshold 0.45 \
    --run_name parallel_deval \
    | awk '{print $NF}')
echo "[8/9] parallel_orch    → job ${JID8}"

# ------------------------------------------------------------------------------
# 9. MORPHEUS — bypass DISABLED (PnR-conformant ablation)
#    direct_answer_threshold=1.1 forces every answer through the LoRA adapter
#    so that ESR reflects the activated specialist (matches exposé R1).
#    The bypass-on result (job 7) is reported as a retrieval-direct ablation.
# ------------------------------------------------------------------------------
JID9=$(sbatch --dependency=afterany:${JID8} --job-name=deval_morpheus_nobypass "${SCRIPT}" \
    --morpheus \
    --morpheus_state_dir "$(pwd)/morpheus_state" \
    --morpheus_direct_answer_threshold 1.1 \
    --run_name morpheus_nobypass_deval \
    | awk '{print $NF}')
echo "[9/9] morpheus_nobypass → job ${JID9}"

echo ""
echo "======================================================================"
echo "Submitted 9 jobs. Results:"
echo "  eval_results/<run_name>/summary.json"
echo ""
echo "Key metrics to compare:"
echo "  summary.esr                   — Edit Success Rate on D_conflict"
echo "  summary.dcontrol_forgetting_rate — Forgetting on D_control (0=perfect)"
echo "  summary.dcontrol_accuracy     — D_control accuracy (1=no forgetting)"
echo "  summary.efficiency            — latency + VRAM"
echo "======================================================================"
