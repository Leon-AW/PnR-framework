#!/bin/bash
# ==============================================================================
# D_eval Sweep — v3 (Apr 30 2026): Factuality Classifier Integration
#
# Extends v2 with two new MORPHEUS variants that replace the hardcoded tau_low
# threshold with a trained MLP factuality classifier
# (all-MiniLM-L6-v2 → LayerNorm → 384→256→64→1, BCE+Adam, val AUC-ROC 0.814).
# Checkpoint: /vol/tmp/wagnerql/checkpoints/factuality_classifier
#
# Changes vs v2:
#   - Re-runs only the MORPHEUS jobs (jobs 7 + 9 from v2), now also as
#     _v3 variants with --morpheus_classifier_path set.
#   - All 9 v2 baselines already exist; they are NOT re-submitted here.
#     Add --rerun_all flag (see below) to resubmit everything if needed.
#   - v3 suffixes: morpheus_clf_deval_v3 / morpheus_clf_nobypass_deval_v3
#
# Thesis comparison (Table X):
#   morpheus_deval_v2          — tau_low=0.65, bypass on
#   morpheus_nobypass_deval_v2 — tau_low=0.65, bypass off
#   morpheus_clf_deval_v3      — MLP classifier, bypass on   ← new
#   morpheus_clf_nobypass_v3   — MLP classifier, bypass off  ← new
#
# Run from project root:
#   bash slurm/submit_deval_sweep_v3.sh
#
# To also re-run all 9 v2 baselines with _v3 suffix:
#   bash slurm/submit_deval_sweep_v3.sh --rerun_all
# ==============================================================================

set -euo pipefail

SCRIPT="slurm/eval_deval.sh"
CKPT="$(pwd)/checkpoints"
ROUTER_STATE="${CKPT}/router_state"
RECIPE_CKPT="$(realpath external/RECIPE/train_records/recipe/mistral-7b/2026.04.14-13.34.10/checkpoints/epoch-159-i-99000-ema_loss-0.2240)"
CLF_PATH="/vol/tmp/wagnerql/checkpoints/factuality_classifier"

RERUN_ALL=false
for arg in "$@"; do
    [[ "$arg" == "--rerun_all" ]] && RERUN_ALL=true
done

echo "D_eval v3 sweep — factuality classifier integration"
echo "Classifier: ${CLF_PATH}"
echo ""

# ==============================================================================
# Baseline re-runs (only submitted when --rerun_all is passed)
# ==============================================================================
if $RERUN_ALL; then
    echo "--- Resubmitting all v2 baselines with _v3 suffix ---"

    JID1=$(sbatch --job-name=deval_frozen_v3 "${SCRIPT}" \
        --no_adapter \
        --compute_logprob \
        --run_name frozen_base_deval_v3 \
        | awk '{print $NF}')
    echo "[1/9] frozen_base_v3        → job ${JID1}"

    JID2=$(sbatch --job-name=deval_mono_v3 "${SCRIPT}" \
        --monolithic "${CKPT}/monolithic_v1" \
        --compute_logprob \
        --run_name monolithic_deval_v3 \
        | awk '{print $NF}')
    echo "[2/9] monolithic_v3         → job ${JID2}"

    JID3=$(sbatch --job-name=deval_lora_rag_v3 "${SCRIPT}" \
        --lora_rag "${CKPT}/monolithic_v1" \
        --lora_rag_index "$(pwd)/data/counterfact_train.jsonl" \
        --compute_logprob \
        --run_name lora_rag_deval_v3 \
        | awk '{print $NF}')
    echo "[3/9] lora_rag_v3           → job ${JID3}"

    JID4=$(sbatch --job-name=deval_pnr_v3 "${SCRIPT}" \
        --router_state "${ROUTER_STATE}" \
        --similarity_threshold 0.45 \
        --compute_logprob \
        --run_name pnr_deval_v3 \
        | awk '{print $NF}')
    echo "[4/9] pnr_v3                → job ${JID4}"

    JID5=$(sbatch --job-name=deval_xlora_v3 "${SCRIPT}" \
        --xlora "${CKPT}/xlora_baseline" \
        --compute_logprob \
        --run_name xlora_deval_v3 \
        | awk '{print $NF}')
    echo "[5/9] xlora_v3              → job ${JID5}"

    JID6=$(sbatch --job-name=deval_recipe_v3 "${SCRIPT}" \
        --recipe_official "${RECIPE_CKPT}" \
        --recipe_official_edits "$(pwd)/data/edit_pairs.json" \
        --compute_logprob \
        --run_name recipe_deval_v3 \
        | awk '{print $NF}')
    echo "[6/9] recipe_official_v3    → job ${JID6}"

    JID8=$(sbatch --job-name=deval_parallel_v3 "${SCRIPT}" \
        --parallel \
        --router_state "${ROUTER_STATE}" \
        --similarity_threshold 0.45 \
        --compute_logprob \
        --run_name parallel_deval_v3 \
        | awk '{print $NF}')
    echo "[8/9] parallel_v3           → job ${JID8}"

    echo ""
fi

# ==============================================================================
# MORPHEUS v3 jobs — tau_low baseline (same as v2 jobs 7+9, for direct pairing)
# ==============================================================================
JID_M=$(sbatch --job-name=deval_morpheus_v3 "${SCRIPT}" \
    --morpheus \
    --morpheus_state_dir "$(pwd)/morpheus_state" \
    --compute_logprob \
    --run_name morpheus_deval_v3 \
    | awk '{print $NF}')
echo "[morpheus_v3]              → job ${JID_M}  (tau_low=0.65, bypass on — reference re-run)"

JID_MN=$(sbatch --job-name=deval_morpheus_nobypass_v3 "${SCRIPT}" \
    --morpheus \
    --morpheus_state_dir "$(pwd)/morpheus_state" \
    --morpheus_direct_answer_threshold 1.1 \
    --compute_logprob \
    --run_name morpheus_nobypass_deval_v3 \
    | awk '{print $NF}')
echo "[morpheus_nobypass_v3]     → job ${JID_MN}  (tau_low=0.65, bypass off — reference re-run)"

# ==============================================================================
# NEW: MORPHEUS + MLP classifier — bypass ON
#
# Routing signal: FactualityClassifier.predict_single(query) instead of max_sim.
# Classifier trained on CF paraphrases (pos) vs TriviaQA+neighborhood (neg);
# val AUC-ROC 0.814, CF recall 0.92.
# ==============================================================================
JID_CLF=$(sbatch --job-name=deval_morpheus_clf_v3 "${SCRIPT}" \
    --morpheus \
    --morpheus_state_dir "$(pwd)/morpheus_state" \
    --morpheus_classifier_path "${CLF_PATH}" \
    --compute_logprob \
    --run_name morpheus_clf_deval_v3 \
    | awk '{print $NF}')
echo "[morpheus_clf_v3]          → job ${JID_CLF}  (MLP classifier, bypass on)  ← NEW"

# ==============================================================================
# NEW: MORPHEUS + MLP classifier — bypass DISABLED (PnR-conformant ablation)
#
# direct_answer_threshold=1.1 forces all answers through the activated LoRA
# adapter, so ESR reflects the specialist output, not the hard_override bypass.
# Matches exposé R1 definition.
# ==============================================================================
JID_CLF_N=$(sbatch --job-name=deval_morpheus_clf_nobypass_v3 "${SCRIPT}" \
    --morpheus \
    --morpheus_state_dir "$(pwd)/morpheus_state" \
    --morpheus_classifier_path "${CLF_PATH}" \
    --morpheus_direct_answer_threshold 1.1 \
    --compute_logprob \
    --run_name morpheus_clf_nobypass_deval_v3 \
    | awk '{print $NF}')
echo "[morpheus_clf_nobypass_v3] → job ${JID_CLF_N}  (MLP classifier, bypass off) ← NEW"

echo ""
echo "======================================================================"
echo "Submitted. Monitor:"
echo "  squeue -u \${USER} --format='%.10i %.35j %.8T %.10M %.10L %R'"
echo "  tail -f logs/deval_morpheus_clf_v3_*.out"
echo "  tail -f logs/deval_morpheus_clf_nobypass_v3_*.out"
echo ""
echo "Results land in:"
echo "  eval_results/morpheus_clf_deval_v3/report.json"
echo "  eval_results/morpheus_clf_nobypass_deval_v3/report.json"
echo ""
echo "Thesis comparison (MORPHEUS ablation table):"
echo "  morpheus_deval_v2            tau_low=0.65  bypass=on"
echo "  morpheus_nobypass_deval_v2   tau_low=0.65  bypass=off"
echo "  morpheus_clf_deval_v3        MLP clf       bypass=on   ← new"
echo "  morpheus_clf_nobypass_v3     MLP clf       bypass=off  ← new"
echo ""
echo "Key metrics:"
echo "  summary.esr                       — generation-based ESR (free decoding)"
echo "  summary.logprob_esr               — teacher-forced ESR (ROME/MEMIT style)"
echo "  summary.dcontrol_forgetting_rate  — FR on D_control (0=perfect)"
echo "======================================================================"
