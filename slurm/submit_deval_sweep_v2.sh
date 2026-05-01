#!/bin/bash
# ==============================================================================
# Re-run D_eval Sweep — v2 (Apr 29 2026)
#
# Re-runs every system on D_eval (D_conflict ESR + D_control FR) after the
# Apr 29 2026 fixes:
#
#   1. `parse_model_output` now truncates at sentence boundaries (matches
#      `build_triviaqa_dcontrol.py::extract_answer` byte-for-byte). Affects
#      ALL splits, every system.
#   2. `GenerationConfig.stop_sequences` defaults to ('\n', '.', '!', '?');
#      `_StopOnSubstrings` halts generation at the first sentence boundary.
#      Cuts off verbose instruction-tuned continuations like "X is located
#      in Singapore." → "X is located in Singapore" (no period).
#   3. `--compute_logprob` enables ROME / MEMIT-style teacher-forced ESR:
#      every result also carries log P(target_new) and log P(target_true);
#      the report adds `summary.logprob_esr` (cf_conflict only) and
#      `summary.logprob_em` (any split).
#
# Differences from `submit_deval_sweep.sh`:
#   - All 9 jobs submitted in PARALLEL (no afterany dependency chain). With
#     gruenau7 + gruenau10 both idle this runs ~4-5 jobs concurrently
#     instead of one at a time.
#   - Every run carries a `_v2` suffix so old results stay intact for
#     before/after comparison in the thesis.
#   - All invocations pass --compute_logprob.
#
# Run from project root:
#   bash slurm/submit_deval_sweep_v2.sh
# ==============================================================================

set -euo pipefail

SCRIPT="slurm/eval_deval.sh"
CKPT="$(pwd)/checkpoints"
ROUTER_STATE="${CKPT}/router_state"
RECIPE_CKPT="$(realpath external/RECIPE/train_records/recipe/mistral-7b/2026.04.14-13.34.10/checkpoints/epoch-159-i-99000-ema_loss-0.2240)"

echo "Submitting D_eval v2 sweep (parsing fix + stop sequences + log-prob ESR)..."
echo "Nodes: gruenau7 (RTX A6000) + gruenau10 (A100 80GB) — see slurm/eval_deval.sh"
echo ""

# ------------------------------------------------------------------------------
# 1. Frozen base — sanity gate (FR should now be ≤ 0.7%, with parsing fix
#    expected closer to 0%). ESR=0% by definition.
# ------------------------------------------------------------------------------
JID1=$(sbatch --job-name=deval_frozen_v2 "${SCRIPT}" \
    --no_adapter \
    --compute_logprob \
    --run_name frozen_base_deval_v2 \
    | awk '{print $NF}')
echo "[1/9] frozen_base_v2       → job ${JID1}"

# ------------------------------------------------------------------------------
# 2. Monolithic LoRA — non-CF adapter, no routing. Establishes interference
#    floor on D_control after the fix.
# ------------------------------------------------------------------------------
JID2=$(sbatch --job-name=deval_mono_v2 "${SCRIPT}" \
    --monolithic "${CKPT}/monolithic_v1" \
    --compute_logprob \
    --run_name monolithic_deval_v2 \
    | awk '{print $NF}')
echo "[2/9] monolithic_v2        → job ${JID2}"

# ------------------------------------------------------------------------------
# 3. LoRA + RAG — monolithic adapter + CF retrieval index.
# ------------------------------------------------------------------------------
JID3=$(sbatch --job-name=deval_lora_rag_v2 "${SCRIPT}" \
    --lora_rag "${CKPT}/monolithic_v1" \
    --lora_rag_index "$(pwd)/data/counterfact_train.jsonl" \
    --compute_logprob \
    --run_name lora_rag_deval_v2 \
    | awk '{print $NF}')
echo "[3/9] lora_rag_v2          → job ${JID3}"

# ------------------------------------------------------------------------------
# 4. PnR — Centroid Router + patch_cf_main, threshold=0.45.
# ------------------------------------------------------------------------------
JID4=$(sbatch --job-name=deval_pnr_v2 "${SCRIPT}" \
    --router_state "${ROUTER_STATE}" \
    --similarity_threshold 0.45 \
    --compute_logprob \
    --run_name pnr_deval_v2 \
    | awk '{print $NF}')
echo "[4/9] pnr_v2               → job ${JID4}"

# ------------------------------------------------------------------------------
# 5. X-LoRA — soft gating baseline.
# ------------------------------------------------------------------------------
JID5=$(sbatch --job-name=deval_xlora_v2 "${SCRIPT}" \
    --xlora "${CKPT}/xlora_baseline" \
    --compute_logprob \
    --run_name xlora_deval_v2 \
    | awk '{print $NF}')
echo "[5/9] xlora_v2             → job ${JID5}"

# ------------------------------------------------------------------------------
# 6. RECIPE Official — best checkpoint (epoch-159).
# ------------------------------------------------------------------------------
JID6=$(sbatch --job-name=deval_recipe_v2 "${SCRIPT}" \
    --recipe_official "${RECIPE_CKPT}" \
    --recipe_official_edits "$(pwd)/data/edit_pairs.json" \
    --compute_logprob \
    --run_name recipe_deval_v2 \
    | awk '{print $NF}')
echo "[6/9] recipe_official_v2   → job ${JID6}"

# ------------------------------------------------------------------------------
# 7. MORPHEUS — bypass active (direct_answer_threshold=0.95 default).
# ------------------------------------------------------------------------------
JID7=$(sbatch --job-name=deval_morpheus_v2 "${SCRIPT}" \
    --morpheus \
    --morpheus_state_dir "$(pwd)/morpheus_state" \
    --compute_logprob \
    --run_name morpheus_deval_v2 \
    | awk '{print $NF}')
echo "[7/9] morpheus_v2          → job ${JID7}"

# ------------------------------------------------------------------------------
# 8. Parallel Orchestrator.
# ------------------------------------------------------------------------------
JID8=$(sbatch --job-name=deval_parallel_v2 "${SCRIPT}" \
    --parallel \
    --router_state "${ROUTER_STATE}" \
    --similarity_threshold 0.45 \
    --compute_logprob \
    --run_name parallel_deval_v2 \
    | awk '{print $NF}')
echo "[8/9] parallel_v2          → job ${JID8}"

# ------------------------------------------------------------------------------
# 9. MORPHEUS — bypass DISABLED (PnR-conformant ablation).
# ------------------------------------------------------------------------------
JID9=$(sbatch --job-name=deval_morpheus_nobypass_v2 "${SCRIPT}" \
    --morpheus \
    --morpheus_state_dir "$(pwd)/morpheus_state" \
    --morpheus_direct_answer_threshold 1.1 \
    --compute_logprob \
    --run_name morpheus_nobypass_deval_v2 \
    | awk '{print $NF}')
echo "[9/9] morpheus_nobypass_v2 → job ${JID9}"

echo ""
echo "======================================================================"
echo "Submitted 9 jobs (parallel — no afterany chain)."
echo ""
echo "Monitor:"
echo "  squeue -u \${USER} --format='%.10i %.30j %.8T %.10M %.10L %R'"
echo "  tail -f logs/deval_*_v2_*.out"
echo ""
echo "Results:"
echo "  eval_results/<run_name>_v2/report.json"
echo ""
echo "Key new fields:"
echo "  summary.esr                       — generation-based ESR (free decoding)"
echo "  summary.logprob_esr               — ROME/MEMIT-style ESR (teacher-forced)"
echo "  summary.dcontrol_forgetting_rate  — should drop after parsing fix"
echo "======================================================================"
