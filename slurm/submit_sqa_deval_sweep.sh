#!/bin/bash
# Submit SituatedQA D_eval for all methods.
# Serialised via --dependency=afterany to avoid MPS contention on gruenau10.
#
# Usage:
#   bash slurm/submit_sqa_deval_sweep.sh
#
# Prerequisites:
#   data/sqa_deval.json must exist (run slurm/build_sqa_deval.sh first).

set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -f data/sqa_deval.json ]; then
    echo "ERROR: data/sqa_deval.json not found. Run slurm/build_sqa_deval.sh first."
    exit 1
fi

CHECKPOINTS="$(realpath checkpoints)"
MORPHEUS_STATE="$(realpath morpheus_state)"

submit() {
    local name="$1"; shift
    if [ -z "${PREV_JOB:-}" ]; then
        JID=$(sbatch --parsable slurm/eval_sqa_deval.sh --run_name "$name" "$@")
    else
        JID=$(sbatch --parsable --dependency=afterany:"${PREV_JOB}" \
              slurm/eval_sqa_deval.sh --run_name "$name" "$@")
    fi
    echo "Submitted $name → job $JID"
    PREV_JOB="$JID"
}

# ── Baselines ────────────────────────────────────────────────────────────────

submit frozen_base_sqa_deval \
    --no_adapter

submit monolithic_sqa_deval \
    --monolithic "${CHECKPOINTS}/monolithic_v1"

submit lora_rag_sqa_deval \
    --lora_rag        "${CHECKPOINTS}/monolithic_v1" \
    --lora_rag_index  data/edit_pairs.json

submit recipe_sqa_deval \
    --recipe_official       "${CHECKPOINTS}/recipe_baseline" \
    --recipe_official_edits data/edit_pairs.json

submit xlora_sqa_deval \
    --xlora /vol/tmp/wagnerql/xlora_baseline

# ── PnR + extensions ─────────────────────────────────────────────────────────

submit pnr_sqa_deval \
    --similarity_threshold 0.45

submit parallel_sqa_deval \
    --parallel

submit morpheus_sqa_deval \
    --morpheus \
    --morpheus_state_dir "${MORPHEUS_STATE}" \
    --morpheus_direct_answer_threshold 0.95

echo ""
echo "All SQA D_eval jobs submitted. Final job: ${PREV_JOB}"
echo "Monitor: squeue -u wagnerql --format='%.10i %.35j %.8T %.10M %.10L %R'"
