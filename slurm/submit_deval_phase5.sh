#!/bin/bash
# ==============================================================================
# Submit D_eval Phase 5 — Two-Stage Routing with 6 CF Relation-Family Clusters
#
# Phase 5 of the May 2026 NF-1/NF-2/NF-4 closure plan. Runs exactly the four
# system × dataset combinations needed to validate the multi-expert claim:
#
#   1. PnR      × CF  D_eval (cf_conflict + cf_control)
#   2. PnR      × SQA D_eval (sqa_train     + cf_control)
#   3. Parallel × CF  D_eval
#   4. Parallel × SQA D_eval
#
# All four runs share:
#   - --router_state with 6 patch_cf_relfam_{0..5} adapters
#   - --domain_classifier_path Stage-1 gate (val macro-F1 0.978)
#   - --compute_logprob (cheap; gives logprob_esr without judge)
#   - --domain_confidence_threshold 0.7  /  --domain_fallback_threshold 0.30
#
# Judge scoring for SQA correctness runs separately after these complete.
#
# Run from project root:
#   bash slurm/submit_deval_phase5.sh
# ==============================================================================

set -euo pipefail

cd "$(dirname "$0")/.."

CKPT="$(pwd)/checkpoints"
ROUTER_STATE="${CKPT}/router_state"
DOMAIN_CLF="${CKPT}/domain_classifier"

# Sanity-check prerequisites — failing here saves a 4-job submit cascade.
for p in "${ROUTER_STATE}/manifest.json" "${DOMAIN_CLF}/classifier.pt" \
         data/sqa_deval.json data/counterfact_eval.json \
         data/triviaqa_dcontrol.json; do
    if [ ! -e "$p" ]; then
        echo "ERROR: required artefact missing: $p"
        exit 1
    fi
done

CF_SCRIPT="slurm/eval_deval.sh"
SQA_SCRIPT="slurm/eval_sqa_deval.sh"

COMMON_ARGS=(
    --router_state            "${ROUTER_STATE}"
    --similarity_threshold    0.45
    --domain_classifier_path  "${DOMAIN_CLF}"
    --domain_confidence_threshold 0.7
    --domain_fallback_threshold   0.30
    --compute_logprob
)

submit() {
    local job_name="$1"; shift
    local script="$1"; shift
    local run_name="$1"; shift
    if [ -z "${PREV_JOB:-}" ]; then
        JID=$(sbatch --parsable --job-name="${job_name}" "${script}" \
              --run_name "${run_name}" "$@" "${COMMON_ARGS[@]}")
    else
        JID=$(sbatch --parsable --dependency=afterany:"${PREV_JOB}" \
              --job-name="${job_name}" "${script}" \
              --run_name "${run_name}" "$@" "${COMMON_ARGS[@]}")
    fi
    echo "Submitted ${run_name} → job ${JID}"
    PREV_JOB="${JID}"
}

echo "============================================================"
echo "Phase 5 D_eval sweep (4 jobs, sequential on gruenau10)"
echo "  Router state : ${ROUTER_STATE}"
echo "  Domain clf   : ${DOMAIN_CLF}"
echo "============================================================"

submit deval_pnr_phase5_cf      "${CF_SCRIPT}"  pnr_phase5_cf_deval
submit deval_pnr_phase5_sqa     "${SQA_SCRIPT}" pnr_phase5_sqa_deval
submit deval_parallel_phase5_cf "${CF_SCRIPT}"  parallel_phase5_cf_deval --parallel
submit deval_parallel_phase5_sqa "${SQA_SCRIPT}" parallel_phase5_sqa_deval --parallel

echo ""
echo "All 4 jobs submitted (sequential via --dependency=afterany)."
echo "Final job: ${PREV_JOB}"
echo "Monitor:"
echo "  squeue -u \$(whoami) --format='%.10i %.30j %.2t %.10M %.10L %R'"
echo "Results land in: eval_results/<run_name>/summary.json"
