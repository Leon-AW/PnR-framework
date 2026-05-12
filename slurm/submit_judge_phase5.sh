#!/bin/bash
# ==============================================================================
# Submit Phase 5 Judge Scoring
#
# Post-hoc LLM-as-a-Judge scoring for the 4 Phase-5 D_eval run dirs:
#   pnr_phase5_cf_deval         (cf_conflict + cf_control)
#   pnr_phase5_sqa_deval        (sqa_train   + cf_control)
#   parallel_phase5_cf_deval    (cf_conflict + cf_control)
#   parallel_phase5_sqa_deval   (sqa_train   + cf_control)
#
# `scripts/score_with_judge.py` auto-routes prompt family by split name:
#   cf_conflict → counterfact judge
#   sqa_train / cf_control → factoid judge
#
# Single SLURM job (4 runs × ~17 min/run ≈ 70 min; 4h budget).
#
# Default behaviour: chains after the last Phase-5 D_eval job via
# --dependency=afterany so the judge fires only once the result.json files
# exist. Pass --no_dep to override (e.g. for re-scoring a completed run).
#
# Usage:
#   bash slurm/submit_judge_phase5.sh                # waits on JID 358567
#   bash slurm/submit_judge_phase5.sh --no_dep       # submit immediately
#   bash slurm/submit_judge_phase5.sh --dep_jid 1234 # custom predecessor
# ==============================================================================

set -euo pipefail

cd "$(dirname "$0")/.."

# Default predecessor: the last D_eval job submitted by submit_deval_phase5.sh.
# Update this if you resubmit the sweep.
DEFAULT_DEP_JID=358567

DEP_JID="${DEFAULT_DEP_JID}"
USE_DEP=true

while [ $# -gt 0 ]; do
    case "$1" in
        --no_dep) USE_DEP=false; shift ;;
        --dep_jid) DEP_JID="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

RUNS=(
    pnr_phase5_cf_deval
    pnr_phase5_sqa_deval
    parallel_phase5_cf_deval
    parallel_phase5_sqa_deval
)

# Sanity-check that the source run dirs are referenced — they may not exist
# yet at submit time (the D_eval jobs are still running), but we can warn.
for run in "${RUNS[@]}"; do
    rdir="eval_results/${run}"
    if [ ! -d "$rdir" ]; then
        echo "[note] ${rdir}/ does not exist yet — will be created by predecessor JID ${DEP_JID}."
    elif [ ! -f "${rdir}/results.json" ]; then
        echo "[note] ${rdir}/results.json missing — predecessor will write it."
    fi
done

SBATCH_ARGS=(--parsable --job-name=judge_phase5)
if [ "${USE_DEP}" = "true" ]; then
    SBATCH_ARGS+=(--dependency=afterany:"${DEP_JID}")
    echo "Submitting Phase 5 judge with --dependency=afterany:${DEP_JID}"
else
    echo "Submitting Phase 5 judge with NO dependency"
fi

JID=$(sbatch "${SBATCH_ARGS[@]}" slurm/score_with_judge.sh "${RUNS[@]}")
echo "Submitted judge_phase5 → job ${JID}"
echo ""
echo "Reads:   eval_results/<run>/results.json"
echo "Writes:  eval_results/<run>/results.json  (with judge_score / judge_correct fields)"
echo "          eval_results/<run>/summary.json (with judge_* aggregates)"
echo ""
echo "Monitor: squeue -j ${JID}; tail -f logs/judge_phase5_${JID}.out"
