#!/bin/bash
# ==============================================================================
# Submit AIT QM Judge Scoring
#
# Post-hoc LLM-as-Judge (Gemma) scoring for the 9 v3 QM D_eval run dirs.
# `scripts/score_with_judge.py` auto-routes prompt family by split name:
#   qm_conflict / qm_stable / qm_control → factoid judge
#
# Sizing: 9 runs × 2000 records ≈ 18 000 judge calls.
# Phase-5 baseline rate was ~17 min / 1000 records on gruenau10, so plan
# ~5 h compute + model-load overhead. Budget set to 8 h.
#
# Usage:
#   bash slurm/submit_judge_qm.sh
# ==============================================================================

set -euo pipefail

cd "$(dirname "$0")/.."

RUNS=(
    qm_deval_frozen_v3/pnr_qm_frozen_v3
    qm_deval_xlora_v3/pnr_qm_xlora_v3
    qm_deval_parallel_v3/pnr_qm_parallel_v3
    qm_deval_monolithic_v3/pnr_qm_monolithic_v3
    qm_deval_lora_rag_v3/pnr_qm_lora_rag_v3
    qm_deval_recipe_v3/recipe_qm_deval_v3
    qm_deval_pnr_v3/pnr_qm_routed_v3
    qm_deval_morpheus_v3/pnr_qm_morpheus_v3
    qm_deval_morpheus_nobypass_v3/pnr_qm_morpheus_nobypass_v3
)

for run in "${RUNS[@]}"; do
    rdir="eval_results/${run}"
    if [ ! -f "${rdir}/results.json" ]; then
        echo "[FATAL] ${rdir}/results.json missing — refusing to submit."
        exit 1
    fi
done

JID=$(sbatch --parsable \
    --job-name=judge_qm \
    --time=08:00:00 \
    slurm/score_with_judge.sh "${RUNS[@]}")
echo "Submitted judge_qm → job ${JID}"
echo ""
echo "Reads:   eval_results/<run>/results.json"
echo "Writes:  eval_results/<run>/results.json  (judge_score / judge_raw per record)"
echo "          eval_results/<run>/report.json  (judge_accuracy_overall + by_split.<split>.judge_accuracy)"
echo ""
echo "Monitor: squeue -j ${JID}; tail -f logs/judge_qm_${JID}.out"
