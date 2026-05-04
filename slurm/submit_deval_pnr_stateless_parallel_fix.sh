#!/bin/bash
# ==============================================================================
# Submit two D_eval jobs (thesis CF + D_control) after roadmap May 2026 code fix:
#
#   1) pnr_deval_stateless — default warm_context=False (honest per-query detach)
#   2) parallel_deval_may2026 — Parallel Orchestrator wiring fix (similarity planner,
#      _score_adapters / per-adapter τ, Source-Replay, Resolver changes)
#
# Serialised via SLURM --dependency so only one occupies gruenau10 at a time.
#
# Usage (from repo root):
#   bash slurm/submit_deval_pnr_stateless_parallel_fix.sh
# ==============================================================================

set -euo pipefail

cd "$(dirname "$0")/.."
SCRIPT="${PWD}/slurm/eval_deval.sh"
ROUTER_STATE="${PWD}/checkpoints/router_state"
mkdir -p logs

echo "Submitting: pnr_deval_stateless → parallel_deval_may2026"
echo ""

JID1=$(
  sbatch --parsable --job-name=deval_pnr_stateless "${SCRIPT}" \
    --router_state "${ROUTER_STATE}" \
    --similarity_threshold 0.45 \
    --run_name pnr_deval_stateless
)
echo "[1/2] PnR (stateless)     → job ${JID1}  → eval_results/pnr_deval_stateless/"

JID2=$(
  sbatch --parsable \
    --dependency=afterany:"${JID1}" \
    --job-name=deval_parallel_may "${SCRIPT}" \
    --parallel \
    --router_state "${ROUTER_STATE}" \
    --similarity_threshold 0.45 \
    --run_name parallel_deval_may2026
)
echo "[2/2] Parallel (fix)       → job ${JID2}  → eval_results/parallel_deval_may2026/"

echo ""
echo "Monitor: squeue -u \${USER} | grep deval"
echo "Logs:    tail -f logs/eval_deval_<jobid>.out"
