#!/bin/bash
#SBATCH --job-name=eval_qm_deval
#SBATCH --partition=longgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a10080gb:1
#SBATCH --nodelist=gruenau9
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL

# ==============================================================================
# AIT QM D_eval — Retention (qm_stable) + ESR (qm_conflict) + Forgetting Rate (qm_control)
#
# Runs on gruenau10 (3x A100-80GB), like the CounterFact / SituatedQA D_evals.
# The AIT QM dataset is semi-synthetic and carries no sensitive data once built,
# so QM evaluation runs on the gruenau cluster — not the AIT server. (Only
# scripts/build_qm_conflict_pairs.py, which reads the proprietary data/DE source
# corpus, is AIT-bound; the generated conflict pairs are not.)
#
# Evaluates a system (mode set by the caller — see Usage) on the 3-bucket
# SQA-style D_eval design (May 19 redesign):
#   qm_stable    — 500 QM facts unchanged 2015→2025      (expected adapter: base_qm)
#   qm_conflict  — 500 semi-synthetic QM conflict pairs  (expected adapter: patch_qm_current — measures R1 / ESR)
#   qm_control   — 1000 TriviaQA D_control records       (measures R2 / forgetting)
#
# QM answers are long free-form documents, so the runner auto-applies a
# long-form generation config to `qm_conflict` only (512 tokens, no
# sentence-boundary stop sequences, no short-answer truncation). `qm_control`
# is TriviaQA D_control and keeps the short `--max_new_tokens` config below so
# the forgetting-rate probe stays byte-identical to cf_control conditions.
# `--compute_logprob` adds the parsing-free, length-normalised TF-ESR
# (logP(answer_new) > logP(answer_old)) alongside the generation ESR; the
# report also carries qm_strict_esr (new_value present AND old_value absent).
#
# --n_samples 1000 is a per-split cap: qm_conflict has exactly 500 records so
# all 500 are used; qm_control draws the full 1000 TriviaQA D_control set,
# matching the CF/SQA forgetting-rate tables.
#
# This script hardcodes only the QM-common args; the system mode (monolithic /
# routing / no_adapter / ...) and the run_name + output_dir are passed via "$@",
# matching slurm/eval_deval.sh. A bare invocation runs PnR routing (no mode flag).
#
# Usage:
#   # Monolithic baseline (patch_qm_current, bypasses routing):
#   sbatch slurm/eval_qm_deval.sh --monolithic checkpoints/patch_qm_current \
#       --run_name pnr_qm_deval_v2 --output_dir eval_results/qm_deval_v2
#
#   # PnR two-adapter routing (base_qm + patch_qm_current, Time-Aware):
#   sbatch slurm/eval_qm_deval.sh --router_state checkpoints/router_state \
#       --similarity_threshold 0.45 \
#       --domain_classifier_path checkpoints/domain_classifier \
#       --domain_confidence_threshold 0.7 --domain_fallback_threshold 0.30 \
#       --run_name pnr_qm_routed_v3 --output_dir eval_results/qm_deval_pnr_v3
# ==============================================================================

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

CONDA_BASE=/usr/local/anaconda3-2024.06
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate pnr

export TQDM_MININTERVAL=10
export TQDM_NCOLS=100

echo "======================================================================"
echo "Job ID  : ${SLURM_JOB_ID}"
echo "Node    : ${SLURMD_NODENAME}"
echo "Started : $(date)"
echo "Args    : $*"
echo "======================================================================"

python eval_pnr.py \
    --eval_sets qm_stable qm_conflict qm_control \
    --n_samples 1000 \
    --qm_stable_path data/qm_stable_facts.json \
    --qm_conflict_path data/qm_conflict_pairs.json \
    --triviaqa_dcontrol_path data/triviaqa_dcontrol.json \
    --qm_base_adapter_name base_qm \
    --qm_adapter_name patch_qm_current \
    --quantization int4 \
    --max_new_tokens 256 \
    --compute_logprob \
    --experiment_name pnr-qm-deval \
    "$@"

echo "======================================================================"
echo "Finished : $(date)"
echo "======================================================================"
