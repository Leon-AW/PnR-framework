#!/bin/bash
#SBATCH --job-name=eval_qm_deval
#SBATCH --partition=shared
#SBATCH --account=aitf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1g.24gb:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/eval_qm_deval_%j.out
#SBATCH --error=logs/eval_qm_deval_%j.err

# ==============================================================================
# AIT QM D_eval — ESR (qm_conflict) + Forgetting Rate (qm_control)
#
# Evaluates patch_qm_current (monolithic, bypasses routing) on:
#   qm_conflict  — semi-synthetic QM conflict pairs (measures R1 / ESR)
#   qm_control   — TriviaQA D_control records       (measures R2 / forgetting)
#
# QM answers are long free-form documents, so the runner auto-applies a
# long-form generation config to `qm_conflict` only (512 tokens, no
# sentence-boundary stop sequences, no short-answer truncation). `qm_control`
# is TriviaQA D_control and keeps the short `--max_new_tokens` config below so
# the forgetting-rate probe stays byte-identical to cf_control conditions.
# `--compute_logprob` adds the parsing-free, length-normalised TF-ESR
# (logP(answer_new) > logP(answer_old)) alongside the generation ESR.
#
# Run inside existing allocation (recommended — reuse job 10427):
#   srun --jobid=10427 --overlap --job-name=eval_qm --time=03:00:00 bash -c '
#     export CUDA_VISIBLE_DEVICES=MIG-24d12bbf-b110-51d8-92d4-6c94334de42b
#     bash slurm/eval_qm_deval.sh' > logs/eval_qm_deval_srun.log 2>&1 &
#
# Or submit as a new SLURM job:
#   sbatch slurm/eval_qm_deval.sh
# ==============================================================================

set -euo pipefail

REPO_ROOT="/gpfs/adafs/home/leon.wagner/PnR-framework"
PYTHON="/gpfs/adafs/home/leon.wagner/miniconda3/envs/pnr/bin/python"

export HF_HUB_OFFLINE=1

cd "$REPO_ROOT"

echo "Node: $(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

"$PYTHON" eval_pnr.py \
    --eval_sets qm_conflict qm_control \
    --n_samples 500 \
    --qm_conflict_path data/qm_conflict_pairs.json \
    --triviaqa_dcontrol_path data/triviaqa_dcontrol.json \
    --qm_adapter_name patch_qm_current \
    --monolithic checkpoints/patch_qm_current \
    --quantization int4 \
    --max_new_tokens 256 \
    --compute_logprob \
    --experiment_name pnr-qm-deval \
    --run_name pnr_qm_deval_v2 \
    --output_dir eval_results/qm_deval_v2 \
    "$@"
