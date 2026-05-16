#!/bin/bash
#SBATCH --job-name=qm_conflict_pairs
#SBATCH --partition=shared
#SBATCH --account=aitf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:2g.48gb:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Build AIT QM conflict pairs with the Gemma-4 judge model.
# Runs on a 2g.48gb MIG slice (48 GB VRAM) on ada-gpu-[02-03].
# Gemma-4-26B-A4B int4 needs ~13 GB; 48 GB leaves ample headroom.
#
# Extra CLI args are forwarded to the Python script. Examples:
#   # full run (partition=shared uses default qos=backfill)
#   sbatch slurm/build_qm_conflict_pairs.sh --target 500
#   # smoke test (interactive partition requires --qos=interactive)
#   sbatch --partition=interactive --qos=interactive --time=01:00:00 \
#     --job-name=qm_smoke slurm/build_qm_conflict_pairs.sh \
#     --limit_candidates 40 --target 10 --output data/qm_conflict_pairs_smoke.json

set -euo pipefail

REPO_ROOT="/gpfs/adafs/home/leon.wagner/PnR-framework"
PYTHON="/gpfs/adafs/home/leon.wagner/miniconda3/envs/pnr/bin/python"

# Gemma-4 weights are pre-downloaded to the shared GPFS HF cache; the ada-gpu
# compute nodes may be air-gapped, so resolve from cache only.
export HF_HUB_OFFLINE=1

cd "$REPO_ROOT"

echo "Node: $(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

"$PYTHON" scripts/build_qm_conflict_pairs.py "$@"
