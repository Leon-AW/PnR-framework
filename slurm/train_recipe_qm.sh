#!/bin/bash
#SBATCH --job-name=recipe_qm
#SBATCH --partition=longgpu
#SBATCH --gres=gpu:a10080gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --exclude=gruenau9

# ==============================================================================
# RECIPE Baseline — QM Meta-Training (official repo, gruenau)
#
# Runs the EMNLP-2024 author implementation on Mistral-7B-Instruct-v0.3
# meta-trained on the AIT QM conflict pairs. Mirrors slurm/train_recipe_official.sh
# (the SituatedQA/zSRE counterpart) — same env, same node convention.
#
# Prereqs (already in place):
#   - external/RECIPE -> /vol/tmp/wagnerql/RECIPE (symlink)
#   - external/RECIPE/configs/recipe/mistral-7b.yaml
#   - external/RECIPE/utils/utils.py mistral dispatch
#   - editors/recipe/data.py 'qm' branch (dispatches to __zsre__)
#   - external/RECIPE/data/meta-train/qm/qm_train.json (build via
#     `python scripts/build_recipe_qm_data.py` if absent)
#
# Notes:
#   - Mistral is loaded at bf16; int4/int8 not supported (prompt injection
#     hook requires gradients through inputs_embeds).
#   - Checkpoints land under
#     /vol/tmp/wagnerql/RECIPE/train_records/recipe/mistral-7b/<timestamp>/checkpoints/.
# ==============================================================================

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

CONDA_BASE=/usr/local/anaconda3-2024.06
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate pnr

export TQDM_MININTERVAL=10
export TQDM_NCOLS=100

echo "======================================================================"
echo "Job ID       : ${SLURM_JOB_ID}"
echo "Node         : ${SLURMD_NODENAME}"
echo "Started      : $(date)"
echo "======================================================================"

# Regenerate QM training data if missing
if [ ! -f external/RECIPE/data/meta-train/qm/qm_train.json ]; then
    python scripts/build_recipe_qm_data.py
fi

# Official repo uses relative paths — run from its root
cd external/RECIPE
python train_recipe.py -mn mistral-7b -dn qm "$@"

echo "======================================================================"
echo "Finished : $(date)"
echo "======================================================================"
