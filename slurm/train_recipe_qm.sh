#!/bin/bash
#SBATCH --job-name=train_recipe_qm
#SBATCH --partition=shared
#SBATCH --account=aitf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:2g.48gb:1
#SBATCH --time=28:00:00
#SBATCH --output=logs/train_recipe_qm_%j.out
#SBATCH --error=logs/train_recipe_qm_%j.err

# ==============================================================================
# Meta-train RECIPE on AIT QM conflict pairs.
#
# Trains the Knowledge Representation Module (KRM) and Prompt Tokens (PT)
# so RECIPE can store/retrieve QM-style factual edits at inference time.
#
# Prerequisite (CPU, <1 s):
#   python scripts/build_recipe_qm_data.py
#
# Run inside existing allocation (recommended — reuse job 10427):
#   srun --jobid=10427 --overlap --job-name=recipe_qm --time=27:00:00 bash -c '
#     export CUDA_VISIBLE_DEVICES=MIG-67a4fc8d-4980-5ba3-8261-8dcb9d94d1d9
#     bash slurm/train_recipe_qm.sh' > logs/train_recipe_qm_srun.log 2>&1 &
#
# Or submit as a new SLURM job:
#   sbatch slurm/train_recipe_qm.sh
# ==============================================================================

set -euo pipefail

REPO_ROOT="/gpfs/adafs/home/leon.wagner/PnR-framework"
PYTHON="/gpfs/adafs/home/leon.wagner/miniconda3/envs/pnr/bin/python"
RECIPE_ROOT="$REPO_ROOT/external/RECIPE"

export HF_HUB_OFFLINE=1

cd "$RECIPE_ROOT"

echo "Node: $(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Build training data if not present
if [ ! -f "$RECIPE_ROOT/data/meta-train/qm/qm_train.json" ]; then
    echo "Building RECIPE QM training data..."
    cd "$REPO_ROOT"
    "$PYTHON" scripts/build_recipe_qm_data.py
    cd "$RECIPE_ROOT"
fi

"$PYTHON" train_recipe.py \
    --model_name mistral-7b \
    --data_name qm \
    --batch_size 2
