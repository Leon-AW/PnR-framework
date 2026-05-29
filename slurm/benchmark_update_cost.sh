#!/bin/bash
#SBATCH --job-name=update_cost_bench
#SBATCH --partition=longgpu
#SBATCH --gres=gpu:a10080gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL

# ==============================================================================
# Update-cost benchmark — PnR incremental patch vs. monolithic retrain.
#
# Measures the exposé R2 efficiency claim left unmeasured in results_analysis.md:
# the wall-clock + peak VRAM of one update operation, at matched per-example
# exposure, plus the analytical cumulative-cost projection over K updates.
#
# Runs both training jobs on ONE GPU back-to-back so latency/VRAM are directly
# comparable (same hardware, same process). Artefacts default to /vol/tmp.
#
# Usage:
#   cd /path/to/PnR-framework
#   sbatch slurm/benchmark_update_cost.sh
#
#   Override (e.g. 2 epochs of exposure):
#   sbatch slurm/benchmark_update_cost.sh --epochs 2.0
#
# Monitor:
#   squeue --me ; tail -f logs/update_cost_bench_<JOBID>.out
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

python - <<'EOF'
import torch
print(f"CUDA available : {torch.cuda.is_available()}")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {p.name} — {p.total_memory/1024**3:.1f} GB")
EOF

# Defaults mirror the real CounterFact patch / monolithic training configs
# (LoRA r=16, eff. batch 16, int4, seq-len 256). Extra CLI args pass through.
python scripts/benchmark_update_cost.py \
    --full_data data/counterfact_train.jsonl \
    --increment_data data/counterfact_relfam_5.jsonl \
    --epochs 1.0 \
    --out /vol/tmp/wagnerql/update_cost_bench \
    "$@"

echo "Finished     : $(date)"
