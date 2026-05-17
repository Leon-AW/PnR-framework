#!/bin/bash
#SBATCH --job-name=sanity_gate_qm
#SBATCH --partition=longgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a10080gb:1
#SBATCH --nodelist=gruenau10
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ==============================================================================
# Sanity gate: verify base_qm emits old_value, patch_qm_current emits new_value
# on 5 qm_conflict samples. Both run monolithic (bypass routing).
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
echo "======================================================================"

echo ""
echo "=== RUN 1: base_qm (should emit old_value) ==="
python eval_pnr.py \
    --eval_sets qm_conflict \
    --n_samples 5 \
    --qm_conflict_path data/qm_conflict_pairs.json \
    --monolithic checkpoints/base_qm \
    --quantization int4 \
    --max_new_tokens 256 \
    --experiment_name pnr-sanity-gate \
    --run_name sanity_base_qm \
    --output_dir eval_results/sanity_gate/base_qm

echo ""
echo "=== RUN 2: patch_qm_current (should emit new_value) ==="
python eval_pnr.py \
    --eval_sets qm_conflict \
    --n_samples 5 \
    --qm_conflict_path data/qm_conflict_pairs.json \
    --monolithic checkpoints/patch_qm_current \
    --quantization int4 \
    --max_new_tokens 256 \
    --experiment_name pnr-sanity-gate \
    --run_name sanity_patch_qm_current \
    --output_dir eval_results/sanity_gate/patch_qm_current

echo ""
echo "=== RESULTS SUMMARY ==="
python - <<'EOF'
import json, pathlib

for name, path in [("base_qm", "eval_results/sanity_gate/base_qm"),
                   ("patch_qm_current", "eval_results/sanity_gate/patch_qm_current")]:
    report_files = list(pathlib.Path(path).glob("**/report.json"))
    if not report_files:
        print(f"[{name}] No report.json found in {path}")
        continue
    report = json.loads(report_files[0].read_text())
    raw_files = list(pathlib.Path(path).glob("**/raw_results.json"))
    print(f"\n--- {name} ---")
    if raw_files:
        raw = json.loads(raw_files[0].read_text())
        qm = [r for r in raw if r.get("split") == "qm_conflict"]
        for r in qm[:5]:
            meta = r.get("metadata", {})
            print(f"  Q: {r.get('question','')[:80]}")
            print(f"  new_value   : {meta.get('new_value','')}")
            print(f"  old_value   : {meta.get('old_value','')}")
            print(f"  prediction  : {r.get('raw_prediction','')[:120]}")
            print(f"  is_em       : {r.get('is_exact_match')}  old_present: {meta.get('old_value_present')}")
            print()
    summary = report.get("summary", {})
    print(f"  ESR={summary.get('esr','N/A')}  F1={summary.get('token_f1','N/A')}")
EOF

echo ""
echo "======================================================================"
echo "Finished : $(date)"
echo "======================================================================"
