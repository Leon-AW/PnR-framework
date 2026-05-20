#!/bin/bash
#SBATCH --job-name=train_qm_base
#SBATCH --partition=longgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a10080gb:1
#SBATCH --nodelist=gruenau10
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL

# ==============================================================================
# Train the AIT QM Base Adapter (base_qm) — v2 redesign (May 19 2026)
#
# base_qm is the "2015 corpus snapshot" adapter — SQA-style, it covers:
#   - 500 stable QM facts (unchanged 2015→2025)  [data/qm_stable_facts.json]
#   - 500 old-conflict answers (the superseded values) [data/qm_train_old.jsonl]
# Combined into data/qm_train_base.jsonl (1000 records, chat format).
#
# This gives base_qm a *distinct* question distribution from patch_qm_current
# (which only sees the 500 changed facts). The router centroids separate because
# base_qm's stable-fact questions are topically different from the conflict
# questions — enabling intra-domain routing (simulated: ~89.5% with multi-cluster
# centroids k=22/24). See docs/roadmap.md §5f QM D_eval redesign.
#
# Hyperparams: same r=16, alpha=32 as patch_qm_current; max_steps=1000 (was 500)
# to maintain ~4 epochs with the doubled training set (1000 records).
# knowledge_timestamp stays at 2015-01-01 (the "old" snapshot).
#
# Usage:
#   sbatch slurm/train_qm_base.sh
#   sbatch slurm/train_qm_base.sh --max_steps 1500   # extra args forwarded
# ==============================================================================

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

CONDA_BASE=/usr/local/anaconda3-2024.06
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate pnr

echo "======================================================================"
echo "Job ID  : ${SLURM_JOB_ID}"
echo "Node    : ${SLURMD_NODENAME}"
echo "Started : $(date)"
echo "Args    : $*"
echo "======================================================================"

# Build the base_qm training JSONL if absent (stable + old-conflict, 1000 recs).
if [ ! -f "data/qm_train_base.jsonl" ]; then
    echo "Building data/qm_train_base.jsonl ..."
    python -c "
import json

with open('data/qm_stable_facts.json') as f:
    stable = json.load(f)
recs = []
for r in stable:
    recs.append({'id': r['id'], 'messages': [{'role':'user','content':r['question']},{'role':'assistant','content':r['answer']}], 'language': r.get('language','de'), 'intention_category': r.get('intention_category','F'), 'split_origin': 'qm_stable'})
with open('data/qm_train_old.jsonl') as f:
    for line in f:
        rec = json.loads(line)
        rec['split_origin'] = 'qm_conflict_old'
        recs.append(rec)
with open('data/qm_train_base.jsonl','w') as f:
    for r in recs:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
print(f'  Written {len(recs)} records')
"
fi

# Storage convention: large checkpoints live on /vol/tmp, symlinked into the
# repo so router-state builds find a stable path.
OUTPUT_DIR=/vol/tmp/wagnerql/checkpoints/base_qm
mkdir -p "$(dirname "$OUTPUT_DIR")"

python train/train_qm_patch.py \
    --data_path data/qm_train_base.jsonl \
    --adapter_name base_qm \
    --adapter_type base_qm \
    --answer_field answer_old \
    --max_steps 1000 \
    --output_dir "$OUTPUT_DIR" \
    "$@"

ln -sfn "$OUTPUT_DIR" checkpoints/base_qm
echo "Symlinked checkpoints/base_qm -> $OUTPUT_DIR"

echo "======================================================================"
echo "Finished : $(date)"
echo "======================================================================"
