#!/usr/bin/env python3
"""
Update-cost benchmark — PnR incremental patch vs. monolithic retrain.
=====================================================================

Answers the exposé R2 efficiency claim that the framework is left *unmeasured*
in `docs/results_analysis.md`: the **cost of an update vs. monolithic
retraining**. This is a TRAINING-cost benchmark, distinct from the inference
latency/VRAM table in `summarize_results.py`.

Why the metric is "cost per update vs. corpus size", not a single number
------------------------------------------------------------------------
Both training paths use the same LoRA config and a `max_steps` budget, so
running them for an identical step count would make wall-clock ~equal and hide
the real effect. The PnR advantage is structural:

  * PnR  : to add increment k, train ONE patch on increment k's data only.
           Base + prior patches are untouched → per-update cost is FLAT,
           independent of how much has already been learned.
  * Mono : to add increment k without forgetting, retrain the single adapter
           on the CUMULATIVE corpus (old ∪ new). To cover a corpus that keeps
           growing, step/pass count scales with it → per-update cost grows
           LINEARLY with the number of accumulated updates.

So the benchmark measures wall-clock / peak-VRAM / params at **matched
per-example exposure** (default: 1 epoch each), then projects the cumulative
cost over K successive updates. The headline is the *slope*, not one cell.

CounterFact reference geometry (defaults below):
  full corpus  = data/counterfact_train.jsonl        (19,728 records)
  one increment= data/counterfact_relfam_5.jsonl     (~3,288 records, 1 of 6)
  → adding the 6th family: monolithic re-processes 6× the data of one patch,
    and must do so on every one of the 6 updates.

Usage
-----
  # See the scaling argument with NO GPU (volumes + analytical projection):
  python scripts/benchmark_update_cost.py --dry-run

  # Measure both on GPU (writes JSON + appends a markdown table):
  python scripts/benchmark_update_cost.py \
      --out /vol/tmp/wagnerql/update_cost_bench

  # via SLURM:  sbatch slurm/benchmark_update_cost.sh
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_OUT = Path("/vol/tmp/wagnerql/update_cost_bench")


# ── data geometry ───────────────────────────────────────────────────────────

def _count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open() as f:
        return sum(1 for _ in f)


def gather_geometry(full_path: Path, increment_path: Path) -> dict:
    n_full = _count_jsonl(full_path)
    n_inc = _count_jsonl(increment_path)
    n_updates = max(1, round(n_full / n_inc)) if n_inc else 0
    return {
        "full_path": str(full_path),
        "increment_path": str(increment_path),
        "n_full_records": n_full,
        "n_increment_records": n_inc,
        "n_updates_to_fill_corpus": n_updates,
    }


def coverage_steps(n_records: int, eff_batch: int, epochs: float) -> int:
    return max(1, math.ceil(epochs * n_records / eff_batch))


def project_cumulative(geom: dict, eff_batch: int, epochs: float) -> dict:
    """Analytical cumulative training-step cost over K successive updates.

    PnR   : each update trains one increment → K × steps(increment).
    Mono  : update k retrains on k increments → Σ_{k=1..K} steps(k·increment).
    """
    n_inc = geom["n_increment_records"]
    K = geom["n_updates_to_fill_corpus"]
    if not n_inc or not K:
        return {}
    inc_steps = coverage_steps(n_inc, eff_batch, epochs)
    pnr_total = K * inc_steps
    mono_total = sum(coverage_steps(k * n_inc, eff_batch, epochs) for k in range(1, K + 1))
    return {
        "K_updates": K,
        "steps_per_increment": inc_steps,
        "pnr_cumulative_steps": pnr_total,
        "mono_cumulative_steps": mono_total,
        "mono_over_pnr_ratio": round(mono_total / pnr_total, 2) if pnr_total else None,
    }


# ── measured training run ───────────────────────────────────────────────────

def _load_iterable(data_path: str, seed: int):
    from datasets import load_dataset
    ds = load_dataset("json", data_files=data_path, split="train")
    n = len(ds)
    return ds.to_iterable_dataset().shuffle(seed=seed, buffer_size=10_000), n


def measure_run(
    label: str,
    data_path: str,
    *,
    epochs: float,
    eff_batch: int,
    grad_accum: int,
    lora_r: int,
    lora_alpha: int,
    max_seq_length: int,
    learning_rate: float,
    quantization: str,
    model_id: str,
    out_dir: Path,
    seed: int,
) -> dict:
    """Train one adapter end-to-end and record wall-clock, peak VRAM, params."""
    import torch
    from src.models.core import (
        PatchAndRouteLLM, FrozenFoundationConfig, ExpertConfig, QuantizationType,
    )
    from src.training.trainer import train_adapter

    quant_map = {"none": QuantizationType.NONE,
                 "int8": QuantizationType.INT8,
                 "int4": QuantizationType.INT4}

    dataset, n_records = _load_iterable(data_path, seed)
    batch_size = max(1, eff_batch // grad_accum)
    max_steps = coverage_steps(n_records, eff_batch, epochs)
    adapter_name = f"bench_{label}_{datetime.now():%H%M%S}"

    llm = PatchAndRouteLLM(
        foundation_config=FrozenFoundationConfig(
            model_id=model_id, quantization=quant_map[quantization],
        )
    )
    llm.load_frozen_foundation()
    llm.attach_expert(ExpertConfig(name=adapter_name, r=lora_r, lora_alpha=lora_alpha))
    model, tokenizer = llm.get_training_components()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    train_adapter(
        model=model, tokenizer=tokenizer, dataset=dataset,
        adapter_name=adapter_name, output_dir=str(out_dir / adapter_name),
        max_steps=max_steps, learning_rate=learning_rate,
        batch_size=batch_size, gradient_accumulation_steps=grad_accum,
        save_steps=max_steps, logging_steps=max(1, max_steps // 10),
        max_seq_length=max_seq_length, seed=seed, optim="paged_adamw_8bit",
    )
    wall = time.perf_counter() - t0
    peak_vram = (torch.cuda.max_memory_allocated() / 1e6
                 if torch.cuda.is_available() else None)

    return {
        "label": label,
        "data_path": data_path,
        "n_records": n_records,
        "epochs": epochs,
        "max_steps": max_steps,
        "eff_batch": eff_batch,
        "wall_clock_s": round(wall, 1),
        "s_per_step": round(wall / max_steps, 3),
        "s_per_1k_records": round(wall / (n_records / 1000), 1),
        "peak_vram_mb": round(peak_vram, 1) if peak_vram else None,
        "trainable_params": trainable,
        "trainable_pct": round(100 * trainable / total, 4) if total else None,
    }


# ── reporting ─────────────────────────────────────────────────────────────────

def _fmt(v, suffix=""):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:,.1f}{suffix}"
    return f"{v:,}{suffix}"


def render_markdown(geom: dict, proj: dict, runs: list[dict]) -> str:
    L = []
    L.append("## Update cost — PnR incremental patch vs. monolithic retrain\n")
    L.append(f"- Full corpus: `{geom['full_path']}` — "
             f"{geom['n_full_records']:,} records")
    L.append(f"- One increment: `{geom['increment_path']}` — "
             f"{geom['n_increment_records']:,} records "
             f"(≈ {geom['n_updates_to_fill_corpus']} increments fill the corpus)\n")

    if runs:
        L.append("| Update operation | Records | Steps | Wall-clock | s/1k rec | "
                 "Peak VRAM (MB) | Trainable params |")
        L.append("| :--------------- | ------: | ----: | ---------: | -------: | "
                 "-------------: | ---------------: |")
        for r in runs:
            L.append(f"| {r['label']} | {_fmt(r['n_records'])} | {_fmt(r['max_steps'])} "
                     f"| {_fmt(r['wall_clock_s'],'s')} | {_fmt(r['s_per_1k_records'])} "
                     f"| {_fmt(r['peak_vram_mb'])} | {_fmt(r['trainable_params'])} |")
        L.append("")
        L.append("_Matched per-example exposure (same epochs over each set). "
                 "Trainable params are identical by construction (same LoRA rank) — "
                 "the differentiator is records/steps processed per update, not "
                 "model footprint._\n")

    if proj:
        L.append("### Cumulative cost over successive updates (analytical)\n")
        L.append(f"Adding all {proj['K_updates']} increments, "
                 f"{proj['steps_per_increment']:,} steps per increment:\n")
        L.append("| Strategy | Total training steps | vs. PnR |")
        L.append("| :------- | -------------------: | ------: |")
        L.append(f"| PnR (one patch per update) | {proj['pnr_cumulative_steps']:,} | 1.00× |")
        L.append(f"| Monolithic (retrain cumulative) | {proj['mono_cumulative_steps']:,} "
                 f"| {proj['mono_over_pnr_ratio']}× |")
        L.append("")
        L.append("_PnR per-update cost is flat (O(increment)); monolithic grows "
                 "linearly with accumulated knowledge (Σ retrains over a growing "
                 "corpus). The ratio widens as more updates arrive — this is the "
                 "structural R2 efficiency claim, now quantified._")
    return "\n".join(L)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--full_data", default="data/counterfact_train.jsonl")
    p.add_argument("--increment_data", default="data/counterfact_relfam_5.jsonl")
    p.add_argument("--epochs", type=float, default=1.0,
                   help="Per-example exposure for matched comparison")
    p.add_argument("--eff_batch", type=int, default=16)
    p.add_argument("--grad_accum", type=int, default=16)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--max_seq_length", type=int, default=256)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--quantization", default="int4", choices=["none", "int8", "int4"])
    p.add_argument("--model_id", default="mistralai/Mistral-7B-Instruct-v0.3")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true",
                   help="No GPU: print data volumes + analytical projection only")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    full = REPO_ROOT / args.full_data if not Path(args.full_data).is_absolute() else Path(args.full_data)
    inc = REPO_ROOT / args.increment_data if not Path(args.increment_data).is_absolute() else Path(args.increment_data)

    geom = gather_geometry(full, inc)
    proj = project_cumulative(geom, args.eff_batch, args.epochs)

    runs: list[dict] = []
    if not args.dry_run:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        runs.append(measure_run(
            "PnR patch (1 increment)", str(inc),
            epochs=args.epochs, eff_batch=args.eff_batch, grad_accum=args.grad_accum,
            lora_r=args.lora_r, lora_alpha=args.lora_alpha,
            max_seq_length=args.max_seq_length, learning_rate=args.learning_rate,
            quantization=args.quantization, model_id=args.model_id,
            out_dir=out_dir, seed=args.seed,
        ))
        runs.append(measure_run(
            "Monolithic (full corpus)", str(full),
            epochs=args.epochs, eff_batch=args.eff_batch, grad_accum=args.grad_accum,
            lora_r=args.lora_r, lora_alpha=args.lora_alpha,
            max_seq_length=args.max_seq_length, learning_rate=args.learning_rate,
            quantization=args.quantization, model_id=args.model_id,
            out_dir=out_dir, seed=args.seed,
        ))
        payload = {"timestamp": datetime.now().isoformat(),
                   "geometry": geom, "projection": proj, "runs": runs}
        (out_dir / "update_cost.json").write_text(json.dumps(payload, indent=2))
        print(f"JSON written to {out_dir / 'update_cost.json'}", file=sys.stderr)

    print(render_markdown(geom, proj, runs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
