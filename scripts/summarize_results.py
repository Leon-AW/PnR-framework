#!/usr/bin/env python3
"""
Unified results table across all evaluation datasets.

One row per method; columns grouped by dataset. Adding a new dataset
means adding an entry to DATASETS and a run key to each METHOD entry.

Usage:
    python scripts/summarize_results.py
    python scripts/summarize_results.py --format csv --out eval_results/summary.csv
    python scripts/summarize_results.py --format both
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "eval_results"

# ── method registry ───────────────────────────────────────────────────────────
# Each method maps dataset keys → run directory under eval_results/.
# Use None when a run doesn't exist yet (shows — in the table).
# MORPHEUS ablation rows share the same situated_qa run (architecture is
# identical for SQA; only CF routing changes).

METHODS: list[dict[str, str | None]] = [
    {
        "name":         "Frozen Base",
        # standardised SQA D_eval (1 000 train + D_control, May 2)
        "situated_qa":  "frozen_base_sqa_deval",
        "counterfact":  "frozen_base_deval_v2",
        # v3 (May 20, job 362311) — 3-bucket QM D_eval (stable/conflict/control)
        "ait_qm":       "qm_deval_frozen_v3/pnr_qm_frozen_v3",
    },
    {
        "name":         "X-LoRA",
        # merged after job 352297 (cf_control) completed May 3
        "situated_qa":  "xlora_sqa_deval",
        "counterfact":  "xlora_v3",
        # v3 QM rerun (May 23, job 362979)
        "ait_qm":       "qm_deval_xlora_v3/pnr_qm_xlora_v3",
    },
    {
        "name":         "Parallel (single-stage)",
        "situated_qa":  "parallel_sqa_deval",
        # logprob re-run (job 352338, May 4): ESR=15.7%, TF-ESR=52.8%, FR=8.5%
        "counterfact":  "parallel_deval_logprob",
    },
    {
        # Phase 5 (May 12): 6 patch_cf_relfam_* clusters + domain classifier Stage-1 gate
        "name":         "Parallel (multi-expert + 2-stage)",
        "situated_qa":  "parallel_phase5_sqa_deval",
        "counterfact":  "parallel_phase5_cf_deval",
        # v3 QM (May 21, job 362314) — short-form synthesis + DEFAULT_SHORT_ANSWER_BOUNDARIES
        # stops on per-adapter generation. Kept as the as-published baseline for the
        # comparison with the long-form variant below.
        "ait_qm":       "qm_deval_parallel_v3/pnr_qm_parallel_v3",
    },
    {
        # May 28 rerun (job 364767) — same Phase-5 architecture, but with the long-form
        # synthesis path: SYNTHESIS_PROMPT_TEMPLATE_LONG_FORM, synthesis budget 1536
        # tokens, and stop_sequences=() on both per-adapter generation and the Resolver
        # pass so multi-paragraph QM answers are not truncated at the first newline.
        # Mirrors the long-form handling the runner already applies to PnR / RECIPE.
        "name":         "Parallel (multi-expert + 2-stage, long-form synthesis)",
        "situated_qa":  "parallel_phase5_sqa_deval",
        "counterfact":  "parallel_phase5_cf_deval",
        "ait_qm":       "qm_deval_parallel_longform_v3/pnr_qm_parallel_longform_v3",
    },
    {
        "name":         "RECIPE",
        "situated_qa":  "recipe_sqa_deval",
        "counterfact":  "recipe_deval_v2",
        # v3 QM rerun (May 25, job 363718) — re-cloned external/RECIPE,
        # retrained QM ckpt epoch-1000-i-63000-ema_loss-1.3794, evaluated
        # via QM-aware long-form path in src/baselines/recipe_official.py.
        "ait_qm":       "qm_deval_recipe_v3/recipe_qm_deval_v3",
    },
    {
        "name":         "Monolithic LoRA",
        "situated_qa":  "monolithic_sqa_deval",
        "counterfact":  "monolithic_deval_v2",
        # v3 QM monolithic (May 22, job 362967) = patch_qm_current only
        # over the 3-bucket 2000-sample design.
        "ait_qm":       "qm_deval_monolithic_v3/pnr_qm_monolithic_v3",
    },
    {
        "name":         "Monolithic LoRA (sequential QM)",
        "situated_qa":  None,
        "counterfact":  None,
        # Legacy May-17 sequential-training row (old→new on same adapter).
        # Kept as a reference; the v3 Monolithic row above is the canonical
        # 2000-sample number on the 3-bucket design.
        "ait_qm":       "qm_deval_monolithic/pnr_qm_monolithic",
    },
    {
        "name":         "LoRA + RAG",
        "situated_qa":  "lora_rag_sqa_deval",
        "counterfact":  "lora_rag_deval_v2",
        # v3 QM rerun (May 22, job 362968) — qm_train.jsonl chat-message
        # _build_index fix landed in src/baselines/lora_rag.py.
        "ait_qm":       "qm_deval_lora_rag_v3/pnr_qm_lora_rag_v3",
    },
    {
        "name":         "PnR Routing (single-stage)",
        "situated_qa":  "pnr_sqa_deval",
        # logprob re-run (job 352307, May 4): ESR=17.2%, TF-ESR=53.2%, FR=8.2%
        "counterfact":  "pnr_deval_logprob",
    },
    {
        # Phase 5 (May 12): 6 patch_cf_relfam_* clusters + domain classifier Stage-1 gate
        "name":         "PnR Routing (multi-expert + 2-stage)",
        "situated_qa":  "pnr_phase5_sqa_deval",
        "counterfact":  "pnr_phase5_cf_deval",
        # v3 QM (May 20, job 362310) — 3-bucket, 2000 samples, k=150 router state
        "ait_qm":       "qm_deval_pnr_v3/pnr_qm_routed_v3",
    },
    {
        "name":         "MORPHEUS (τ, bypass)",
        "situated_qa":  "morpheus_sqa_deval",
        "counterfact":  "morpheus_deval_v3",
        # v3 QM (May 23, job 363704) — morpheus_state_qm/ seeded from
        # qm_train.jsonl + qm_train_base.jsonl (1000 unique records).
        "ait_qm":       "qm_deval_morpheus_v3/pnr_qm_morpheus_v3",
    },
    {
        "name":         "MORPHEUS (τ, no-bypass)",
        "situated_qa":  "morpheus_sqa_deval",
        "counterfact":  "morpheus_nobypass_deval_v3",
        # v3 QM nobypass ablation (May 23, job 363705) — same KS, but
        # --morpheus_direct_answer_threshold 1.1 forces routing through
        # the QM specialist (no KS short-circuit).
        "ait_qm":       "qm_deval_morpheus_nobypass_v3/pnr_qm_morpheus_nobypass_v3",
    },
    {
        "name":         "MORPHEUS (clf, bypass)",
        "situated_qa":  "morpheus_sqa_deval",
        "counterfact":  "morpheus_clf_deval_v3",
    },
    {
        "name":         "MORPHEUS (clf, no-bypass)",
        "situated_qa":  "morpheus_sqa_deval",
        "counterfact":  "morpheus_clf_nobypass_deval_v3",
    },
]

# ── dataset definitions ───────────────────────────────────────────────────────
# Each dataset defines:
#   key      – matches the key in METHODS
#   label    – shown as group header
#   columns  – (header, field, align, min_width)
#   extract  – fn(report) → dict[field, float|None]

def _get(d: Any, *keys: str, default=None) -> Any:
    for k in keys:
        if isinstance(d, dict) and k in d:
            d = d[k]
        else:
            return default
    return d


def _extract_situated_qa(report: dict) -> dict[str, float | None]:
    s = report.get("summary", {})
    sp = report.get("by_split", {})
    # Standardised D_eval (sqa_train + cf_control): per-split entry is canonical;
    # summary.exact_match_overall conflates with cf_control's near-perfect EM and
    # summary.judge_accuracy_overall conflates with cf_control's ~98 % FR-probe.
    # Legacy SQA-only runs (no by_split) fall back to summary fields.
    if "sqa_train" in sp:
        return {
            "sqa_em":    _get(sp, "sqa_train", "exact_match"),
            "sqa_f1":    _get(sp, "sqa_train", "f1"),
            "sqa_judge": _get(sp, "sqa_train", "judge_accuracy"),
        }
    return {
        "sqa_em":    s.get("exact_match_overall", s.get("exact_match")),
        "sqa_f1":    s.get("f1_overall",          s.get("f1")),
        "sqa_judge": s.get("judge_accuracy_overall"),
    }


def _extract_counterfact(report: dict) -> dict[str, float | None]:
    s  = report.get("summary", {})
    sp = report.get("by_split", {})

    # by_split.cf_conflict is canonical for ESR/F1/judge: it isolates the
    # conflict split from the cf_control FR probe.  summary.esr is unreliable
    # for newer runs (Phase 5 records 0.0 in CF runs even when cf_conflict EM
    # is 30 %+), so prefer the by-split entry whenever present.  Fall back to
    # summary fields for legacy reports without cf_conflict.
    esr  = _get(sp, "cf_conflict", "exact_match", default=_get(s, "esr"))
    ctrl = _get(sp, "cf_control",  "exact_match")
    cf_fr = _get(s, "dcontrol_forgetting_rate")
    if cf_fr is None and ctrl is not None:
        cf_fr = 1.0 - ctrl

    return {
        "cf_esr":         esr,
        "cf_f1":          _get(sp, "cf_conflict", "f1"),
        "cf_logp_esr":    _get(sp, "cf_conflict", "logprob_esr",
                                default=_get(s, "logprob_esr")),
        "cf_fr":          cf_fr,
        "cf_judge":       _get(sp, "cf_conflict", "judge_accuracy"),
        "sys_judge_ctrl": _get(sp, "cf_control",  "judge_accuracy"),
    }


def _extract_ait_qm(report: dict) -> dict[str, float | None]:
    s  = report.get("summary", {})
    sp = report.get("by_split", {})
    # qm_conflict: primary ESR metrics; qm_control: forgetting rate
    esr      = _get(sp, "qm_conflict", "exact_match", default=_get(s, "esr"))
    f1       = _get(sp, "qm_conflict", "f1")
    logp_esr = _get(sp, "qm_conflict", "logprob_esr",
                    default=_get(s, "qm_logprob_esr"))
    strict   = _get(sp, "qm_conflict", "strict_esr",
                    default=_get(s, "qm_strict_esr"))
    judge    = _get(sp, "qm_conflict", "judge_accuracy")
    ctrl_acc = _get(sp, "qm_control",  "exact_match",
                    default=_get(s, "dcontrol_accuracy"))
    qm_fr    = _get(s, "dcontrol_forgetting_rate")
    if qm_fr is None and ctrl_acc is not None:
        qm_fr = 1.0 - ctrl_acc
    return {
        "qm_esr":      esr,
        "qm_f1":       f1,
        "qm_logp_esr": logp_esr,
        "qm_strict":   strict,
        "qm_judge":    judge,
        "qm_fr":       qm_fr,
    }


DATASETS: list[dict] = [
    {
        "key":     "situated_qa",
        "label":   "SituatedQA",
        "columns": [
            # (header,     field,      align, min_width)
            ("EM",         "sqa_em",   ">", 6),
            ("F1",         "sqa_f1",   ">", 6),
            ("Judge",      "sqa_judge",">", 6),
        ],
        "extract": _extract_situated_qa,
    },
    {
        "key":     "counterfact",
        "label":   "CounterFact",
        "columns": [
            ("ESR (EM)",   "cf_esr",      ">", 8),
            ("F1",         "cf_f1",       ">", 6),
            ("TF-ESR",     "cf_logp_esr", ">", 7),
            ("Judge CF",   "cf_judge",    ">", 9),
        ],
        "extract": _extract_counterfact,
    },
    {
        "key":     "ait_qm",
        "label":   "AIT QM",
        "columns": [
            ("ESR",        "qm_esr",      ">", 6),
            ("F1",         "qm_f1",       ">", 6),
            ("TF-ESR",     "qm_logp_esr", ">", 7),
            ("Strict ESR", "qm_strict",   ">", 10),
            ("Judge",      "qm_judge",    ">", 6),
        ],
        "extract": _extract_ait_qm,
    },
]

# System-level columns sourced from the counterfact run (FR is dataset-agnostic)
SYSTEM_COLUMNS: list[tuple] = [
    ("FR",           "cf_fr",          ">", 6),
    ("Judge D_ctrl", "sys_judge_ctrl", ">", 11),
]

# ── data loading ──────────────────────────────────────────────────────────────

def load_report(run_dir: str | None) -> dict | None:
    if run_dir is None:
        return None
    path = RESULTS_DIR / run_dir / "report.json"
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


# ── inference efficiency ────────────────────────────────────────────────────────
# Sourced from per-record results.json (report.json carries no efficiency block).
# Comparison is fixed to the CounterFact D_eval run so every method is timed on the
# same short-answer workload — this isolates the architectural per-query overhead
# (single-adapter hard route vs. N-adapter ensemble + synthesis vs. retrieval bypass)
# rather than conflating it with dataset answer length.
EFFICIENCY_SOURCE_KEY = "counterfact"


def _load_records(run_dir: str | None) -> list[dict] | None:
    if run_dir is None:
        return None
    path = RESULTS_DIR / run_dir / "results.json"
    if not path.exists():
        return None
    with path.open() as f:
        data = json.load(f)
    return data if isinstance(data, list) else data.get("results", data)


def _extract_efficiency(run_dir: str | None) -> dict[str, float | None]:
    """Per-query inference cost from a run's results.json.

    latency_ms is wall-clock per query (perf_counter); the first query carries a
    cold-start / adapter-load spike, so the *median* is reported as the robust
    typical-cost figure alongside p95. vram_mb is the per-query peak
    (torch.cuda.max_memory_allocated, reset each query), so the run peak is the
    max across records — the true VRAM footprint the method requires.
    """
    empty = {"eff_med_ms": None, "eff_p95_ms": None, "eff_vram": None, "eff_n": None}
    recs = _load_records(run_dir)
    if not recs:
        return empty
    lat = sorted(r["latency_ms"] for r in recs if r.get("latency_ms") is not None)
    vram = [r["vram_mb"] for r in recs if r.get("vram_mb") is not None]
    n = len(lat)
    if n == 0:
        return empty
    import statistics
    return {
        "eff_med_ms": statistics.median(lat),
        "eff_p95_ms": lat[min(int(n * 0.95), n - 1)],
        "eff_vram":   max(vram) if vram else None,
        "eff_n":      n,
    }


def build_rows() -> list[dict]:
    rows = []
    for method in METHODS:
        row: dict[str, Any] = {"name": method["name"]}
        for ds in DATASETS:
            report = load_report(method.get(ds["key"]))
            if report is None:
                for _, field, _, _ in ds["columns"]:
                    row[field] = None
            else:
                row.update(ds["extract"](report))
        # System columns come from the counterfact run
        cf_report = load_report(method.get("counterfact"))
        if cf_report is not None:
            cf_data = _extract_counterfact(cf_report)
            for _, field, _, _ in SYSTEM_COLUMNS:
                row.setdefault(field, cf_data.get(field))
        else:
            for _, field, _, _ in SYSTEM_COLUMNS:
                row.setdefault(field, None)
        rows.append(row)
    return rows


# ── formatting ────────────────────────────────────────────────────────────────

def pct(v: Any, decimals: int = 1) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v) * 100:.{decimals}f}%"
    except (TypeError, ValueError):
        return "—"


def _all_columns() -> list[tuple]:
    cols = [("Method", "name", "<", 38)]
    for ds in DATASETS:
        cols.extend(ds["columns"])
    cols.extend(SYSTEM_COLUMNS)
    return cols


def _group_headers() -> str:
    """Return a markdown comment-style group header line."""
    # Method column
    parts = ["Method" + " " * (38 - len("Method"))]
    for ds in DATASETS:
        total_w = sum(max(c[3], len(c[0])) + 3 for c in ds["columns"]) - 3
        label = ds["label"]
        parts.append(label.center(total_w))
    sys_w = sum(max(c[3], len(c[0])) + 3 for c in SYSTEM_COLUMNS) - 3
    parts.append("System".center(sys_w))
    return "| " + " | ".join(parts) + " |"


def print_markdown(rows: list[dict]) -> None:
    cols = _all_columns()
    widths = [max(c[3], len(c[0])) for c in cols]

    # Group header
    print(_group_headers())

    # Column headers
    header_cells = []
    for (hdr, _, align, _), w in zip(cols, widths):
        header_cells.append(hdr.rjust(w) if align == ">" else hdr.ljust(w))
    print("| " + " | ".join(header_cells) + " |")

    # Separator
    sep_cells = []
    for (_, _, align, _), w in zip(cols, widths):
        dash = "-" * w
        sep_cells.append(dash + ":" if align == ">" else ":" + dash)
    print("| " + " | ".join(sep_cells) + " |")

    # Data rows
    for row in rows:
        cells = []
        for (_, field, align, _), w in zip(cols, widths):
            if field == "name":
                v = str(row.get("name", "—"))
                cells.append(v.ljust(w))
            else:
                v = pct(row.get(field))
                cells.append(v.rjust(w) if align == ">" else v.ljust(w))
        print("| " + " | ".join(cells) + " |")

    print()
    ds_labels = " | ".join(f"**{ds['label']}**: " + ", ".join(c[0] for c in ds["columns"])
                            for ds in DATASETS)
    print(f"_{ds_labels}_")
    print("_**System**: FR = D_control forgetting rate (lower is better). "
          "Judge = Gemma-4 binary correctness verdict._")
    print("_TF-ESR = teacher-forcing log-prob ESR: P(target\\_new|prompt) > P(target\\_true|prompt). Standard metric in ROME, MEMIT, GRACE, RECIPE._")
    print("_'—' = run missing or not yet scored._")


def _fmt_num(v: Any, suffix: str = "") -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):,.0f}{suffix}"
    except (TypeError, ValueError):
        return "—"


def print_efficiency_markdown(methods: list[dict]) -> None:
    """Separate inference-efficiency table (CounterFact D_eval, per query).

    Kept distinct from the accuracy table on purpose: this answers a *different*
    axis (inference-time cost), and is NOT the exposé's "cost of updates vs
    monolithic retraining" claim — that one is an update/training-cost benchmark
    reported elsewhere.
    """
    print()
    print("### Inference efficiency — CounterFact D_eval, per query")
    print()
    cols = [("Method", "<", 38), ("Median ms/q", ">", 11),
            ("p95 ms/q", ">", 9), ("Peak VRAM (MB)", ">", 14), ("n", ">", 6)]
    widths = [w for _, _, w in cols]
    print("| " + " | ".join(
        (h.ljust(w) if a == "<" else h.rjust(w)) for (h, a, w) in cols) + " |")
    print("| " + " | ".join(
        (":" + "-" * w if a == "<" else "-" * w + ":") for (_, a, w) in cols) + " |")
    for m in methods:
        eff = _extract_efficiency(m.get(EFFICIENCY_SOURCE_KEY))
        cells = [
            str(m["name"]).ljust(widths[0]),
            _fmt_num(eff["eff_med_ms"]).rjust(widths[1]),
            _fmt_num(eff["eff_p95_ms"]).rjust(widths[2]),
            _fmt_num(eff["eff_vram"]).rjust(widths[3]),
            _fmt_num(eff["eff_n"]).rjust(widths[4]),
        ]
        print("| " + " | ".join(cells) + " |")
    print()
    print("_Per-query inference cost on the CounterFact D_eval run "
          "(conflict + control, short-answer). Median is robust to the "
          "first-query cold-start/adapter-load spike; peak VRAM is the max "
          "per-query allocation. **Caveat:** latency is comparable only insofar "
          "as runs shared GPU type/load; treat as relative, not absolute. "
          "MORPHEUS-bypass is fast precisely because it returns a stored value "
          "without running the LLM._")


def write_efficiency_csv(methods: list[dict], out_path: Path) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Method", "median_ms_per_query", "p95_ms_per_query",
                    "peak_vram_mb", "n"])
        for m in methods:
            eff = _extract_efficiency(m.get(EFFICIENCY_SOURCE_KEY))
            def _r(v):
                return "" if v is None else f"{float(v):.1f}"
            w.writerow([m["name"], _r(eff["eff_med_ms"]), _r(eff["eff_p95_ms"]),
                        _r(eff["eff_vram"]), eff["eff_n"] or ""])
    print(f"Efficiency CSV written to {out_path}", file=sys.stderr)


def write_csv(rows: list[dict], out_path: Path) -> None:
    cols = _all_columns()
    dataset_labels = (
        ["Method"]
        + [ds["label"] for ds in DATASETS for _ in ds["columns"]]
        + ["System"] * len(SYSTEM_COLUMNS)
    )
    col_headers = [c[0] for c in cols]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(dataset_labels)
        w.writerow(col_headers)
        for row in rows:
            cells = []
            for _, field, _, _ in cols:
                if field == "name":
                    cells.append(row.get("name", ""))
                else:
                    v = row.get(field)
                    if v is None:
                        cells.append("")
                    else:
                        try:
                            cells.append(f"{float(v) * 100:.1f}")
                        except (TypeError, ValueError):
                            cells.append(str(v))
            w.writerow(cells)
    print(f"CSV written to {out_path}", file=sys.stderr)


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--format", choices=["markdown", "csv", "both"], default="markdown")
    p.add_argument("--out", default=None,
                   help="CSV output path (default: eval_results/summary.csv)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rows = build_rows()

    if args.format in ("markdown", "both"):
        print_markdown(rows)
        print_efficiency_markdown(METHODS)

    if args.format in ("csv", "both"):
        out = Path(args.out) if args.out else RESULTS_DIR / "summary.csv"
        write_csv(rows, out)
        eff_out = out.with_name(out.stem + "_efficiency.csv")
        write_efficiency_csv(METHODS, eff_out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
