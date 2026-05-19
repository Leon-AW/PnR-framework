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
        "ait_qm":       "qm_deval_frozen/pnr_qm_frozen",
    },
    {
        "name":         "X-LoRA",
        # merged after job 352297 (cf_control) completed May 3
        "situated_qa":  "xlora_sqa_deval",
        "counterfact":  "xlora_v3",
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
    },
    {
        "name":         "RECIPE",
        "situated_qa":  "recipe_sqa_deval",
        "counterfact":  "recipe_deval_v2",
    },
    {
        "name":         "Monolithic LoRA",
        "situated_qa":  "monolithic_sqa_deval",
        "counterfact":  "monolithic_deval_v2",
        # QM monolithic = patch_qm_current only (new facts, no base_qm)
        "ait_qm":       "qm_deval_v2/pnr_qm_deval_v2",
    },
    {
        "name":         "Monolithic LoRA (sequential QM)",
        "situated_qa":  None,
        "counterfact":  None,
        # QM sequential monolithic = old→new, demonstrates catastrophic forgetting
        "ait_qm":       "qm_deval_monolithic/pnr_qm_monolithic",
    },
    {
        "name":         "LoRA + RAG",
        "situated_qa":  "lora_rag_sqa_deval",
        "counterfact":  "lora_rag_deval_v2",
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
        "ait_qm":       "qm_deval_pnr/pnr_qm_routed",
    },
    {
        "name":         "MORPHEUS (τ, bypass)",
        "situated_qa":  "morpheus_sqa_deval",
        "counterfact":  "morpheus_deval_v3",
    },
    {
        "name":         "MORPHEUS (τ, no-bypass)",
        "situated_qa":  "morpheus_sqa_deval",
        "counterfact":  "morpheus_nobypass_deval_v3",
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

    if args.format in ("csv", "both"):
        out = Path(args.out) if args.out else RESULTS_DIR / "summary.csv"
        write_csv(rows, out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
