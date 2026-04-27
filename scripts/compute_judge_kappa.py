#!/usr/bin/env python3
"""Compute Cohen's kappa for human vs Gemma judge labels."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.external_judge import JUDGE_MODEL_ID, JUDGE_PROMPT_VERSION


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute judge-human agreement from calibration CSV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        default="eval_results/_calibration/calibration_to_annotate.csv",
    )
    parser.add_argument(
        "--output",
        default="eval_results/_calibration/calibration_report.json",
    )
    return parser.parse_args()


def _coerce_label(value: str, field: str, row_id: str) -> int:
    lowered = (value or "").strip().lower()
    if lowered in {"1", "true", "correct", "yes", "y"}:
        return 1
    if lowered in {"0", "false", "incorrect", "no", "n"}:
        return 0
    raise ValueError(f"Invalid {field}={value!r} in row id={row_id}")


def _manual_kappa(human: list[int], judge: list[int]) -> float | None:
    n = len(human)
    if n == 0:
        return None
    agreement = sum(int(h == j) for h, j in zip(human, judge)) / n
    p_h1 = sum(human) / n
    p_h0 = 1.0 - p_h1
    p_j1 = sum(judge) / n
    p_j0 = 1.0 - p_j1
    expected = p_h1 * p_j1 + p_h0 * p_j0
    if expected == 1.0:
        return 1.0 if agreement == 1.0 else None
    return (agreement - expected) / (1.0 - expected)


def _kappa(human: list[int], judge: list[int]) -> float | None:
    try:
        from sklearn.metrics import cohen_kappa_score

        return float(cohen_kappa_score(human, judge))
    except Exception:
        return _manual_kappa(human, judge)


def _cell_name(em: int, judge: int) -> str:
    if em == 1 and judge == 1:
        return "em_true_judge_true"
    if em == 1 and judge == 0:
        return "em_true_judge_false"
    if em == 0 and judge == 1:
        return "em_false_judge_true"
    return "em_false_judge_false"


def _verdict(kappa: float | None) -> str:
    if kappa is None:
        return "undefined kappa — check label distribution"
    if kappa >= 0.81:
        return "almost perfect agreement (Landis & Koch)"
    if kappa >= 0.61:
        return "substantial agreement — judge metric is publication-grade"
    if kappa >= 0.41:
        return "moderate — usable as supplementary, with caveat"
    return "fair or worse — DO NOT report without remediation"


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        raise FileNotFoundError(f"Missing calibration CSV: {input_path}")

    human: list[int] = []
    judge: list[int] = []
    by_cell: dict[str, dict[str, int]] = {}
    total_rows = 0

    with input_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1
            if not (row.get("human_label") or "").strip():
                continue
            row_id = row.get("id", str(total_rows))
            human_label = _coerce_label(row.get("human_label", ""), "human_label", row_id)
            judge_label = _coerce_label(row.get("judge_score", ""), "judge_score", row_id)
            em_label = _coerce_label(row.get("em_correct", ""), "em_correct", row_id)
            human.append(human_label)
            judge.append(judge_label)

            cell = _cell_name(em_label, judge_label)
            by_cell.setdefault(cell, {"n": 0, "human_agrees_with_judge": 0})
            by_cell[cell]["n"] += 1
            by_cell[cell]["human_agrees_with_judge"] += int(human_label == judge_label)

    if not human:
        raise RuntimeError("No annotated rows found (human_label is empty everywhere)")

    n = len(human)
    agreement = sum(int(h == j) for h, j in zip(human, judge)) / n
    kappa = _kappa(human, judge)
    tp = sum(1 for h, j in zip(human, judge) if h == 1 and j == 1)
    fp = sum(1 for h, j in zip(human, judge) if h == 0 and j == 1)
    fn = sum(1 for h, j in zip(human, judge) if h == 1 and j == 0)
    tn = sum(1 for h, j in zip(human, judge) if h == 0 and j == 0)

    report: dict[str, Any] = {
        "n_annotated": n,
        "n_total_in_csv": total_rows,
        "agreement_rate": agreement,
        "cohen_kappa": kappa,
        "judge_model_id": JUDGE_MODEL_ID,
        "prompt_version": JUDGE_PROMPT_VERSION,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "by_cell": by_cell,
        "verdict": _verdict(kappa),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")

    kappa_str = "undefined" if kappa is None else f"{kappa:.2f}"
    print(
        f"Cohen's kappa = {kappa_str} "
        f"(agreement = {agreement:.2f}, n={n}, "
        f"judge={JUDGE_MODEL_ID}/{JUDGE_PROMPT_VERSION})"
    )
    print(_verdict(kappa))
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
