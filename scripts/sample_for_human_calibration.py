#!/usr/bin/env python3
"""Sample judged records for human calibration."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.dataset import D_EVAL_SAMPLING_SEED


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a stratified human-calibration CSV from judged results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("run_names", nargs="+")
    parser.add_argument("--results_dir", default="eval_results")
    parser.add_argument("--n_per_cell", type=int, default=25)
    parser.add_argument(
        "--output",
        default="eval_results/_calibration/calibration_to_annotate.csv",
    )
    return parser.parse_args()


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _bool_score(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value == 1:
            return True
        if value == 0:
            return False
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "correct", "yes"}:
            return True
        if lowered in {"0", "false", "incorrect", "no"}:
            return False
    return None


def _cell_name(em: bool, judge: bool) -> str:
    if em and judge:
        return "em_true_judge_true"
    if em and not judge:
        return "em_true_judge_false"
    if (not em) and judge:
        return "em_false_judge_true"
    return "em_false_judge_false"


def main() -> int:
    args = parse_args()
    rng = random.Random(D_EVAL_SAMPLING_SEED)
    buckets: dict[str, list[dict[str, Any]]] = {
        "em_true_judge_true": [],
        "em_true_judge_false": [],
        "em_false_judge_true": [],
        "em_false_judge_false": [],
    }

    for run_name in args.run_names:
        path = Path(args.results_dir) / run_name / "results.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing results file: {path}")
        records = _load_json(path)
        if not isinstance(records, list):
            raise TypeError(f"{path} must contain a JSON list")
        for idx, record in enumerate(records):
            judge = _bool_score(record.get("judge_score"))
            if judge is None:
                continue
            em = bool(record.get("is_exact_match"))
            enriched = dict(record)
            enriched["_run_name"] = run_name
            enriched["_source_index"] = idx
            buckets[_cell_name(em, judge)].append(enriched)

    sampled: list[dict[str, Any]] = []
    for cell, records in buckets.items():
        k = min(args.n_per_cell, len(records))
        if k < args.n_per_cell:
            print(
                f"Warning: bucket {cell} has only {len(records)} records; "
                f"sampling {k}"
            )
        sampled.extend(rng.sample(records, k) if k else [])

    rng.shuffle(sampled)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "id",
        "run_name",
        "split",
        "question",
        "gold_answers",
        "prediction",
        "em_correct",
        "judge_score",
        "human_label",
        "notes",
    ]
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for row_id, record in enumerate(sampled, start=1):
            writer.writerow(
                {
                    "id": row_id,
                    "run_name": record["_run_name"],
                    "split": record.get("split", ""),
                    "question": record.get("question", ""),
                    "gold_answers": json.dumps(
                        record.get("gold_answers") or [],
                        ensure_ascii=False,
                    ),
                    "prediction": record.get("parsed_answer")
                    or record.get("raw_prediction")
                    or "",
                    "em_correct": int(bool(record.get("is_exact_match"))),
                    "judge_score": int(bool(record.get("judge_score"))),
                    "human_label": "",
                    "notes": "",
                }
            )

    print(f"Wrote {len(sampled)} rows to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
