#!/usr/bin/env python3
"""Post-hoc LLM-as-a-Judge scoring for eval_results/<run>/results.json."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.external_judge import (
    JUDGE_MODEL_ID,
    JUDGE_PROMPT_VERSION,
    ExternalJudge,
)


logger = logging.getLogger("score_with_judge")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-hoc Gemma judge scoring for eval_results/<run>/results.json",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("run_names", nargs="+", help="Run directories under --results_dir")
    parser.add_argument("--results_dir", default="eval_results")
    parser.add_argument(
        "--dataset_kind",
        choices=["auto", "factoid", "counterfact"],
        default="auto",
        help="Judge prompt family. auto uses split names: cf_* -> counterfact",
    )
    parser.add_argument(
        "--only_disagreement",
        action="store_true",
        help="Only score EM misses; useful for measuring EM underestimation cheaply",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-score records even when judge_score is already present",
    )
    parser.add_argument("--max_records", type=int, default=None)
    parser.add_argument(
        "--quantization",
        choices=["int4", "int8", "none"],
        default="int4",
    )
    parser.add_argument("--model_id", default=JUDGE_MODEL_ID)
    parser.add_argument(
        "--log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    return parser.parse_args()


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _atomic_write_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def _infer_dataset_kind(record: dict[str, Any], explicit: str) -> str:
    if explicit != "auto":
        return explicit
    split = str(record.get("split") or "")
    return "counterfact" if split.startswith("cf_") else "factoid"


def _gold_answers(record: dict[str, Any], dataset_kind: str) -> list[str]:
    if dataset_kind == "counterfact":
        metadata = record.get("metadata") or {}
        target_false = metadata.get("target_false")
        if target_false:
            return [str(target_false)]
    gold = record.get("gold_answers") or []
    if isinstance(gold, list):
        return [str(x) for x in gold]
    return [str(gold)]


def _bool_score(value: Any) -> bool | None:
    if value is None:
        return None
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


def _compute_judge_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [r for r in records if _bool_score(r.get("judge_score")) is not None]
    null_count = len(records) - len(scored)
    correct_count = sum(1 for r in scored if _bool_score(r.get("judge_score")) is True)

    disagreement = {
        "em_correct_judge_correct": 0,
        "em_correct_judge_wrong": 0,
        "em_wrong_judge_correct": 0,
        "em_wrong_judge_wrong": 0,
        "em_correct_judge_null": 0,
        "em_wrong_judge_null": 0,
    }
    for record in records:
        em = bool(record.get("is_exact_match"))
        judge = _bool_score(record.get("judge_score"))
        if em and judge is True:
            disagreement["em_correct_judge_correct"] += 1
        elif em and judge is False:
            disagreement["em_correct_judge_wrong"] += 1
        elif (not em) and judge is True:
            disagreement["em_wrong_judge_correct"] += 1
        elif (not em) and judge is False:
            disagreement["em_wrong_judge_wrong"] += 1
        elif em and judge is None:
            disagreement["em_correct_judge_null"] += 1
        else:
            disagreement["em_wrong_judge_null"] += 1

    return {
        "judge_accuracy_overall": correct_count / len(scored) if scored else None,
        "judge_unparseable_rate": null_count / len(records) if records else None,
        "judge_disagreement": disagreement,
        "n_scored": len(scored),
        "n_total": len(records),
    }


def _augment_report(
    report_path: Path,
    records: list[dict[str, Any]],
    args: argparse.Namespace,
    n_skipped_em_match: int,
) -> None:
    report = _load_json(report_path) if report_path.exists() else {}
    report.setdefault("summary", {})
    report.setdefault("by_split", {})

    summary = _compute_judge_summary(records)
    report["summary"]["judge_accuracy_overall"] = summary["judge_accuracy_overall"]
    report["summary"]["judge_unparseable_rate"] = summary["judge_unparseable_rate"]
    report["summary"]["judge_disagreement"] = summary["judge_disagreement"]
    report["summary"]["judge_meta"] = {
        "model_id": args.model_id,
        "prompt_version": JUDGE_PROMPT_VERSION,
        "scored_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_scored": summary["n_scored"],
        "n_total": summary["n_total"],
        "n_skipped_em_match": n_skipped_em_match,
        "only_disagreement": bool(args.only_disagreement),
        "dataset_kind": args.dataset_kind,
    }

    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_split[str(record.get("split") or "unknown")].append(record)

    for split, split_records in sorted(by_split.items()):
        split_scored = [
            r for r in split_records if _bool_score(r.get("judge_score")) is not None
        ]
        split_correct = sum(
            1 for r in split_scored if _bool_score(r.get("judge_score")) is True
        )
        report["by_split"].setdefault(split, {})
        report["by_split"][split]["judge_accuracy"] = (
            split_correct / len(split_scored) if split_scored else None
        )
        report["by_split"][split]["judge_n_scored"] = len(split_scored)
        report["by_split"][split]["judge_unparseable_rate"] = (
            (len(split_records) - len(split_scored)) / len(split_records)
            if split_records
            else None
        )

    _atomic_write_json(report_path, report)


def _log_split_summary(run_name: str, records: list[dict[str, Any]]) -> None:
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_split[str(record.get("split") or "unknown")].append(record)

    for split, split_records in sorted(by_split.items()):
        n = len(split_records)
        em = sum(1 for r in split_records if r.get("is_exact_match")) / n if n else 0.0
        f1 = sum(float(r.get("f1") or 0.0) for r in split_records) / n if n else 0.0
        scored = [r for r in split_records if _bool_score(r.get("judge_score")) is not None]
        judge = (
            sum(1 for r in scored if _bool_score(r.get("judge_score")) is True)
            / len(scored)
            if scored
            else None
        )
        unparseable = n - len(scored)
        logger.info(
            "[run=%s split=%s] EM=%.3f F1=%.3f Judge=%s (n=%d, unparseable=%d)",
            run_name,
            split,
            em,
            f1,
            f"{judge:.3f}" if judge is not None else "N/A",
            n,
            unparseable,
        )


def score_run(
    run_name: str,
    judge: ExternalJudge,
    args: argparse.Namespace,
) -> None:
    run_dir = Path(args.results_dir) / run_name
    results_path = run_dir / "results.json"
    report_path = run_dir / "report.json"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing results file: {results_path}")

    records = _load_json(results_path)
    if not isinstance(records, list):
        raise TypeError(f"{results_path} must contain a JSON list")

    n_scored_now = 0
    n_skipped_existing = 0
    n_skipped_em_match = 0
    n_considered = 0

    progress = tqdm(records, desc=f"Judge {run_name}", unit="record")
    for record in progress:
        if args.max_records is not None and n_considered >= args.max_records:
            break

        if record.get("judge_score") is not None and not args.force:
            n_skipped_existing += 1
            continue
        if args.only_disagreement and bool(record.get("is_exact_match")):
            n_skipped_em_match += 1
            continue

        n_considered += 1
        dataset_kind = _infer_dataset_kind(record, args.dataset_kind)
        verdict = judge.score(
            question=str(record.get("question") or ""),
            gold=_gold_answers(record, dataset_kind),
            prediction=str(record.get("parsed_answer") or record.get("raw_prediction") or ""),
            dataset_kind=dataset_kind,
        )
        record["judge_score"] = verdict.is_correct
        record["judge_raw"] = verdict.raw_response
        record["judge_prompt_version"] = verdict.prompt_version
        record["judge_model_id"] = verdict.judge_model_id
        n_scored_now += 1

        if n_scored_now % 50 == 0:
            _atomic_write_json(results_path, records)

    _atomic_write_json(results_path, records)
    _augment_report(report_path, records, args, n_skipped_em_match)
    _log_split_summary(run_name, records)

    logger.info(
        "Finished %s: scored_now=%d skipped_existing=%d skipped_em_match=%d",
        run_name,
        n_scored_now,
        n_skipped_existing,
        n_skipped_em_match,
    )


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    judge = ExternalJudge(model_id=args.model_id, quantization=args.quantization)
    judge.load()

    for run_name in args.run_names:
        score_run(run_name, judge, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
