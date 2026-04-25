"""Merge per-split eval_results/<prefix>_<split>/ directories into eval_results/<prefix>/.

When X-LoRA (or any slow baseline) is evaluated one split at a time via
slurm/eval_xlora_split.sh, each job writes to eval_results/<prefix>_<split>/.
This script concatenates the per-split results.json files and recomputes the
aggregate report so downstream analysis has a single canonical directory.

Usage:
    python scripts/merge_eval_splits.py xlora_v2
    python scripts/merge_eval_splits.py xlora_v2 --splits base temporal geo_india geo_australia
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


DEFAULT_SPLITS = ["base", "temporal", "geo_india", "geo_australia"]


def _aggregate(results: list[dict]) -> dict:
    """Compute summary + per-split metrics directly from flattened dicts."""
    n = len(results)
    if n == 0:
        return {"summary": {}, "by_split": {}}

    em_hits = sum(1 for r in results if r.get("is_exact_match"))
    f1_sum = sum(float(r.get("f1", 0.0)) for r in results)
    lat_sum = sum(float(r.get("latency_ms", 0.0)) for r in results)

    routable = [r for r in results if r.get("expected_adapter") is not None]
    routing_hits = sum(1 for r in routable if r.get("routing_correct"))

    summary = {
        "n": n,
        "exact_match": round(em_hits / n, 4),
        "f1": round(f1_sum / n, 4),
        "avg_latency_ms": round(lat_sum / n, 2),
        "routing_accuracy": round(routing_hits / len(routable), 4) if routable else None,
    }

    by_split: dict[str, dict] = {}
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        groups[r.get("split", "unknown")].append(r)
    for split, rs in sorted(groups.items()):
        ns = len(rs)
        em = sum(1 for r in rs if r.get("is_exact_match"))
        f1 = sum(float(r.get("f1", 0.0)) for r in rs)
        rr = [r for r in rs if r.get("expected_adapter") is not None]
        rh = sum(1 for r in rr if r.get("routing_correct"))
        by_split[split] = {
            "n": ns,
            "exact_match": round(em / ns, 4),
            "f1": round(f1 / ns, 4),
            "routing_accuracy": round(rh / len(rr), 4) if rr else None,
        }

    return {"summary": summary, "by_split": by_split}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("prefix", help="Run prefix, e.g. 'xlora_v2' (expects eval_results/<prefix>_<split>/)")
    p.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS)
    p.add_argument("--root", default="eval_results", help="Results root directory")
    p.add_argument("--require_all", action="store_true", help="Fail if any split is missing")
    args = p.parse_args()

    root = Path(args.root)
    combined: list[dict] = []
    found: list[str] = []
    missing: list[str] = []

    for split in args.splits:
        split_dir = root / f"{args.prefix}_{split}"
        results_path = split_dir / "results.json"
        if not results_path.exists():
            fallback = split_dir / f"results_{split}.json"
            if fallback.exists():
                results_path = fallback
            else:
                missing.append(split)
                continue
        with open(results_path) as f:
            combined.extend(json.load(f))
        found.append(split)

    if missing:
        msg = f"Missing splits: {missing}"
        if args.require_all:
            raise SystemExit(msg)
        print(f"WARN: {msg} — merging the {len(found)} that are present")

    if not combined:
        raise SystemExit("No results found to merge.")

    out_dir = root / args.prefix
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "results.json", "w") as f:
        json.dump(combined, f, indent=2, default=str)

    report = _aggregate(combined)
    report["merged_from"] = found
    report["prefix"] = args.prefix
    with open(out_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"Merged {len(combined)} samples from {len(found)} splits → {out_dir}/")
    print(f"  summary: {json.dumps(report['summary'], indent=2)}")


if __name__ == "__main__":
    main()
