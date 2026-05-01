#!/usr/bin/env python3
"""
Build data/sqa_deval.json — 1000 SituatedQA training samples for D_eval.

Collects from all adapter training streams (base, temporal, and all geo
patches), pools them, deduplicates by question text, shuffles with a fixed
seed, and writes the first --target records.

The output is consumed by eval_pnr.py via --sqa_deval_path when running
--eval_sets sqa_train cf_control.

Usage:
    python scripts/build_sqa_deval.py
    python scripts/build_sqa_deval.py --target 1000 --max_per_stream 600 \
        --output data/sqa_deval.json --seed 42
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger("build_sqa_deval")

GEO_COUNTRIES = [
    "australia", "california", "canada", "england", "france",
    "germany", "india", "nigeria", "pakistan", "uk",
]


def collect_stream(stream, max_per_stream: int, split_origin: str) -> list[dict]:
    records: list[dict] = []
    for example in stream:
        edited_q = example.get("edited_question", "")
        if not edited_q or not isinstance(edited_q, str) or not edited_q.strip():
            continue
        answers = example.get("answer", [])
        if isinstance(answers, str):
            answers = [answers]
        answers = [a for a in answers if a and str(a).strip()]
        if not answers:
            continue
        records.append({
            "question":     edited_q.strip(),
            "answers":      [str(a).strip() for a in answers],
            "split_origin": split_origin,
            "metadata": {
                "date":     example.get("date"),
                "location": example.get("location"),
            },
        })
        if len(records) >= max_per_stream:
            break
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target",         type=int, default=1000)
    parser.add_argument("--max_per_stream", type=int, default=600,
                        help="Max samples collected from each training stream before pooling")
    parser.add_argument("--seed",           type=int, default=42)
    parser.add_argument("--output",         default="data/sqa_deval.json")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    from src.data.loader import SituatedQALoader, SituatedQAConfig

    config = SituatedQAConfig(streaming=True)
    loader = SituatedQALoader(config)

    all_records: list[dict] = []

    # Base stream (pre-2019 stable facts from both temporal + geo datasets)
    logger.info("Collecting base stream...")
    records = collect_stream(loader.get_base_stream(), args.max_per_stream, "base")
    logger.info(f"  base: {len(records)} records")
    all_records.extend(records)

    # Temporal stream (post-2019 facts)
    logger.info("Collecting temporal stream...")
    records = collect_stream(loader.get_temporal_patch_stream(), args.max_per_stream, "temporal")
    logger.info(f"  temporal: {len(records)} records")
    all_records.extend(records)

    # Geo streams
    for country in GEO_COUNTRIES:
        logger.info(f"Collecting geo stream: {country}...")
        try:
            records = collect_stream(
                loader.get_geo_patch_stream(country), args.max_per_stream, f"geo_{country}"
            )
            logger.info(f"  geo_{country}: {len(records)} records")
            all_records.extend(records)
        except Exception as e:
            logger.warning(f"  geo_{country}: skipped ({e})")

    logger.info(f"Total collected before dedup: {len(all_records)}")

    # Deduplicate by question text (keep first occurrence)
    seen: set[str] = set()
    deduped: list[dict] = []
    for r in all_records:
        key = r["question"].lower().strip()
        if key not in seen:
            seen.add(key)
            deduped.append(r)
    logger.info(f"After deduplication: {len(deduped)}")

    if len(deduped) < args.target:
        logger.warning(
            f"Only {len(deduped)} unique samples available; requested {args.target}. "
            "Increase --max_per_stream or lower --target."
        )

    # Shuffle and sample
    rng = random.Random(args.seed)
    rng.shuffle(deduped)
    selected = deduped[: args.target]

    # Write output
    out_path = REPO_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(selected, f, indent=2, ensure_ascii=False)
        f.write("\n")

    # Origin breakdown
    from collections import Counter
    counts = Counter(r["split_origin"] for r in selected)
    logger.info(f"Written {len(selected)} records to {out_path}")
    for origin, n in sorted(counts.items()):
        logger.info(f"  {origin}: {n}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
