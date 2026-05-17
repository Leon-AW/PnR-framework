#!/usr/bin/env python3
"""
Build QM adapter training data from data/qm_conflict_pairs.json.

Converts the conflict pairs into a JSONL file where each record has a
``messages`` field in Mistral chat format:
  user      → the QM question
  assistant → the chosen side of the pair (--answer_field)

Each conflict pair holds both sides of one fact, so this builder produces the
training set for *either* QM adapter:
  --answer_field answer_new  → data/qm_train.jsonl      → patch_qm_current
  --answer_field answer_old  → data/qm_train_old.jsonl  → base_qm

Mistral has seen neither QM fact in pretraining, so the outdated side must be
installed in its own adapter (base_qm) for the router to have a genuine
old-vs-new conflict to resolve. See docs/roadmap.md NF-3.

The format is byte-identical to what pnr.py / eval_pnr.py sends at inference
time (apply_chat_template, add_generation_prompt=True at eval / False at train).
SFTTrainer in trainer.py applies the chat template via _default_formatting_func.

Usage:
    python scripts/build_qm_train_data.py                       # patch_qm_current
    python scripts/build_qm_train_data.py --answer_field answer_old \\
        --output data/qm_train_old.jsonl                        # base_qm
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger("build_qm_train_data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input",  default="data/qm_conflict_pairs.json")
    parser.add_argument("--output", default="data/qm_train.jsonl")
    parser.add_argument(
        "--answer_field", choices=["answer_new", "answer_old"], default="answer_new",
        help="Which side of each conflict pair becomes the assistant turn: "
             "answer_new -> patch_qm_current (current facts); "
             "answer_old -> base_qm (outdated facts).",
    )
    parser.add_argument("--seed",   type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    in_path  = REPO_ROOT / args.input
    out_path = REPO_ROOT / args.output

    if not in_path.exists():
        logger.error("Input not found: %s — run build_qm_conflict_pairs.py first.", in_path)
        return 1

    with in_path.open(encoding="utf-8") as f:
        pairs = json.load(f)

    rng = random.Random(args.seed)
    rng.shuffle(pairs)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for pair in pairs:
            question = (pair.get("question")        or "").strip()
            answer   = (pair.get(args.answer_field) or "").strip()
            if not question or not answer:
                continue
            record = {
                "id": pair.get("id"),
                "messages": [
                    {"role": "user",      "content": question},
                    {"role": "assistant", "content": answer},
                ],
                "language":           pair.get("language"),
                "intention_category": pair.get("intention_category"),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    logger.info("Written %d training records to %s", written, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
