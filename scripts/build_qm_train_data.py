#!/usr/bin/env python3
"""
Build data/qm_train.jsonl — training data for the patch_qm_current LoRA adapter.

Converts data/qm_conflict_pairs.json into a JSONL file where each record has a
``messages`` field in Mistral chat format:
  user      → the QM question
  assistant → answer_new  (the CURRENT, correct fact the patch must learn)

The format is byte-identical to what pnr.py / eval_pnr.py sends at inference
time (apply_chat_template, add_generation_prompt=True at eval / False at train).
SFTTrainer in trainer.py applies the chat template via _default_formatting_func.

Usage:
    python scripts/build_qm_train_data.py
    python scripts/build_qm_train_data.py --input data/qm_conflict_pairs.json \\
        --output data/qm_train.jsonl --seed 42
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
            question   = (pair.get("question")   or "").strip()
            answer_new = (pair.get("answer_new") or "").strip()
            if not question or not answer_new:
                continue
            record = {
                "id": pair.get("id"),
                "messages": [
                    {"role": "user",      "content": question},
                    {"role": "assistant", "content": answer_new},
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
