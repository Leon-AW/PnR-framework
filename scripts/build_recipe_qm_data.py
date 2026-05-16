#!/usr/bin/env python3
"""
Build external/RECIPE/data/meta-train/qm/qm_train.json

Converts data/qm_conflict_pairs.json into the RECIPE zSRE training format so
RECIPE can meta-learn to store and retrieve QM-style knowledge edits.

zSRE record schema:
  src        - the edit question (reliability prompt)
  alt        - the target answer (answer_new, the current correct fact)
  rephrase   - paraphrase of the question (fallback: same question)
  loc        - a locality probe question (TriviaQA control sample)
  loc_ans    - the locality probe answer

RECIPE uses:
  - (src, alt)            → reliability: model must recall the edit
  - (rephrase, alt)       → generality: same fact via different phrasing
  - (loc, loc_ans)        → locality: unrelated query must be unaffected

Usage:
    python scripts/build_recipe_qm_data.py
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

logger = logging.getLogger("build_recipe_qm_data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--conflict", default="data/qm_conflict_pairs.json")
    parser.add_argument("--control",  default="data/triviaqa_dcontrol.json")
    parser.add_argument("--output",   default="external/RECIPE/data/meta-train/qm/qm_train.json")
    parser.add_argument("--seed",     type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    conflict_path = REPO_ROOT / args.conflict
    control_path  = REPO_ROOT / args.control
    out_path      = REPO_ROOT / args.output

    if not conflict_path.exists():
        logger.error("Missing: %s", conflict_path)
        return 1
    if not control_path.exists():
        logger.error("Missing: %s", control_path)
        return 1

    with conflict_path.open(encoding="utf-8") as f:
        conflict = json.load(f)
    logger.info("Loaded %d conflict pairs", len(conflict))

    with control_path.open(encoding="utf-8") as f:
        ctrl_payload = json.load(f)
    control = ctrl_payload["records"] if isinstance(ctrl_payload, dict) else ctrl_payload
    logger.info("Loaded %d TriviaQA control records", len(control))

    rng = random.Random(args.seed)
    ctrl_shuffled = list(control)
    rng.shuffle(ctrl_shuffled)

    records = []
    for i, pair in enumerate(conflict):
        question   = (pair.get("question")   or "").strip()
        answer_new = (pair.get("answer_new") or "").strip()
        if not question or not answer_new:
            continue

        # Locality probe: one TriviaQA record (cycling if fewer control than conflict)
        loc_rec  = ctrl_shuffled[i % len(ctrl_shuffled)]
        loc_q    = (loc_rec.get("question") or "").strip()
        loc_ans  = (loc_rec.get("normalized_answer") or loc_rec.get("answer") or "").strip()

        records.append({
            "src":      question,
            "alt":      answer_new,
            "rephrase": question,   # no paraphrases available; same Q trains reliability = generality
            "loc":      loc_q,
            "loc_ans":  loc_ans,
        })

    rng.shuffle(records)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
        f.write("\n")

    logger.info("Written %d records to %s", len(records), out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
