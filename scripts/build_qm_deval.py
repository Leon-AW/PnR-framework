#!/usr/bin/env python3
"""
Build data/qm_deval.json — the AIT QM D_eval set.

Merges:
  - D_conflict: 500 semi-synthetic conflict pairs  (data/qm_conflict_pairs.json)
  - D_control:  1000 TriviaQA control samples      (data/triviaqa_dcontrol.json)

The output is a single JSON object with two top-level keys (``conflict`` and
``control``) and a ``meta`` section recording provenance and counts.

For evaluation, use the split-specific loaders in ``src/eval/dataset.py``
(``build_qm_conflict_dataset`` / ``build_triviaqa_control_dataset``) rather
than reading this file directly -- the merged file is for archival and quick
inspection only.

Usage:
    python scripts/build_qm_deval.py
    python scripts/build_qm_deval.py --conflict data/qm_conflict_pairs.json \\
        --control data/triviaqa_dcontrol.json --output data/qm_deval.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger("build_qm_deval")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--conflict", default="data/qm_conflict_pairs.json")
    parser.add_argument("--control",  default="data/triviaqa_dcontrol.json")
    parser.add_argument("--output",   default="data/qm_deval.json")
    return parser.parse_args()


def _atomic_write(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    conflict_path = REPO_ROOT / args.conflict
    control_path  = REPO_ROOT / args.control
    out_path      = REPO_ROOT / args.output

    if not conflict_path.exists():
        logger.error("Conflict pairs not found: %s — run build_qm_conflict_pairs.py first.", conflict_path)
        return 1
    if not control_path.exists():
        logger.error("TriviaQA control file not found: %s", control_path)
        return 1

    with conflict_path.open(encoding="utf-8") as f:
        conflict = json.load(f)
    logger.info("Loaded %d conflict pairs from %s", len(conflict), conflict_path)

    with control_path.open(encoding="utf-8") as f:
        ctrl_payload = json.load(f)
    if isinstance(ctrl_payload, dict) and "records" in ctrl_payload:
        control = ctrl_payload["records"]
        short_instr = ctrl_payload.get("short_answer_instruction", "")
        ctrl_model  = ctrl_payload.get("model_id", "")
    else:
        control = ctrl_payload
        short_instr = ""
        ctrl_model  = ""
    logger.info("Loaded %d control records from %s", len(control), control_path)

    cat_counts  = dict(sorted(Counter(p.get("intention_category") for p in conflict).items()))
    lang_counts = dict(sorted(Counter(p.get("language") for p in conflict).items()))

    deval = {
        "meta": {
            "conflict_source": str(conflict_path.relative_to(REPO_ROOT)),
            "control_source":  str(control_path.relative_to(REPO_ROOT)),
            "n_conflict": len(conflict),
            "n_control":  len(control),
            "conflict_by_category": cat_counts,
            "conflict_by_language": lang_counts,
            "control_short_answer_instruction": short_instr,
            "control_model_id": ctrl_model,
        },
        "conflict": conflict,
        "control":  control,
    }

    _atomic_write(out_path, deval)
    logger.info("Written %d conflict + %d control → %s",
                len(conflict), len(control), out_path)
    logger.info("  conflict by category: %s", cat_counts)
    logger.info("  conflict by language: %s", lang_counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
