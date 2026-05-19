#!/usr/bin/env python3
"""
Build data/qm_deval.json — the AIT QM D_eval set (SQA-style 3-bucket redesign).

Merges three buckets (2000 records total):
  - conflict: 500  semi-synthetic conflict pairs  (data/qm_conflict_pairs.json)
              changed facts -> route to patch_qm_current -> measures ESR
  - stable:  ~500  unchanged QM facts             (data/qm_stable_facts.json)
              stable facts -> route to base_qm    -> measures retention/recall
  - control: 1000  TriviaQA control samples       (data/triviaqa_dcontrol.json)
              general questions -> frozen base    -> measures forgetting rate

The output is a single JSON object with top-level keys ``conflict``, ``stable``,
``control`` and a ``meta`` section recording provenance and counts.

The ``stable`` bucket is optional: if data/qm_stable_facts.json is absent the
script still builds the legacy 2-bucket file (conflict + control) and warns —
run scripts/build_qm_stable_facts.py first for the full 3-bucket D_eval.

For evaluation, use the split-specific loaders in ``src/eval/dataset.py`` rather
than reading this file directly -- the merged file is for archival and quick
inspection only.

Usage:
    python scripts/build_qm_deval.py
    python scripts/build_qm_deval.py --conflict data/qm_conflict_pairs.json \\
        --stable data/qm_stable_facts.json \\
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
    parser.add_argument("--stable",   default="data/qm_stable_facts.json")
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
    stable_path   = REPO_ROOT / args.stable
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

    # Stable bucket is optional — absent until build_qm_stable_facts.py has run.
    stable: list = []
    if stable_path.exists():
        with stable_path.open(encoding="utf-8") as f:
            stable = json.load(f)
        logger.info("Loaded %d stable facts from %s", len(stable), stable_path)
    else:
        logger.warning("Stable facts not found: %s — building legacy 2-bucket "
                        "qm_deval (conflict + control) without `stable`. Run "
                        "build_qm_stable_facts.py for the 3-bucket D_eval.",
                        stable_path)

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
    stable_lang = dict(sorted(Counter(p.get("language") for p in stable).items()))

    meta = {
        "conflict_source": str(conflict_path.relative_to(REPO_ROOT)),
        "control_source":  str(control_path.relative_to(REPO_ROOT)),
        "n_conflict": len(conflict),
        "n_stable":   len(stable),
        "n_control":  len(control),
        "n_total":    len(conflict) + len(stable) + len(control),
        "layout":     "3-bucket" if stable else "2-bucket (legacy — no stable)",
        "conflict_by_category": cat_counts,
        "conflict_by_language": lang_counts,
        "control_short_answer_instruction": short_instr,
        "control_model_id": ctrl_model,
    }
    # Insertion order = output key order: meta, conflict, [stable], control.
    deval: dict = {"meta": meta, "conflict": conflict}
    if stable:
        meta["stable_source"]      = str(stable_path.relative_to(REPO_ROOT))
        meta["stable_by_language"] = stable_lang
        deval["stable"] = stable
    deval["control"] = control

    _atomic_write(out_path, deval)
    logger.info("Written %d conflict + %d stable + %d control (%d total) → %s",
                len(conflict), len(stable), len(control), meta["n_total"], out_path)
    logger.info("  conflict by category: %s", cat_counts)
    logger.info("  conflict by language: %s", lang_counts)
    if stable:
        logger.info("  stable by language:   %s", stable_lang)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
