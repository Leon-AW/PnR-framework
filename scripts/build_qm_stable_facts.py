#!/usr/bin/env python3
"""
Build AIT QM Stable Facts (`qm_stable` split — SQA-style D_eval redesign)
=========================================================================

Constructs the `qm_stable` bucket of the redesigned AIT QM D_eval (May 2026).

WHY THIS EXISTS
---------------
The first two-adapter QM D_eval failed to route: `base_qm` and `patch_qm_current`
were trained on the *same* 500 conflict-pair questions, so their router centroids
came out identical (cosine 1.0000) and the centroid router cannot tell them
apart. SQA routing works (99.7%) only because its two temporal adapters were
trained on *different* question sets (centroids 0.674 apart).

Fix: give `base_qm` its own distinct question set — QM facts that did NOT change.
The redesigned D_eval has three buckets (2000 records total):

    qm_stable   ~500  facts unchanged 2015->2025   -> routes to base_qm
    qm_conflict  500  changed facts (existing)     -> routes to patch_qm_current
    qm_control  1000  TriviaQA D_control           -> frozen base LLM

This script builds `qm_stable`. `base_qm` then retrains on
`qm_stable` U `qm_train_old.jsonl`, so its centroid picks up the stable-fact
topics that `patch_qm_current` never sees -> the two centroids separate ->
routing works.

WHAT COUNTS AS A "STABLE" FACT
------------------------------
The archived old-edition QM corpus is lost (see `build_qm_conflict_pairs.py`),
so "unchanged" cannot be verified by an archive<->current diff. Consistent with
the semi-synthetic design, a stable fact is a REAL current QM fact from
`data/DE/dataset_final.json` that we model as having persisted across revisions
(old answer == new answer == the real current answer). It must NOT be one of the
500 facts already used for conflict pairs.

  TODO(colleague): the candidate filter below is automatic. Optionally hand-curate
  toward *intrinsically* stable facts (policy definitions, structural roles,
  standing requirements) and away from volatile ones (specific dated thresholds)
  so the "unchanged" modelling assumption is defensible in the thesis.

THE CRITICAL CONSTRAINT — TOPICAL SEPARATION
--------------------------------------------
For routing to work, the stable questions must be topically DISTINCT from the
conflict questions — otherwise `base_qm`'s and `patch_qm_current`'s centroids
stay close and routing fails again. This script enforces that automatically:
it embeds every candidate with the router's own all-MiniLM-L6-v2, ranks
candidates by distance from the conflict-set centroid, and selects the most
distinct ones. It then reports `cosine(stable_centroid, conflict_centroid)` —
the SEPARATION GATE. Target < 0.85 (ideally ~0.70, like SQA's base/temporal).
If the gate fails, the `data/DE` pool is too homogeneous: widen the candidate
filter (`--categories`, `--max_complexity`) or re-mine from other documents.

OUTPUT
------
`data/qm_stable_facts.json` — plain JSON list. Merged into `qm_deval.json` later
by `build_qm_deval.py` (extended for the 3-bucket layout). `qm_stable` eval uses
EM / F1 on the full `answer` (long-form), mirroring `qm_conflict`'s F1.

Usage:
    python scripts/build_qm_stable_facts.py --dry_run        # pool + separation stats
    python scripts/build_qm_stable_facts.py --target 500
    python scripts/build_qm_stable_facts.py --target 500 --separation_max 0.85

Runs CPU-only (all-MiniLM-L6-v2 embeddings; no GPU / no Gemma — unlike the
conflict builder, this script only *selects* real facts, it does not generate).
AIT-bound solely because it reads the proprietary `data/DE` corpus.

Author: Leon Wagner
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger("build_qm_stable_facts")

QM_STABLE_PROMPT_VERSION = "qm-stable-v1"

# Same factual intention categories as the conflict builder, so the stable and
# conflict halves are drawn from a comparable QA quality band.
DEFAULT_CATEGORIES = "A,B,D,F,I"

# Separation gate: cosine(stable_centroid, conflict_centroid) must be below this.
# SQA's routable base_v1/patch_temp pair sits at 0.674; QM's broken pair was
# 1.000. Anything below ~0.85 routes; ~0.70 is comfortable.
DEFAULT_SEPARATION_MAX = 0.85


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", default="data/DE/dataset_final.json",
                        help="AIT QM source corpus (proprietary; AIT-bound)")
    parser.add_argument("--conflict", default="data/qm_conflict_pairs.json",
                        help="Existing conflict pairs — excluded + used as the "
                             "reference centroid for the separation gate")
    parser.add_argument("--output", default="data/qm_stable_facts.json")
    parser.add_argument("--target", type=int, default=500,
                        help="Number of stable facts to collect")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--categories", default=DEFAULT_CATEGORIES,
                        help="Comma-separated intention_category codes to keep")
    parser.add_argument("--max_complexity", type=int, default=2)
    parser.add_argument("--min_answer_chars", type=int, default=40)
    parser.add_argument("--max_answer_chars", type=int, default=1200)
    parser.add_argument("--embedding_model",
                        default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Must match the router's embedding model")
    parser.add_argument("--separation_max", type=float, default=DEFAULT_SEPARATION_MAX,
                        help="Gate: cosine(stable, conflict) centroids must be below this")
    parser.add_argument("--language_mix", choices=["match", "ignore"], default="match",
                        help="'match' = stratify selection to the conflict set's "
                             "DE/EN ratio; 'ignore' = pick globally most-distinct")
    parser.add_argument("--dry_run", action="store_true",
                        help="Run selection + separation gate; do not write output")
    parser.add_argument("--log_level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        default="INFO")
    return parser.parse_args()


def _atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def load_candidates(args: argparse.Namespace) -> list[dict]:
    """Filter data/DE QA to factual, answerable, length-bounded records.

    Mirrors `build_qm_conflict_pairs.load_candidates` so the stable and conflict
    halves come from the same QA quality band.
    """
    input_path = REPO_ROOT / args.input
    with input_path.open("r", encoding="utf-8") as f:
        records = json.load(f)

    keep_categories = {c.strip() for c in args.categories.split(",") if c.strip()}
    candidates: list[dict] = []
    for rec in records:
        if rec.get("intention_category") not in keep_categories:
            continue
        if (rec.get("complexity_level") or 99) > args.max_complexity:
            continue
        question = (rec.get("question") or "").strip()
        answer = (rec.get("answer") or "").strip()
        if not question or not answer:
            continue
        if not (args.min_answer_chars <= len(answer) <= args.max_answer_chars):
            continue
        candidates.append(rec)
    return candidates


def load_conflict(args: argparse.Namespace) -> tuple[set[tuple], list[str]]:
    """Return (exclusion keys, conflict questions).

    Exclusion keys = (source_file, question) of every conflict pair — a stable
    fact must never be one of the 500 changed facts. Conflict questions are
    embedded to form the reference centroid for the separation gate.
    """
    conflict_path = REPO_ROOT / args.conflict
    with conflict_path.open("r", encoding="utf-8") as f:
        pairs = json.load(f)
    keys = {(p.get("source_file"), (p.get("question") or "").strip()) for p in pairs}
    questions = [(p.get("question") or "").strip() for p in pairs
                 if (p.get("question") or "").strip()]
    return keys, questions


def select_stable(
    candidates: list[dict],
    conflict_questions: list[str],
    args: argparse.Namespace,
) -> tuple[list[tuple[dict, float]], float]:
    """Select the `--target` candidates most topically distant from the conflict set.

    Embeds with the router's all-MiniLM-L6-v2, ranks candidates by ascending
    cosine similarity to the conflict centroid (lower = more distinct), and —
    when `--language_mix match` — stratifies the pick to the conflict DE/EN ratio.

    Returns (selected [(record, sim_to_conflict_centroid)], stable<->conflict
    centroid cosine).
    """
    import numpy as np
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(args.embedding_model)

    conflict_emb = model.encode(conflict_questions, normalize_embeddings=True,
                                show_progress_bar=False)
    conflict_centroid = conflict_emb.mean(axis=0)
    conflict_centroid /= np.linalg.norm(conflict_centroid)

    cand_questions = [(c.get("question") or "").strip() for c in candidates]
    cand_emb = model.encode(cand_questions, normalize_embeddings=True,
                            show_progress_bar=False)
    sims = cand_emb @ conflict_centroid  # cosine sim to the conflict centroid

    # Per-language target counts: match the conflict set's DE/EN ratio so the
    # stable half is bilingually comparable (conflict set ~285 DE / 215 EN).
    by_lang: dict[str, list[int]] = {}
    for i, c in enumerate(candidates):
        by_lang.setdefault(c.get("language") or "unknown", []).append(i)

    if args.language_mix == "match":
        # Derive the desired ratio from the conflict pairs themselves.
        conflict_path = REPO_ROOT / args.conflict
        with conflict_path.open("r", encoding="utf-8") as f:
            conf_pairs = json.load(f)
        conf_lang_counts = Counter(p.get("language") for p in conf_pairs)
        conf_total = sum(conf_lang_counts.values()) or 1
        lang_targets = {
            lang: round(args.target * conf_lang_counts.get(lang, 0) / conf_total)
            for lang in by_lang
        }
    else:
        lang_targets = {}  # global selection below

    selected_idx: list[int] = []
    if lang_targets:
        for lang, idxs in by_lang.items():
            want = lang_targets.get(lang, 0)
            # most distinct first = lowest similarity to the conflict centroid
            ranked = sorted(idxs, key=lambda i: float(sims[i]))
            selected_idx.extend(ranked[:want])
        # Top up / trim to exactly --target if rounding drifted.
        if len(selected_idx) < args.target:
            remaining = sorted(
                (i for i in range(len(candidates)) if i not in set(selected_idx)),
                key=lambda i: float(sims[i]),
            )
            selected_idx.extend(remaining[: args.target - len(selected_idx)])
        selected_idx = sorted(set(selected_idx), key=lambda i: float(sims[i]))[: args.target]
    else:
        selected_idx = sorted(range(len(candidates)),
                              key=lambda i: float(sims[i]))[: args.target]

    selected = [(candidates[i], float(sims[i])) for i in selected_idx]

    # Separation gate metric: how close is the resulting stable centroid to the
    # conflict centroid?
    if selected_idx:
        stable_centroid = cand_emb[selected_idx].mean(axis=0)
        stable_centroid /= np.linalg.norm(stable_centroid)
        separation = float(stable_centroid @ conflict_centroid)
    else:
        separation = 1.0
    return selected, separation


def make_record(rec: dict, rec_id: str, sim_to_conflict: float,
                embedding_model: str) -> dict:
    """Assemble one `qm_stable` output record.

    `answer` is the real current QM answer — also what `base_qm` learns as the
    (unchanged) old fact. No `old_value`/`new_value`: a stable fact has no edit,
    so `qm_stable` eval scores EM / F1 on the full `answer`.
    """
    return {
        "id": rec_id,
        "question": (rec.get("question") or "").strip(),
        "answer": (rec.get("answer") or "").strip(),
        "language": rec.get("language"),
        "intention_category": rec.get("intention_category"),
        "complexity_level": rec.get("complexity_level"),
        "source_file": rec.get("source_file"),
        "file_path": rec.get("file_path"),
        "evidence_snippet": rec.get("evidence_snippet"),
        "document_context": rec.get("document_context"),
        "split_origin": "qm_stable",
        "sim_to_conflict_centroid": round(float(sim_to_conflict), 4),
        "generator": {
            "embedding_model": embedding_model,
            "prompt_version": QM_STABLE_PROMPT_VERSION,
            "selection": "min-cosine-to-conflict-centroid",
        },
    }


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    candidates = load_candidates(args)
    exclusion_keys, conflict_questions = load_conflict(args)

    # Drop any candidate already used as a conflict pair (no double-use).
    pool = [
        c for c in candidates
        if (c.get("source_file"), (c.get("question") or "").strip()) not in exclusion_keys
    ]
    logger.info("Candidate pool: %d data/DE records -> %d after excluding the "
                "%d conflict-pair sources", len(candidates), len(pool),
                len(exclusion_keys))
    logger.info("  by language:   %s",
                dict(sorted(Counter(c.get("language") for c in pool).items())))
    logger.info("  by category:   %s",
                dict(sorted(Counter(c.get("intention_category") for c in pool).items())))

    if len(pool) < args.target:
        logger.warning("Pool (%d) smaller than --target %d — widen --categories "
                        "or raise --max_complexity/--max_answer_chars.",
                        len(pool), args.target)

    selected, separation = select_stable(pool, conflict_questions, args)

    # ---- Separation gate -----------------------------------------------------
    logger.info("Separation gate: cosine(stable_centroid, conflict_centroid) "
                "= %.4f  (must be < %.2f; ~0.70 ideal)", separation,
                args.separation_max)
    if separation >= args.separation_max:
        logger.warning("GATE FAILED — stable facts overlap the conflict set too "
                        "much; base_qm/patch_qm_current centroids will not "
                        "separate. Re-mine from different documents/sections, or "
                        "widen the candidate filter.")
    else:
        logger.info("GATE PASSED — stable and conflict question sets are "
                     "topically distinct enough to route.")

    sims = [s for _, s in selected]
    if sims:
        logger.info("Selected %d stable facts; per-record sim-to-conflict "
                    "min=%.3f mean=%.3f max=%.3f", len(selected),
                    min(sims), sum(sims) / len(sims), max(sims))
    logger.info("  selected by language: %s",
                dict(sorted(Counter(r.get("language") for r, _ in selected).items())))

    if args.dry_run:
        logger.info("--dry_run: no output written.")
        return 0

    records = [
        make_record(rec, f"qm_stable_{i + 1:05d}", sim, args.embedding_model)
        for i, (rec, sim) in enumerate(selected)
    ]
    out_path = REPO_ROOT / args.output
    _atomic_write_json(out_path, records)
    logger.info("Done. Wrote %d stable facts to %s", len(records), out_path)
    if separation >= args.separation_max:
        logger.warning("Output written, but the separation gate FAILED — do not "
                        "retrain base_qm on this set until it passes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
