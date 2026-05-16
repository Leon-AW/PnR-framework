#!/usr/bin/env python3
"""
Build AIT QM Conflict Pairs (D_conflict, semi-synthetic)
=========================================================

Constructs the conflict half of the AIT QM D_eval. The original old-edition
QM corpus (data/Archiv) was lost, so instead of mining contradictions from two
independently-generated QA sets, we build *controlled* conflict pairs:

  - answer_new : a REAL, validated current fact from data/DE/dataset_final.json
                 -> trains the Knowledge Patch; the target the system must output
  - answer_old : a CONSTRUCTED earlier-revision answer that contradicts it on
                 exactly one attribute -> trains the Base Adapter

This mirrors how CounterFact is built (a real fact + a counterfactual): the
*evaluated* target stays real and grounded; only the prior version is synthetic.
The methodology is semi-synthetic and must be documented as such in the thesis
methodology section.

Pipeline (one Gemma-4 call each for generation + verification):
  1. Filter data/DE/dataset_final.json to factual, answerable, short-enough QA
     (categories A/B/D/F/I by default; complexity <= 2).
  2. For each candidate: ask the Gemma-4 judge model to produce a minimal
     single-attribute contradicting "old" answer (JSON output).
  3. Verify with the same model that old vs. new genuinely contradict.
  4. Keep verified pairs until --target is reached.

Output: data/qm_conflict_pairs.json (plain JSON list). This is the D_conflict
source. The final qm_deval.json (D_conflict union D_control) is assembled in a
later step once data/triviaqa_dcontrol.json has been copied onto this server.

Usage:
    # GPU required (loads Gemma-4-26B int4). Submit via SLURM on this server.
    python scripts/build_qm_conflict_pairs.py --dry_run          # filter stats only
    python scripts/build_qm_conflict_pairs.py --target 500
    python scripts/build_qm_conflict_pairs.py --target 500 --resume

Author: Leon Wagner
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.external_judge import JUDGE_MODEL_ID, ExternalJudge

logger = logging.getLogger("build_qm_conflict_pairs")

QM_CONFLICT_PROMPT_VERSION = "qm-conflict-v1"

# Factual intention categories where a single-attribute contradiction is clean.
# Excludes C (multi-step how-to), G/H (summaries/synthesis), N (hard negatives).
DEFAULT_CATEGORIES = "A,B,D,F,I"

_LANGUAGE_NAMES = {"de": "German", "en": "English"}

QM_EDIT_PROMPT = """You are helping construct a knowledge-editing benchmark for an enterprise Quality Management (QM) system.

You are given a CURRENT, factually correct question-answer pair extracted from a company's QM documentation. Imagine the PREVIOUS revision of that document -- the version in force before the most recent update -- and write what the answer USED TO say.

Strict requirements:
1. Change exactly ONE concrete, checkable attribute: a date, a number, a quantity, a threshold/limit, a responsible role or department, a location, a standard/norm identifier, or a named requirement.
2. The OLD answer must DIRECTLY CONTRADICT the current answer on that single attribute -- a reader comparing the two must see a clear factual disagreement.
3. Everything else stays the same: same meaning, same structure, same length, same terminology, same formatting. Only the one attribute differs.
4. The old value must be PLAUSIBLE for an earlier revision of a real QM document (a realistic earlier date, a realistic earlier threshold, a real-sounding role). Never absurd or obviously fake.
5. Write the OLD answer in <<LANGUAGE>>, the same language as the current answer.
6. Do NOT change a value, name, or entity that already appears in the QUESTION below. The attribute you change must occur only in the answer -- otherwise the question itself gives away the current answer.

Question:
<<QUESTION>>

Current (correct) answer:
<<ANSWER>>

Respond with ONLY a single JSON object and nothing else:
{"changed_attribute": "<short description of the attribute you changed>", "new_value": "<the value stated in the current answer>", "old_value": "<the contradicting earlier value>", "old_answer": "<the full earlier-revision answer>"}"""

QM_VERIFY_PROMPT = """You are checking a knowledge-editing benchmark pair.

Two answers to the SAME question are given: a CURRENT answer and an EARLIER-REVISION answer. They are supposed to DIRECTLY CONTRADICT each other on exactly one factual point.

Question:
<<QUESTION>>

Current answer:
<<ANSWER_NEW>>

Earlier-revision answer:
<<ANSWER_OLD>>

Decide: do the two answers state DIFFERENT, MUTUALLY EXCLUSIVE values for the same factual point, such that they cannot both be true?

Respond with EXACTLY one word: CONTRADICT or COMPATIBLE. No other text."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", default="data/DE/dataset_final.json")
    parser.add_argument("--output", default="data/qm_conflict_pairs.json")
    parser.add_argument("--target", type=int, default=500,
                        help="Number of verified conflict pairs to collect")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--categories", default=DEFAULT_CATEGORIES,
                        help="Comma-separated intention_category codes to keep")
    parser.add_argument("--max_complexity", type=int, default=2)
    parser.add_argument("--min_answer_chars", type=int, default=40)
    parser.add_argument("--max_answer_chars", type=int, default=1200,
                        help="Skip long multi-fact answers; bad for single-attribute edits")
    parser.add_argument("--gen_max_new_tokens", type=int, default=768)
    parser.add_argument("--verify_max_new_tokens", type=int, default=8)
    parser.add_argument("--quantization", choices=["int4", "int8", "none"], default="int4")
    parser.add_argument("--model_id", default=JUDGE_MODEL_ID)
    parser.add_argument("--checkpoint_every", type=int, default=25)
    parser.add_argument("--limit_candidates", type=int, default=None,
                        help="Cap candidate pool (smoke testing)")
    parser.add_argument("--resume", action="store_true",
                        help="Append to an existing output file, skipping done sources")
    parser.add_argument("--dry_run", action="store_true",
                        help="Report candidate-pool stats only; do not load the model")
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


def _extract_json(text: str) -> dict | None:
    """Pull the first JSON object out of a model response (tolerates fences/prose)."""
    t = (text or "").strip()
    t = re.sub(r"^```(?:json)?", "", t).strip()
    t = re.sub(r"```$", "", t).strip()
    match = re.search(r"\{.*\}", t, re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _verify_verdict(raw: str) -> bool | None:
    """True = contradicts (valid pair), False = compatible, None = off-spec."""
    upper = (raw or "").upper()
    has_c = bool(re.search(r"\bCONTRADICT\b", upper))
    has_k = bool(re.search(r"\bCOMPATIBLE\b", upper))
    if has_c and not has_k:
        return True
    if has_k and not has_c:
        return False
    return None


def load_candidates(args: argparse.Namespace) -> list[dict]:
    """Filter data/DE QA to factual, answerable, short-enough records."""
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


def build_edit_prompt(rec: dict) -> str:
    lang = _LANGUAGE_NAMES.get(rec.get("language"), "the same language as the question")
    return (
        QM_EDIT_PROMPT
        .replace("<<LANGUAGE>>", lang)
        .replace("<<QUESTION>>", (rec.get("question") or "").strip())
        .replace("<<ANSWER>>", (rec.get("answer") or "").strip())
    )


def build_verify_prompt(question: str, answer_new: str, answer_old: str) -> str:
    return (
        QM_VERIFY_PROMPT
        .replace("<<QUESTION>>", question)
        .replace("<<ANSWER_NEW>>", answer_new)
        .replace("<<ANSWER_OLD>>", answer_old)
    )


def make_pair(rec: dict, edit: dict, verify_raw: str, pair_id: str,
              model_id: str) -> dict:
    """Assemble one output record. answer_new is real; answer_old is synthetic."""
    return {
        "id": pair_id,
        "question": (rec.get("question") or "").strip(),
        "answer_new": (rec.get("answer") or "").strip(),
        "answer_old": str(edit["old_answer"]).strip(),
        "changed_attribute": str(edit.get("changed_attribute") or "").strip(),
        "old_value": str(edit.get("old_value") or "").strip(),
        "new_value": str(edit.get("new_value") or "").strip(),
        "language": rec.get("language"),
        "intention_category": rec.get("intention_category"),
        "complexity_level": rec.get("complexity_level"),
        "source_file": rec.get("source_file"),
        "file_path": rec.get("file_path"),
        "evidence_snippet": rec.get("evidence_snippet"),
        "document_context": rec.get("document_context"),
        "split_origin": "qm_conflict",
        "verification": {"verdict": "CONTRADICT", "raw": verify_raw},
        "generator": {
            "model": model_id,
            "prompt_version": QM_CONFLICT_PROMPT_VERSION,
            "decoding": "greedy",
        },
    }


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    candidates = load_candidates(args)
    logger.info("Candidate pool: %d records", len(candidates))
    logger.info("  by category:   %s",
                dict(sorted(Counter(c["intention_category"] for c in candidates).items())))
    logger.info("  by language:   %s",
                dict(sorted(Counter(c.get("language") for c in candidates).items())))
    logger.info("  by complexity: %s",
                dict(sorted(Counter(c.get("complexity_level") for c in candidates).items())))

    import random
    random.Random(args.seed).shuffle(candidates)
    if args.limit_candidates is not None:
        candidates = candidates[: args.limit_candidates]

    if args.dry_run:
        logger.info("--dry_run: pool of %d candidates for target=%d. No model loaded.",
                    len(candidates), args.target)
        if len(candidates) < args.target:
            logger.warning("Pool smaller than target -- widen --categories or "
                            "raise --max_complexity/--max_answer_chars.")
        return 0

    out_path = REPO_ROOT / args.output
    pairs: list[dict] = []
    # source key = (source_file, question); rejected candidates are NOT recorded,
    # so --resume re-attempts them (deterministic, so wasteful but correct). A
    # future version could persist an attempted-log to skip rejects on resume.
    seen: set[tuple] = set()
    if args.resume and out_path.exists():
        with out_path.open("r", encoding="utf-8") as f:
            pairs = json.load(f)
        seen = {(p.get("source_file"), p.get("question")) for p in pairs}
        logger.info("Resume: loaded %d existing pairs", len(pairs))

    # Pin the whole 4-bit model to the single (MIG) GPU. device_map="auto"
    # spuriously offloads to CPU on a 48 GB MIG slice -> bnb 4-bit rejects it.
    judge = ExternalJudge(
        model_id=args.model_id,
        quantization=args.quantization,
        device_map={"": 0},
    )
    judge.load()

    stats = Counter()
    progress = tqdm(candidates, desc="QM conflict pairs", unit="cand")
    for rec in progress:
        if len(pairs) >= args.target:
            break
        key = (rec.get("source_file"), (rec.get("question") or "").strip())
        if key in seen:
            stats["skip_already_done"] += 1
            continue
        stats["attempted"] += 1

        edit = _extract_json(judge.generate(build_edit_prompt(rec),
                                            max_new_tokens=args.gen_max_new_tokens))
        if not edit or not str(edit.get("old_answer") or "").strip():
            stats["reject_bad_json"] += 1
            continue

        question = (rec.get("question") or "").strip()
        answer_new = (rec.get("answer") or "").strip()
        answer_old = str(edit["old_answer"]).strip()
        old_value = str(edit.get("old_value") or "").strip()
        new_value = str(edit.get("new_value") or "").strip()
        if (len(answer_old) < args.min_answer_chars
                or answer_old == answer_new
                or not old_value
                or old_value.lower() == new_value.lower()):
            stats["reject_no_change"] += 1
            continue
        # Question-leak: if the new value already appears in the question, the
        # question gives away the answer and the old answer is non-responsive.
        if new_value and new_value.lower() in question.lower():
            stats["reject_value_in_question"] += 1
            continue

        verify_raw = judge.generate(
            build_verify_prompt(question, answer_new, answer_old),
            max_new_tokens=args.verify_max_new_tokens,
        )
        verdict = _verify_verdict(verify_raw)
        if verdict is None:
            stats["reject_verify_offspec"] += 1
            continue
        if verdict is False:
            stats["reject_verify_compatible"] += 1
            continue

        pair_id = f"qm_conflict_{len(pairs) + 1:05d}"
        pairs.append(make_pair(rec, edit, verify_raw, pair_id, args.model_id))
        seen.add(key)
        stats["accepted"] += 1
        progress.set_postfix(accepted=len(pairs))

        if len(pairs) % args.checkpoint_every == 0:
            _atomic_write_json(out_path, pairs)

    _atomic_write_json(out_path, pairs)

    logger.info("Done. Wrote %d conflict pairs to %s", len(pairs), out_path)
    for reason, count in sorted(stats.items()):
        logger.info("  %-26s %d", reason, count)
    if stats["attempted"]:
        logger.info("  yield: %.1f%%", 100.0 * stats["accepted"] / stats["attempted"])
    if len(pairs) < args.target:
        logger.warning("Collected %d < target %d -- widen --categories or rerun "
                        "with --resume on a larger pool.", len(pairs), args.target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
