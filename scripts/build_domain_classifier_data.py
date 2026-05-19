#!/usr/bin/env python3
"""Build train/val splits for the 3-class domain classifier (Phase 4).

The domain classifier is the Stage-1 gate of the two-stage router added to
close NF-1 (SQA ``routing_acc=0``). Its job is to predict, from the query
text alone, which knowledge family the query belongs to so Stage-2 can
restrict centroid scoring to the matching adapter pool:

  Class            | Stage-2 candidate set
  -----------------+----------------------------------------------------
  cf               | patch_cf_relfam_{0..5}
  sqa              | base_v1, patch_temp_2019_plus, patch_geo_*
  ood_trivia       | <NONE>  (frozen base, short-circuit return)

Class sources (all queries — no labels other than the class itself):

* cf (5,000):  uniform sample of ``question`` from
  ``data/counterfact_train.jsonl`` (the same training corpus the cluster
  adapters see, so the classifier is calibrated against the actual
  populations Stage 2 will route).

* sqa (5,000): ``edited_question`` from SituatedQA temporal+geo train
  splits, fetched live from the GitHub raw URLs that
  ``src/data/loader.py`` already declares in ``GITHUB_DATA_FILES``.
  ``edited_question`` is the right field — it is what reaches the LLM
  at eval time after `_build_prompt` (the un-edited ``question`` lacks
  the temporal/geo trigger that distinguishes SQA from a generic factoid).

* ood_trivia (5,000): TriviaQA rc.nocontext train, filtered to exclude
  every ``question_id`` already used by ``data/triviaqa_dcontrol.json``
  and ``data/triviaqa_dcalibration.json``. Reusing those IDs would leak
  D_control through the classifier's training signal and re-introduce
  the kind of evaluation contamination the disjoint-slice exposé
  paragraph rules out (see ``scripts/build_triviaqa_dcontrol.py``
  ``--exclude_path`` rationale).

Output schema mirrors ``data/factuality_classifier_data.json``:

    {
      "metadata": {
        "n_train": N, "n_val": M,
        "class_counts_train": {"cf": ..., "sqa": ..., "ood_trivia": ...},
        "labels": ["cf", "sqa", "ood_trivia"],
      },
      "train": [{"text": ..., "label": <int>, "class": "cf|sqa|ood_trivia",
                 "source": "..."}, ...],
      "val":   [...]
    }

``label`` is the integer index into ``metadata.labels`` so the trainer
can use ``CrossEntropyLoss`` directly.

Author: Leon Wagner
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import urllib.request
from pathlib import Path

# Class label ordering — index = integer label used by CrossEntropyLoss.
# Must stay in sync with src/routing/domain_classifier.py CLASS_LABELS.
CLASS_LABELS: list[str] = ["cf", "sqa", "qm", "ood_trivia"]

GITHUB_SQA_URLS: list[tuple[str, str]] = [
    (
        "temp_train",
        "https://raw.githubusercontent.com/mikejqzhang/SituatedQA/master/"
        "data/qa_data/temp.train.jsonl",
    ),
    (
        "geo_train",
        "https://raw.githubusercontent.com/mikejqzhang/SituatedQA/master/"
        "data/qa_data/geo.train.jsonl",
    ),
]


def _load_qm_questions(path: Path) -> list[str]:
    """Return user questions from a QM training JSONL (chat-message format)."""
    questions: list[str] = []
    with path.open() as f:
        for line in f:
            rec = json.loads(line)
            messages = rec.get("messages", [])
            if messages:
                q = (messages[0].get("content") or "").strip()
                if q:
                    questions.append(q)
    return questions


def _load_cf_questions(path: Path) -> list[str]:
    questions: list[str] = []
    with path.open() as f:
        for line in f:
            rec = json.loads(line)
            q = (rec.get("question") or "").strip()
            if q:
                questions.append(q)
    return questions


def _load_sqa_questions(cache_dir: Path) -> list[tuple[str, str]]:
    """Return ``(source_tag, edited_question)`` pairs from temp+geo train.

    Files are downloaded once into ``cache_dir`` and reused on subsequent
    runs (the URLs are large enough to be worth caching, small enough
    that on-disk size is not a concern).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    questions: list[tuple[str, str]] = []
    for tag, url in GITHUB_SQA_URLS:
        local_path = cache_dir / f"{tag}.jsonl"
        if not local_path.exists():
            print(f"  Downloading {tag} from {url} ...", flush=True)
            urllib.request.urlretrieve(url, local_path)
        with local_path.open() as f:
            for line in f:
                rec = json.loads(line)
                q = (rec.get("edited_question") or rec.get("question") or "").strip()
                if q:
                    questions.append((tag, q))
    return questions


def _load_excluded_triviaqa_ids(*paths: Path) -> set[str]:
    excluded: set[str] = set()
    for path in paths:
        if not path.exists():
            print(f"  WARN: exclusion file missing, skipping: {path}", file=sys.stderr)
            continue
        with path.open() as f:
            data = json.load(f)
        records = data.get("records", data) if isinstance(data, dict) else data
        for r in records:
            qid = r.get("question_id")
            if qid:
                excluded.add(str(qid))
    return excluded


def _load_triviaqa_questions(
    target: int,
    excluded_ids: set[str],
    rng: random.Random,
) -> list[str]:
    """Sample ``target`` TriviaQA train questions excluding the given IDs."""
    from datasets import load_dataset

    print(f"  Loading TriviaQA rc.nocontext train ...", flush=True)
    tqa = load_dataset("trivia_qa", "rc.nocontext", split="train")
    print(f"    {len(tqa):,} questions available", flush=True)

    candidate_indices = list(range(len(tqa)))
    rng.shuffle(candidate_indices)

    questions: list[str] = []
    for idx in candidate_indices:
        rec = tqa[idx]
        qid = str(rec.get("question_id") or "")
        if qid in excluded_ids:
            continue
        q = (rec.get("question") or "").strip()
        if q:
            questions.append(q)
        if len(questions) >= target:
            break

    if len(questions) < target:
        print(
            f"  WARN: only collected {len(questions):,} / {target:,} TriviaQA "
            f"questions after exclusions",
            file=sys.stderr,
        )
    return questions


def _stratified_split(
    items: list[dict],
    val_frac: float,
    rng: random.Random,
) -> tuple[list[dict], list[dict]]:
    """Stratified train/val split preserving class ratios."""
    by_class: dict[int, list[dict]] = {i: [] for i in range(len(CLASS_LABELS))}
    for x in items:
        by_class[x["label"]].append(x)

    train: list[dict] = []
    val: list[dict] = []
    for lst in by_class.values():
        rng.shuffle(lst)
        n_val = int(len(lst) * val_frac)
        val.extend(lst[:n_val])
        train.extend(lst[n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def build(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)

    cf_questions = _load_cf_questions(Path(args.cf_train_path))
    print(f"  CF train: {len(cf_questions):,} questions available")
    rng.shuffle(cf_questions)
    cf_sample = cf_questions[: args.n_per_class]

    sqa_pairs = _load_sqa_questions(Path(args.sqa_cache_dir))
    print(f"  SQA train: {len(sqa_pairs):,} questions available")
    rng.shuffle(sqa_pairs)
    sqa_sample = sqa_pairs[: args.n_per_class]

    qm_questions = _load_qm_questions(Path(args.qm_train_path))
    print(f"  QM train: {len(qm_questions):,} questions available")
    rng.shuffle(qm_questions)
    qm_sample = qm_questions[: args.qm_n_samples]
    print(f"  QM using: {len(qm_sample):,} questions (--qm_n_samples cap)")

    excluded_ids = _load_excluded_triviaqa_ids(
        Path(args.dcontrol_path),
        Path(args.dcalibration_path),
    )
    print(f"  TriviaQA exclusions: {len(excluded_ids):,} IDs from D_control + D_calibration")
    trivia_sample = _load_triviaqa_questions(args.n_per_class, excluded_ids, rng)

    items: list[dict] = []
    for q in cf_sample:
        items.append({"text": q, "label": CLASS_LABELS.index("cf"),
                      "class": "cf", "source": "counterfact_train"})
    for tag, q in sqa_sample:
        items.append({"text": q, "label": CLASS_LABELS.index("sqa"),
                      "class": "sqa", "source": f"sqa_{tag}"})
    for q in qm_sample:
        items.append({"text": q, "label": CLASS_LABELS.index("qm"),
                      "class": "qm", "source": "qm_train"})
    for q in trivia_sample:
        items.append({"text": q, "label": CLASS_LABELS.index("ood_trivia"),
                      "class": "ood_trivia", "source": "triviaqa_train"})

    train, val = _stratified_split(items, val_frac=args.val_frac, rng=rng)

    class_counts_train: dict[str, int] = {c: 0 for c in CLASS_LABELS}
    for x in train:
        class_counts_train[x["class"]] += 1

    output = {
        "metadata": {
            "labels": CLASS_LABELS,
            "n_train": len(train),
            "n_val": len(val),
            "class_counts_train": class_counts_train,
            "n_per_class_target": args.n_per_class,
            "val_frac": args.val_frac,
            "seed": args.seed,
            "exclusions": {
                "dcontrol_path": str(args.dcontrol_path),
                "dcalibration_path": str(args.dcalibration_path),
                "n_excluded_triviaqa_ids": len(excluded_ids),
            },
            "note": (
                "4-class Stage-1 router gate (cf/sqa/qm/ood_trivia). "
                "QM class uses all available questions (~500, imbalanced vs "
                "cf/sqa/ood_trivia at 5000). Downstream test = SQA/QM "
                "routing_acc + TriviaQA D_control FR. See "
                "scripts/build_domain_classifier_data.py."
            ),
        },
        "train": train,
        "val": val,
    }

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(output, f)

    print()
    print(f"Wrote {len(train):,} train / {len(val):,} val → {out_path}")
    print(f"  Train class counts: {class_counts_train}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output_path", default="data/domain_classifier_data.json")
    p.add_argument("--cf_train_path", default="data/counterfact_train.jsonl")
    p.add_argument(
        "--sqa_cache_dir",
        default="data/sqa_train_cache",
        help="Where to cache the GitHub-hosted SituatedQA train JSONLs.",
    )
    p.add_argument("--dcontrol_path", default="data/triviaqa_dcontrol.json")
    p.add_argument("--dcalibration_path", default="data/triviaqa_dcalibration.json")
    p.add_argument("--qm_train_path", default="data/qm_train.jsonl",
                   help="QM training JSONL (chat-message format); used for the 'qm' class.")
    p.add_argument("--qm_n_samples", type=int, default=500,
                   help="Max QM questions to use (all available ~500; "
                        "separate from --n_per_class to allow explicit imbalance control).")
    p.add_argument("--n_per_class", type=int, default=5000)
    p.add_argument("--val_frac", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    build(parse_args())
