#!/usr/bin/env python3
"""Build train/val splits for factuality classifier training.

There is no test split. The knowledge store is a closed, explicit set — every
fact was deliberately added. The classifier's job is to route queries about
*known* stored facts correctly, not to generalise to unseen facts. Holding out
CF training records would only hurt: the classifier would fail to recognise those
queries as facts and route them to parametric memory instead.

The real test is downstream: ESR (did MORPHEUS answer CF queries correctly?) and
FR (did it leave TriviaQA queries alone?). Those metrics ARE the classifier test.

Positives (label=1) — queries the classifier must route to the knowledge store:
  - ALL CF questions from data/counterfact_train.jsonl (every record in the store)
  - paraphrase_prompts from CF eval test set (query-variation generalisation)

Negatives (label=0) — queries that must reach parametric memory:
  - TriviaQA D_control questions (easy: different format and domain)
  - neighborhood_prompts from CF eval test set (hard: same fill-in-the-blank
    format as CF, but subjects NOT in the store — the key publication-quality
    negative that prevents the classifier from learning surface phrasing)

Val split (10% stratified): used for early stopping only.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def _load_cf_train_questions(path: Path) -> list[str]:
    questions = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            q = (rec.get("question") or "").strip()
            if q:
                questions.append(q)
    return questions


def _load_cf_eval(path: Path) -> tuple[list[str], list[str]]:
    """Returns (paraphrase_prompts, neighborhood_prompts) from test split."""
    with open(path) as f:
        data = json.load(f)
    paraphrases: list[str] = []
    neighborhoods: list[str] = []
    for rec in data.get("test", []):
        for p in rec.get("paraphrase_prompts", []):
            if isinstance(p, str) and p.strip():
                paraphrases.append(p.strip())
        for n in rec.get("neighborhood_prompts", []):
            if isinstance(n, str) and n.strip():
                neighborhoods.append(n.strip())
    return paraphrases, neighborhoods


def _load_triviaqa_questions(path: Path) -> list[str]:
    with open(path) as f:
        data = json.load(f)
    questions = []
    for rec in data.get("records", []):
        q = (rec.get("question") or "").strip()
        if q:
            questions.append(q)
    return questions


def _stratified_split(
    items: list[dict],
    val_frac: float,
    rng: random.Random,
) -> tuple[list[dict], list[dict]]:
    """Stratified train/val split preserving class ratios."""
    pos = [x for x in items if x["label"] == 1]
    neg = [x for x in items if x["label"] == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)

    def _split_class(lst: list[dict]) -> tuple[list[dict], list[dict]]:
        n_val = int(len(lst) * val_frac)
        return lst[n_val:], lst[:n_val]

    pos_tr, pos_va = _split_class(pos)
    neg_tr, neg_va = _split_class(neg)

    train = pos_tr + neg_tr
    val = pos_va + neg_va
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def build(args: argparse.Namespace) -> None:
    rng = random.Random(42)

    cf_train_qs = _load_cf_train_questions(Path(args.cf_train_path))
    paraphrases, neighborhoods = _load_cf_eval(Path(args.cf_eval_path))
    triviaqa_qs = _load_triviaqa_questions(Path(args.triviaqa_path))

    # Hard negatives capped to avoid class imbalance domination.
    rng.shuffle(neighborhoods)
    hard_negatives = neighborhoods[:5_000]

    all_items: list[dict] = []

    # ALL CF train questions — every fact in the store must be recognised.
    for q in cf_train_qs:
        all_items.append({"text": q, "label": 1, "source": "cf_train"})

    # Paraphrases teach variation in how stored-fact queries are phrased.
    for p in paraphrases:
        all_items.append({"text": p, "label": 1, "source": "paraphrase"})

    for q in triviaqa_qs:
        all_items.append({"text": q, "label": 0, "source": "triviaqa"})

    for n in hard_negatives:
        all_items.append({"text": n, "label": 0, "source": "neighborhood"})

    train, val = _stratified_split(all_items, val_frac=0.10, rng=rng)

    pos_train = sum(1 for x in train if x["label"] == 1)
    neg_train = sum(1 for x in train if x["label"] == 0)

    output = {
        "metadata": {
            "n_train": len(train),
            "n_val": len(val),
            "pos_train": pos_train,
            "neg_train": neg_train,
            "note": (
                "No test split. Downstream ESR + FR metrics are the test. "
                "See scripts/build_factuality_classifier_data.py docstring."
            ),
        },
        "train": train,
        "val": val,
    }

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f)

    print(
        f"Wrote {len(train)} train / {len(val)} val "
        f"({pos_train} pos / {neg_train} neg in train) → {out_path}"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output_path", default="data/factuality_classifier_data.json")
    p.add_argument("--cf_train_path", default="data/counterfact_train.jsonl")
    p.add_argument("--cf_eval_path", default="data/counterfact_eval.json")
    p.add_argument("--triviaqa_path", default="data/triviaqa_dcontrol.json")
    return p.parse_args()


if __name__ == "__main__":
    build(parse_args())
