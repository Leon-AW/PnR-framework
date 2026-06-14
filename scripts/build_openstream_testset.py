#!/usr/bin/env python3
"""Build the held-out open-stream test set for the routing stress test.

Motivation
----------
The Stage-1 domain classifier (``checkpoints/domain_classifier``) is trained on
and evaluated against the same four-class partition ``{cf, sqa, qm, ood_trivia}``.
Every query in the thesis evaluation comes from one of those four distributions,
so the architecture's stability guarantee (FR ≈ 0) has never been tested against
queries from genuinely novel domains — the motivating enterprise scenario. This
script collects 1,000 queries from five domains the classifier has *never* seen,
so ``scripts/run_openstream_stress.py`` can measure how often such queries leak
out of the frozen base into an expert adapter.

Domains (200 each = 1,000 total; balanced to match the D_eval n=1,000 convention):

  domain   | HF dataset                          | config                | split      | field
  ---------+-------------------------------------+-----------------------+------------+---------
  medical  | qiaojin/PubMedQA                    | pqa_labeled           | train      | question
  legal    | nguha/legalbench                    | consumer_contracts_qa | test       | question
  finance  | virattt/financial-qa-10K            | —                     | train      | question
  science  | allenai/sciq                        | —                     | train      | question
  nq_open  | google-research-datasets/nq_open    | —                     | validation | question

``nq_open`` is deliberately trivia-adjacent (open factoids). It probes the gate
boundary, but it can legitimately free-ride on ``ood_trivia`` — the stress-test
runner therefore reports every headline number both overall and
overall-excluding-nq_open.

Output schema (mirrors the project's data-builder conventions):

    {
      "metadata": {"n_total":, "per_domain":, "sources":, "seed":, "note":},
      "records": [{"text":, "domain":, "source":, "id":}]
    }

Author: Leon Wagner
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# domain -> (hf_name, config, split, [candidate question fields])
# Listed candidate fields are tried in order; the first present is used.
DOMAIN_SOURCES: dict[str, tuple[str, str | None, str, list[str]]] = {
    "medical": ("qiaojin/PubMedQA", "pqa_labeled", "train", ["question"]),
    "legal": ("nguha/legalbench", "consumer_contracts_qa", "test", ["question"]),
    "finance": ("virattt/financial-qa-10K", None, "train", ["question"]),
    "science": ("allenai/sciq", None, "train", ["question"]),
    "nq_open": ("google-research-datasets/nq_open", None, "validation", ["question"]),
}


def _load_domain_questions(
    domain: str,
    n: int,
    rng: random.Random,
) -> tuple[list[str], str]:
    """Load ``n`` deduped questions for ``domain``; return (questions, source_tag)."""
    from datasets import load_dataset

    name, config, split, fields = DOMAIN_SOURCES[domain]
    source_tag = f"{name}:{config}:{split}" if config else f"{name}:{split}"
    print(f"  [{domain}] loading {source_tag} ...", flush=True)
    ds = load_dataset(name, config, split=split) if config else load_dataset(name, split=split)

    field = next((f for f in fields if f in ds.column_names), None)
    if field is None:
        raise KeyError(
            f"None of {fields} present in {source_tag} columns {ds.column_names}"
        )

    # Dedup while keeping insertion order, then shuffle deterministically.
    seen: set[str] = set()
    questions: list[str] = []
    for q in ds[field]:
        q = (q or "").strip()
        if q and q not in seen:
            seen.add(q)
            questions.append(q)

    rng.shuffle(questions)
    if len(questions) < n:
        print(
            f"  WARN: [{domain}] only {len(questions)} unique questions available "
            f"(< {n} requested) — using all of them",
            flush=True,
        )
    return questions[:n], source_tag


def build(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)

    records: list[dict] = []
    per_domain: dict[str, int] = {}
    sources: dict[str, str] = {}

    for domain in DOMAIN_SOURCES:
        questions, source_tag = _load_domain_questions(domain, args.n_per_domain, rng)
        sources[domain] = source_tag
        per_domain[domain] = len(questions)
        prefix = domain[:4]
        for i, q in enumerate(questions):
            records.append(
                {"text": q, "domain": domain, "source": source_tag, "id": f"{prefix}_{i}"}
            )

    rng.shuffle(records)

    output = {
        "metadata": {
            "n_total": len(records),
            "per_domain": per_domain,
            "sources": sources,
            "seed": args.seed,
            "n_per_domain_target": args.n_per_domain,
            "trivia_adjacent_domain": "nq_open",
            "note": (
                "Held-out open-stream test set for the routing stress test. Five "
                "domains the Stage-1 domain classifier has never seen "
                "(cf/sqa/qm/ood_trivia). nq_open is trivia-adjacent and may "
                "legitimately route to the frozen base via ood_trivia — report "
                "leak rate both overall and excluding nq_open. See "
                "scripts/run_openstream_stress.py."
            ),
        },
        "records": records,
    }

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(output, f, indent=2)

    print()
    print(f"Wrote {len(records):,} records → {out_path}")
    print(f"  Per-domain counts: {per_domain}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output_path", default="data/openstream_heldout.json")
    p.add_argument("--n_per_domain", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    build(parse_args())
