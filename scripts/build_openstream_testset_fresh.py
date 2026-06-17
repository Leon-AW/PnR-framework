#!/usr/bin/env python3
"""Build the FRESH held-out OOD test set for the open-stream leak *mitigation*.

Why a second test set
---------------------
``scripts/build_openstream_testset.py`` built the *diagnosis* set (5 domains:
PubMedQA, LegalBench, financial-QA-10K, SciQ, NQ) that surfaced the ~31% routing
leak. That set has done its job — it motivated the Mahalanobis open-set gate.

If we now calibrated and reported the mitigation on those *same* 5 domains we
would be tuning on the test set (adaptive overfitting). So this script builds a
genuinely **fresh** OOD test set from domains the Stage-1 classifier has never
seen AND that are disjoint from the 5 diagnosis domains. The open-set detector
is fitted on in-domain training data and its threshold is calibrated on a held-out
in-domain split (see ``tasks/todo.md`` validity design) — it never touches these
records. A leak reduction measured here is therefore valid: it holds on domains
used in neither fitting nor calibration.

Domains (disjoint from the 4 training classes {cf, sqa, qm, ood_trivia} AND from
the 5 diagnosis domains; modern, well-known, parquet-native benchmarks chosen to
span the leak-difficulty range, as the diagnosis set did — SciQ 3% ... finance 50%):

  domain      | HF dataset              | config   | split | field    | expected leak
  ------------+-------------------------+----------+-------+----------+--------------
  professional| TIGER-Lab/MMLU-Pro      | —        | test  | question | high (pro jargon, multi-field exam)
  math        | HuggingFaceH4/MATH-500  | —        | test  | problem  | medium (competition math)
  german_rc   | facebook/belebele       | deu_Latn | test  | question | ? (DE boundary)

``german_rc`` is the deliberate bilingual probe: the ``qm`` in-domain class is
DE/EN, but all 5 diagnosis domains were English, so the German OOD boundary is so
far untested. Include it to check the detector does not over-trust the German
``qm`` manifold. (Older script-based sets like QASPER / GermanQuAD are not used —
``datasets`` 4.x dropped dataset-script loading; these three are parquet-native.)

Output schema mirrors ``build_openstream_testset.py`` exactly so
``run_openstream_stress.py`` consumes it unchanged:

    {
      "metadata": {"n_total":, "per_domain":, "sources":, "seed":, "note":},
      "records": [{"text":, "domain":, "source":, "id":}]
    }

Run with the project env:
    /vol/fob-vol1/mi23/wagnerql/.conda/envs/pnr/bin/python \
        scripts/build_openstream_testset_fresh.py

Author: Leon Wagner
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# domain -> (hf_name, config, split, [candidate flat question fields])
# Listed candidate fields are tried in order; the first present is used.
DOMAIN_SOURCES: dict[str, tuple[str, str | None, str, list[str]]] = {
    "professional": ("TIGER-Lab/MMLU-Pro", None, "test", ["question"]),
    "math": ("HuggingFaceH4/MATH-500", None, "test", ["problem", "question"]),
    "german_rc": ("facebook/belebele", "deu_Latn", "test", ["question"]),
}

# domain -> callable(dataset) -> list[str]. Overrides the flat-field path for any
# dataset with a nested schema. All current domains are flat, so this is empty.
EXTRACTORS: dict[str, "callable"] = {}


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

    if domain in EXTRACTORS:
        raw = EXTRACTORS[domain](ds)
    else:
        field = next((f for f in fields if f in ds.column_names), None)
        if field is None:
            raise KeyError(
                f"None of {fields} present in {source_tag} columns {ds.column_names}"
            )
        raw = ds[field]

    # Dedup while keeping insertion order, then shuffle deterministically.
    seen: set[str] = set()
    questions: list[str] = []
    for q in raw:
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
    failed: dict[str, str] = {}

    for domain in DOMAIN_SOURCES:
        try:
            questions, source_tag = _load_domain_questions(domain, args.n_per_domain, rng)
        except Exception as e:  # one broken dataset must not kill the whole build
            print(f"  ERROR: [{domain}] failed to load: {type(e).__name__}: {e}", flush=True)
            failed[domain] = f"{type(e).__name__}: {e}"
            continue
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
            "failed": failed,
            "seed": args.seed,
            "n_per_domain_target": args.n_per_domain,
            "purpose": "fresh OOD test set for the open-stream leak MITIGATION",
            "note": (
                "Fresh held-out OOD test set, disjoint from the 4 training classes "
                "{cf,sqa,qm,ood_trivia} AND from the 5 diagnosis domains "
                "(PubMedQA/LegalBench/financial-QA-10K/SciQ/NQ). Used ONLY for the "
                "final leak-reduction measurement; the open-set detector is fitted on "
                "in-domain training data and calibrated on a held-out in-domain split, "
                "neither of which sees these records. german_qa is the DE bilingual "
                "boundary probe. See tasks/todo.md (validity design) and "
                "scripts/run_openstream_stress.py."
            ),
        },
        "records": records,
    }

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print()
    print(f"Wrote {len(records):,} records → {out_path}")
    print(f"  Per-domain counts: {per_domain}")
    if failed:
        print(f"  FAILED domains (skipped): {failed}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output_path", default="data/openstream_test_fresh.json")
    p.add_argument("--n_per_domain", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    build(parse_args())
