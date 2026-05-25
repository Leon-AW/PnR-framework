#!/usr/bin/env python3
"""Seed the MORPHEUS KnowledgeStore from training data.

The graduated-factuality protocol (System 5) is a no-op unless the store
contains records. This script populates a ``knowledge_store/records.json``
from the same sources that trained the LoRA adapters:

- SituatedQA streams (base, temporal, geo_<country>) for each registered
  adapter domain — mirrors ``src/data/loader.py`` streams.
- CounterFact training pairs (``data/counterfact_train.jsonl``) when the
  ``patch_cf_main`` adapter is in play.
- AIT QM SFT chat-message JSONL (``data/qm_train.jsonl`` for the patch
  domain and ``data/qm_train_base.jsonl`` for the base domain) when one
  of the QM adapters is in play. Long-form German answers are stored
  verbatim as ``object_value`` (same no-triple pattern as SituatedQA).

Records are embedded with the same sentence-transformer used by the
PrototypeRouter so that retrieval operates in a consistent similarity
space, and written to the canonical ``store_dir`` so that MORPHEUS picks
them up at inference time (after the ``__init__`` auto-load patch in
``src/morpheus/inference.py``).

The three domain blocks (SQA, CF, QM) can be enabled/disabled independently
via ``--skip_sqa`` / ``--skip_counterfact`` / by omitting the QM paths, so a
domain-specific KS (e.g. ``morpheus_state_qm/``) can be built in isolation.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.morpheus.knowledge_store import KnowledgeRecord, KnowledgeStore
from src.morpheus.config import KnowledgeStoreConfig


logger = logging.getLogger(__name__)


GEO_COUNTRIES: tuple[str, ...] = (
    "India", "Canada", "Australia", "UK", "Nigeria",
    "England", "Pakistan", "California", "France", "Germany",
)

# SituatedQA fields vary slightly: prefer ``edited_question`` (the time/geo-
# situated rewrite that is what adapters saw during training), fall back to
# ``question``. ``answer`` is sometimes a list, sometimes a string.
_QUESTION_KEYS = ("edited_question", "question")
_ANSWER_KEYS = ("answer",)


@dataclass
class Fact:
    """Intermediate representation carried through the pipeline.

    ``lookup_text`` is what gets embedded — it must mirror the form of a
    future user query so that similarity search fires. For both SituatedQA
    and CounterFact that is the natural-language question / completion prompt.

    ``subject``/``predicate``/``object_value`` are stored on the
    ``KnowledgeRecord`` and are what the consolidation engine would
    normally populate. For CounterFact we preserve the real triple
    (e.g. subject="Danielle Darrieux", predicate="P103") so that
    ``search_by_subject`` / future CRUD edits can target a specific
    entity-relation pair. For SituatedQA there is no explicit triple,
    so we store the full question as the subject.
    """

    domain: str
    lookup_text: str
    subject: str
    predicate: str
    object_value: str


def _extract_qa(example: dict) -> tuple[str, str] | None:
    q = None
    for k in _QUESTION_KEYS:
        v = example.get(k)
        if isinstance(v, str) and v.strip():
            q = v.strip()
            break
    if q is None:
        return None
    for k in _ANSWER_KEYS:
        v = example.get(k)
        if isinstance(v, list):
            v = next((x for x in v if isinstance(x, str) and x.strip()), None)
        if isinstance(v, str) and v.strip():
            return q, v.strip()
    return None


def _iter_stream(stream: Iterable, limit: int, skip: int = 0) -> list[dict]:
    """Pull `limit` examples from an iterable dataset, skipping the first `skip`."""
    out: list[dict] = []
    for i, ex in enumerate(stream):
        if i < skip:
            continue
        if not isinstance(ex, dict):
            continue
        out.append(ex)
        if len(out) >= limit:
            break
    return out


def collect_situated_qa(
    n_per_split: int,
    skip_first: int,
    include_splits: set[str],
) -> list[Fact]:
    """Return Facts from SituatedQA (no explicit triple structure)."""
    from src.data.loader import SituatedQAConfig, SituatedQALoader

    # streaming=False is required: the loader caches `_geo_dataset` and
    # `_temp_dataset`, so when streaming=True the first country's filter
    # exhausts the shared HTTP iterator and every subsequent country yields
    # zero examples. Non-streaming loads the full JSONL into memory once
    # (a few MB) and lets each filter view iterate independently.
    loader = SituatedQALoader(SituatedQAConfig(streaming=False))
    facts: list[Fact] = []

    streams: list[tuple[str, Iterable]] = []
    if "base" in include_splits:
        streams.append(("base", loader.get_base_stream()))
    if "temporal" in include_splits:
        streams.append(("temporal", loader.get_temporal_patch_stream()))
    for country in GEO_COUNTRIES:
        domain = f"geo_{country.lower()}"
        if domain not in include_splits:
            continue
        try:
            streams.append((domain, loader.get_geo_patch_stream(country)))
        except Exception as e:
            logger.warning("Skipping geo stream %s: %s", country, e)

    for domain, stream in streams:
        examples = _iter_stream(stream, limit=n_per_split, skip=skip_first)
        n_kept = 0
        for ex in examples:
            qa = _extract_qa(ex)
            if qa is None:
                continue
            q, a = qa
            # SituatedQA is open QA with no canonical (s,p,o) decomposition,
            # so the full question plays the role of subject. Predicate is
            # a display connector read by ``build_override_context``.
            facts.append(Fact(
                domain=domain,
                lookup_text=q,
                subject=q,
                predicate="— answer:",
                object_value=a,
            ))
            n_kept += 1
        logger.info("  situated/%s: kept %d (from %d scanned)", domain, n_kept, len(examples))

    return facts


def collect_qm(path: Path, domain: str, exclude_ids: set[str] | None = None) -> list[Fact]:
    """Return Facts from an AIT QM SFT JSONL (chat-message format).

    QM training records are ``{"id": ..., "messages": [{"role": "user",
    "content": <question>}, {"role": "assistant", "content": <answer>}],
    ...}``. Like SituatedQA, there is no atomic (subject, predicate,
    object) triple — the full question plays the role of subject and the
    assistant message (a long-form German markdown answer) is the
    object_value. The KS bypass mechanism returns ``object_value``
    verbatim, which is exactly what the QM eval expects.

    ``domain`` is the label we want on the resulting KnowledgeRecord
    (e.g. ``"qm_patch"`` for ``qm_train.jsonl`` or ``"qm_base"`` for
    ``qm_train_base.jsonl``).

    ``exclude_ids`` lets the caller drop records whose ``id`` is already
    represented by a more-current source. Critical for QM: qm_train_base
    contains the OLD (pre-edit) answer for conflict items, while
    qm_train contains the NEW (post-edit) answer for the same IDs. Both
    seeded with the same lookup_text → identical embeddings → tie-break
    in search returns one of them arbitrarily. Excluding the conflict
    IDs from the base pass guarantees the KS holds one answer per
    question (conflict→new, stable→base).
    """
    facts: list[Fact] = []
    n_scanned = 0
    n_excluded = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_scanned += 1
            rec = json.loads(line)
            if exclude_ids is not None and rec.get("id") in exclude_ids:
                n_excluded += 1
                continue
            messages = rec.get("messages") or []
            q = next(
                (m.get("content", "").strip() for m in messages if m.get("role") == "user"),
                "",
            )
            a = next(
                (m.get("content", "").strip() for m in messages if m.get("role") == "assistant"),
                "",
            )
            if not q or not a:
                continue
            facts.append(Fact(
                domain=domain,
                lookup_text=q,
                subject=q,
                predicate="— answer:",
                object_value=a,
            ))
    logger.info("  qm/%s: kept %d records (from %d scanned, %d excluded, %s)",
                domain, len(facts), n_scanned, n_excluded, path)
    return facts


def _collect_ids(path: Path) -> set[str]:
    """Return the set of record IDs in a chat-message JSONL file."""
    ids: set[str] = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rid = rec.get("id")
            if rid:
                ids.add(rid)
    return ids


def collect_counterfact(path: Path, limit: int | None) -> list[Fact]:
    """Return Facts from CounterFact preserving the (subject, relation, object) triple.

    The architecture spec (``docs/new architecture to solve continual
    learning.md``) describes System 5 as a store of "discrete facts,
    events, entities, relationships — structured information". CounterFact
    natively provides that structure, so we keep ``subject`` and
    ``relation_id`` intact rather than flattening to question text. This
    also keeps ``KnowledgeStore.search_by_subject`` usable for future
    targeted edits.

    ``lookup_text`` is still the natural-language question so that future
    user queries land in the same embedding neighbourhood.
    """
    facts: list[Fact] = []
    with open(path) as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            rec = json.loads(line)
            q = (rec.get("question") or "").strip()
            a = (rec.get("answer") or "").strip()
            subj = (rec.get("subject") or "").strip()
            rel = (rec.get("relation_id") or "").strip()
            if not q or not a or not subj or not rel:
                continue
            facts.append(Fact(
                domain="counterfact",
                lookup_text=q,
                subject=subj,
                predicate=rel,
                object_value=a,
            ))
    logger.info("  counterfact: kept %d records", len(facts))
    return facts


def embed_facts(
    facts: list[Fact],
    embedding_model: str,
    batch_size: int,
    use_gpu: bool,
) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    logger.info("Loading embedding model: %s", embedding_model)
    device = "cuda" if use_gpu else "cpu"
    enc = SentenceTransformer(embedding_model, device=device)

    # Embed the lookup text (the user-facing question / completion prompt).
    # The factuality-assessment path searches by query embedding, so the
    # stored vector must live in the same space as future queries — even
    # when the stored subject is a structured entity like "Croatia".
    texts = [f.lookup_text for f in facts]
    logger.info("Encoding %d facts (batch=%d, device=%s) ...", len(texts), batch_size, device)
    embs = enc.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    return embs


def build_records(
    facts: list[Fact],
    embeddings: np.ndarray,
) -> list[KnowledgeRecord]:
    now = time.time()
    records: list[KnowledgeRecord] = []
    for i, (fact, emb) in enumerate(zip(facts, embeddings)):
        rid = f"{fact.domain}_{i:07d}"
        # Keep the lookup text on the record so that its original surface
        # form (what was embedded) can be reconstructed for audit /
        # display — especially relevant for CounterFact where subject
        # alone is the structured entity, not the full prompt.
        metadata = {"lookup_text": fact.lookup_text} if fact.lookup_text != fact.subject else {}
        records.append(KnowledgeRecord(
            record_id=rid,
            subject=fact.subject,
            predicate=fact.predicate,
            object_value=fact.object_value,
            source=f"training/{fact.domain}",
            timestamp=now,
            confidence=1.0,
            domain=fact.domain,
            embedding=emb,
            metadata=metadata,
        ))
    return records


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output_dir",
        default="morpheus_state/knowledge_store",
        help="Where to write records.json (must match KnowledgeStoreConfig.store_dir).",
    )
    p.add_argument(
        "--embedding_model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Must match the PrototypeRouter's embedding model.",
    )
    p.add_argument("--n_per_split", type=int, default=500,
                   help="Max records per SituatedQA domain.")
    # skip_first was originally defaulted to 200 to avoid overlap with the
    # first 200 eval samples. Small geo splits (AU/UK/FR/DE ~60–90 rows)
    # got completely skipped, seeding them with zero facts. We set the
    # default to 0: System 5 is *designed* to hold training facts, and
    # those facts are the same data the adapters were trained on.
    p.add_argument("--skip_first", type=int, default=0,
                   help="Skip the first N records per split (default 0).")
    p.add_argument("--splits", nargs="+", default=None,
                   help="Which SituatedQA splits to include (default: all registered).")
    p.add_argument("--counterfact_path", default="data/counterfact_train.jsonl")
    p.add_argument("--counterfact_limit", type=int, default=None,
                   help="Cap on CounterFact records (default: all).")
    p.add_argument("--skip_counterfact", action="store_true")
    p.add_argument("--skip_sqa", action="store_true",
                   help="Skip SituatedQA collection (useful for QM-only builds).")
    p.add_argument("--qm_train_path", default=None,
                   help="Path to qm_train.jsonl (chat-message SFT data for "
                        "patch_qm_current). Records seeded under domain 'qm_patch'.")
    p.add_argument("--qm_train_base_path", default=None,
                   help="Path to qm_train_base.jsonl (chat-message SFT data for "
                        "base_qm). Records seeded under domain 'qm_base'.")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--cpu", action="store_true", help="Force CPU encoding.")
    p.add_argument("--log_level", default="INFO")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")

    if args.splits is None:
        include_splits = {"base", "temporal"} | {f"geo_{c.lower()}" for c in GEO_COUNTRIES}
    else:
        include_splits = set(args.splits)

    facts: list[Fact] = []
    if not args.skip_sqa:
        logger.info("Collecting SituatedQA facts ...")
        facts.extend(collect_situated_qa(
            n_per_split=args.n_per_split,
            skip_first=args.skip_first,
            include_splits=include_splits,
        ))
    else:
        logger.info("Skipping SituatedQA collection (--skip_sqa)")

    if not args.skip_counterfact:
        cf_path = Path(args.counterfact_path)
        if cf_path.exists():
            logger.info("Collecting CounterFact facts from %s ...", cf_path)
            facts.extend(collect_counterfact(cf_path, args.counterfact_limit))
        else:
            logger.warning("CounterFact path %s not found — skipping.", cf_path)

    # The QM patch (qm_train.jsonl) overrides the QM base (qm_train_base.jsonl)
    # for any shared IDs — qm_train_base carries the OLD answer for conflict
    # items, qm_train carries the NEW answer. Seeding both unfiltered would
    # leave both records in the KS with identical embeddings (tie-break
    # arbitrary), causing bypass to occasionally return the old value.
    qm_patch_ids: set[str] = set()
    if args.qm_train_path:
        qm_patch_path = Path(args.qm_train_path)
        if qm_patch_path.exists():
            logger.info("Collecting QM facts from %s (domain=qm_patch) ...", qm_patch_path)
            facts.extend(collect_qm(qm_patch_path, "qm_patch"))
            qm_patch_ids = _collect_ids(qm_patch_path)
        else:
            logger.warning("QM path %s not found — skipping qm_patch.", qm_patch_path)

    if args.qm_train_base_path:
        qm_base_path = Path(args.qm_train_base_path)
        if qm_base_path.exists():
            logger.info(
                "Collecting QM facts from %s (domain=qm_base, excluding %d patch IDs) ...",
                qm_base_path, len(qm_patch_ids),
            )
            facts.extend(collect_qm(qm_base_path, "qm_base", exclude_ids=qm_patch_ids))
        else:
            logger.warning("QM path %s not found — skipping qm_base.", qm_base_path)

    if not facts:
        raise SystemExit("No facts collected. Aborting.")

    logger.info("Total facts collected: %d", len(facts))

    import torch
    use_gpu = (not args.cpu) and torch.cuda.is_available()
    embeddings = embed_facts(
        facts,
        embedding_model=args.embedding_model,
        batch_size=args.batch_size,
        use_gpu=use_gpu,
    )

    logger.info("Building %d KnowledgeRecord instances ...", len(facts))
    records = build_records(facts, embeddings)

    store = KnowledgeStore(KnowledgeStoreConfig(store_dir=args.output_dir))
    for r in records:
        store.create(r)

    out = store.save(args.output_dir)
    logger.info("Saved %d records to %s", store.num_records, out)

    # Manifest for downstream audit (domain counts + embedding source).
    manifest = {
        "embedding_model": args.embedding_model,
        "n_records": store.num_records,
        "n_per_split": args.n_per_split,
        "skip_first": args.skip_first,
        "sqa_included": not args.skip_sqa,
        "counterfact_included": not args.skip_counterfact,
        "qm_train_path": args.qm_train_path,
        "qm_train_base_path": args.qm_train_base_path,
        "domain_counts": {},
    }
    for r in records:
        manifest["domain_counts"][r.domain] = manifest["domain_counts"].get(r.domain, 0) + 1
    with open(Path(args.output_dir) / "seed_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Manifest: %s", json.dumps(manifest["domain_counts"], indent=2))


if __name__ == "__main__":
    main()
