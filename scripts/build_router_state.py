#!/usr/bin/env python3
"""
Build Router State
==================

Builds the offline state for the Time-Aware Centroid Router from each
adapter's training stream. Three things are produced per adapter:

  1. **Per-chunk routing anchors** (cluster_centroids) — each training
     query is its own anchor; routing uses ``max_i cos(q, anchor_i)``.
     This realises the architecture's ``Adapter_i = {Weights_i,
     DataIndices_i}`` tuple directly: the same training rows drive both
     routing and Source-Replay. Mean-only centroids collapse near origin
     for broad adapters (e.g. ``patch_cf_main`` covering ~20k diverse
     facts) and are insufficient on their own.
  2. **Per-adapter calibrated similarity threshold** — derived from
     in-domain training similarity (5th percentile, leave-one-out) and a
     held-out OOD negative pool. Stored in
     ``AdapterEntry.metadata['similarity_threshold']``. Eliminates the
     Pareto trap of a single global threshold (geo wants τ≈0.6, CF wants
     τ≈0.35).
  3. **Source-Replay FAISS index** per adapter — built from the same
     (question, answer) pairs used for routing. Auto-loaded by
     ``CentroidRouter.load()`` when the ``.faiss`` sidecars are present
     so the router can perform always-on retrieval at inference.

Adapter → data source mapping:
  - base_v1                  : SituatedQA base stream (pre-2019, global)
  - patch_temp_2019_plus     : SituatedQA temporal stream (post-2019)
  - patch_geo_<country>      : SituatedQA geo stream filtered by country
  - patch_geo_others         : SituatedQA rest-of-world stream
  - patch_cf_main (and any   : CounterFact training JSONL (fill-in-the-blank
    future patch_cf_*)         completion queries from CF training split)

Usage:
    python scripts/build_router_state.py \\
        --checkpoints_dir checkpoints/ \\
        --output_dir checkpoints/router_state/ \\
        --cf_data_path data/counterfact_train.jsonl \\
        --calibration_neg_path data/triviaqa_dcalibration.json \\
        --max_samples 5000

Author: Leon Wagner
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import SituatedQALoader, SituatedQAConfig
from src.routing import CentroidRouter, AdapterManifest
from src.utils.logging import setup_logger, configure_framework_logging


# ---------------------------------------------------------------------------
# Adapter → stream configuration
# ---------------------------------------------------------------------------

# Countries that have dedicated patches (used to build the "others" exclusion list)
GEO_PATCH_COUNTRIES = [
    "Australia",
    "California",
    "Canada",
    "England",
    "France",
    "Germany",
    "India",
    "Nigeria",
    "Pakistan",
    "UK",
    "United Kingdom",
]

# Adapters that come from the geographic split (not temporal)
GEO_ADAPTERS = {
    "patch_geo_australia": "Australia",
    "patch_geo_california": "California",
    "patch_geo_canada": "Canada",
    "patch_geo_england": "England",
    "patch_geo_france": "France",
    "patch_geo_germany": "Germany",
    "patch_geo_india": "India",
    "patch_geo_nigeria": "Nigeria",
    "patch_geo_pakistan": "Pakistan",
    "patch_geo_uk": "UK",
}

# Adapters to skip (not part of the routing pool).
# The current CF routing pool is the six `patch_cf_relfam_{0..5}` adapters
# (one per Wikidata relation-family cluster). The legacy single-expert
# `patch_cf_main` adapter and the descoped KMeans cluster adapters are kept.
# `monolithic_qm` is the catastrophic-forgetting baseline — it must not be
# registered as a routing target; eval uses it via --monolithic flag only.
# on disk for ablations but must NOT be registered in the routing manifest —
# they would either duplicate the CF anchor block (main + relfam read the
# same training distribution) or split CF routing across stale checkpoints
# (kmeans clusters were superseded by relfam).
# NOTE: the KMeans cluster checkpoints have `"adapter_name": "patch_cf_{i}"`
# in their training_config.json (legacy naming) even though the on-disk
# directories are `patch_cf_kmeans_{i}`. The manifest registers them under
# the training_config name, so the legacy ids are what we must skip.
SKIP_ADAPTERS = {
    "monolithic_v1",
    "monolithic_qm",
    "xlora_baseline",
    "patch_cf_main",
    "patch_cf_0",
    "patch_cf_1",
    "patch_cf_2",
    "patch_cf_3",
    "patch_cf_4",
    "patch_cf_5",
    "domain_classifier",
}


# ---------------------------------------------------------------------------
# Sample collectors — return list[dict] with at least {question, answer}
# so the same payload feeds both per-chunk centroids and Source-Replay.
# ---------------------------------------------------------------------------

def collect_sqa_samples(stream, max_samples: int) -> list[dict]:
    """Collect (edited_question, answer) records from a SituatedQA stream."""
    samples: list[dict] = []
    for example in stream:
        q = example.get("edited_question", "")
        if not q or not q.strip():
            continue
        answers = example.get("answer") or []
        if isinstance(answers, list) and answers:
            answer = answers[0]
        elif isinstance(answers, str):
            answer = answers
        else:
            answer = ""
        samples.append({
            "edited_question": q.strip(),
            "answer": answer,
            "date": example.get("date"),
            "location": example.get("location"),
        })
        if len(samples) >= max_samples:
            break
    return samples


def collect_cf_samples(jsonl_path: Path, max_samples: int) -> list[dict]:
    """Collect (question, answer) records from a CounterFact JSONL file.

    Uses the ``question`` field (e.g. "The mother tongue of X is") which is
    the exact format presented to the model at eval time. Mapped to the same
    schema (``edited_question``/``answer``) as the SituatedQA collector so
    the Source-Replay indexer doesn't need to know which adapter it's
    indexing.
    """
    samples: list[dict] = []
    with jsonl_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            q = record.get("question", "")
            if not q or not q.strip():
                continue
            samples.append({
                "edited_question": q.strip(),
                "answer": record.get("answer", ""),
                "subject": record.get("subject"),
                "relation_id": record.get("relation_id"),
            })
            if len(samples) >= max_samples:
                break
    return samples


def collect_qm_samples(jsonl_path: Path, max_samples: int) -> list[dict]:
    """Collect (question, answer) records from a QM training JSONL file.

    QM training data uses the chat-message format produced by
    ``scripts/build_qm_train_data.py``: each record has a ``messages`` list
    with a user turn (the question) and an assistant turn (the answer).
    Extracts the user content as ``edited_question`` so the Source-Replay
    indexer sees the same schema as SituatedQA and CounterFact collectors.
    """
    samples: list[dict] = []
    with jsonl_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            messages = record.get("messages", [])
            if len(messages) < 2:
                continue
            q = messages[0].get("content", "")
            a = messages[1].get("content", "")
            if not q or not q.strip():
                continue
            samples.append({
                "edited_question": q.strip(),
                "answer": a,
                "id": record.get("id"),
                "language": record.get("language"),
            })
            if len(samples) >= max_samples:
                break
    return samples


# ---------------------------------------------------------------------------
# Per-chunk routing anchors + calibrated thresholds
# ---------------------------------------------------------------------------

def embed_questions(questions: list[str], encoder, batch_size: int = 64) -> np.ndarray:
    """L2-normalised embeddings for a list of questions, shape (n, dim)."""
    if not questions:
        raise ValueError("Cannot embed an empty question list")
    emb = encoder.encode(
        questions,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=batch_size,
    )
    return np.asarray(emb, dtype=np.float32)


def calibrate_threshold(
    own_anchors: np.ndarray,
    neg_anchors: np.ndarray | None,
    *,
    in_domain_percentile: float = 5.0,
    neg_percentile: float = 99.0,
    margin: float = 0.02,
    fallback: float,
) -> tuple[float, dict]:
    """Compute a calibrated routing threshold τ_i for one adapter.

    The threshold lies between two distributions:

    - **In-domain** (``s_pos``): every training-query embedding scored
      against the adapter's *other* anchors (leave-one-out max). We take
      the ``in_domain_percentile``-th percentile (default 5) — a soft
      lower bound that still admits queries from sparser regions of the
      training distribution.
    - **Out-of-domain** (``s_neg``): a calibration negative pool (e.g.
      unused TriviaQA D-pool slice) scored against the same anchors.
      We take the ``neg_percentile``-th percentile (default 99) of the
      per-query max-sim distribution. Using the raw max would let a
      single noisy OOD query dominate the threshold — empirically this
      pushed ``patch_cf_main`` τ to 0.745 (vs. an in-domain median of
      0.564) on the May-1 build, rejecting most CF queries.

    Final τ_i = max(s_neg + margin, s_pos - margin), clamped to [0, 1].
    Falls back to ``fallback`` if the negative pool is empty or in-domain
    statistics are degenerate (e.g. fewer than 2 anchors).

    A ``calibration_quality`` field (``s_pos - s_neg``) is recorded; if
    it is negative, the in-domain and OOD distributions overlap and no
    threshold can perfectly separate them — the diagnostics surface
    this so callers can decide whether to fall back to a global floor.
    """
    info: dict = {
        "in_domain_percentile": in_domain_percentile,
        "neg_percentile": neg_percentile,
        "margin": margin,
        "fallback_used": False,
    }

    if own_anchors.shape[0] < 2:
        info["reason"] = "fewer_than_2_anchors"
        info["fallback_used"] = True
        return fallback, info

    # In-domain: leave-one-out max similarity. own_anchors are L2-normalised.
    sim_own = own_anchors @ own_anchors.T
    np.fill_diagonal(sim_own, -np.inf)
    s_pos_loo = sim_own.max(axis=1)
    s_pos = float(np.percentile(s_pos_loo, in_domain_percentile))
    info["s_pos_p{:.0f}".format(in_domain_percentile)] = s_pos
    info["s_pos_p25"] = float(np.percentile(s_pos_loo, 25))
    info["s_pos_min"] = float(s_pos_loo.min())
    info["s_pos_median"] = float(np.median(s_pos_loo))

    if neg_anchors is None or neg_anchors.shape[0] == 0:
        info["reason"] = "no_negatives_provided"
        info["fallback_used"] = True
        return fallback, info

    sim_neg = neg_anchors @ own_anchors.T
    s_neg_max_per_q = sim_neg.max(axis=1)
    s_neg = float(np.percentile(s_neg_max_per_q, neg_percentile))
    info["s_neg_p{:.0f}".format(neg_percentile)] = s_neg
    info["s_neg_p95"] = float(np.percentile(s_neg_max_per_q, 95))
    info["s_neg_max"] = float(s_neg_max_per_q.max())

    tau = max(s_neg + margin, s_pos - margin)
    tau = float(np.clip(tau, 0.0, 1.0))
    quality = s_pos - s_neg
    info["tau"] = tau
    info["calibration_quality"] = quality

    # When quality < 0 the in-domain and OOD distributions overlap: the
    # embedding model cannot discriminate this adapter's domain from the
    # TriviaQA calibration negatives. A derived τ in this regime is
    # HIGHER than the in-domain median (empirically: patch_cf_main τ=0.703
    # vs in-domain median=0.564 on May-1), so the calibrated threshold
    # silently kills routing — worse than the global fallback.
    # We record the finding in diagnostics (it is a genuine architectural
    # insight about the embedding model's discrimination power) and fall
    # back to the global τ so the eval run is not sabotaged.
    if quality < 0:
        info["reason"] = "distribution_overlap_fallback_to_global"
        info["fallback_used"] = True
        info["tau"] = fallback
        return fallback, info

    return tau, info


# ---------------------------------------------------------------------------
# Calibration negatives loader (TriviaQA D_calibration)
# ---------------------------------------------------------------------------

def load_calibration_questions(path: Path) -> list[str]:
    """Read a TriviaQA-style calibration JSON and return the question strings.

    Accepts both the wrapped ``{records: [...]}`` format produced by
    ``scripts/build_triviaqa_dcontrol.py`` and a bare list. Returns the raw
    questions (not wrapped in any chat template) — what the centroid router
    sees at inference time is the user query, not the rendered chat prompt.
    """
    with path.open() as f:
        data = json.load(f)
    records = data.get("records", data) if isinstance(data, dict) else data
    return [r["question"] for r in records if r.get("question")]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute per-chunk routing anchors, calibrated thresholds "
                    "and Source-Replay indices for all adapters",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoints_dir", default="checkpoints",
                        help="Directory containing adapter checkpoints")
    parser.add_argument("--output_dir", default="checkpoints/router_state",
                        help="Directory to save manifest.json + sidecars")
    parser.add_argument("--embedding_model", default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Embedding model (HuggingFace ID or local path)")
    parser.add_argument("--max_samples", type=int, default=5000,
                        help="Max questions per adapter for anchors / replay. "
                             "With per-chunk anchors this directly sets the "
                             "router's resolution for the adapter's domain.")
    parser.add_argument("--similarity_threshold", type=float, default=0.45,
                        help="Global fallback similarity threshold. Overridden "
                             "per-adapter by the calibrated value when a "
                             "calibration negative pool is supplied.")
    parser.add_argument("--cf_data_path", default="data/counterfact_train.jsonl",
                        help="Path to CounterFact training JSONL used to build "
                             "patch_cf_* anchors. Must use the 'question' field.")
    parser.add_argument(
        "--calibration_neg_path",
        default="data/triviaqa_dcalibration.json",
        help="Path to the TriviaQA D_calibration JSON used as the OOD negative "
             "pool for per-adapter threshold calibration. MUST be disjoint "
             "from data/triviaqa_dcontrol.json (the eval-time D_control "
             "probe) — calibrating on D_control would be test-set leakage "
             "and break the exposé's 'any drop = routing-induced forgetting' "
             "guarantee.",
    )
    parser.add_argument("--threshold_margin", type=float, default=0.02,
                        help="Slack term added to the OOD upper bound when "
                             "deriving τ_i. Larger = more conservative routing.")
    parser.add_argument("--in_domain_percentile", type=float, default=5.0,
                        help="Percentile of in-domain leave-one-out similarities "
                             "used as the lower bound for τ_i.")
    parser.add_argument("--neg_percentile", type=float, default=99.0,
                        help="Percentile of OOD per-query max-sim used as the "
                             "upper bound for τ_i. Default 99 = tolerate ~5 "
                             "OOD outliers per 500-question calibration pool. "
                             "Using 100 (= raw max) lets a single noisy "
                             "TriviaQA query dominate τ; verified empirically "
                             "in the May-1 first build (patch_cf_main τ=0.745 "
                             "vs in-domain median 0.564 → CF queries rejected).")
    parser.add_argument("--no_source_replay", action="store_true",
                        help="Skip building Source-Replay FAISS indices (faster, "
                             "but disables Change 3 / always-on replay).")
    parser.add_argument("--no_gpu", action="store_true",
                        help="Disable GPU for embedding model")
    parser.add_argument("--qm_old_data_path", default="data/qm_train_old.jsonl",
                        help="Path to QM old-facts training JSONL (used for base_qm anchors)")
    parser.add_argument("--qm_new_data_path", default="data/qm_train.jsonl",
                        help="Path to QM new-facts training JSONL (used for patch_qm_current anchors)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_framework_logging(level="INFO")
    logger = setup_logger("build_router_state", level="INFO")

    checkpoints_dir = Path(args.checkpoints_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cf_data_path = Path(args.cf_data_path)
    calib_neg_path = Path(args.calibration_neg_path) if args.calibration_neg_path else None
    qm_old_data_path = Path(args.qm_old_data_path)
    qm_new_data_path = Path(args.qm_new_data_path)

    logger.info("=" * 70)
    logger.info("BUILD ROUTER STATE")
    logger.info("=" * 70)
    logger.info(f"Checkpoints:           {checkpoints_dir}")
    logger.info(f"Output:                {output_dir}")
    logger.info(f"Model:                 {args.embedding_model}")
    logger.info(f"Max samples / adapter: {args.max_samples}")
    logger.info(f"CF data path:          {cf_data_path} (exists={cf_data_path.exists()})")
    logger.info(f"QM old data path:      {qm_old_data_path} (exists={qm_old_data_path.exists()})")
    logger.info(f"QM new data path:      {qm_new_data_path} (exists={qm_new_data_path.exists()})")
    logger.info(
        f"Calib neg path:        {calib_neg_path} "
        f"(exists={calib_neg_path.exists() if calib_neg_path else False})"
    )
    logger.info(f"Global fallback τ:     {args.similarity_threshold}")
    logger.info(f"In-domain percentile:  {args.in_domain_percentile}")
    logger.info(f"Threshold margin:      {args.threshold_margin}")
    logger.info(f"Source-Replay:         {'OFF' if args.no_source_replay else 'ON'}")
    logger.info("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load embedding model
    # ------------------------------------------------------------------
    logger.info("\n[1/5] Loading embedding model...")
    from sentence_transformers import SentenceTransformer
    import torch

    device = "cuda" if (not args.no_gpu and torch.cuda.is_available()) else "cpu"
    logger.info(f"  Device: {device}")
    encoder = SentenceTransformer(args.embedding_model, device=device)
    embedding_dim = encoder.get_sentence_embedding_dimension()
    logger.info(f"  Embedding dim: {embedding_dim}")

    # ------------------------------------------------------------------
    # 2. Build CentroidRouter and discover adapters
    # ------------------------------------------------------------------
    logger.info("\n[2/5] Discovering adapters...")

    router = CentroidRouter(
        embedding_fn=lambda text: encoder.encode(
            text, normalize_embeddings=True, show_progress_bar=False
        ).astype(np.float32),
        similarity_threshold=args.similarity_threshold,
        use_gpu=(not args.no_gpu),
        store_dir=output_dir,
    )
    n = router.register_from_checkpoints(checkpoints_dir)
    logger.info(f"  Discovered {n} adapters")

    # Remove adapters that are not part of the routing pool
    for skip_id in SKIP_ADAPTERS:
        if router.unregister_adapter(skip_id):
            logger.info(f"  Skipped (not a routing target): {skip_id}")

    routing_adapters = router.get_registered_adapters()
    logger.info(f"  Routing adapters: {routing_adapters}")

    # ------------------------------------------------------------------
    # 3. Load training data and collect (q, a) samples per adapter
    # ------------------------------------------------------------------
    logger.info("\n[3/5] Loading training data streams...")

    loader_config = SituatedQAConfig(streaming=True, seed=42)
    loader = SituatedQALoader(loader_config)

    adapter_samples: dict[str, list[dict]] = {}

    for adapter_id in routing_adapters:
        logger.info(f"\n  [{adapter_id}]")
        t0 = time.time()

        try:
            if adapter_id == "base_v1":
                stream = loader.get_base_stream()
                samples = collect_sqa_samples(stream, args.max_samples)

            elif adapter_id == "patch_temp_2019_plus":
                stream = loader.get_temporal_patch_stream()
                samples = collect_sqa_samples(stream, args.max_samples)

            elif adapter_id == "patch_geo_others":
                stream = loader.get_rest_of_world_stream(GEO_PATCH_COUNTRIES)
                samples = collect_sqa_samples(stream, args.max_samples)

            elif adapter_id in GEO_ADAPTERS:
                country = GEO_ADAPTERS[adapter_id]
                stream = loader.get_geo_patch_stream(country)
                samples = collect_sqa_samples(stream, args.max_samples)

            elif adapter_id.startswith("patch_cf_relfam_"):
                cluster_idx = adapter_id.rsplit("_", 1)[-1]
                relfam_path = Path("data") / f"counterfact_relfam_{cluster_idx}.jsonl"
                if not relfam_path.exists():
                    logger.error(
                        f"  Per-cluster CF data file not found: {relfam_path} — "
                        f"build it with scripts/build_counterfact_relation_clusters.py. "
                        f"Skipping {adapter_id}."
                    )
                    continue
                samples = collect_cf_samples(relfam_path, args.max_samples)
                logger.info(f"  Source: {relfam_path} ({len(samples)} CF samples)")

            elif adapter_id.startswith("patch_cf"):
                if not cf_data_path.exists():
                    logger.error(
                        f"  CF data file not found: {cf_data_path} — "
                        f"pass --cf_data_path. Skipping {adapter_id}."
                    )
                    continue
                samples = collect_cf_samples(cf_data_path, args.max_samples)
                logger.info(f"  Source: {cf_data_path} ({len(samples)} CF samples)")

            elif adapter_id == "base_qm":
                if not qm_old_data_path.exists():
                    logger.error(
                        f"  QM old-facts data not found: {qm_old_data_path} — "
                        f"pass --qm_old_data_path. Skipping {adapter_id}."
                    )
                    continue
                samples = collect_qm_samples(qm_old_data_path, args.max_samples)
                logger.info(f"  Source: {qm_old_data_path} ({len(samples)} QM old-facts samples)")

            elif adapter_id == "patch_qm_current":
                if not qm_new_data_path.exists():
                    logger.error(
                        f"  QM new-facts data not found: {qm_new_data_path} — "
                        f"pass --qm_new_data_path. Skipping {adapter_id}."
                    )
                    continue
                samples = collect_qm_samples(qm_new_data_path, args.max_samples)
                logger.info(f"  Source: {qm_new_data_path} ({len(samples)} QM new-facts samples)")

            else:
                logger.warning(f"  Unknown adapter type for {adapter_id}, skipping")
                continue

            logger.info(f"  Collected {len(samples)} samples in {time.time()-t0:.1f}s")

            if not samples:
                logger.warning(f"  No samples found for {adapter_id}!")
                continue

            adapter_samples[adapter_id] = samples

        except Exception as e:
            logger.error(f"  Failed to collect samples for {adapter_id}: {e}")
            continue

    # ------------------------------------------------------------------
    # 4. Embed calibration negatives once (shared across adapters)
    # ------------------------------------------------------------------
    logger.info("\n[4/5] Loading + embedding calibration negative pool...")
    neg_anchors: np.ndarray | None = None
    calib_count = 0
    if calib_neg_path and calib_neg_path.exists():
        try:
            neg_questions = load_calibration_questions(calib_neg_path)
            calib_count = len(neg_questions)
            if neg_questions:
                neg_anchors = embed_questions(neg_questions, encoder)
                logger.info(
                    f"  Calibration pool: {calib_count} questions "
                    f"→ shape {neg_anchors.shape}"
                )
            else:
                logger.warning(f"  Calibration pool {calib_neg_path} is empty")
        except Exception as e:
            logger.error(f"  Failed to load calibration pool: {e}")
    else:
        logger.warning(
            f"  No calibration negatives — every adapter will use the global "
            f"fallback τ={args.similarity_threshold}. To enable per-adapter "
            f"calibrated thresholds, build {calib_neg_path or 'data/triviaqa_dcalibration.json'} "
            f"first via scripts/build_triviaqa_dcalibration.sh."
        )

    # ------------------------------------------------------------------
    # 5. Compute anchors + thresholds + Source-Replay indices
    # ------------------------------------------------------------------
    logger.info("\n[5/5] Computing anchors / thresholds / indices...")

    if not args.no_source_replay:
        router.initialize_source_replay(output_dir)

    computed = 0
    failed: list[str] = []
    diagnostics: dict[str, dict] = {}

    for adapter_id, samples in adapter_samples.items():
        logger.info(f"\n  ── {adapter_id} ({len(samples)} samples) ──")
        t0 = time.time()
        try:
            questions = [s["edited_question"] for s in samples]

            # (a) per-chunk routing anchors
            anchors = embed_questions(questions, encoder)
            anchor_list = [anchors[i] for i in range(anchors.shape[0])]
            router._manifest.update_cluster_centroids(adapter_id, anchor_list)

            # (b) calibrated per-adapter threshold
            tau, info = calibrate_threshold(
                own_anchors=anchors,
                neg_anchors=neg_anchors,
                in_domain_percentile=args.in_domain_percentile,
                neg_percentile=args.neg_percentile,
                margin=args.threshold_margin,
                fallback=args.similarity_threshold,
            )
            entry = router._manifest[adapter_id]
            entry.metadata["similarity_threshold"] = tau
            entry.metadata["calibration"] = {
                "source": (
                    str(calib_neg_path) if (calib_neg_path and not info["fallback_used"]) else None
                ),
                "num_anchors": int(anchors.shape[0]),
                "num_negatives": int(neg_anchors.shape[0]) if neg_anchors is not None else 0,
                **info,
            }

            quality = info.get("calibration_quality", float("nan"))
            quality_flag = " ⚠ overlap" if quality < 0 else ""
            logger.info(
                f"  ✓ Anchors={anchors.shape[0]}  τ={tau:.3f}  "
                f"(fallback={'yes' if info['fallback_used'] else 'no'}, "
                f"s_pos_p{args.in_domain_percentile:.0f}≈{info.get(f's_pos_p{args.in_domain_percentile:.0f}', float('nan')):.3f}, "
                f"s_pos_med≈{info.get('s_pos_median', float('nan')):.3f}, "
                f"s_neg_p{args.neg_percentile:.0f}≈{info.get(f's_neg_p{args.neg_percentile:.0f}', float('nan')):.3f}, "
                f"qual≈{quality:+.3f}{quality_flag})"
            )

            # (c) Source-Replay FAISS index from the same samples
            if not args.no_source_replay:
                num_chunks = router.index_samples_for_replay(adapter_id, samples)
                logger.info(f"  ✓ Source-Replay: indexed {num_chunks} chunks")

            diagnostics[adapter_id] = {
                "num_anchors": int(anchors.shape[0]),
                "tau": tau,
                **info,
            }
            computed += 1
            logger.info(f"  Total: {time.time()-t0:.1f}s")

        except Exception as e:
            logger.error(f"  ✗ Failed: {e}", exc_info=True)
            failed.append(adapter_id)

    # ------------------------------------------------------------------
    # Save manifest + diagnostics
    # ------------------------------------------------------------------
    router.save(output_dir)
    logger.info(f"\nRouter state saved to: {output_dir}/manifest.json")

    diag_path = output_dir / "build_diagnostics.json"
    with diag_path.open("w") as f:
        json.dump({
            "global_fallback_threshold": args.similarity_threshold,
            "in_domain_percentile": args.in_domain_percentile,
            "margin": args.threshold_margin,
            "calibration_negatives_path": str(calib_neg_path) if calib_neg_path else None,
            "calibration_negatives_count": calib_count,
            "embedding_model": args.embedding_model,
            "embedding_dim": embedding_dim,
            "always_on_source_replay": not args.no_source_replay,
            "per_adapter": diagnostics,
        }, f, indent=2)
    logger.info(f"Build diagnostics saved to: {diag_path}")

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("DONE")
    logger.info(f"  Adapters discovered: {len(routing_adapters)}")
    logger.info(f"  Adapters built:      {computed}")
    logger.info(f"  Failed:              {len(failed)}")
    if failed:
        logger.info(f"  Failed adapters:     {failed}")
    logger.info("\nNext step:")
    logger.info(f"  eval_pnr.py --router_state {output_dir} ...")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
