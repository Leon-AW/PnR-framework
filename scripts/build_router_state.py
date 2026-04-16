#!/usr/bin/env python3
"""
Build Router State
==================

Computes centroid embeddings for all adapters from SituatedQA training streams
and saves the manifest to disk for use by the CentroidRouter at eval time.

This fixes the EM=0.0 issue where routing returns None because the manifest
has no centroid vectors (register_from_checkpoints does not compute them).

Usage:
    python scripts/build_router_state.py \\
        --checkpoints_dir checkpoints/ \\
        --output_dir checkpoints/router_state/ \\
        --max_samples 500

Author: Leon Wagner
"""

from __future__ import annotations

import argparse
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

# Adapters to skip (not part of the routing pool)
SKIP_ADAPTERS = {"monolithic_v1", "xlora_baseline"}


def collect_questions(stream, max_samples: int) -> list[str]:
    """Collect edited_question strings from a SituatedQA stream.

    Args:
        stream: Filtered IterableDataset.
        max_samples: Maximum questions to collect.

    Returns:
        List of question strings.
    """
    questions = []
    for example in stream:
        q = example.get("edited_question", "")
        if q and q.strip():
            questions.append(q.strip())
        if len(questions) >= max_samples:
            break
    return questions


def compute_centroid(questions: list[str], encoder) -> np.ndarray:
    """Compute mean embedding centroid for a list of questions.

    Args:
        questions: List of question strings.
        encoder: SentenceTransformer instance.

    Returns:
        Normalized centroid vector as float32 array.
    """
    embeddings = encoder.encode(
        questions,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=64,
    )
    centroid = embeddings.mean(axis=0)
    norm = np.linalg.norm(centroid)
    if norm > 0:
        centroid = centroid / norm
    return centroid.astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute centroids for all adapters and save router state",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoints_dir", default="checkpoints",
                        help="Directory containing adapter checkpoints")
    parser.add_argument("--output_dir", default="checkpoints/router_state",
                        help="Directory to save manifest.json")
    parser.add_argument("--embedding_model", default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Embedding model (HuggingFace ID or local path)")
    parser.add_argument("--max_samples", type=int, default=500,
                        help="Max questions per adapter for centroid computation")
    parser.add_argument("--similarity_threshold", type=float, default=0.65,
                        help="Similarity threshold stored in manifest")
    parser.add_argument("--no_gpu", action="store_true",
                        help="Disable GPU for embedding model")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_framework_logging(level="INFO")
    logger = setup_logger("build_router_state", level="INFO")

    checkpoints_dir = Path(args.checkpoints_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("BUILD ROUTER STATE")
    logger.info("=" * 70)
    logger.info(f"Checkpoints: {checkpoints_dir}")
    logger.info(f"Output:      {output_dir}")
    logger.info(f"Model:       {args.embedding_model}")
    logger.info(f"Max samples: {args.max_samples}")
    logger.info("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load embedding model
    # ------------------------------------------------------------------
    logger.info("\n[1/4] Loading embedding model...")
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
    logger.info("\n[2/4] Discovering adapters...")

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
    # 3. Load SituatedQA data and collect questions per adapter
    # ------------------------------------------------------------------
    logger.info("\n[3/4] Loading SituatedQA data streams...")

    loader_config = SituatedQAConfig(streaming=True, seed=42)
    loader = SituatedQALoader(loader_config)

    # Collect questions per adapter
    adapter_questions: dict[str, list[str]] = {}

    for adapter_id in routing_adapters:
        logger.info(f"\n  [{adapter_id}]")
        t0 = time.time()

        try:
            if adapter_id == "base_v1":
                stream = loader.get_base_stream()
                questions = collect_questions(stream, args.max_samples)

            elif adapter_id == "patch_temp_2019_plus":
                stream = loader.get_temporal_patch_stream()
                questions = collect_questions(stream, args.max_samples)

            elif adapter_id == "patch_geo_others":
                stream = loader.get_rest_of_world_stream(GEO_PATCH_COUNTRIES)
                questions = collect_questions(stream, args.max_samples)

            elif adapter_id in GEO_ADAPTERS:
                country = GEO_ADAPTERS[adapter_id]
                stream = loader.get_geo_patch_stream(country)
                questions = collect_questions(stream, args.max_samples)

            else:
                logger.warning(f"  Unknown adapter type for {adapter_id}, skipping")
                continue

            logger.info(f"  Collected {len(questions)} questions in {time.time()-t0:.1f}s")

            if not questions:
                logger.warning(f"  No questions found for {adapter_id}!")
                continue

            adapter_questions[adapter_id] = questions

        except Exception as e:
            logger.error(f"  Failed to collect questions for {adapter_id}: {e}")
            continue

    # ------------------------------------------------------------------
    # 4. Compute centroids and save manifest
    # ------------------------------------------------------------------
    logger.info("\n[4/4] Computing centroids and saving manifest...")

    computed = 0
    failed = []

    for adapter_id, questions in adapter_questions.items():
        logger.info(f"\n  Computing centroid for {adapter_id} ({len(questions)} questions)...")
        t0 = time.time()
        try:
            centroid = compute_centroid(questions, encoder)
            router._manifest.update_centroid(adapter_id, centroid)
            computed += 1
            logger.info(f"  ✓ Done in {time.time()-t0:.1f}s  (dim={centroid.shape[0]})")
        except Exception as e:
            logger.error(f"  ✗ Failed: {e}")
            failed.append(adapter_id)

    # Save
    router.save(output_dir)
    logger.info(f"\nRouter state saved to: {output_dir}/manifest.json")

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("DONE")
    logger.info(f"  Adapters discovered: {len(routing_adapters)}")
    logger.info(f"  Centroids computed:  {computed}")
    logger.info(f"  Failed:              {len(failed)}")
    if failed:
        logger.info(f"  Failed adapters:     {failed}")
    logger.info("\nNext step:")
    logger.info(f"  eval_pnr.py --router_state {output_dir} ...")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
