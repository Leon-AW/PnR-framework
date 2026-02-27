#!/usr/bin/env python3
"""
Compute Centroids Script
========================

Offline utility to compute centroid embeddings for all registered adapters.

This script:
1. Discovers adapters from the checkpoints directory
2. Loads training data for each adapter
3. Computes mean embedding (centroid) using the embedding model
4. Saves the manifest with centroids for online routing

Usage:
    python scripts/compute_centroids.py \\
        --checkpoints_dir checkpoints/ \\
        --embedding_model /path/to/KaLM-Embedding-Gemma3-12B \\
        --output_dir router_state/

Options:
    --checkpoints_dir   Directory containing adapter checkpoints
    --embedding_model   Path to embedding model (e.g., KaLM-Embedding-Gemma3-12B)
    --output_dir        Directory to save manifest and indices
    --training_data_dir Directory containing training JSONL files
    --max_samples       Maximum samples per adapter for centroid computation
    --index_for_replay  Also build FAISS indices for Source-Replay
    --max_chunks        Maximum chunks per adapter for Source-Replay

Author: Leon Wagner
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.routing import CentroidRouter, AdapterManifest
from src.utils.logging import setup_logger, configure_framework_logging


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute centroid embeddings for adapters",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Required paths
    parser.add_argument(
        "--checkpoints_dir",
        type=str,
        default="checkpoints",
        help="Directory containing adapter checkpoints",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        required=True,
        help="Path to embedding model (e.g., /path/to/KaLM-Embedding-Gemma3-12B)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="router_state",
        help="Directory to save manifest and indices",
    )
    
    # Training data
    parser.add_argument(
        "--training_data_dir",
        type=str,
        default=None,
        help="Directory containing training JSONL files (auto-detected if None)",
    )
    parser.add_argument(
        "--training_data_mapping",
        type=str,
        default=None,
        help="JSON file mapping adapter_id -> training_data_path",
    )
    
    # Centroid computation
    parser.add_argument(
        "--max_samples",
        type=int,
        default=1000,
        help="Maximum samples per adapter for centroid computation",
    )
    parser.add_argument(
        "--text_field",
        type=str,
        default="edited_question",
        help="Field to extract text from training data",
    )
    
    # Source-Replay indexing
    parser.add_argument(
        "--index_for_replay",
        action="store_true",
        help="Also build FAISS indices for Source-Replay",
    )
    parser.add_argument(
        "--max_chunks",
        type=int,
        default=5000,
        help="Maximum chunks per adapter for Source-Replay indexing",
    )
    
    # Hardware
    parser.add_argument(
        "--no_gpu",
        action="store_true",
        help="Disable GPU usage",
    )
    
    # Misc
    parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=0.65,
        help="Similarity threshold for routing (saved in manifest)",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging verbosity",
    )
    
    return parser.parse_args()


def discover_training_data(
    checkpoints_dir: Path,
    training_data_dir: Path | None = None,
) -> dict[str, str]:
    """Discover training data paths for each adapter.
    
    Attempts to find training data by:
    1. Looking for training_data.jsonl in adapter directory
    2. Matching adapter name to files in training_data_dir
    3. Using patterns from adapter type (geo/temporal)
    
    Args:
        checkpoints_dir: Directory containing adapter checkpoints.
        training_data_dir: Optional directory with training data files.
        
    Returns:
        Dictionary mapping adapter_id -> training_data_path.
    """
    mapping = {}
    
    for subdir in checkpoints_dir.iterdir():
        if not subdir.is_dir():
            continue
        
        # Check for training_data.jsonl in adapter dir
        local_data = subdir / "training_data.jsonl"
        if local_data.exists():
            mapping[subdir.name] = str(local_data)
            continue
        
        # Check training_data_dir if provided
        if training_data_dir:
            # Try exact match
            external_data = training_data_dir / f"{subdir.name}.jsonl"
            if external_data.exists():
                mapping[subdir.name] = str(external_data)
                continue
            
            # Try pattern matching for geo patches
            if "geo_" in subdir.name:
                country = subdir.name.replace("patch_geo_", "")
                geo_data = training_data_dir / f"geo_{country}.jsonl"
                if geo_data.exists():
                    mapping[subdir.name] = str(geo_data)
                    continue
        
        # Could not find training data
        logging.warning(f"No training data found for adapter: {subdir.name}")
    
    return mapping


def main() -> None:
    """Main script entry point."""
    args = parse_args()
    
    # Setup logging
    configure_framework_logging(level=args.log_level)
    logger = setup_logger("compute_centroids", level=args.log_level)
    
    logger.info("=" * 70)
    logger.info("CENTROID COMPUTATION SCRIPT")
    logger.info("=" * 70)
    logger.info(f"Checkpoints: {args.checkpoints_dir}")
    logger.info(f"Embedding model: {args.embedding_model}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Max samples: {args.max_samples}")
    logger.info(f"Index for replay: {args.index_for_replay}")
    logger.info("=" * 70)
    
    # Paths
    checkpoints_dir = Path(args.checkpoints_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    training_data_dir = Path(args.training_data_dir) if args.training_data_dir else None
    
    # Step 1: Initialize router with embedding model
    logger.info("\n[1/4] Loading embedding model...")
    
    router = CentroidRouter(
        embedding_model_path=args.embedding_model,
        similarity_threshold=args.similarity_threshold,
        use_gpu=not args.no_gpu,
        store_dir=output_dir,
    )
    
    logger.info("✓ Embedding model loaded")
    
    # Step 2: Discover adapters
    logger.info("\n[2/4] Discovering adapters...")
    
    num_adapters = router.register_from_checkpoints(checkpoints_dir)
    
    logger.info(f"✓ Found {num_adapters} adapters")
    
    # Step 3: Discover/load training data mapping
    logger.info("\n[3/4] Mapping training data...")
    
    if args.training_data_mapping:
        # Load from JSON file
        with open(args.training_data_mapping) as f:
            training_data_mapping = json.load(f)
        logger.info(f"✓ Loaded mapping from {args.training_data_mapping}")
    else:
        # Auto-discover
        training_data_mapping = discover_training_data(checkpoints_dir, training_data_dir)
        logger.info(f"✓ Auto-discovered {len(training_data_mapping)} training data files")
    
    # Update manifest with source data paths
    for adapter_id, data_path in training_data_mapping.items():
        entry = router._manifest.get(adapter_id)
        if entry:
            entry.source_data_path = data_path
    
    # Step 4: Compute centroids
    logger.info("\n[4/4] Computing centroids...")
    
    computed = 0
    failed = []
    
    for adapter_id in router.get_registered_adapters():
        entry = router._manifest.get(adapter_id)
        
        if entry.has_centroid:
            logger.info(f"  [SKIP] {adapter_id} (already has centroid)")
            continue
        
        if not entry.source_data_path:
            logger.warning(f"  [SKIP] {adapter_id} (no training data)")
            continue
        
        try:
            logger.info(f"  [COMPUTE] {adapter_id}...")
            router.compute_adapter_centroid(
                adapter_id=adapter_id,
                text_field=args.text_field,
                max_samples=args.max_samples,
            )
            computed += 1
            logger.info(f"    ✓ Centroid computed")
            
        except Exception as e:
            logger.error(f"    ✗ Failed: {e}")
            failed.append(adapter_id)
    
    logger.info(f"\nComputed {computed} centroids, {len(failed)} failed")
    
    # Step 5: Index for Source-Replay (if requested)
    if args.index_for_replay:
        logger.info("\n[5/5] Building Source-Replay indices...")
        
        router.initialize_source_replay(output_dir)
        
        indexed = 0
        for adapter_id in router.get_registered_adapters():
            entry = router._manifest.get(adapter_id)
            
            if not entry.source_data_path:
                continue
            
            try:
                logger.info(f"  [INDEX] {adapter_id}...")
                num_chunks = router.index_adapter_for_replay(
                    adapter_id=adapter_id,
                    max_chunks=args.max_chunks,
                )
                indexed += 1
                logger.info(f"    ✓ Indexed {num_chunks} chunks")
                
            except Exception as e:
                logger.error(f"    ✗ Failed: {e}")
        
        logger.info(f"\nIndexed {indexed} adapters for Source-Replay")
    
    # Step 6: Save manifest
    logger.info("\nSaving router state...")
    
    router.save(output_dir)
    
    # Save summary
    summary_path = output_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Centroid Computation Summary\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"=" * 60 + "\n\n")
        f.write(router.summary())
    
    logger.info(f"✓ Router state saved to {output_dir}")
    
    # Final summary
    logger.info("\n" + "=" * 70)
    logger.info("CENTROID COMPUTATION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Adapters discovered: {num_adapters}")
    logger.info(f"Centroids computed: {computed}")
    logger.info(f"Failed: {len(failed)}")
    if failed:
        logger.info(f"  Failed adapters: {failed}")
    logger.info(f"\nOutput saved to: {output_dir}")
    logger.info("\nNext steps:")
    logger.info("  1. Review manifest.json for correctness")
    logger.info("  2. Use with PatchAndRouteInference for inference")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()

