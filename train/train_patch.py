#!/usr/bin/env python3
"""
Train Patch Adapter Script
==========================

Entry point for training specialized Knowledge Patches (Temporal or Geographic).

This script extends the Patch-and-Route framework by training domain-specific adapters:
- Temporal Patches: Knowledge updates from specific time periods (year >= cutoff)
- Geographic Patches: Country-specific knowledge (non-US locations)

Matrix Split Strategy:
- Temporal Patch: Uses `edited_question` with temporal triggers ("as of 2021")
- Geo Patch: Uses `edited_question` with location triggers ("in India")

Usage:
    # Train Temporal Patch for 2019+ data
    python train_patch.py --type temporal --cutoff_year 2019

    # Train Geographic Patch for India
    python train_patch.py --type geo --country India

    # Train Geographic Patch for UK
    python train_patch.py --type geo --country UK --max_steps 500

Author: Leon Wagner
Thesis: "A Modular 'Patch-and-Route' Framework for Continual Learning in Enterprise LLMs"
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.data.loader import SituatedQALoader, SituatedQAConfig
from src.models.core import (
    PatchAndRouteLLM,
    FrozenFoundationConfig,
    ExpertConfig,
    QuantizationType,
)
from src.training.trainer import train_adapter
from src.utils.logging import setup_logger, configure_framework_logging
from src.utils.config import save_config


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train specialized Knowledge Patches (Temporal or Geographic)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Patch type (required)
    parser.add_argument(
        "--type",
        type=str,
        required=True,
        choices=["temporal", "geo", "geo_generic"],
        help="Type of patch: 'temporal', 'geo' (specific country), or 'geo_generic' (rest of world)",
    )
    
    # Temporal patch options
    parser.add_argument(
        "--cutoff_year",
        type=int,
        default=2019,
        help="[Temporal] Year cutoff - train on data >= this year",
    )
    
    # Geographic patch options
    parser.add_argument(
        "--country",
        type=str,
        default=None,
        help="[Geo] Country name to filter for (e.g., 'India', 'UK', 'Germany')",
    )
    parser.add_argument(
        "--exclude_countries",
        type=str,
        default=None,
        help="[Geo Generic] Comma-separated list of countries to exclude (e.g. 'India,UK')",
    )
    
    # Model configuration
    parser.add_argument(
        "--model_id",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.3",
        help="HuggingFace model identifier",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        choices=["none", "int8", "int4"],
        default="int4",
        help="Quantization type for memory efficiency",
    )
    parser.add_argument(
        "--base_adapter",
        type=str,
        default=None,
        help="Optional: Path to base adapter to initialize from (transfer learning)",
    )
    
    # LoRA configuration
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
        help="LoRA rank (higher = more capacity)",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha scaling factor",
    )
    
    # Training configuration
    parser.add_argument(
        "--max_steps",
        type=int,
        default=500,
        help="Maximum training steps (patches typically need fewer steps)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Per-device batch size",
    )
    parser.add_argument(
        "--gradient_accumulation",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Peak learning rate",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="Maximum sequence length",
    )
    
    # Data configuration
    parser.add_argument(
        "--buffer_size",
        type=int,
        default=10000,
        help="Shuffle buffer size for streaming",
    )
    
    # Output configuration
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (auto-generated if not specified)",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=100,
        help="Steps between checkpoint saves",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Steps between logging",
    )
    
    # Misc
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging verbosity",
    )
    
    args = parser.parse_args()
    
    # Validate geo patch requires country
    if args.type == "geo" and args.country is None:
        parser.error("--country is required for geo patches")
    
    # Validate generic geo patch requires exclusions (warning only, but good practice)
    if args.type == "geo_generic" and not args.exclude_countries:
        # It's valid to have no exclusions (train on ALL non-US), but usually unintended in PnR context
        pass
    
    return args


def get_adapter_name(patch_type: str, cutoff_year: int, country: str | None) -> str:
    """Generate a standardized adapter name based on patch type."""
    if patch_type == "temporal":
        return f"patch_temp_{cutoff_year}_plus"
    elif patch_type == "geo_generic":
        return "patch_geo_others"
    else:  # geo
        # Normalize country name for filename
        country_slug = country.lower().replace(" ", "_")
        return f"patch_geo_{country_slug}"


def main() -> None:
    """Main training pipeline for patches."""
    args = parse_args()
    
    # Derive adapter name
    adapter_name = get_adapter_name(args.type, args.cutoff_year, args.country)
    
    # Derive output directory if not specified
    output_dir = args.output_dir or f"checkpoints/{adapter_name}"
    
    # =========================================================================
    # Setup Logging
    # =========================================================================
    configure_framework_logging(level=args.log_level)
    logger = setup_logger("train_patch", level=args.log_level)
    
    logger.info("=" * 70)
    logger.info("PATCH-AND-ROUTE: KNOWLEDGE PATCH TRAINING")
    logger.info("=" * 70)
    logger.info(f"Patch Type: {args.type.upper()}")
    logger.info(f"Adapter Name: {adapter_name}")
    if args.type == "temporal":
        logger.info(f"Temporal Filter: date >= {args.cutoff_year}")
    else:
        logger.info(f"Geographic Filter: location contains '{args.country}'")
    logger.info(f"Model: {args.model_id}")
    logger.info(f"Quantization: {args.quantization}")
    logger.info(f"Max steps: {args.max_steps}")
    logger.info(f"Output: {output_dir}")
    logger.info("=" * 70)
    
    # =========================================================================
    # Initialize Data Loader
    # =========================================================================
    logger.info("\n[1/4] Loading SituatedQA dataset (streaming)...")
    
    data_config = SituatedQAConfig(
        streaming=True,
        buffer_size=args.buffer_size,
        seed=args.seed,
        temporal_cutoff_year=args.cutoff_year,
    )
    
    loader = SituatedQALoader(config=data_config)
    
    # Get appropriate stream based on patch type
    if args.type == "temporal":
        patch_stream = loader.get_temporal_patch_stream()
        logger.info(f"✓ Temporal Patch stream (date >= {args.cutoff_year})")
    elif args.type == "geo_generic":
        exclusions = args.exclude_countries.split(",") if args.exclude_countries else []
        patch_stream = loader.get_rest_of_world_stream(exclusions)
        logger.info(f"✓ Generic Geo Patch stream (excluding {len(exclusions)} countries)")
    else:  # geo
        patch_stream = loader.get_geo_patch_stream(args.country)
        logger.info(f"✓ Geographic Patch stream (location = '{args.country}')")
    
    # Format for training
    train_dataset = loader.format_stream(patch_stream, shuffle=True)
    
    logger.info(f"✓ Data stream formatted with chat template")
    logger.info(f"  Using 'edited_question' as prompt (contains explicit triggers)")
    
    # =========================================================================
    # Initialize Model
    # =========================================================================
    logger.info("\n[2/4] Loading Frozen Foundation (base LLM)...")
    
    quant_map = {
        "none": QuantizationType.NONE,
        "int8": QuantizationType.INT8,
        "int4": QuantizationType.INT4,
    }
    
    foundation_config = FrozenFoundationConfig(
        model_id=args.model_id,
        quantization=quant_map[args.quantization],
    )
    
    llm = PatchAndRouteLLM(foundation_config=foundation_config)
    llm.load_frozen_foundation()
    
    # Optionally load base adapter for transfer learning
    if args.base_adapter:
        logger.info(f"\n[2.5/4] Loading base adapter for transfer learning: {args.base_adapter}")
        llm.load_expert(args.base_adapter)
        logger.info("✓ Base adapter loaded - will fine-tune from this checkpoint")
    
    logger.info("\n[3/4] Attaching Expert Adapter (LoRA)...")
    
    # If we loaded a base adapter, we don't need to attach a new one
    if not args.base_adapter:
        expert_config = ExpertConfig(
            name=adapter_name,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
        )
        llm.attach_expert(expert_config)
    
    llm.print_model_info()
    
    # Get training components
    model, tokenizer = llm.get_training_components()
    
    # =========================================================================
    # Training
    # =========================================================================
    logger.info("\n[4/4] Starting Patch training...")
    
    metrics = train_adapter(
        model=model,
        tokenizer=tokenizer,
        dataset=train_dataset,
        adapter_name=adapter_name,
        output_dir=output_dir,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        max_seq_length=args.max_seq_length,
        seed=args.seed,
        dataset_buffer_size=args.buffer_size,
    )
    
    # =========================================================================
    # Save Configuration
    # =========================================================================
    output_path = Path(output_dir)
    
    config_dict = {
        "adapter_name": adapter_name,
        "adapter_type": f"patch_{args.type}",
        "patch_type": args.type,
        "model_id": args.model_id,
        "quantization": args.quantization,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "data_strategy": "matrix_split",
        "prompt_field": "edited_question",
        "metrics": metrics,
    }
    
    # Add type-specific fields
    if args.type == "temporal":
        config_dict["temporal_cutoff_year"] = args.cutoff_year
        config_dict["temporal_filter"] = f">= {args.cutoff_year}"
    elif args.type == "geo_generic":
        config_dict["geo_type"] = "rest_of_world"
        config_dict["excluded_countries"] = args.exclude_countries
    else:
        config_dict["country"] = args.country
        config_dict["geo_filter"] = f"location contains '{args.country}'"
    
    if args.base_adapter:
        config_dict["initialized_from"] = args.base_adapter
    
    save_config(config_dict, output_path / "training_config.json")
    
    logger.info("\n" + "=" * 70)
    logger.info("PATCH TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Patch '{adapter_name}' saved to: {output_path}")
    logger.info("\nPatch Registry:")
    logger.info(f"  - checkpoints/base_v1 (Base Adapter)")
    logger.info(f"  - {output_path} (This Patch)")
    logger.info("\nNext steps:")
    logger.info("  1. Train additional patches for other domains")
    logger.info("  2. Implement Knowledge Router to select patches dynamically")
    logger.info("  3. Evaluate adapter performance on held-out test sets")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()

