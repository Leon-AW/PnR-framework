#!/usr/bin/env python3
"""
Train Base Adapter Script
=========================

Main entry point for training the Base Expert Adapter on SituatedQA stable facts.

This script demonstrates the Patch-and-Route framework for continual learning:
1. Load Frozen Foundation (Mistral-7B with 4-bit quantization)
2. Attach Expert Adapter (LoRA)
3. Train on Matrix Split data:
   - Temporal: date < 2019 (The Frozen Past)
   - Geographic: US locations (Standard Knowledge)
4. Save checkpoint for future knowledge updates

The Matrix Split Strategy:
- Base Stream: Pre-2019 temporal facts + US geographic facts
- Uses `edited_question` as prompt (contains explicit temporal/geo triggers)
- Random answer sampling for mild data augmentation

Usage:
    python train_base_adapter.py [OPTIONS]
    
    Options:
        --model_id      HuggingFace model ID (default: mistralai/Mistral-7B-Instruct-v0.3)
        --max_steps     Training steps (default: 1000)
        --batch_size    Per-device batch size (default: 4)
        --learning_rate Learning rate (default: 2e-4)
        --output_dir    Checkpoint directory (default: checkpoints/base_v1)
        --lora_r        LoRA rank (default: 16)
        --cutoff_year   Temporal cutoff for stable facts (default: 2019)

Example:
    python train_base_adapter.py --max_steps 2000 --batch_size 2

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
from src.training.trainer import train_adapter, TrainingConfig
from src.utils.logging import setup_logger, configure_framework_logging
from src.utils.config import save_config


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Base Expert Adapter on SituatedQA stable facts",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
        default=1000,
        help="Maximum training steps",
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
        "--cutoff_year",
        type=int,
        default=2019,
        help="Temporal cutoff year for stable facts",
    )
    parser.add_argument(
        "--buffer_size",
        type=int,
        default=10000,
        help="Shuffle buffer size for streaming",
    )
    parser.add_argument(
        "--include_geo",
        action="store_true",
        default=True,
        help="Include US geographic data in Base stream",
    )
    parser.add_argument(
        "--temporal_only",
        action="store_true",
        help="Use only temporal data (exclude geographic)",
    )
    
    # Output configuration
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/base_v1",
        help="Output directory for checkpoints",
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

    # MLflow experiment tracking
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="pnr-training",
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="MLflow run name (defaults to 'base_v1')",
    )

    return parser.parse_args()


def main() -> None:
    """Main training pipeline."""
    args = parse_args()
    
    # =========================================================================
    # Setup Logging
    # =========================================================================
    configure_framework_logging(level=args.log_level)
    logger = setup_logger("train_base_adapter", level=args.log_level)
    
    logger.info("=" * 70)
    logger.info("PATCH-AND-ROUTE: BASE EXPERT ADAPTER TRAINING")
    logger.info("=" * 70)
    logger.info(f"Model: {args.model_id}")
    logger.info(f"Quantization: {args.quantization}")
    logger.info(f"LoRA rank: {args.lora_r}")
    logger.info(f"Max steps: {args.max_steps}")
    logger.info(f"Temporal cutoff: year < {args.cutoff_year}")
    logger.info(f"Include geo (US): {not args.temporal_only}")
    logger.info(f"Output: {args.output_dir}")
    logger.info("=" * 70)
    
    # =========================================================================
    # Initialize Data Loader (Matrix Split Strategy)
    # =========================================================================
    logger.info("\n[1/4] Loading SituatedQA dataset (streaming)...")
    
    data_config = SituatedQAConfig(
        streaming=True,
        buffer_size=args.buffer_size,
        seed=args.seed,
        temporal_cutoff_year=args.cutoff_year,
    )
    
    loader = SituatedQALoader(config=data_config)
    
    # Get appropriate stream based on configuration
    if args.temporal_only:
        # Temporal only: pre-2019 facts
        base_stream = loader.get_temporal_base_stream()
        logger.info(f"✓ Using temporal-only Base stream (date < {args.cutoff_year})")
    else:
        # Full Base: temporal + geo (US)
        base_stream = loader.get_base_stream()
        logger.info(f"✓ Using combined Base stream:")
        logger.info(f"    - Temporal: date < {args.cutoff_year}")
        logger.info(f"    - Geographic: US locations")
    
    # Format for training (applies chat template, shuffles)
    train_dataset = loader.format_stream(base_stream, shuffle=True)
    
    logger.info(f"✓ Data stream formatted with chat template")
    logger.info(f"  Using 'edited_question' as prompt (contains explicit triggers)")
    logger.info(f"  Random answer sampling for augmentation")
    
    # =========================================================================
    # Initialize Model (Frozen Foundation + Expert Adapter)
    # =========================================================================
    logger.info("\n[2/4] Loading Frozen Foundation (base LLM)...")
    
    # Map string to enum
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
    
    logger.info("\n[3/4] Attaching Expert Adapter (LoRA)...")
    
    expert_config = ExpertConfig(
        name="base_v1",
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
    logger.info("\n[4/4] Starting Expert Adapter training...")
    
    metrics = train_adapter(
        model=model,
        tokenizer=tokenizer,
        dataset=train_dataset,
        adapter_name="base_v1",
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        max_seq_length=args.max_seq_length,
        seed=args.seed,
        dataset_buffer_size=args.buffer_size,
        mlflow_experiment=args.experiment_name,
        mlflow_run_name=args.run_name or "base_v1",
    )
    
    # =========================================================================
    # Save Configuration
    # =========================================================================
    output_path = Path(args.output_dir)
    
    # Save training configuration for reproducibility
    config_dict = {
        "adapter_name": "base_v1",
        "adapter_type": "base",
        "model_id": args.model_id,
        "quantization": args.quantization,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "learning_rate": args.learning_rate,
        "temporal_cutoff_year": args.cutoff_year,
        "include_geo": not args.temporal_only,
        "seed": args.seed,
        "data_strategy": "matrix_split",
        "prompt_field": "edited_question",
        "metrics": metrics,
    }
    save_config(config_dict, output_path / "training_config.json")
    
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Base Adapter saved to: {output_path}")
    logger.info("\nNext steps:")
    logger.info("  1. Train Temporal Patch: python train_patch.py --type temporal --year 2021")
    logger.info("  2. Train Geo Patch: python train_patch.py --type geo --country India")
    logger.info("  3. Implement Knowledge Router for dynamic adapter selection")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
