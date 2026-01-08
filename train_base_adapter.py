#!/usr/bin/env python3
"""
Train Base Adapter Script
=========================

Main entry point for training the Base Expert Adapter on SituatedQA stable facts.

This script demonstrates the Patch-and-Route framework for continual learning:
1. Load Frozen Foundation (Mistral-7B with 4-bit quantization)
2. Attach Expert Adapter (LoRA)
3. Train on temporally-filtered SituatedQA data (year < 2019)
4. Save checkpoint for future knowledge updates

Usage:
    python train_base_adapter.py [OPTIONS]
    
    Options:
        --model_id      HuggingFace model ID (default: mistralai/Mistral-7B-Instruct-v0.3)
        --max_steps     Training steps (default: 1000)
        --batch_size    Per-device batch size (default: 4)
        --learning_rate Learning rate (default: 2e-4)
        --output_dir    Checkpoint directory (default: checkpoints/situatedqa_base_v1)
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
from src.training.trainer import PatchAndRouteTrainer, TrainingConfig
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
    
    # Output configuration
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/situatedqa_base_v1",
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
    logger.info(f"Output: {args.output_dir}")
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
    
    data_loader = SituatedQALoader(config=data_config)
    
    # Get temporally-filtered streams
    stream_stable, stream_update = data_loader.get_temporal_streams(split="train")
    
    # Format stable stream for training
    train_dataset = data_loader.get_formatted_stream(stream_stable, shuffle=True)
    
    logger.info(f"✓ Data loaded (temporal cutoff: {args.cutoff_year})")
    logger.info(f"  stream_stable: Facts before {args.cutoff_year} (for Base Adapter)")
    logger.info(f"  stream_update: Facts from {args.cutoff_year}+ (reserved for updates)")
    
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
        name="situatedqa_base_v1",
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
    
    training_config = TrainingConfig(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        max_seq_length=args.max_seq_length,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        seed=args.seed,
        dataset_buffer_size=args.buffer_size,
    )
    
    trainer = PatchAndRouteTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        config=training_config,
    )
    
    # Run training
    metrics = trainer.train()
    
    # =========================================================================
    # Save Final Checkpoint
    # =========================================================================
    output_path = trainer.save_model()
    
    # Save training configuration for reproducibility
    config_dict = {
        "model_id": args.model_id,
        "quantization": args.quantization,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "temporal_cutoff_year": args.cutoff_year,
        "seed": args.seed,
        "metrics": metrics,
    }
    save_config(config_dict, output_path / "training_config.json")
    
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Expert Adapter saved to: {output_path}")
    logger.info("\nNext steps:")
    logger.info("  1. Evaluate on held-out temporal data (stream_update)")
    logger.info("  2. Train additional Expert Adapters for knowledge updates")
    logger.info("  3. Implement Knowledge Router for dynamic adapter selection")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()

