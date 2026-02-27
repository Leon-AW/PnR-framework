#!/usr/bin/env python3
"""
Train Monolithic Baseline
=========================

Main entry point for training a single LoRA adapter on combined local JSON datasets.

This script implements the monolithic baseline approach:
1. Load multiple JSON QA files and combine them
2. Apply simple chat format (question -> answer)
3. Train a single LoRA adapter
4. Save checkpoint

Usage:
    python train_monolithic_baseline.py --data_paths data/archive.json data/current.json

    Options:
        --data_paths        Paths to JSON files (can specify multiple)
        --output_dir        Checkpoint directory
        --model_id          Base model (default: mistralai/Mistral-7B-Instruct-v0.3)
        --quantization      none, int8, int4 (default: int4)
        --lora_r            LoRA rank (default: 16)
        --lora_alpha        LoRA alpha (default: 32)
        --max_steps         Training steps (default: 2000)
        --batch_size        Per-device batch size (default: 4)
        --learning_rate     Peak LR (default: 2e-4)
        --system_prompt     Custom system prompt (optional)

Example:
    python train_monolithic_baseline.py \\
        --data_paths data/archive.json data/current.json \\
        --output_dir checkpoints/monolithic_v1 \\
        --max_steps 2000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loaders.local_loader import LocalJSONLoader, LocalJSONConfig
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
        description="Train monolithic LoRA adapter on combined JSON datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data configuration
    parser.add_argument(
        "--data_paths",
        type=str,
        nargs="+",
        required=True,
        help="Paths to JSON files (can specify multiple)",
    )
    parser.add_argument(
        "--include_negatives",
        action="store_true",
        default=True,
        help="Include negative (unanswerable) samples",
    )
    parser.add_argument(
        "--no_negatives",
        action="store_true",
        help="Exclude negative samples",
    )
    parser.add_argument(
        "--language_filter",
        type=str,
        default=None,
        help="Filter by language code (e.g., 'en', 'de')",
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=None,
        help="Custom system prompt (uses default if not specified)",
    )
    parser.add_argument(
        "--validation_split",
        type=float,
        default=0.1,
        help="Fraction of data for validation",
    )

    # Model configuration
    parser.add_argument(
        "--model_id",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        help="HuggingFace model identifier",
    )
    parser.add_argument(
        "--target_devices",
        type=int,
        nargs="+",
        default=None,
        help="Specific GPU/MIG device IDs to use (e.g., 0 1 2 for multi-GPU)",
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
        default=2000,
        help="Maximum training steps",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,  # Reduced for 14B model on 24GB GPU
        help="Per-device batch size",
    )
    parser.add_argument(
        "--gradient_accumulation",
        type=int,
        default=16,  # Increased to maintain effective batch size
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
        default=4096,  # Increased for Chain-of-Thought (analysis field can be long)
        help="Maximum sequence length (use 4096+ for CoT to avoid truncation)",
    )

    # Output configuration
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/monolithic_v1",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=200,
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
    
    # Chain-of-Thought (DeepSeek-R1)
    parser.add_argument(
        "--no_chain_of_thought",
        action="store_true",
        help="Disable Chain-of-Thought training (omit <think> blocks from analysis field)",
    )

    return parser.parse_args()


def validate_gpu_configuration(target_devices: list[int] | None) -> None:
    """Validate GPU configuration before training.

    Catches common issues that cause training failures:
    - Distributed training with insufficient GPUs
    - MIG device mismatch
    - Memory constraints
    """
    import torch
    import os

    # Check if running in distributed mode (launched by torchrun/accelerate)
    world_size = os.environ.get("WORLD_SIZE")
    local_rank = os.environ.get("LOCAL_RANK")

    if world_size is not None:
        world_size = int(world_size)
        device_count = torch.cuda.device_count()

        if world_size > device_count:
            raise RuntimeError(
                f"Distributed training misconfiguration detected!\n"
                f"  WORLD_SIZE={world_size} but only {device_count} CUDA devices visible.\n"
                f"  This will cause 'CUDA error: invalid device ordinal'.\n\n"
                f"Solutions:\n"
                f"  1. For single-GPU training: unset WORLD_SIZE LOCAL_RANK RANK\n"
                f"  2. For multi-GPU: ensure --gres=gpu:N matches num_processes in accelerate\n"
                f"  3. Use --target_devices to restrict to specific devices"
            )

        print(f"[INFO] Distributed training: WORLD_SIZE={world_size}, LOCAL_RANK={local_rank}")

    # Check device availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Check your GPU drivers and CUDA installation.")

    device_count = torch.cuda.device_count()
    if device_count == 0:
        raise RuntimeError("No CUDA devices found. Check CUDA_VISIBLE_DEVICES setting.")

    # Validate target devices
    if target_devices is not None:
        for d in target_devices:
            if d >= device_count:
                raise RuntimeError(
                    f"Target device {d} does not exist. "
                    f"Only {device_count} devices available (0-{device_count-1}).\n"
                    f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<not set>')}"
                )

    # Check memory on first device
    device_id = target_devices[0] if target_devices else 0
    props = torch.cuda.get_device_properties(device_id)
    memory_gb = props.total_memory / 1024**3

    if memory_gb < 20:
        print(f"[WARNING] Device {device_id} has only {memory_gb:.1f} GB VRAM.")
        print("          DeepSeek-R1-14B with 4-bit quantization needs ~18-20 GB.")
        print("          Training may fail with OOM errors.")
    else:
        print(f"[OK] Device {device_id}: {props.name} with {memory_gb:.1f} GB VRAM")


def main() -> None:
    """Main training pipeline."""
    args = parse_args()

    # Validate GPU configuration early to catch issues before loading model
    validate_gpu_configuration(args.target_devices)

    # Handle negatives flag
    include_negatives = args.include_negatives and not args.no_negatives

    # =========================================================================
    # Setup Logging
    # =========================================================================
    configure_framework_logging(level=args.log_level)
    logger = setup_logger("train_monolithic", level=args.log_level)

    logger.info("=" * 70)
    logger.info("MONOLITHIC BASELINE: COMBINED DATASET TRAINING")
    logger.info("=" * 70)
    logger.info(f"Model: {args.model_id}")
    logger.info(f"Quantization: {args.quantization}")
    logger.info(f"LoRA rank: {args.lora_r}")
    logger.info(f"Max steps: {args.max_steps}")
    logger.info(f"Data files: {len(args.data_paths)}")
    for p in args.data_paths:
        logger.info(f"  - {p}")
    logger.info(f"Include negatives: {include_negatives}")
    logger.info(f"Output: {args.output_dir}")
    logger.info("=" * 70)

    # =========================================================================
    # Load Data
    # =========================================================================
    logger.info("\n[1/4] Loading JSON datasets...")

    data_config = LocalJSONConfig(
        data_paths=args.data_paths,
        format_type="simple",
        include_negatives=include_negatives,
        validation_split=args.validation_split,
        language_filter=args.language_filter,
        user_prefix=args.system_prompt,  # DeepSeek-R1: No system prompt, use as user prefix
        seed=args.seed,
        use_chain_of_thought=not args.no_chain_of_thought,  # Enable CoT by default
    )

    data_loader = LocalJSONLoader(config=data_config)
    dataset = data_loader.load()

    # Get statistics
    stats = data_loader.get_statistics()
    logger.info(f"Total samples: {stats['total_samples']}")
    logger.info(f"Languages: {stats['languages']}")
    logger.info(f"Categories: {stats['intention_categories']}")

    # Handle split dataset
    if isinstance(dataset, dict):
        train_dataset = dataset['train']
        eval_dataset = dataset['test']
        logger.info(f"Train: {len(train_dataset)}, Validation: {len(eval_dataset)}")
    else:
        train_dataset = dataset
        eval_dataset = None
        logger.info(f"Train: {len(train_dataset)}, Validation: None")

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
        target_devices=args.target_devices,
    )

    llm = PatchAndRouteLLM(foundation_config=foundation_config)
    llm.load_frozen_foundation()

    logger.info("\n[3/4] Attaching Expert Adapter (LoRA)...")

    expert_config = ExpertConfig(
        name="monolithic_baseline",
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )

    llm.attach_expert(expert_config)
    llm.print_model_info()

    model, tokenizer = llm.get_training_components()

    # =========================================================================
    # Training
    # =========================================================================
    logger.info("\n[4/4] Starting training...")

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
    )

    # Convert Dataset to format expected by trainer
    # The trainer expects a 'messages' field which our loader provides
    def formatting_func(example):
        return tokenizer.apply_chat_template(
            example['messages'],
            tokenize=False,
            add_generation_prompt=False,
        )

    trainer = PatchAndRouteTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        config=training_config,
        formatting_func=formatting_func,
    )

    metrics = trainer.train()

    # =========================================================================
    # Save Final Checkpoint
    # =========================================================================
    output_path = trainer.save_model()

    # Save training configuration for reproducibility
    config_dict = {
        "training_type": "monolithic_baseline",
        "model_id": args.model_id,
        "quantization": args.quantization,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "data_paths": args.data_paths,
        "include_negatives": include_negatives,
        "language_filter": args.language_filter,
        "system_prompt": args.system_prompt,
        "seed": args.seed,
        "data_statistics": stats,
        "metrics": metrics,
    }
    save_config(config_dict, output_path / "training_config.json")

    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Adapter saved to: {output_path}")
    logger.info("\nTo use this adapter:")
    logger.info("  from src.models.core import PatchAndRouteLLM")
    logger.info("  llm = PatchAndRouteLLM()")
    logger.info("  llm.load_frozen_foundation()")
    logger.info(f"  llm.load_expert('{output_path}')")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
