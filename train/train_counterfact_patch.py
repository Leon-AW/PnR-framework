#!/usr/bin/env python3
"""
Train CounterFact Patch Adapter
================================

Trains a single LoRA adapter on all 21,919 CounterFact QA pairs.

The adapter learns: given a counterfactual relation prompt, output the
counterfactual answer (target_false). This is the "D_conflict" side of the
CounterFact evaluation:
  - D_conflict: CF questions → adapter must output target_false (ESR)
  - D_control:  TriviaQA questions → router must NOT activate adapter (stability)

Prerequisite: run scripts/build_counterfact_data.py first to create
  data/counterfact_train.jsonl

Usage:
    python train/train_counterfact_patch.py
    python train/train_counterfact_patch.py --max_steps 3000 --adapter_name patch_cf_v2

Author: Leon Wagner
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

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
    parser = argparse.ArgumentParser(
        description="Train CounterFact patch adapter",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    parser.add_argument("--data_path", type=str,
                        default="data/counterfact_train.jsonl",
                        help="Path to counterfact_train.jsonl")

    # Adapter
    parser.add_argument("--adapter_name", type=str,
                        default="patch_cf_main",
                        help="Name for the adapter checkpoint")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: checkpoints/<adapter_name>)")

    # Model
    parser.add_argument("--model_id", type=str,
                        default="mistralai/Mistral-7B-Instruct-v0.3",
                        help="Base model HuggingFace ID")
    parser.add_argument("--quantization", type=str,
                        choices=["none", "int8", "int4"], default="int4",
                        help="Quantization type")

    # LoRA
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha")

    # Training
    parser.add_argument("--max_steps", type=int, default=2000,
                        help="Max training steps (~1.6 epochs of 19,728 records at eff. batch 16)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Per-device train batch size")
    parser.add_argument("--gradient_accumulation", type=int, default=16,
                        help="Gradient accumulation steps (effective batch = batch_size × grad_accum)")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Peak learning rate")
    parser.add_argument("--max_seq_length", type=int, default=256,
                        help="Max sequence length (CF QA pairs are short)")
    parser.add_argument("--save_steps", type=int, default=200,
                        help="Steps between checkpoint saves")
    parser.add_argument("--logging_steps", type=int, default=25,
                        help="Steps between logging")
    parser.add_argument("--optim", type=str, default="paged_adamw_8bit",
                        help="Optimizer")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    return parser.parse_args()


def load_counterfact_dataset(data_path: str, seed: int):
    """Load counterfact JSONL and return as IterableDataset.

    The JSONL already has a `messages` field in the format SFTTrainer expects:
      [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    """
    from datasets import load_dataset

    ds = load_dataset("json", data_files=data_path, split="train")
    n = len(ds)

    # Convert to iterable for streaming-compatible training
    iterable = ds.to_iterable_dataset().shuffle(seed=seed, buffer_size=10_000)

    return iterable, n


def main() -> None:
    args = parse_args()

    adapter_name = args.adapter_name
    output_dir = args.output_dir or f"checkpoints/{adapter_name}"

    configure_framework_logging(level=args.log_level)
    logger = setup_logger("train_counterfact_patch", level=args.log_level)

    logger.info("=" * 70)
    logger.info("PATCH-AND-ROUTE: COUNTERFACT PATCH TRAINING")
    logger.info("=" * 70)
    logger.info(f"Adapter name : {adapter_name}")
    logger.info(f"Data path    : {args.data_path}")
    logger.info(f"Model        : {args.model_id}")
    logger.info(f"Quantization : {args.quantization}")
    logger.info(f"Max steps    : {args.max_steps}")
    logger.info(f"LoRA r/alpha : {args.lora_r}/{args.lora_alpha}")
    logger.info(f"Eff. batch   : {args.batch_size * args.gradient_accumulation}")
    logger.info(f"Output       : {output_dir}")
    logger.info("=" * 70)

    # Verify data exists
    if not Path(args.data_path).exists():
        logger.error(f"Data file not found: {args.data_path}")
        logger.error("Run: python scripts/build_counterfact_data.py")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    logger.info("\n[1/4] Loading CounterFact dataset...")
    train_dataset, n_records = load_counterfact_dataset(args.data_path, args.seed)
    logger.info(f"  {n_records:,} training records loaded")

    # ------------------------------------------------------------------
    # 2. Initialize model
    # ------------------------------------------------------------------
    logger.info("\n[2/4] Loading frozen foundation...")

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

    # ------------------------------------------------------------------
    # 3. Attach LoRA adapter
    # ------------------------------------------------------------------
    logger.info("\n[3/4] Attaching LoRA adapter...")

    expert_config = ExpertConfig(
        name=adapter_name,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )
    llm.attach_expert(expert_config)
    llm.print_model_info()

    model, tokenizer = llm.get_training_components()

    # ------------------------------------------------------------------
    # 4. Train
    # ------------------------------------------------------------------
    logger.info("\n[4/4] Starting training...")

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
        optim=args.optim,
    )

    # ------------------------------------------------------------------
    # 5. Save training config
    # ------------------------------------------------------------------
    config_dict = {
        "adapter_name": adapter_name,
        "adapter_type": "patch_cf",
        "model_id": args.model_id,
        "quantization": args.quantization,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "data_path": str(Path(args.data_path).resolve()),
        "n_train_records": n_records,
        "dataset": "NeelNanda/counterfact-tracing",
        "prompt_field": "prompt",
        "answer_field": "target_false",
        "metrics": metrics,
    }

    save_config(config_dict, Path(output_dir) / "training_config.json")

    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Checkpoint: {output_dir}")
    final_loss = metrics.get("train_loss", metrics.get("final_loss", "N/A"))
    logger.info(f"Final loss: {final_loss}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
