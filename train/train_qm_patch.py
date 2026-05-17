#!/usr/bin/env python3
"""
Train AIT QM Patch Adapter (patch_qm_current)
=============================================

Trains a LoRA adapter on the 500 AIT QM conflict pairs so that the frozen
Mistral-7B-Instruct-v0.3 base, when routed through this adapter, outputs the
CURRENT correct answer (answer_new) for QM-domain questions.

Data:   data/qm_train.jsonl   (built by scripts/build_qm_train_data.py)
Output: checkpoints/patch_qm_current/

Adapter design mirrors patch_cf_main:
  - LoRA r=16, alpha=32 on q/v projections
  - int4 quantization of the frozen base (QLoRA)
  - Cosine LR schedule with 10% warmup

With 500 records and eff_batch=4, max_steps=500 ≈ 4 epochs — sufficient for
single-attribute memorisation in a small knowledge patch.

Usage:
    python train/train_qm_patch.py
    python train/train_qm_patch.py --max_steps 1000 --adapter_name patch_qm_v2

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
        description="Train AIT QM patch adapter (patch_qm_current)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--data_path",  default="data/qm_train.jsonl")
    parser.add_argument("--adapter_name", default="patch_qm_current")
    parser.add_argument("--adapter_type", default="patch_qm",
                        help="Recorded in training_config.json: 'patch_qm' for "
                             "the current-facts patch, 'base_qm' for the "
                             "outdated-facts base adapter.")
    parser.add_argument("--answer_field", default="answer_new",
                        choices=["answer_new", "answer_old"],
                        help="Which conflict-pair side --data_path was built "
                             "from; recorded in training_config.json.")
    parser.add_argument("--output_dir", default=None,
                        help="Checkpoint dir (default: checkpoints/<adapter_name>)")
    parser.add_argument("--model_id",   default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--quantization", choices=["none", "int8", "int4"], default="int4")
    parser.add_argument("--lora_r",     type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--max_steps",  type=int, default=500,
                        help="Training steps (500 records ÷ eff_batch 4 ≈ 125 steps/epoch → 4 epochs)")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation", type=int, default=4,
                        help="Effective batch = batch_size × grad_accum = 4")
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_seq_length", type=int, default=512,
                        help="QM answers are longer than CounterFact short answers")
    parser.add_argument("--save_steps",    type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=25)
    parser.add_argument("--optim", default="paged_adamw_8bit")
    parser.add_argument("--seed",  type=int, default=42)
    parser.add_argument("--log_level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def load_qm_dataset(data_path: str, seed: int):
    from datasets import load_dataset
    ds = load_dataset("json", data_files=data_path, split="train")
    n = len(ds)
    iterable = ds.to_iterable_dataset().shuffle(seed=seed, buffer_size=n)
    return iterable, n


def main() -> None:
    args = parse_args()

    adapter_name = args.adapter_name
    output_dir   = args.output_dir or f"checkpoints/{adapter_name}"

    configure_framework_logging(level=args.log_level)
    logger = setup_logger("train_qm_patch", level=args.log_level)

    logger.info("=" * 70)
    logger.info("PATCH-AND-ROUTE: AIT QM PATCH TRAINING")
    logger.info("=" * 70)
    logger.info(f"Adapter name   : {adapter_name}")
    logger.info(f"Data path      : {args.data_path}")
    logger.info(f"Model          : {args.model_id}")
    logger.info(f"Quantization   : {args.quantization}")
    logger.info(f"Max steps      : {args.max_steps}")
    logger.info(f"LoRA r/alpha   : {args.lora_r}/{args.lora_alpha}")
    logger.info(f"Eff. batch     : {args.batch_size * args.gradient_accumulation}")
    logger.info(f"Output         : {output_dir}")
    logger.info("=" * 70)

    if not Path(args.data_path).exists():
        logger.error("Data file not found: %s", args.data_path)
        logger.error("Run: python scripts/build_qm_train_data.py")
        sys.exit(1)

    logger.info("\n[1/4] Loading QM training dataset...")
    train_dataset, n_records = load_qm_dataset(args.data_path, args.seed)
    logger.info("  %d training records loaded", n_records)

    logger.info("\n[2/4] Loading frozen foundation...")
    quant_map = {"none": QuantizationType.NONE, "int8": QuantizationType.INT8,
                 "int4": QuantizationType.INT4}
    foundation_config = FrozenFoundationConfig(
        model_id=args.model_id,
        quantization=quant_map[args.quantization],
    )
    llm = PatchAndRouteLLM(foundation_config=foundation_config)
    llm.load_frozen_foundation()

    logger.info("\n[3/4] Attaching LoRA adapter...")
    expert_config = ExpertConfig(name=adapter_name, r=args.lora_r, lora_alpha=args.lora_alpha)
    llm.attach_expert(expert_config)
    llm.print_model_info()
    model, tokenizer = llm.get_training_components()

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

    config_dict = {
        "adapter_name":       adapter_name,
        "adapter_type":       args.adapter_type,
        "model_id":           args.model_id,
        "quantization":       args.quantization,
        "lora_r":             args.lora_r,
        "lora_alpha":         args.lora_alpha,
        "max_steps":          args.max_steps,
        "batch_size":         args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "learning_rate":      args.learning_rate,
        "seed":               args.seed,
        "data_path":          str(Path(args.data_path).resolve()),
        "n_train_records":    n_records,
        "dataset":            "qm_conflict_pairs (semi-synthetic, AIT QM)",
        "prompt_field":       "question",
        "answer_field":       args.answer_field,
        "metrics":            metrics,
    }

    from src.utils.config import save_config
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
