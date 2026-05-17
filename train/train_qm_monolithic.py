#!/usr/bin/env python3
"""
Train AIT QM Sequential Monolithic Adapter (monolithic_qm)
===========================================================

Demonstrates catastrophic forgetting in the QM domain by training a single
LoRA adapter sequentially:

  Phase 1 — old facts  (answer_old, 500 steps)
             The adapter learns the 500 outdated QM facts.
  Phase 2 — new facts  (answer_new, 500 steps)
             The adapter continues training on the 500 current facts.
             Old QM knowledge is overwritten → catastrophic forgetting.

Compared against PnR routing (base_qm + patch_qm_current with Time-Aware
routing), which preserves both old and new knowledge in separate frozen
adapters.  This is the QM-domain analogue of the CounterFact monolithic
baseline — except that here both knowledge states must be installed
explicitly, since Mistral-7B-Instruct has no prior QM knowledge.

Training is done in a single Python process: the LoRA weights survive
between the two train_adapter() calls, so phase 2 genuinely starts from
the weights produced by phase 1 (no save/reload needed).

Data:
  Phase 1: data/qm_train_old.jsonl  (built by build_qm_train_data.py --answer_field answer_old)
  Phase 2: data/qm_train.jsonl      (current facts, default output of build_qm_train_data.py)

Output: checkpoints/monolithic_qm/  (symlinked from /vol/tmp/wagnerql/checkpoints/monolithic_qm)

Usage:
    python train/train_qm_monolithic.py
    python train/train_qm_monolithic.py --max_steps_per_phase 1000

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
        description="Train AIT QM sequential monolithic adapter (monolithic_qm)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--old_data_path", default="data/qm_train_old.jsonl",
                        help="Phase-1 training data (outdated facts, answer_old)")
    parser.add_argument("--new_data_path", default="data/qm_train.jsonl",
                        help="Phase-2 training data (current facts, answer_new)")
    parser.add_argument("--adapter_name", default="monolithic_qm")
    parser.add_argument("--output_dir",   default=None,
                        help="Checkpoint dir (default: checkpoints/<adapter_name>)")
    parser.add_argument("--model_id",     default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--quantization", choices=["none", "int8", "int4"], default="int4")
    parser.add_argument("--lora_r",       type=int, default=16)
    parser.add_argument("--lora_alpha",   type=int, default=32)
    parser.add_argument("--max_steps_per_phase", type=int, default=500,
                        help="Training steps per phase (same as base_qm / patch_qm_current)")
    parser.add_argument("--batch_size",   type=int, default=1)
    parser.add_argument("--gradient_accumulation", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--save_steps",   type=int, default=100)
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
    phase1_dir   = f"{output_dir}/phase1"

    configure_framework_logging(level=args.log_level)
    logger = setup_logger("train_qm_monolithic", level=args.log_level)

    logger.info("=" * 70)
    logger.info("PATCH-AND-ROUTE: AIT QM SEQUENTIAL MONOLITHIC TRAINING")
    logger.info("=" * 70)
    logger.info(f"Adapter name        : {adapter_name}")
    logger.info(f"Phase-1 data (old)  : {args.old_data_path}")
    logger.info(f"Phase-2 data (new)  : {args.new_data_path}")
    logger.info(f"Steps per phase     : {args.max_steps_per_phase}")
    logger.info(f"Model               : {args.model_id}")
    logger.info(f"LoRA r/alpha        : {args.lora_r}/{args.lora_alpha}")
    logger.info(f"Eff. batch          : {args.batch_size * args.gradient_accumulation}")
    logger.info(f"Output              : {output_dir}")
    logger.info("=" * 70)

    for path in (args.old_data_path, args.new_data_path):
        if not Path(path).exists():
            logger.error("Data file not found: %s", path)
            sys.exit(1)

    logger.info("\n[1/5] Loading datasets...")
    old_dataset, n_old = load_qm_dataset(args.old_data_path, args.seed)
    new_dataset, n_new = load_qm_dataset(args.new_data_path, args.seed)
    logger.info("  Phase-1 (old): %d records", n_old)
    logger.info("  Phase-2 (new): %d records", n_new)

    logger.info("\n[2/5] Loading frozen foundation...")
    quant_map = {"none": QuantizationType.NONE, "int8": QuantizationType.INT8,
                 "int4": QuantizationType.INT4}
    foundation_config = FrozenFoundationConfig(
        model_id=args.model_id,
        quantization=quant_map[args.quantization],
    )
    llm = PatchAndRouteLLM(foundation_config=foundation_config)
    llm.load_frozen_foundation()

    logger.info("\n[3/5] Attaching fresh LoRA adapter...")
    expert_config = ExpertConfig(name=adapter_name, r=args.lora_r, lora_alpha=args.lora_alpha)
    llm.attach_expert(expert_config)
    llm.print_model_info()
    model, tokenizer = llm.get_training_components()

    # ── Phase 1: old facts ────────────────────────────────────────────────────
    logger.info("\n[4/5] PHASE 1 — training on outdated QM facts (answer_old)...")
    logger.info("  LoRA weights start from random init.")
    metrics_phase1 = train_adapter(
        model=model,
        tokenizer=tokenizer,
        dataset=old_dataset,
        adapter_name=f"{adapter_name}_phase1",
        output_dir=phase1_dir,
        max_steps=args.max_steps_per_phase,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        max_seq_length=args.max_seq_length,
        seed=args.seed,
        optim=args.optim,
    )
    logger.info("  Phase-1 complete. Final loss: %s",
                metrics_phase1.get("train_loss", "N/A"))

    # ── Phase 2: new facts ────────────────────────────────────────────────────
    logger.info("\n[5/5] PHASE 2 — training on current QM facts (answer_new)...")
    logger.info("  LoRA weights continue from Phase-1 state → old knowledge is overwritten.")
    metrics_phase2 = train_adapter(
        model=model,
        tokenizer=tokenizer,
        dataset=new_dataset,
        adapter_name=adapter_name,
        output_dir=output_dir,
        max_steps=args.max_steps_per_phase,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        max_seq_length=args.max_seq_length,
        seed=args.seed,
        optim=args.optim,
    )
    logger.info("  Phase-2 complete. Final loss: %s",
                metrics_phase2.get("train_loss", "N/A"))

    config_dict = {
        "adapter_name":          adapter_name,
        "adapter_type":          "monolithic_qm_sequential",
        "model_id":              args.model_id,
        "quantization":          args.quantization,
        "lora_r":                args.lora_r,
        "lora_alpha":            args.lora_alpha,
        "max_steps_per_phase":   args.max_steps_per_phase,
        "batch_size":            args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "learning_rate":         args.learning_rate,
        "seed":                  args.seed,
        "phase1_data_path":      str(Path(args.old_data_path).resolve()),
        "phase2_data_path":      str(Path(args.new_data_path).resolve()),
        "n_train_old":           n_old,
        "n_train_new":           n_new,
        "dataset":               "qm_conflict_pairs (semi-synthetic, AIT QM)",
        "training_order":        "sequential: answer_old → answer_new",
        "phase1_metrics":        metrics_phase1,
        "phase2_metrics":        metrics_phase2,
    }
    save_config(config_dict, Path(output_dir) / "training_config.json")

    logger.info("\n" + "=" * 70)
    logger.info("SEQUENTIAL MONOLITHIC TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Checkpoint : {output_dir}")
    logger.info(f"Phase-1 loss (old facts): {metrics_phase1.get('train_loss', 'N/A')}")
    logger.info(f"Phase-2 loss (new facts): {metrics_phase2.get('train_loss', 'N/A')}")
    logger.info("Catastrophic forgetting of old QM facts is expected.")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
