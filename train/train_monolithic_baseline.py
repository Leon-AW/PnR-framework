#!/usr/bin/env python3
"""
Train Monolithic Baseline
=========================

Trains a single LoRA adapter on all available knowledge — the direct
apples-to-apples comparison against the PnR multi-adapter approach.

Two data modes:

  --situatedqa       (default for thesis evaluation)
      Combines all SituatedQA streams that PnR trains separate adapters on:
        • base stream       (pre-cutoff stable facts + US geo)
        • temporal stream   (post-cutoff temporal updates)
        • all-non-US stream (11 geographic regions)
      This is a fair comparison: same data, one adapter vs. many.

  --data_paths f1.json f2.json ...
      Local JSON files (QM-AIT or custom datasets).

Usage:
    # SituatedQA mode (thesis eval):
    CUDA_VISIBLE_DEVICES=0 python train/train_monolithic_baseline.py \\
        --situatedqa \\
        --output_dir checkpoints/monolithic_v1 \\
        --max_steps 2000

    # Local JSON mode:
    CUDA_VISIBLE_DEVICES=0 python train/train_monolithic_baseline.py \\
        --data_paths data/archive.json data/current.json \\
        --output_dir checkpoints/monolithic_v1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Project root is one level up from train/
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.core import (
    PatchAndRouteLLM,
    FrozenFoundationConfig,
    ExpertConfig,
    QuantizationType,
)
from src.training.trainer import PatchAndRouteTrainer, TrainingConfig
from src.utils.logging import setup_logger, configure_framework_logging
from src.utils.config import save_config


# Geographic regions that PnR trains separate adapters for
SITUATEDQA_GEO_REGIONS = [
    "australia", "california", "canada", "england", "france",
    "germany", "india", "nigeria", "others", "pakistan", "uk",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train monolithic LoRA baseline (single adapter on all data)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- Data mode (mutually exclusive) -------------------------------------
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument(
        "--situatedqa",
        action="store_true",
        help="Train on all SituatedQA streams combined (base + temporal + geo)",
    )
    data_group.add_argument(
        "--data_paths",
        type=str,
        nargs="+",
        metavar="FILE",
        help="Local JSON files for QM-AIT or custom datasets",
    )

    # ---- SituatedQA options -------------------------------------------------
    parser.add_argument(
        "--cutoff_year",
        type=int,
        default=2019,
        help="Year boundary for temporal split (base = before, patch = after)",
    )
    parser.add_argument(
        "--buffer_size",
        type=int,
        default=1000,
        help="Shuffle buffer size for streaming datasets",
    )

    # ---- Local JSON options --------------------------------------------------
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
        help="Filter by language code (e.g. 'en')",
    )
    parser.add_argument(
        "--validation_split",
        type=float,
        default=0.1,
        help="Fraction held out for validation (local JSON mode only)",
    )

    # ---- Model --------------------------------------------------------------
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
        help="Quantization type",
    )

    # ---- LoRA ---------------------------------------------------------------
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")

    # ---- Training -----------------------------------------------------------
    parser.add_argument("--max_steps", type=int, default=2000, help="Max training steps")
    parser.add_argument("--batch_size", type=int, default=1, help="Per-device batch size")
    parser.add_argument(
        "--gradient_accumulation", type=int, default=16, help="Gradient accumulation steps"
    )
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Peak learning rate")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Max sequence length")
    parser.add_argument("--save_steps", type=int, default=200, help="Steps between saves")
    parser.add_argument("--logging_steps", type=int, default=25, help="Steps between logs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # ---- Output -------------------------------------------------------------
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/monolithic_v1",
        help="Checkpoint output directory",
    )

    # ---- Misc ---------------------------------------------------------------
    parser.add_argument(
        "--log_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    parser.add_argument(
        "--experiment_name", type=str, default="pnr-training", help="MLflow experiment name"
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="MLflow run name (defaults to 'monolithic_baseline')",
    )

    return parser.parse_args()


def validate_gpu() -> None:
    import torch
    import os

    world_size = os.environ.get("WORLD_SIZE")
    if world_size is not None:
        world_size = int(world_size)
        device_count = torch.cuda.device_count()
        if world_size > device_count:
            raise RuntimeError(
                f"WORLD_SIZE={world_size} but only {device_count} CUDA devices visible. "
                "Use CUDA_VISIBLE_DEVICES to pin to a single GPU."
            )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available.")

    props = torch.cuda.get_device_properties(0)
    mem_gb = props.total_memory / 1024**3
    print(f"[OK] GPU 0: {props.name} — {mem_gb:.1f} GB VRAM")


def load_situatedqa_combined(args, logger):
    """Load and interleave all SituatedQA streams that PnR trains on."""
    from datasets import interleave_datasets
    from src.data.loader import SituatedQALoader, SituatedQAConfig

    cfg = SituatedQAConfig(
        streaming=True,
        temporal_cutoff_year=args.cutoff_year,
        buffer_size=args.buffer_size,
        seed=args.seed,
    )
    loader = SituatedQALoader(config=cfg)

    logger.info("Loading SituatedQA streams (combined monolithic mode):")
    base_stream = loader.get_base_stream()
    logger.info("  ✓ base stream (pre-%d stable facts + US geo)", args.cutoff_year)

    temporal_stream = loader.get_temporal_patch_stream()
    logger.info("  ✓ temporal stream (post-%d updates)", args.cutoff_year)

    geo_stream = loader.get_all_non_us_stream()
    logger.info("  ✓ all-non-US geo stream (%d regions)", len(SITUATEDQA_GEO_REGIONS))

    combined = interleave_datasets([base_stream, temporal_stream, geo_stream])
    train_dataset = loader.format_stream(combined, shuffle=True)
    logger.info("✓ Combined stream ready (interleaved, shuffled)")
    return train_dataset, loader


def load_local_json(args, logger):
    """Load local JSON files."""
    from src.data.local_loader import LocalJSONLoader, LocalJSONConfig

    include_negatives = args.include_negatives and not args.no_negatives
    cfg = LocalJSONConfig(
        data_paths=args.data_paths,
        format_type="simple",
        include_negatives=include_negatives,
        validation_split=args.validation_split,
        language_filter=args.language_filter,
        seed=args.seed,
        use_chain_of_thought=False,
    )
    loader = LocalJSONLoader(config=cfg)
    dataset = loader.load()
    stats = loader.get_statistics()
    logger.info("Loaded %d samples from %d file(s)", stats["total_samples"], len(args.data_paths))

    if isinstance(dataset, dict):
        return dataset["train"], dataset["test"], stats
    return dataset, None, stats


def main() -> None:
    args = parse_args()
    validate_gpu()

    configure_framework_logging(level=args.log_level)
    logger = setup_logger("train_monolithic", level=args.log_level)

    logger.info("=" * 70)
    logger.info("MONOLITHIC BASELINE — single LoRA on all knowledge")
    logger.info("=" * 70)
    logger.info("Model       : %s", args.model_id)
    logger.info("Quantization: %s", args.quantization)
    logger.info("LoRA rank   : %d  alpha: %d", args.lora_r, args.lora_alpha)
    logger.info("Max steps   : %d", args.max_steps)
    logger.info("Data mode   : %s", "SituatedQA (combined)" if args.situatedqa else f"local JSON ({len(args.data_paths)} files)")
    logger.info("Output      : %s", args.output_dir)
    logger.info("=" * 70)

    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------
    logger.info("\n[1/4] Loading training data...")
    eval_dataset = None
    stats = {}

    if args.situatedqa:
        train_dataset, _ = load_situatedqa_combined(args, logger)
    else:
        train_dataset, eval_dataset, stats = load_local_json(args, logger)

    # -------------------------------------------------------------------------
    # Load model
    # -------------------------------------------------------------------------
    logger.info("\n[2/4] Loading frozen foundation...")

    quant_map = {"none": QuantizationType.NONE, "int8": QuantizationType.INT8, "int4": QuantizationType.INT4}
    foundation_cfg = FrozenFoundationConfig(
        model_id=args.model_id,
        quantization=quant_map[args.quantization],
    )
    llm = PatchAndRouteLLM(foundation_config=foundation_cfg)
    llm.load_frozen_foundation()

    logger.info("\n[3/4] Attaching LoRA adapter...")
    expert_cfg = ExpertConfig(name="monolithic_baseline", r=args.lora_r, lora_alpha=args.lora_alpha)
    llm.attach_expert(expert_cfg)
    llm.print_model_info()

    model, tokenizer = llm.get_training_components()

    # -------------------------------------------------------------------------
    # Train
    # -------------------------------------------------------------------------
    logger.info("\n[4/4] Training...")

    training_cfg = TrainingConfig(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        max_seq_length=args.max_seq_length,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        seed=args.seed,
        mlflow_experiment=args.experiment_name,
        mlflow_run_name=args.run_name or "monolithic_baseline",
    )

    def formatting_func(example):
        return tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )

    trainer = PatchAndRouteTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        config=training_cfg,
        formatting_func=formatting_func,
    )

    metrics = trainer.train()

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------
    output_path = trainer.save_model()

    config_dict = {
        "training_type": "monolithic_baseline",
        "data_mode": "situatedqa" if args.situatedqa else "local_json",
        "model_id": args.model_id,
        "quantization": args.quantization,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "cutoff_year": args.cutoff_year if args.situatedqa else None,
        "data_paths": args.data_paths if not args.situatedqa else None,
        "seed": args.seed,
        "data_statistics": stats,
        "metrics": metrics,
    }
    save_config(config_dict, output_path / "training_config.json")

    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE — adapter saved to: %s", output_path)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
