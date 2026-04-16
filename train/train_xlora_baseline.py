#!/usr/bin/env python3
"""
Train X-LoRA Baseline
======================

Trains only the X-LoRA gating classifier on top of pre-trained LoRA adapters.
No LoRA weights are updated; only the gating network learns to mix adapters
per-layer at inference time.

This implements the X-LoRA baseline (arXiv:2402.07148) for comparison against
PnR's discrete routing strategy.

Data source: SituatedQA (same as all PnR adapters) — streams the full combined
dataset (base + temporal patch + all geo) so the gating network sees every
query type covered by the expert pool.

Usage:
    python train_xlora_baseline.py \\
        --checkpoints_dir checkpoints/ \\
        --output_dir checkpoints/xlora_baseline \\
        --max_steps 2000 \\
        --run_name xlora_baseline

Options:
    --output_dir        Checkpoint directory (default: checkpoints/xlora_baseline)
    --adapter_paths     Explicit list of LoRA adapter directories
    --checkpoints_dir   Auto-discover adapters from this dir (default: checkpoints/)
    --xlora_depth       X-LoRA gating network depth (default: 8)
    --cutoff_year       Temporal cutoff matching base adapter (default: 2019)
    --buffer_size       Shuffle buffer size for streaming (default: 10000)
    --model_id          Base model (default: mistralai/Mistral-7B-Instruct-v0.3)
    --quantization      none, int8, int4 (default: int4)
    --max_steps         Training steps (default: 2000)
    --batch_size        Per-device batch size (default: 1)
    --gradient_accumulation  Gradient accumulation steps (default: 16)
    --learning_rate     Peak LR (default: 1e-4)
    --max_seq_length    Maximum sequence length (default: 4096)
    --experiment_name   MLflow experiment name (default: pnr-training)
    --run_name          MLflow run name (default: xlora_baseline)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def _check_xlora_installed() -> None:
    """Exit with a clear error if xlora is not installed."""
    try:
        import xlora  # noqa: F401
    except ImportError:
        print(
            "ERROR: xlora is not installed.\n"
            "Install with:\n"
            "  pip install git+https://github.com/EricLBuehler/xlora.git\n",
            file=sys.stderr,
        )
        sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train X-LoRA gating classifier on pre-trained LoRA adapters",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Adapter discovery
    parser.add_argument(
        "--adapter_paths",
        type=str,
        nargs="*",
        default=None,
        help="Explicit list of pre-trained LoRA adapter directories (overrides --checkpoints_dir)",
    )
    parser.add_argument(
        "--checkpoints_dir",
        type=str,
        default="checkpoints",
        help="Auto-discover LoRA adapters from this directory",
    )

    # X-LoRA config
    parser.add_argument(
        "--xlora_depth",
        type=int,
        default=8,
        help="X-LoRA gating network depth",
    )

    # Data (SituatedQA streaming — same source as all PnR adapters)
    parser.add_argument(
        "--cutoff_year",
        type=int,
        default=2019,
        help="Temporal cutoff year (must match the base adapter's cutoff)",
    )
    parser.add_argument(
        "--buffer_size",
        type=int,
        default=10_000,
        help="Shuffle buffer size for streaming",
    )

    # Model
    parser.add_argument(
        "--model_id",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.3",
        help="HuggingFace model identifier — must match the model the adapters were trained on",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        choices=["none", "int8", "int4"],
        default="int4",
        help="Quantization type for the base model",
    )

    # Training
    parser.add_argument(
        "--max_steps",
        type=int,
        default=2000,
        help="Maximum training steps",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Per-device batch size",
    )
    parser.add_argument(
        "--gradient_accumulation",
        type=int,
        default=16,
        help="Gradient accumulation steps (effective batch = batch_size × this)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Peak learning rate for the gating classifier",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=4096,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/xlora_baseline",
        help="Output directory for the X-LoRA gating checkpoint",
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
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Gradient clipping max norm (paper uses 0.3)",
    )

    # MLflow
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
        help="MLflow run name (defaults to 'xlora_baseline')",
    )

    # Misc
    parser.add_argument(
        "--log_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging verbosity",
    )

    return parser.parse_args()


def discover_adapters(checkpoints_dir: str, model_id: str) -> list[str]:
    """Auto-discover final LoRA adapter directories under checkpoints_dir.

    A directory is treated as a LoRA adapter if it contains adapter_config.json.
    Intermediate HuggingFace checkpoint subdirectories (checkpoint-N) are skipped
    — only the top-level final adapter directory for each expert is included.

    Adapters whose base_model_name_or_path does not match *model_id* are skipped
    (they were trained on a different foundation model and will cause shape mismatches).
    """
    import json as _json

    root = Path(checkpoints_dir)
    if not root.exists():
        return []

    adapters = []
    for path in sorted(root.rglob("adapter_config.json")):
        parent = path.parent
        # Skip checkpoint-N subdirs — they are intermediate saves, not final adapters
        if any(part.startswith("checkpoint-") for part in parent.parts):
            continue
        # Skip adapters trained on a different base model
        try:
            cfg = _json.loads(path.read_text())
            base = cfg.get("base_model_name_or_path", "")
            if base and base != model_id:
                logger.warning(
                    "Skipping %s — trained on %s, expected %s",
                    parent, base, model_id,
                )
                continue
        except Exception:
            pass  # if we can't read the config, try loading it anyway
        adapters.append(str(parent))

    return adapters


def validate_gpu_configuration() -> None:
    """Validate GPU configuration before training."""
    import os
    import torch

    world_size = os.environ.get("WORLD_SIZE")
    if world_size is not None:
        world_size = int(world_size)
        device_count = torch.cuda.device_count()
        if world_size > device_count:
            raise RuntimeError(
                f"Distributed training misconfiguration!\n"
                f"  WORLD_SIZE={world_size} but only {device_count} CUDA devices visible."
            )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    props = torch.cuda.get_device_properties(0)
    memory_gb = props.total_memory / 1024**3
    if memory_gb < 20:
        print(f"[WARNING] Device 0 has only {memory_gb:.1f} GB VRAM — may OOM.")
    else:
        print(f"[OK] Device 0: {props.name} with {memory_gb:.1f} GB VRAM")


def main() -> None:
    _check_xlora_installed()
    import xlora

    args = parse_args()
    validate_gpu_configuration()

    from src.utils.logging import setup_logger, configure_framework_logging
    from src.utils.config import save_config

    configure_framework_logging(level=args.log_level)
    logger = setup_logger("train_xlora", level=args.log_level)

    logger.info("=" * 70)
    logger.info("X-LoRA BASELINE: GATING CLASSIFIER TRAINING")
    logger.info("=" * 70)
    logger.info(f"Model:          {args.model_id}")
    logger.info(f"Quantization:   {args.quantization}")
    logger.info(f"X-LoRA depth:   {args.xlora_depth}")
    logger.info(f"Max steps:      {args.max_steps}")
    logger.info(f"Effective BS:   {args.batch_size * args.gradient_accumulation}")
    logger.info(f"Data:           SituatedQA (full combined stream)")
    logger.info(f"Output:         {args.output_dir}")
    logger.info("=" * 70)

    # =========================================================================
    # [1/5] Load SituatedQA — full combined stream
    # =========================================================================
    logger.info("\n[1/5] Loading SituatedQA dataset (streaming)...")

    from datasets import concatenate_datasets
    from src.data.loader import SituatedQALoader, SituatedQAConfig

    data_config = SituatedQAConfig(
        streaming=True,
        buffer_size=args.buffer_size,
        seed=args.seed,
        temporal_cutoff_year=args.cutoff_year,
    )
    loader = SituatedQALoader(config=data_config)

    # Combine every stream that the expert pool covers:
    #   - base:          temporal pre-cutoff + US geo  (base_v1)
    #   - temporal patch: temporal post-cutoff          (patch_temp_2019_plus)
    #   - all non-US geo: all geographic patches        (patch_geo_*)
    base_stream = loader.get_base_stream()
    temporal_patch_stream = loader.get_temporal_patch_stream()
    non_us_stream = loader.get_all_non_us_stream()

    combined_raw = concatenate_datasets([base_stream, temporal_patch_stream, non_us_stream])
    train_dataset = loader.format_stream(combined_raw, shuffle=True)

    logger.info("Combined stream: base + temporal patch + all non-US geo")

    # =========================================================================
    # [2/5] Discover / validate adapter paths
    # =========================================================================
    logger.info("\n[2/5] Resolving adapter paths...")

    if args.adapter_paths:
        adapter_paths = args.adapter_paths
    else:
        adapter_paths = discover_adapters(args.checkpoints_dir, args.model_id)

    if not adapter_paths:
        logger.error(
            f"No LoRA adapters found. Provide --adapter_paths or ensure "
            f"checkpoints exist under --checkpoints_dir={args.checkpoints_dir}"
        )
        sys.exit(1)

    logger.info(f"Found {len(adapter_paths)} adapter(s):")
    for p in adapter_paths:
        logger.info(f"  {p}")

    # =========================================================================
    # [3/5] Load base model + tokenizer, wrap with X-LoRA
    # =========================================================================
    logger.info("\n[3/5] Loading base model and wrapping with X-LoRA...")

    import torch
    from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM

    bnb_config = None
    if args.quantization == "int4":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif args.quantization == "int8":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if bnb_config is None else None,
    )

    # Required by xlora before wrapping
    base_model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # adapters dict: name → path (xlora loads them via PeftModel.from_pretrained internally)
    adapters = {f"adapter_{i}": p for i, p in enumerate(adapter_paths)}

    xlora_config = xlora.xLoRAConfig(
        hidden_size=base_model.config.hidden_size,
        base_model_id=args.model_id,
        device=torch.device(device),
        adapters=adapters,
        xlora_depth=args.xlora_depth,
        layerwise_scalings=True,       # predict per-layer scalings (paper: Λ ∈ R^{s×l×n})
        enable_relu_and_dropout=True,  # ReLU + Dropout(0.2) between layers
        xlora_dropout_p=0.2,           # matches paper's p=0.2
        use_trainable_adapters=False,  # freeze LoRA weights, train gating only
    )

    # add_xlora_to_model loads all adapters via PEFT and wraps the gating classifier.
    # It automatically freezes all LoRA weights (use_trainable_adapters=False).
    model = xlora.add_xlora_to_model(
        model=base_model,
        xlora_config=xlora_config,
        verbose=True,
    )

    # Count trainable params — only the xLoRA classifier should be trainable
    classifier = model.internal_xlora_classifier
    trainable = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.4f}%)")
    logger.info("(Only the xLoRA gating classifier trains; all LoRA and base weights are frozen)")

    # =========================================================================
    # [4/5] Train gating classifier
    # =========================================================================
    logger.info("\n[4/5] Training X-LoRA gating classifier...")

    from src.training.trainer import PatchAndRouteTrainer, TrainingConfig

    training_config = TrainingConfig(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        max_seq_length=args.max_seq_length,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        max_grad_norm=args.max_grad_norm,
        eval_strategy="no",
        load_best_model_at_end=False,
        seed=args.seed,
        mlflow_experiment=args.experiment_name,
        mlflow_run_name=args.run_name or "xlora_baseline",
        neftune_noise_alpha=None,  # disabled: NEFTune is for full fine-tuning, not gating classifier training
    )

    # SituatedQA stream already has a 'text' field from format_stream()
    trainer = PatchAndRouteTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=None,
        config=training_config,
        formatting_func=None,  # stream already formatted
    )

    # Print metrics to stdout so they appear in the SLURM .out log file
    from transformers import TrainerCallback
    class StdoutMetricsCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is None:
                return
            step = state.global_step
            parts = [f"step={step:>5}"]
            for k in ["loss", "mean_token_accuracy", "entropy", "grad_norm", "learning_rate"]:
                if k in logs:
                    parts.append(f"{k}={logs[k]:.4f}")
            print(" | ".join(parts), flush=True)

    trainer.build_trainer()
    trainer.trainer.add_callback(StdoutMetricsCallback())

    metrics = trainer.train()

    # =========================================================================
    # [5/5] Save gating checkpoint
    # =========================================================================
    logger.info("\n[5/5] Saving X-LoRA gating checkpoint...")

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save the gating classifier weights — filename matches what from_pretrained expects
    classifier = model.internal_xlora_classifier
    torch.save(classifier.state_dict(), output_path / "xlora_classifier.pt")

    # Save xlora_config.json in the format from_pretrained reconstructs via xLoRAConfig(**conf).
    # Note: 'device' is intentionally omitted — from_pretrained injects it at load time.
    with open(output_path / "xlora_config.json", "w") as f:
        json.dump({
            "hidden_size": base_model.config.hidden_size,
            "base_model_id": args.model_id,
            "adapters": adapters,
            "xlora_depth": args.xlora_depth,
            "xlora_size": 2048,
            "enable_softmax": True,
            "enable_softmax_topk": False,
            "layerwise_scalings": True,
            "enable_relu_and_dropout": True,
            "use_bias": True,
            "xlora_dropout_p": 0.2,
            "use_trainable_adapters": False,
            "softmax_temperature": 1.0,
            "top_k_lora": None,
            "scaling_pass_value": 0.0,
            "global_scaling_weight": 1.0,
        }, f, indent=2)

    # Save training provenance
    save_config({
        "training_type": "xlora_baseline",
        "model_id": args.model_id,
        "quantization": args.quantization,
        "xlora_depth": args.xlora_depth,
        "adapter_paths": adapter_paths,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "learning_rate": args.learning_rate,
        "cutoff_year": args.cutoff_year,
        "data_source": "SituatedQA (base + temporal patch + all non-US geo)",
        "seed": args.seed,
        "metrics": metrics,
    }, output_path / "training_config.json")

    logger.info("\n" + "=" * 70)
    logger.info("X-LoRA TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Gating checkpoint saved to: {output_path}")
    logger.info("\nTo evaluate:")
    logger.info(f"  python eval_pnr.py --xlora {output_path} --eval_sets base temporal")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
