#!/usr/bin/env python3
"""
Train RAG Baseline
==================

Main entry point for training a LoRA adapter optimized for RAG retrieval context.

This script implements the RAG baseline approach:
1. Load JSON QA file with evidence snippets
2. Chunk source documents semantically
3. Match evidence snippets to relevant chunks
4. Inject 1-2 noise chunks from other documents
5. Train LoRA adapter with RAG-style context
6. Save checkpoint

Usage:
    python train_rag_baseline.py \\
        --data_path data/archive.json \\
        --docs_path data/documents/ \\
        --adapter_name archive_rag

    Options:
        --data_path         Path to single JSON file
        --docs_path         Base path to source documents
        --adapter_name      Name for adapter (default: rag_baseline)
        --output_dir        Checkpoint directory
        --noise_min         Min noise chunks (default: 1)
        --noise_max         Max noise chunks (default: 2)
        --system_prompt     Custom system prompt (optional)
        --model_id          Base model (default: mistralai/Mistral-7B-Instruct-v0.3)
        --quantization      none, int8, int4 (default: int4)
        --lora_r            LoRA rank (default: 16)
        --max_steps         Training steps (default: 2000)

Example:
    # Train separate adapters for different domains
    python train_rag_baseline.py \\
        --data_path data/archive.json \\
        --docs_path data/documents/ \\
        --adapter_name archive_rag \\
        --output_dir checkpoints/

    python train_rag_baseline.py \\
        --data_path data/current.json \\
        --docs_path data/documents/ \\
        --adapter_name current_rag \\
        --output_dir checkpoints/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.data.local_loader import LocalJSONLoader, LocalJSONConfig
from src.data.chunker import ChunkConfig
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
        description="Train RAG-optimized LoRA adapter with document chunking",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data configuration
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to JSON file",
    )
    parser.add_argument(
        "--docs_path",
        type=str,
        required=True,
        help="Base path to source documents",
    )
    parser.add_argument(
        "--adapter_name",
        type=str,
        default="rag_baseline",
        help="Name for this adapter",
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
        help="Custom system prompt (uses default RAG prompt if not specified)",
    )
    parser.add_argument(
        "--validation_split",
        type=float,
        default=0.1,
        help="Fraction of data for validation",
    )

    # Chunking configuration
    parser.add_argument(
        "--noise_min",
        type=int,
        default=1,
        help="Minimum noise chunks to inject",
    )
    parser.add_argument(
        "--noise_max",
        type=int,
        default=2,
        help="Maximum noise chunks to inject",
    )
    parser.add_argument(
        "--max_doc_tokens",
        type=int,
        default=2500,
        help="Threshold for whole-doc vs chunking",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=750,
        help="Target chunk size in tokens",
    )
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=75,
        help="Overlap tokens between chunks",
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
        default=2000,
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
        default=4096,  # Larger for RAG context
        help="Maximum sequence length",
    )

    # Output configuration
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints",
        help="Base output directory for checkpoints",
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

    return parser.parse_args()


def main() -> None:
    """Main training pipeline."""
    args = parse_args()

    # Handle negatives flag
    include_negatives = args.include_negatives and not args.no_negatives

    # Build output directory path
    output_dir = Path(args.output_dir) / args.adapter_name

    # =========================================================================
    # Setup Logging
    # =========================================================================
    configure_framework_logging(level=args.log_level)
    logger = setup_logger("train_rag", level=args.log_level)

    logger.info("=" * 70)
    logger.info("RAG BASELINE: DOCUMENT-AWARE TRAINING")
    logger.info("=" * 70)
    logger.info(f"Model: {args.model_id}")
    logger.info(f"Quantization: {args.quantization}")
    logger.info(f"LoRA rank: {args.lora_r}")
    logger.info(f"Max steps: {args.max_steps}")
    logger.info(f"Adapter name: {args.adapter_name}")
    logger.info(f"Data file: {args.data_path}")
    logger.info(f"Documents path: {args.docs_path}")
    logger.info(f"Noise chunks: {args.noise_min}-{args.noise_max}")
    logger.info(f"Chunk size: {args.chunk_size} tokens")
    logger.info(f"Include negatives: {include_negatives}")
    logger.info(f"Output: {output_dir}")
    logger.info("=" * 70)

    # =========================================================================
    # Load Data with Chunking
    # =========================================================================
    logger.info("\n[1/4] Loading JSON data and chunking documents...")

    chunk_config = ChunkConfig(
        max_doc_tokens=args.max_doc_tokens,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    data_config = LocalJSONConfig(
        data_paths=[args.data_path],
        docs_base_path=args.docs_path,
        format_type="rag",
        noise_chunks=(args.noise_min, args.noise_max),
        include_negatives=include_negatives,
        validation_split=args.validation_split,
        language_filter=args.language_filter,
        system_prompt=args.system_prompt,
        chunk_config=chunk_config,
        seed=args.seed,
    )

    data_loader = LocalJSONLoader(config=data_config)
    dataset = data_loader.load()

    # Get statistics
    stats = data_loader.get_statistics()
    logger.info(f"Total samples: {stats['total_samples']}")
    logger.info(f"Languages: {stats['languages']}")
    logger.info(f"Categories: {stats['intention_categories']}")
    logger.info(f"Samples with evidence: {stats['has_evidence']}")
    logger.info(f"Samples with file_path: {stats['has_file_path']}")
    if 'total_chunks' in stats:
        logger.info(f"Total chunks: {stats['total_chunks']}")
        logger.info(f"Documents chunked: {stats['documents_chunked']}")

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
    )

    llm = PatchAndRouteLLM(foundation_config=foundation_config)
    llm.load_frozen_foundation()

    logger.info("\n[3/4] Attaching Expert Adapter (LoRA)...")

    expert_config = ExpertConfig(
        name=args.adapter_name,
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
        output_dir=str(output_dir),
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
        "training_type": "rag_baseline",
        "adapter_name": args.adapter_name,
        "model_id": args.model_id,
        "quantization": args.quantization,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_seq_length": args.max_seq_length,
        "data_path": args.data_path,
        "docs_path": args.docs_path,
        "noise_chunks": [args.noise_min, args.noise_max],
        "chunk_config": {
            "max_doc_tokens": args.max_doc_tokens,
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap,
        },
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
    logger.info("\nRAG Usage Notes:")
    logger.info("  - This adapter expects context in RAG format")
    logger.info("  - Prepend documents to user query:")
    logger.info("    [Documents:]")
    logger.info("    --- Document 1 ---")
    logger.info("    {chunk_content}")
    logger.info("    ")
    logger.info("    [Question:]")
    logger.info("    {user_question}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
