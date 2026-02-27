#!/usr/bin/env python3
"""
Evaluate Patch-and-Route Framework
====================================

CLI entry point for running evaluation on the PnR framework.

Measures answer quality (EM, F1), routing accuracy, ESR, stability score,
and optionally LLM-as-a-judge scoring. Results are logged to MLflow and
saved as JSON files.

Usage:
    python eval_pnr.py --eval_sets base temporal --n_samples 200

    Baseline comparison (two-pass):
        python eval_pnr.py --monolithic checkpoints/monolithic_v1 --eval_sets base --n_samples 100 --run_name baseline
        python eval_pnr.py --eval_sets base --n_samples 100 --run_name pnr_v1

    Options:
        --eval_sets         Splits to evaluate (base, temporal, geo_india, local, ...)
        --n_samples         Max samples per split (default: 200)
        --local_data_paths  JSON files for "local" split
        --model_id          Base model (default: deepseek-ai/DeepSeek-R1-Distill-Qwen-14B)
        --checkpoints_dir   Adapter checkpoints directory
        --monolithic        Path to monolithic adapter (bypasses routing)
        --use_llm_judge     Enable LLM-as-a-judge scoring
        --output_dir        Results output directory

Example:
    python eval_pnr.py \\
        --eval_sets base temporal geo_india \\
        --n_samples 50 \\
        --experiment_name pnr-evaluation \\
        --run_name pnr_v1_50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.eval.dataset import KNOWN_GEO_ADAPTERS
from src.eval.runner import EvalConfig, EvalRunner, VALID_SPLITS
from src.utils.logging import setup_logger, configure_framework_logging


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate the Patch-and-Route framework",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Evaluation configuration
    parser.add_argument(
        "--eval_sets",
        type=str,
        nargs="+",
        default=["base", "temporal"],
        choices=sorted(VALID_SPLITS),
        help="Dataset splits to evaluate on",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=200,
        help="Maximum samples per split",
    )
    parser.add_argument(
        "--local_data_paths",
        type=str,
        nargs="+",
        default=[],
        help="Paths to local JSON files (for 'local' split)",
    )

    # Model configuration
    parser.add_argument(
        "--model_id",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        help="HuggingFace model identifier",
    )
    parser.add_argument(
        "--checkpoints_dir",
        type=str,
        default="checkpoints",
        help="Directory containing adapter checkpoints",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default=None,
        help="Path to embedding model for the router",
    )
    parser.add_argument(
        "--router_state",
        type=str,
        default=None,
        help="Path to saved router state",
    )
    parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=0.65,
        help="Router similarity threshold",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        choices=["none", "int8", "int4"],
        default="int4",
        help="Quantization type for memory efficiency",
    )

    # Monolithic baseline
    parser.add_argument(
        "--monolithic",
        type=str,
        default=None,
        help="Path to monolithic adapter (bypasses routing for baseline comparison)",
    )

    # Generation configuration
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate per sample",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature (low for reproducibility)",
    )

    # LLM Judge
    parser.add_argument(
        "--use_llm_judge",
        action="store_true",
        help="Enable LLM-as-a-judge scoring",
    )

    # MLflow
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="pnr-evaluation",
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="MLflow run name (auto-generated if not specified)",
    )

    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="eval_results",
        help="Directory for saving evaluation results",
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
    """Main evaluation pipeline."""
    args = parse_args()

    # =========================================================================
    # Setup Logging
    # =========================================================================
    configure_framework_logging(level=args.log_level)
    logger = setup_logger("eval_pnr", level=args.log_level)

    logger.info("=" * 70)
    logger.info("PATCH-AND-ROUTE EVALUATION")
    logger.info("=" * 70)

    # =========================================================================
    # [1/4] Validate Configuration
    # =========================================================================
    logger.info("\n[1/4] Validating configuration...")

    if "local" in args.eval_sets and not args.local_data_paths:
        logger.error("--eval_sets includes 'local' but no --local_data_paths provided")
        sys.exit(1)

    config = EvalConfig(
        model_id=args.model_id,
        checkpoints_dir=args.checkpoints_dir,
        embedding_model=args.embedding_model,
        router_state_path=args.router_state,
        similarity_threshold=args.similarity_threshold,
        quantization=args.quantization,
        eval_sets=args.eval_sets,
        n_samples=args.n_samples,
        local_data_paths=args.local_data_paths,
        monolithic_adapter=args.monolithic,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        do_sample=False,
        mlflow_experiment=args.experiment_name,
        mlflow_run_name=args.run_name,
        output_dir=args.output_dir,
        use_llm_judge=args.use_llm_judge,
    )

    logger.info(f"  Model: {config.model_id}")
    logger.info(f"  Quantization: {config.quantization}")
    logger.info(f"  Eval sets: {config.eval_sets}")
    logger.info(f"  Samples per split: {config.n_samples}")
    logger.info(f"  Monolithic adapter: {config.monolithic_adapter or 'None (using routing)'}")
    logger.info(f"  LLM Judge: {config.use_llm_judge}")
    logger.info(f"  Output: {config.output_dir}")
    logger.info("=" * 70)

    # =========================================================================
    # [2/4]-[4/4] Run Evaluation (handled by EvalRunner)
    # =========================================================================
    runner = EvalRunner(config)
    report = runner.run()

    # =========================================================================
    # Print Summary
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 70)

    summary = report.get("summary", {})
    if summary:
        logger.info(f"  Total samples: {summary.get('n_samples', 0)}")
        logger.info(f"  Exact Match:   {summary.get('exact_match_overall', 'N/A')}")
        logger.info(f"  F1 Score:      {summary.get('f1_overall', 'N/A')}")

        ra = summary.get("routing_accuracy")
        if ra is not None:
            logger.info(f"  Routing Acc:   {ra}")

        esr = summary.get("esr")
        if esr is not None:
            logger.info(f"  ESR:           {esr}")

        ss = summary.get("stability_score")
        if ss is not None:
            logger.info(f"  Stability:     {ss}")

        cfr = summary.get("cfr")
        if cfr is not None:
            logger.info(f"  CFR:           {cfr}")

        eff = summary.get("efficiency", {})
        if eff:
            logger.info(f"  Avg Latency:   {eff.get('avg_latency_ms', 'N/A')} ms")
            logger.info(f"  P95 Latency:   {eff.get('p95_latency_ms', 'N/A')} ms")
            logger.info(f"  Peak VRAM:     {eff.get('peak_vram_mb', 'N/A')} MB")

    run_name = config.mlflow_run_name or "latest"
    logger.info(f"\n  Results: {config.output_dir}/{run_name}/")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
