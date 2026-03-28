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

    # Baseline modes
    parser.add_argument(
        "--monolithic",
        type=str,
        default=None,
        help="Path to monolithic adapter (bypasses routing for baseline comparison)",
    )
    parser.add_argument(
        "--no_adapter",
        action="store_true",
        help=(
            "Evaluate the frozen base model with no adapter and no routing. "
            "Use as Pass 1 of the CFR two-pass protocol to get the foundation "
            "baseline before any patches are applied."
        ),
    )

    # X-LoRA baseline
    parser.add_argument(
        "--xlora",
        type=str,
        default=None,
        help="Path to X-LoRA gating checkpoint (replaces PnR routing with soft adapter blending)",
    )

    # RLEdit baseline
    parser.add_argument(
        "--rledit",
        type=str,
        default=None,
        help=(
            "Path to RLEdit hypernetwork checkpoint directory produced by "
            "train_rledit_baseline.py (contains rledit_hypernetwork.pt + rledit_config.json). "
            "Applies direct weight edits via a trained RL hypernetwork."
        ),
    )
    parser.add_argument(
        "--rledit_edits",
        type=str,
        default=None,
        help=(
            "Path to JSON file with edit pairs to apply before evaluation. "
            "Each entry: {\"question\": str, \"answer\": str} or [question, answer]. "
            "If omitted, evaluates the unedited base model through the hypernetwork."
        ),
    )

    # Parallel Orchestrator
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Use the Parallel-Orchestrator architecture (multi-adapter generation + synthesis)",
    )
    parser.add_argument(
        "--parallel_max_adapters",
        type=int,
        default=5,
        help="Maximum adapters for parallel execution",
    )
    parser.add_argument(
        "--parallel_planner",
        type=str,
        choices=["heuristic", "llm"],
        default="heuristic",
        help="Query planner mode for the Parallel Orchestrator",
    )
    parser.add_argument(
        "--parallel_synth_tokens",
        type=int,
        default=512,
        help="Maximum tokens for the synthesis pass",
    )

    # RECIPE baseline
    parser.add_argument(
        "--recipe",
        type=str,
        default=None,
        help=(
            "Path to RECIPE module checkpoint directory produced by "
            "train_recipe_baseline.py (contains recipe_module.pt + recipe_config.json). "
            "Applies lifelong knowledge editing via retrieval-augmented continuous prompts."
        ),
    )
    parser.add_argument(
        "--recipe_edits",
        type=str,
        default=None,
        help=(
            "Path to JSON file with knowledge edits to load into the RECIPE repository "
            "before evaluation.  Each entry: {\"question\": str, \"answer\": str}. "
            "If omitted, evaluates the base model through the trained RECIPE module "
            "with an empty repository (no edits applied)."
        ),
    )

    # MORPHEUS architecture
    parser.add_argument(
        "--morpheus",
        action="store_true",
        help="Use the MORPHEUS multi-system architecture (prototype-based routing, knowledge store, meta-controller)",
    )
    parser.add_argument(
        "--morpheus_state_dir",
        type=str,
        default=None,
        help="Path to MORPHEUS state directory (router state, expert bank, knowledge store)",
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
        no_adapter=args.no_adapter,
        xlora_checkpoint=args.xlora,
        parallel_orchestrator=args.parallel,
        parallel_max_adapters=args.parallel_max_adapters,
        parallel_query_planner=args.parallel_planner,
        parallel_synthesis_tokens=args.parallel_synth_tokens,
        morpheus=args.morpheus,
        morpheus_state_dir=args.morpheus_state_dir,
        rledit_checkpoint=args.rledit,
        rledit_edits_path=args.rledit_edits,
        recipe_checkpoint=args.recipe,
        recipe_edits_path=args.recipe_edits,
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
    if config.no_adapter:
        logger.info("  Mode: Frozen base model only (no adapter, no routing) — CFR baseline")
    elif config.parallel_orchestrator:
        logger.info(
            f"  Mode: Parallel Orchestrator "
            f"(max_adapters={config.parallel_max_adapters}, "
            f"planner={config.parallel_query_planner})"
        )
    elif config.morpheus:
        logger.info(f"  Mode: MORPHEUS multi-system architecture")
        if config.morpheus_state_dir:
            logger.info(f"         State dir: {config.morpheus_state_dir}")
    elif config.recipe_checkpoint:
        logger.info(f"  Mode: RECIPE — {config.recipe_checkpoint}")
        if config.recipe_edits_path:
            logger.info(f"         Edits: {config.recipe_edits_path}")
    elif config.rledit_checkpoint:
        logger.info(f"  Mode: RLEdit — {config.rledit_checkpoint}")
        if config.rledit_edits_path:
            logger.info(f"         Edits: {config.rledit_edits_path}")
    elif config.xlora_checkpoint:
        logger.info(f"  Mode: X-LoRA — {config.xlora_checkpoint}")
    elif config.monolithic_adapter:
        logger.info(f"  Mode: Monolithic adapter — {config.monolithic_adapter}")
    else:
        logger.info("  Mode: PnR routing")
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
