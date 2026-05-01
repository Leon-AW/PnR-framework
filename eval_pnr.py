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
        --model_id          Base model (default: mistralai/Mistral-7B-Instruct-v0.3)
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
        default="mistralai/Mistral-7B-Instruct-v0.3",
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
        default="sentence-transformers/all-MiniLM-L6-v2",
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

    # LoRA + RAG baseline (Baseline 2 from exposé)
    parser.add_argument(
        "--lora_rag",
        type=str,
        default=None,
        help=(
            "Path to monolithic LoRA adapter for the LoRA+RAG hybrid baseline. "
            "Combines fine-tuned adapter inference with QA-pair retrieval at "
            "inference time (requires --lora_rag_index)."
        ),
    )
    parser.add_argument(
        "--lora_rag_index",
        type=str,
        default=None,
        help=(
            "Path to JSON file of {question, answer} pairs to index for retrieval "
            "in the LoRA+RAG baseline.  Typically data/edit_pairs.json."
        ),
    )

    # X-LoRA baseline
    parser.add_argument(
        "--xlora",
        type=str,
        default=None,
        help="Path to X-LoRA gating checkpoint (replaces PnR routing with soft adapter blending)",
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

    # RECIPE baseline (official EMNLP-2024 implementation)
    parser.add_argument(
        "--recipe_official",
        type=str,
        default=None,
        help=(
            "Path to a checkpoint FILE produced by external/RECIPE/train_recipe.py "
            "(the official EMNLP-2024 implementation). Loads the base LLM and the "
            "trained knowl_rep_model + prompt_transformer from the official repo."
        ),
    )
    parser.add_argument(
        "--recipe_official_edits",
        type=str,
        default=None,
        help=(
            "JSON file of edits to populate the official-RECIPE knowledge base "
            "before evaluation. Each entry: {\"question\": str, \"answer\": str}."
        ),
    )

    # CounterFact + TriviaQA (D_conflict / D_control splits)
    parser.add_argument(
        "--counterfact_eval_path",
        type=str,
        default=None,
        help=(
            "Path to data/counterfact_eval.json (produced by "
            "scripts/build_counterfact_data.py). Required when --eval_sets "
            "includes 'cf_conflict'."
        ),
    )
    parser.add_argument(
        "--triviaqa_dcontrol_path",
        type=str,
        default=None,
        help=(
            "Path to data/triviaqa_dcontrol.json (produced by "
            "scripts/build_triviaqa_dcontrol.py). Required when --eval_sets "
            "includes 'cf_control'."
        ),
    )
    parser.add_argument(
        "--sqa_deval_path",
        type=str,
        default=None,
        help=(
            "Path to data/sqa_deval.json (produced by "
            "scripts/build_sqa_deval.py). Required when --eval_sets "
            "includes 'sqa_train'."
        ),
    )
    parser.add_argument(
        "--cf_adapter_name",
        type=str,
        default="patch_cf_main",
        help="Adapter the router should pick for CounterFact D_conflict samples",
    )
    parser.add_argument(
        "--cf_split_name",
        type=str,
        choices=["train", "test"],
        default="test",
        help="Which CounterFact split to evaluate on (held-out 'test' by default)",
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
    parser.add_argument(
        "--morpheus_similarity_threshold",
        type=float,
        default=0.55,
        help=(
            "Similarity threshold for the MORPHEUS PrototypeRouter. "
            "The router applies a JL random projection before cosine "
            "similarity, so the raw-space default (0.65) is too strict "
            "and suppresses most valid routes."
        ),
    )
    parser.add_argument(
        "--morpheus_factuality_threshold_low",
        type=float,
        default=0.65,
        help=(
            "Knowledge store tau_low: queries with sim < this go to parametric_freedom "
            "(no CF injection). Must exceed the max D_control similarity (≤0.619 on "
            "TriviaQA) to prevent CF triples leaking into TriviaQA answers."
        ),
    )
    parser.add_argument(
        "--morpheus_direct_answer_threshold",
        type=float,
        default=0.95,
        help=(
            "Authoritative-override bypass threshold for the MORPHEUS "
            "Knowledge Store. When confidence >= this value AND zone is "
            "hard_override, the LLM is skipped and the stored object_value "
            "is returned verbatim. Set > 1.0 to disable the bypass and "
            "force adapter-based generation (recommended for Patch-and-Route "
            "evaluation, since the bypass reduces MORPHEUS to retrieval "
            "and decouples the result from the activated specialist)."
        ),
    )
    parser.add_argument(
        "--morpheus_classifier_path",
        type=str,
        default=None,
        help=(
            "Path to a trained FactualityClassifier checkpoint directory "
            "(produced by scripts/train_factuality_classifier.py). When set, "
            "the MLP classifier score replaces max_sim as the factuality_score "
            "passed to KnowledgeStore.assess_factuality. Omit to use the "
            "hardcoded tau_low / max_sim fallback."
        ),
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

    # ROME / MEMIT-style log-probability ESR
    parser.add_argument(
        "--compute_logprob",
        action="store_true",
        help=(
            "Additionally compute teacher-forced log P(target | prompt) "
            "for each sample (ROME / MEMIT-style ESR). For 'cf_conflict' "
            "this scores both target_new and target_true so the report "
            "can show whether the edited model assigns higher probability "
            "to the counterfactual than to the original fact — a metric "
            "that sidesteps generation-parsing artefacts."
        ),
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

    if "cf_conflict" in args.eval_sets and not args.counterfact_eval_path:
        logger.error("--eval_sets includes 'cf_conflict' but no --counterfact_eval_path provided")
        sys.exit(1)
    if "cf_control" in args.eval_sets and not args.triviaqa_dcontrol_path:
        logger.error("--eval_sets includes 'cf_control' but no --triviaqa_dcontrol_path provided")
        sys.exit(1)
    if "sqa_train" in args.eval_sets and not args.sqa_deval_path:
        logger.error("--eval_sets includes 'sqa_train' but no --sqa_deval_path provided")
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
        lora_rag_adapter=args.lora_rag,
        lora_rag_index_path=args.lora_rag_index,
        xlora_checkpoint=args.xlora,
        parallel_orchestrator=args.parallel,
        parallel_max_adapters=args.parallel_max_adapters,
        parallel_query_planner=args.parallel_planner,
        parallel_synthesis_tokens=args.parallel_synth_tokens,
        morpheus=args.morpheus,
        morpheus_state_dir=args.morpheus_state_dir,
        morpheus_similarity_threshold=args.morpheus_similarity_threshold,
        morpheus_direct_answer_threshold=args.morpheus_direct_answer_threshold,
        morpheus_factuality_threshold_low=args.morpheus_factuality_threshold_low,
        morpheus_classifier_path=args.morpheus_classifier_path,
        recipe_official_checkpoint=args.recipe_official,
        recipe_official_edits_path=args.recipe_official_edits,
        counterfact_eval_path=args.counterfact_eval_path,
        triviaqa_dcontrol_path=args.triviaqa_dcontrol_path,
        sqa_deval_path=args.sqa_deval_path,
        cf_adapter_name=args.cf_adapter_name,
        cf_split_name=args.cf_split_name,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        do_sample=False,
        mlflow_experiment=args.experiment_name,
        mlflow_run_name=args.run_name,
        output_dir=args.output_dir,
        use_llm_judge=args.use_llm_judge,
        compute_logprob=args.compute_logprob,
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
        if config.morpheus_classifier_path:
            logger.info(f"         Classifier: {config.morpheus_classifier_path}")
        else:
            logger.info(f"         Classifier: tau_low={config.morpheus_factuality_threshold_low} (no classifier)")
    elif config.recipe_official_checkpoint:
        logger.info(f"  Mode: RECIPE (official repo) — {config.recipe_official_checkpoint}")
        if config.recipe_official_edits_path:
            logger.info(f"         Edits: {config.recipe_official_edits_path}")
    elif config.lora_rag_adapter:
        logger.info(f"  Mode: LoRA+RAG — adapter={config.lora_rag_adapter}")
        logger.info(f"         Index:   {config.lora_rag_index_path}")
    elif config.xlora_checkpoint:
        logger.info(f"  Mode: X-LoRA — {config.xlora_checkpoint}")
    elif config.monolithic_adapter:
        logger.info(f"  Mode: Monolithic adapter — {config.monolithic_adapter}")
    else:
        logger.info("  Mode: PnR routing")
    if config.counterfact_eval_path:
        logger.info(
            f"  CounterFact: {config.counterfact_eval_path} "
            f"(split={config.cf_split_name!r}, cf_adapter={config.cf_adapter_name!r})"
        )
    if config.triviaqa_dcontrol_path:
        logger.info(f"  TriviaQA D_control: {config.triviaqa_dcontrol_path}")
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

        cfr_ctrl = summary.get("cfr_control")
        if cfr_ctrl is not None:
            logger.info(f"  CFR (control): {cfr_ctrl}")

        fr = summary.get("dcontrol_forgetting_rate")
        if fr is not None:
            acc = summary.get("dcontrol_accuracy", "N/A")
            logger.info(f"  D_ctrl FR:     {fr}   (acc={acc})")

        lp_esr = summary.get("logprob_esr")
        if lp_esr is not None:
            logger.info(f"  ESR (log-prob):{lp_esr}   (ROME/MEMIT-style)")
        lp_em = summary.get("logprob_em")
        if lp_em is not None:
            logger.info(f"  Log-prob match:{lp_em}")

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
