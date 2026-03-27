"""
Evaluation Runner
=================

Orchestrates end-to-end evaluation of the Patch-and-Route framework.

Provides:
- EvalConfig: Configuration for an evaluation run
- EvalResult: Result of evaluating a single sample
- EvalRunner: Main orchestrator class
"""

from __future__ import annotations

import dataclasses
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .dataset import (
    EvalSample,
    KNOWN_GEO_ADAPTERS,
    build_local_json_dataset,
    build_situated_qa_dataset,
)
from .metrics import (
    compute_efficiency,
    compute_esr,
    compute_routing_accuracy,
    compute_stability_score,
    compute_cfr,
    exact_match,
    parse_model_output,
    token_f1,
)

logger = logging.getLogger(__name__)

# All valid split names
VALID_SPLITS: set[str] = {"base", "temporal", "local"} | {f"geo_{c}" for c in KNOWN_GEO_ADAPTERS}


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class EvalConfig:
    """Configuration for an evaluation run.

    Attributes:
        model_id: HuggingFace model ID for the base LLM.
        checkpoints_dir: Directory containing adapter checkpoints.
        embedding_model: Path to embedding model for the router.
        router_state_path: Path to saved router state.
        similarity_threshold: Router similarity threshold.
        quantization: Quantization type (int4, int8, none).
        eval_sets: List of splits to evaluate on.
        n_samples: Max samples per split.
        local_data_paths: Paths to local JSON files (for "local" split).
        monolithic_adapter: Path to monolithic adapter (bypasses routing).
        no_adapter: Evaluate the frozen base model with no adapter and no routing.
            Used as Pass 1 in the CFR two-pass protocol to measure the foundation
            baseline before any patches are applied.
        max_new_tokens: Maximum tokens to generate per sample.
        temperature: Sampling temperature (low for reproducibility).
        do_sample: Whether to use sampling.
        mlflow_experiment: MLflow experiment name.
        mlflow_run_name: MLflow run name.
        mlflow_tracking_uri: MLflow tracking URI.
        output_dir: Directory for saving results.
        use_llm_judge: Whether to run LLM-as-a-judge scoring.
        use_gpu: Whether to use GPU for inference.
    """
    model_id: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    checkpoints_dir: str = "checkpoints"
    embedding_model: str | None = None
    router_state_path: str | None = None
    similarity_threshold: float = 0.65
    quantization: str = "int4"
    eval_sets: list[str] = field(default_factory=lambda: ["base", "temporal"])
    n_samples: int = 200
    local_data_paths: list[str] = field(default_factory=list)
    monolithic_adapter: str | None = None
    no_adapter: bool = False  # Frozen base model only — no routing, no adapter (CFR baseline)
    max_new_tokens: int = 256
    temperature: float = 0.1
    do_sample: bool = False
    mlflow_experiment: str = "pnr-evaluation"
    mlflow_run_name: str | None = None
    mlflow_tracking_uri: str = "sqlite:///mlruns.db"
    output_dir: str = "eval_results"
    use_llm_judge: bool = False
    use_gpu: bool = True
    xlora_checkpoint: str | None = None  # Path to X-LoRA gating checkpoint
    morpheus: bool = False  # Use MORPHEUS multi-system architecture
    morpheus_state_dir: str | None = None  # Path to MORPHEUS state directory


# =============================================================================
# Result
# =============================================================================

@dataclass
class EvalResult:
    """Result of evaluating a single sample.

    Attributes:
        sample: The evaluation sample.
        raw_prediction: Full model output (with <think> block).
        parsed_answer: Answer after parse_model_output().
        is_exact_match: Whether parsed answer matches any gold answer.
        f1: Token-level F1 score.
        adapter_used: Which adapter was loaded (from InferenceResult).
        routing_correct: True if expected=None OR adapter_used==expected.
        winner_similarity: Router similarity score for the winner.
        has_conflict: Whether routing detected a conflict.
        latency_ms: Inference latency in milliseconds.
        vram_mb: Peak VRAM usage in megabytes.
        judge_score: Optional LLM judge score (1-5).
    """
    sample: EvalSample
    raw_prediction: str
    parsed_answer: str
    is_exact_match: bool
    f1: float
    adapter_used: str | None
    routing_correct: bool
    winner_similarity: float | None
    has_conflict: bool
    latency_ms: float
    vram_mb: float | None
    judge_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Flatten sample + result fields for JSON output."""
        return {
            "question": self.sample.question,
            "gold_answers": self.sample.gold_answers,
            "expected_adapter": self.sample.expected_adapter,
            "split": self.sample.split,
            "raw_prediction": self.raw_prediction,
            "parsed_answer": self.parsed_answer,
            "is_exact_match": self.is_exact_match,
            "f1": self.f1,
            "adapter_used": self.adapter_used,
            "routing_correct": self.routing_correct,
            "winner_similarity": self.winner_similarity,
            "has_conflict": self.has_conflict,
            "latency_ms": self.latency_ms,
            "vram_mb": self.vram_mb,
            "judge_score": self.judge_score,
            "metadata": self.sample.metadata,
        }


# =============================================================================
# Runner
# =============================================================================

class EvalRunner:
    """Orchestrates evaluation of the PnR framework.

    Example::

        config = EvalConfig(eval_sets=["base", "temporal"], n_samples=50)
        runner = EvalRunner(config)
        report = runner.run()
    """

    def __init__(self, config: EvalConfig) -> None:
        self.config = config

    # -------------------------------------------------------------------------
    # Pipeline Construction
    # -------------------------------------------------------------------------

    def _build_xlora_pipeline(self):
        """Build an XLoRAInference pipeline.

        Returns:
            XLoRAInference instance wrapping the gating checkpoint.
        """
        from src.inference.xlora_inference import XLoRAInference

        import torch
        use_gpu = self.config.use_gpu and torch.cuda.is_available()

        return XLoRAInference(
            xlora_checkpoint=self.config.xlora_checkpoint,
            model_id=self.config.model_id,
            quantization=self.config.quantization,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            do_sample=self.config.do_sample,
            use_gpu=use_gpu,
        )

    def _build_morpheus_pipeline(self):
        """Build a MorpheusInference pipeline.

        Returns:
            MorpheusInference instance using the MORPHEUS multi-system architecture.
        """
        from src.morpheus import (
            MorpheusInference,
            MorpheusConfig,
            MorpheusGenerationConfig,
            StableCoreConfig,
            PrototypeRouterConfig,
        )
        import torch

        use_gpu = self.config.use_gpu and torch.cuda.is_available()

        core_config = StableCoreConfig(
            model_id=self.config.model_id,
            quantization=self.config.quantization,
        )
        router_config = PrototypeRouterConfig(
            embedding_model_path=self.config.embedding_model,
            similarity_threshold=self.config.similarity_threshold,
            use_gpu=use_gpu,
        )
        morpheus_config = MorpheusConfig(
            stable_core=core_config,
            router=router_config,
        )
        if self.config.morpheus_state_dir:
            morpheus_config.state_dir = self.config.morpheus_state_dir

        gen_config = MorpheusGenerationConfig(
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            do_sample=self.config.do_sample,
        )

        pipeline = MorpheusInference(
            config=morpheus_config,
            generation_config=gen_config,
        )

        # Register adapters from checkpoints directory
        checkpoints_dir = Path(self.config.checkpoints_dir)
        if checkpoints_dir.exists():
            router = pipeline.get_router()
            n_registered = 0
            for adapter_dir in sorted(checkpoints_dir.iterdir()):
                if adapter_dir.is_dir() and (adapter_dir / "adapter_config.json").exists():
                    adapter_id = adapter_dir.name
                    router.register_adapter(
                        adapter_id=adapter_id,
                        path=str(adapter_dir),
                        timestamp=adapter_dir.stat().st_mtime,
                    )
                    n_registered += 1
            logger.info(f"Registered {n_registered} adapters with MORPHEUS router")

        return pipeline

    def _build_pipeline(self):
        """Build the inference pipeline (PnR, X-LoRA, or MORPHEUS).

        Returns:
            Configured inference pipeline instance.
        """
        if self.config.morpheus:
            return self._build_morpheus_pipeline()
        if self.config.xlora_checkpoint:
            return self._build_xlora_pipeline()
        import torch
        from src.inference import PatchAndRouteInference, GenerationConfig
        from src.models.core import QuantizationType
        from src.routing import CentroidRouter

        # Resolve quantization
        quant_map = {"none": QuantizationType.NONE, "int8": QuantizationType.INT8, "int4": QuantizationType.INT4}
        quantization = quant_map.get(self.config.quantization, QuantizationType.INT4)

        # GPU availability
        use_gpu = self.config.use_gpu and torch.cuda.is_available()
        if self.config.use_gpu and not torch.cuda.is_available():
            logger.warning("GPU requested but CUDA not available — falling back to CPU")

        # Build router
        if self.config.router_state_path and Path(self.config.router_state_path).exists():
            router = CentroidRouter.load(
                path=self.config.router_state_path,
                embedding_model_path=self.config.embedding_model,
                similarity_threshold=self.config.similarity_threshold,
                use_gpu=use_gpu,
            )
        else:
            router = CentroidRouter(
                embedding_model_path=self.config.embedding_model,
                similarity_threshold=self.config.similarity_threshold,
                use_gpu=use_gpu,
            )
            checkpoints_dir = Path(self.config.checkpoints_dir)
            if checkpoints_dir.exists():
                n_registered = router.register_from_checkpoints(str(checkpoints_dir))
                logger.info(f"Registered {n_registered} adapters from {checkpoints_dir}")
            else:
                logger.warning(f"Checkpoints dir not found: {checkpoints_dir}")

        # Build generation config
        gen_config = GenerationConfig(
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            do_sample=self.config.do_sample,
        )

        # Build inference pipeline
        pipeline = PatchAndRouteInference(
            model_id=self.config.model_id,
            router=router,
            quantization=quantization,
            generation_config=gen_config,
            use_gpu=use_gpu,
        )

        return pipeline

    # -------------------------------------------------------------------------
    # Single Sample Evaluation
    # -------------------------------------------------------------------------

    def _run_single(self, sample: EvalSample, pipeline) -> EvalResult:
        """Evaluate a single sample.

        Args:
            sample: The evaluation sample.
            pipeline: PatchAndRouteInference instance.

        Returns:
            EvalResult for this sample.
        """
        import torch

        # Reset VRAM tracking
        vram_mb = None
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # Time the inference
        t_start = time.perf_counter()

        if self.config.xlora_checkpoint:
            result = pipeline.generate(query=sample.question)
        elif self.config.no_adapter:
            # Frozen base model only — skip routing and load no adapter.
            # This is Pass 1 of the CFR protocol: measures what the foundation
            # already knows, providing the true pre-patch baseline.
            result = pipeline.generate(query=sample.question, skip_routing=True)
        elif self.config.monolithic_adapter:
            result = pipeline.generate(
                query=sample.question,
                force_adapter=self.config.monolithic_adapter,
            )
        else:
            result = pipeline.generate(query=sample.question)

        t_end = time.perf_counter()
        latency_ms = (t_end - t_start) * 1000.0

        # VRAM measurement
        if torch.cuda.is_available():
            vram_mb = torch.cuda.max_memory_allocated() / 1e6

        # Parse answer
        parsed = parse_model_output(result.response)

        # Compute metrics
        is_em = exact_match(parsed, sample.gold_answers)
        f1 = token_f1(parsed, sample.gold_answers)

        # Routing correctness
        adapter_used = result.adapter_loaded
        if self.config.morpheus:
            if sample.expected_adapter is None:
                routing_correct = True
            else:
                routing_correct = adapter_used == sample.expected_adapter
            routing_result = result.routing_result
            winner_sim = routing_result.winner_similarity if routing_result else None
            has_conflict = routing_result.has_conflict if routing_result else False
        elif self.config.xlora_checkpoint:
            # X-LoRA blends softly — no discrete routing to evaluate
            routing_correct = True
            winner_sim = None
            has_conflict = False
        elif self.config.no_adapter:
            # Base model only — routing deliberately skipped
            routing_correct = True
            winner_sim = None
            has_conflict = False
        else:
            if sample.expected_adapter is None:
                routing_correct = True
            else:
                routing_correct = adapter_used == sample.expected_adapter

            routing_result = result.routing_result
            winner_sim = routing_result.winner_similarity if routing_result else None
            has_conflict = routing_result.has_conflict if routing_result else False

        return EvalResult(
            sample=sample,
            raw_prediction=result.response,
            parsed_answer=parsed,
            is_exact_match=is_em,
            f1=f1,
            adapter_used=adapter_used,
            routing_correct=routing_correct,
            winner_similarity=winner_sim,
            has_conflict=has_conflict,
            latency_ms=latency_ms,
            vram_mb=vram_mb,
        )

    # -------------------------------------------------------------------------
    # Split-Level Evaluation
    # -------------------------------------------------------------------------

    def _run_split(self, samples: list[EvalSample], pipeline, split_name: str) -> list[EvalResult]:
        """Evaluate all samples in a split.

        Args:
            samples: List of EvalSample for this split.
            pipeline: PatchAndRouteInference instance.
            split_name: Name of the split (for logging).

        Returns:
            List of EvalResult objects.
        """
        results: list[EvalResult] = []

        for sample in tqdm(samples, desc=f"Eval [{split_name}]", unit="sample"):
            try:
                result = self._run_single(sample, pipeline)
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed on sample (split={split_name}): {e}")
                continue

        return results

    # -------------------------------------------------------------------------
    # Report Computation
    # -------------------------------------------------------------------------

    def _compute_report(
        self,
        all_results: list[EvalResult],
        baseline_results: list[EvalResult] | None = None,
    ) -> dict[str, Any]:
        """Compute the evaluation report from all results.

        Args:
            all_results: Combined results across all splits.
            baseline_results: Optional baseline results for CFR computation.

        Returns:
            Report dictionary with summary, per-split breakdowns, and config.
        """
        # Overall metrics
        n_total = len(all_results)
        em_overall = sum(1 for r in all_results if r.is_exact_match) / n_total if n_total else 0.0
        f1_overall = sum(r.f1 for r in all_results) / n_total if n_total else 0.0

        summary: dict[str, Any] = {
            "n_samples": n_total,
            "exact_match_overall": round(em_overall, 4),
            "f1_overall": round(f1_overall, 4),
            "routing_accuracy": compute_routing_accuracy(all_results),
            "esr": compute_esr(all_results),
            "stability_score": compute_stability_score(all_results),
            "efficiency": compute_efficiency(all_results),
        }

        # CFR (requires baseline)
        if baseline_results:
            summary["cfr"] = compute_cfr(all_results, baseline_results)

        # Per-split breakdown
        splits: dict[str, Any] = {}
        split_names = {r.sample.split for r in all_results}
        for split_name in sorted(split_names):
            split_results = [r for r in all_results if r.sample.split == split_name]
            n = len(split_results)
            splits[split_name] = {
                "n": n,
                "exact_match": round(sum(1 for r in split_results if r.is_exact_match) / n, 4) if n else 0.0,
                "f1": round(sum(r.f1 for r in split_results) / n, 4) if n else 0.0,
                "routing_accuracy": compute_routing_accuracy(split_results),
            }

        # Round optional floats in summary
        for key in ("routing_accuracy", "esr", "stability_score", "cfr"):
            if key in summary and summary[key] is not None:
                summary[key] = round(summary[key], 4)

        return {
            "summary": summary,
            "by_split": splits,
            "config": dataclasses.asdict(self.config),
            "timestamp": datetime.now().isoformat(),
        }

    # -------------------------------------------------------------------------
    # Main Entry Point
    # -------------------------------------------------------------------------

    def run(self, baseline_results: list[EvalResult] | None = None) -> dict[str, Any]:
        """Run the full evaluation.

        Args:
            baseline_results: Optional baseline results for CFR computation.

        Returns:
            Report dictionary.
        """
        from src.utils.mlflow_tracker import PnRTracker

        # 1. Validate eval_sets
        unknown = set(self.config.eval_sets) - VALID_SPLITS
        if unknown:
            raise ValueError(f"Unknown eval sets: {unknown}. Valid: {sorted(VALID_SPLITS)}")

        if "local" in self.config.eval_sets and not self.config.local_data_paths:
            raise ValueError("eval_sets includes 'local' but no local_data_paths provided")

        run_name = self.config.mlflow_run_name or f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        with PnRTracker(
            experiment_name=self.config.mlflow_experiment,
            run_name=run_name,
            tracking_uri=self.config.mlflow_tracking_uri,
            tags={"task": "evaluation"},
        ) as tracker:
            # 2. Build pipeline
            logger.info("[2/4] Building inference pipeline...")
            pipeline = self._build_pipeline()

            # 3. Build judge if requested
            judge = None
            if self.config.use_llm_judge:
                from .judge import LLMJudge
                judge = LLMJudge(pipeline)

            # 4. Run evaluation per split
            logger.info("[3/4] Running evaluation...")
            all_results: list[EvalResult] = []

            for split_name in self.config.eval_sets:
                logger.info(f"Building dataset for split={split_name!r}...")

                try:
                    if split_name == "local":
                        samples = build_local_json_dataset(
                            data_paths=self.config.local_data_paths,
                            n_samples=self.config.n_samples,
                        )
                    else:
                        samples = build_situated_qa_dataset(
                            split=split_name,
                            n_samples=self.config.n_samples,
                        )
                except (RuntimeError, ValueError) as e:
                    logger.warning(f"Skipping split={split_name!r}: {e}")
                    continue

                if not samples:
                    logger.warning(f"No samples for split={split_name!r}, skipping")
                    continue

                split_results = self._run_split(samples, pipeline, split_name)
                all_results.extend(split_results)

            if not all_results:
                logger.error("No evaluation results produced!")
                return {"summary": {}, "by_split": {}, "config": dataclasses.asdict(self.config)}

            # 5. Optional LLM judge scoring
            if judge and all_results:
                logger.info("Running LLM-as-judge scoring...")
                all_results = judge.score_batch(all_results)

            # 6. Compute report
            logger.info("[4/4] Computing report and saving results...")
            report = self._compute_report(all_results, baseline_results)

            # Log summary metrics to MLflow
            summary_metrics = {k: v for k, v in report["summary"].items() if isinstance(v, (int, float))}
            tracker.log_metrics(summary_metrics)

            # Save results to disk
            output_dir = Path(self.config.output_dir) / run_name
            output_dir.mkdir(parents=True, exist_ok=True)

            results_path = output_dir / "results.json"
            with open(results_path, "w") as f:
                json.dump([r.to_dict() for r in all_results], f, indent=2, default=str)

            report_path = output_dir / "report.json"
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2, default=str)

            logger.info(f"Results saved to {output_dir}")
            logger.info(f"  results.json: {len(all_results)} samples")
            logger.info(f"  report.json: summary + per-split breakdown")

        return report
