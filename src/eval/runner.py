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
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .dataset import (
    EvalSample,
    KNOWN_GEO_ADAPTERS,
    build_counterfact_conflict_dataset,
    build_local_json_dataset,
    build_qm_conflict_dataset,
    build_qm_stable_dataset,
    build_situated_qa_dataset,
    build_sqa_train_dataset,
    build_triviaqa_control_dataset,
)
from .metrics import (
    compute_efficiency,
    compute_esr,
    compute_logprob_em,
    compute_logprob_esr,
    compute_strict_esr,
    compute_routing_accuracy,
    compute_stability_score,
    compute_cfr,
    compute_dcontrol_forgetting_rate,
    exact_match,
    normalize_answer,
    parse_model_output,
    token_f1,
)

logger = logging.getLogger(__name__)


def _normalised_value_present(value: str, normalised_text: str) -> bool:
    """True if ``value`` occurs in ``normalised_text`` as a standalone token run.

    Both arguments must already be ``normalize_answer``-ed. Uses non-word
    lookarounds rather than plain ``in`` containment so a short edit value
    ("3", "GG2", "M50") cannot spuriously match inside a longer token (the
    "3" in a normalised "634"). Used for the AIT QM atomic-edit ESR check.
    """
    if not value:
        return False
    return re.search(rf"(?<!\w){re.escape(value)}(?!\w)", normalised_text) is not None


# All valid split names
VALID_SPLITS: set[str] = (
    {"base", "temporal", "local", "cf_conflict", "cf_control", "sqa_train",
     "qm_conflict", "qm_stable", "qm_control"}
    | {f"geo_{c}" for c in KNOWN_GEO_ADAPTERS}
)


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
    model_id: str = "mistralai/Mistral-7B-Instruct-v0.3"
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
    # Splits whose gold answers are long free-form documents (AIT QM document
    # QA). For these, `_run_single` generates with `long_form_max_new_tokens`,
    # disables sentence-boundary stop sequences, and keeps the full untruncated
    # text. Short-answer splits (cf_*, sqa_train, qm_control = TriviaQA
    # D_control) are unaffected and keep the `max_new_tokens` short config.
    # A list (not a set) so `dataclasses.asdict` → `json.dump` serialises it
    # cleanly into report.json's config block.
    long_form_splits: list[str] = field(default_factory=lambda: ["qm_conflict", "qm_stable"])
    long_form_max_new_tokens: int = 512
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
    # MORPHEUS' PrototypeRouter applies a fixed JL random projection (384→256)
    # before computing cosine similarity, so similarities are systematically
    # lower than the raw-space CentroidRouter used by PnR. Using the shared
    # `similarity_threshold=0.65` there would reject most valid routes. The
    # value below is the native default of `PrototypeRouterConfig`.
    morpheus_similarity_threshold: float = 0.55
    # Authoritative-override bypass threshold (KnowledgeStoreConfig.direct_answer_threshold).
    # Default 0.95 keeps current behaviour. Set > 1.0 to disable bypass and
    # force the activated specialist (LoRA adapter) to generate every answer
    # — the Patch-and-Route-conformant evaluation path.
    morpheus_direct_answer_threshold: float = 0.95
    # KnowledgeStore tau_low: queries below this go to parametric_freedom.
    # Must exceed max D_control sim (≤0.619) to prevent CF injection into TriviaQA.
    morpheus_factuality_threshold_low: float = 0.65
    # Path to a trained FactualityClassifier checkpoint directory.
    # When set, the classifier score replaces max_sim as the factuality_score
    # passed to KnowledgeStore.assess_factuality (the learned routing signal).
    # Leave None to use the hardcoded tau_low / max_sim fallback.
    morpheus_classifier_path: str | None = None
    parallel_orchestrator: bool = False  # Use Parallel-Orchestrator architecture
    parallel_max_adapters: int = 5  # Max adapters for parallel execution
    # "similarity" (default, post-May-1): similarity-distribution rule;
    # "keyword" (legacy heuristic, retained for ablation);
    # "llm": classification via the frozen base.
    parallel_query_planner: str = "similarity"
    parallel_synthesis_tokens: int = 512  # Max tokens for synthesis pass
    # Sticky-adapter policy (Change 0). When False (default) PnR explicitly
    # detaches the previously-loaded adapter on routing miss — the honest
    # per-query state required for unbiased D_control. Set True to restore
    # the legacy "warm-context" behaviour (kept as an ablation knob).
    warm_context: bool = False
    recipe_official_checkpoint: str | None = None  # Path to official-repo RECIPE checkpoint file
    recipe_official_edits_path: str | None = None  # JSON file with edits for the official-RECIPE repo
    lora_rag_adapter: str | None = None  # Path to monolithic adapter for LoRA+RAG baseline
    lora_rag_index_path: str | None = None  # JSON file of QA pairs to index for retrieval
    # CounterFact / TriviaQA (cf_conflict + cf_control splits)
    counterfact_eval_path: str | None = None  # data/counterfact_eval.json
    triviaqa_dcontrol_path: str | None = None  # data/triviaqa_dcontrol.json
    sqa_deval_path: str | None = None          # data/sqa_deval.json
    cf_adapter_name: str = "patch_cf_main"  # Adapter the router should pick for D_conflict
    cf_split_name: str = "test"  # Which split of counterfact_eval.json to use ('train' or 'test')
    # AIT QM D_eval (qm_stable + qm_conflict + qm_control splits)
    qm_conflict_path: str | None = None   # data/qm_conflict_pairs.json
    qm_stable_path: str | None = None     # data/qm_stable_facts.json
    qm_adapter_name: str = "patch_qm_current"  # Adapter the router should pick for qm_conflict
    qm_base_adapter_name: str = "base_qm"  # Adapter the router should pick for qm_stable
    # Log-probability scoring (ROME / MEMIT-style ESR). When enabled, the
    # runner asks the active pipeline for a teacher-forced log P(target |
    # prompt) on every sample, after generation. Reports include:
    #   - logprob_target_new / logprob_target_true   (CounterFact)
    #   - logprob_gold (max over gold aliases)        (all splits)
    #   - is_logprob_match  (gold beats target_true / random distractor)
    #   - logprob_esr  (cf_conflict only, gold > target_true rate)
    # See docs/roadmap.md §"Architecture Improvement TODOs" for context.
    compute_logprob: bool = False
    # Phase 4 — Stage-1 domain classifier checkpoint (3-class: cf, sqa,
    # ood_trivia). When provided and loadable, the CentroidRouter and
    # ParallelOrchestrator gate routing on the classifier's prediction:
    # confident OOD → frozen base; confident CF/SQA → mask Stage 2 to
    # the in-domain adapter pool. Closes NF-1 (routing_acc=0 on SQA).
    # Graceful fallback: missing path or failed load logs a warning and
    # eval proceeds without Stage-1 (pre-Phase-4 behaviour).
    domain_classifier_path: str | None = None
    # Confidence threshold for acting on a Stage-1 prediction. Below this,
    # the Stage-1 mask is ignored and Stage 2 considers all adapters.
    domain_confidence_threshold: float = 0.7
    # Replacement value for the per-adapter τ fallback when Stage 1 is
    # active (CF/SQA class with confident probability). Lowered from
    # `similarity_threshold` because OOD has already been filtered out.
    # Per-adapter calibrated τ in the manifest still wins.
    domain_fallback_threshold: float = 0.30


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
    # MORPHEUS-specific diagnostics: populated only for --morpheus runs.
    # Without these, we cannot tell whether the graduated-factuality
    # hierarchy is firing or collapsing to the boundary zone on every query.
    morpheus_zone: str | None = None
    morpheus_factuality_score: float | None = None
    morpheus_max_sim: float | None = None
    morpheus_knowledge_override: bool | None = None
    # Log-probability metrics (ROME / MEMIT-style ESR). Populated only when
    # ``EvalConfig.compute_logprob`` is True and the pipeline exposes a
    # ``score_targets`` method. Defaults of None encode "not measured".
    logprob_gold: float | None = None  # max over gold aliases
    logprob_target_new: float | None = None  # cf_conflict only
    logprob_target_true: float | None = None  # cf_conflict only
    is_logprob_match: bool | None = None  # gold beats distractor

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
            "morpheus_zone": self.morpheus_zone,
            "morpheus_factuality_score": self.morpheus_factuality_score,
            "morpheus_max_sim": self.morpheus_max_sim,
            "morpheus_knowledge_override": self.morpheus_knowledge_override,
            "logprob_gold": self.logprob_gold,
            "logprob_target_new": self.logprob_target_new,
            "logprob_target_true": self.logprob_target_true,
            "is_logprob_match": self.is_logprob_match,
            "metadata": self.sample.metadata,
        }


# =============================================================================
# MORPHEUS zone aggregation
# =============================================================================

def _summarise_morpheus(results: list["EvalResult"]) -> dict[str, Any]:
    """Aggregate graduated-factuality signals over a set of results.

    Reports zone histogram, hard_override hit-rate, and — critically —
    conditional EM within each zone. If hard_override fires but its EM
    is no better than boundary, the override is advisory in name only
    (the prompt isn't forcing deference) and W2 from the improvement plan
    becomes the next lever.
    """
    zoned = [r for r in results if r.morpheus_zone is not None]
    n = len(zoned)
    if n == 0:
        return {}
    zones: dict[str, int] = {}
    em_by_zone: dict[str, list[bool]] = {}
    sims: list[float] = []
    for r in zoned:
        z = r.morpheus_zone
        zones[z] = zones.get(z, 0) + 1
        em_by_zone.setdefault(z, []).append(r.is_exact_match)
        if r.morpheus_max_sim is not None:
            sims.append(r.morpheus_max_sim)
    kb_hit = sum(1 for r in zoned if r.morpheus_knowledge_override)
    summary = {
        "n": n,
        "zones": {z: round(c / n, 4) for z, c in zones.items()},
        "zone_counts": zones,
        "knowledge_override_rate": round(kb_hit / n, 4),
        "em_by_zone": {
            z: round(sum(e) / len(e), 4) for z, e in em_by_zone.items()
        },
    }
    if sims:
        sims_arr = sorted(sims)
        summary["max_sim"] = {
            "mean": round(sum(sims) / len(sims), 4),
            "p50": round(sims_arr[len(sims_arr) // 2], 4),
            "p90": round(sims_arr[int(0.9 * (len(sims_arr) - 1))], 4),
            "max": round(sims_arr[-1], 4),
        }
    return summary


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

    def _build_recipe_official_pipeline(self):
        """Build inference pipeline using the official RECIPE repo."""
        import json
        import torch
        from src.baselines.recipe_official import RECIPEOfficialInference

        use_gpu = self.config.use_gpu and torch.cuda.is_available()
        pipeline = RECIPEOfficialInference(
            checkpoint_path=self.config.recipe_official_checkpoint,
            model_id=self.config.model_id,
            quantization=self.config.quantization,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            do_sample=self.config.do_sample,
            use_gpu=use_gpu,
        )

        if self.config.recipe_official_edits_path:
            edits_path = Path(self.config.recipe_official_edits_path)
            if not edits_path.exists():
                raise FileNotFoundError(
                    f"RECIPE-official edits file not found: {edits_path}"
                )
            with open(edits_path) as f:
                edits = json.load(f)
            logger.info(
                "Populating official RECIPE repository with %d edits from %s ...",
                len(edits), edits_path,
            )
            pipeline.apply_edits(edits)
            logger.info("RECIPE-official: repository populated (%d entries).", len(edits))
        else:
            logger.info(
                "No --recipe_official_edits provided; evaluating with empty repository."
            )

        return pipeline

    def _build_lora_rag_pipeline(self):
        """Build a LoRA+RAG inference pipeline (Baseline 2).

        Wraps the trained monolithic adapter with QA-pair retrieval: the top-k
        most similar QA pairs from the index file are prepended as context to
        every query before generation.

        Returns:
            LoRARAGInference instance ready for inference.
        """
        import torch
        from src.baselines.lora_rag import LoRARAGInference

        use_gpu = self.config.use_gpu and torch.cuda.is_available()

        index_path = self.config.lora_rag_index_path
        if not index_path:
            raise ValueError(
                "--lora_rag_index is required when --lora_rag is set. "
                "Provide a JSON file of {question, answer} pairs to index."
            )

        logger.info(
            "LoRA+RAG: monolithic adapter=%s  index=%s",
            self.config.lora_rag_adapter,
            index_path,
        )

        return LoRARAGInference(
            monolithic_adapter_path=self.config.lora_rag_adapter,
            qa_pairs_path=index_path,
            model_id=self.config.model_id,
            quantization=self.config.quantization,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            do_sample=self.config.do_sample,
            use_gpu=use_gpu,
        )

    def _build_xlora_pipeline(self):
        """Build an XLoRAInference pipeline.

        Returns:
            XLoRAInference instance wrapping the gating checkpoint.
        """
        from src.baselines.xlora import XLoRAInference

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
            similarity_threshold=self.config.morpheus_similarity_threshold,
            use_gpu=use_gpu,
        )
        morpheus_config = MorpheusConfig(
            stable_core=core_config,
            router=router_config,
        )
        if self.config.morpheus_state_dir:
            morpheus_config.state_dir = self.config.morpheus_state_dir
            # MorpheusInference auto-loads records.json from
            # knowledge_store.store_dir, NOT from MorpheusConfig.state_dir.
            # Without this propagation, --morpheus_state_dir silently leaves
            # the KS pointing at the default "morpheus_state/knowledge_store"
            # — which would load CF triples into a QM run (or vice versa).
            morpheus_config.knowledge_store.store_dir = str(
                Path(self.config.morpheus_state_dir) / "knowledge_store"
            )
        morpheus_config.knowledge_store.direct_answer_threshold = (
            self.config.morpheus_direct_answer_threshold
        )
        morpheus_config.knowledge_store.factuality_threshold_low = (
            self.config.morpheus_factuality_threshold_low
        )
        if self.config.morpheus_classifier_path:
            morpheus_config.knowledge_store.classifier_path = (
                self.config.morpheus_classifier_path
            )

        gen_config = MorpheusGenerationConfig(
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            do_sample=self.config.do_sample,
        )

        # Load embedding model so PrototypeRouter can compute similarities.
        # PrototypeRouterConfig stores the path but doesn't auto-load; we build
        # the callable here and pass it explicitly.
        embedding_fn = None
        if self.config.embedding_model:
            from sentence_transformers import SentenceTransformer
            import numpy as np
            _enc = SentenceTransformer(self.config.embedding_model)
            embedding_fn = lambda text: _enc.encode(text, normalize_embeddings=True).astype(np.float32)

        pipeline = MorpheusInference(
            config=morpheus_config,
            generation_config=gen_config,
            embedding_fn=embedding_fn,
        )

        # Register adapters with prototype centroids.
        #
        # CRITICAL: `PrototypeRouter.register_adapter` defaults to a ZERO
        # centroid when none is provided, which collapses routing (sim == 0
        # for every expert → every query falls below `similarity_threshold`
        # → winner_adapter=None).  We therefore load real centroids from the
        # same `router_state/manifest.json` that the PnR/Parallel Orchestrator
        # use, and fall back to the checkpoints directory only if no manifest
        # is available (logging a loud warning in that case).
        import json
        import numpy as np

        router = pipeline.get_router()
        n_registered = 0
        manifest_path = None
        if self.config.router_state_path:
            candidate = Path(self.config.router_state_path) / "manifest.json"
            if candidate.exists():
                manifest_path = candidate

        if manifest_path is not None:
            with open(manifest_path) as f:
                manifest = json.load(f)
            for adapter_id, info in manifest.get("adapters", {}).items():
                centroid_list = info.get("centroid")
                if centroid_list is None:
                    logger.warning(
                        f"Manifest entry for {adapter_id} has no centroid — skipping"
                    )
                    continue
                centroid = np.asarray(centroid_list, dtype=np.float32)
                router.register_adapter(
                    adapter_id=adapter_id,
                    path=info.get("adapter_path", ""),
                    timestamp=float(info.get("timestamp", 0.0)),
                    centroid=centroid,
                )
                n_registered += 1
            logger.info(
                f"Registered {n_registered} adapters with MORPHEUS router "
                f"from {manifest_path}"
            )
        else:
            logger.warning(
                "No router_state manifest found; registering adapters WITHOUT "
                "centroids. MORPHEUS routing will not work until centroids are "
                "provided. Pass --router_state <dir> pointing at a manifest.json."
            )
            checkpoints_dir = Path(self.config.checkpoints_dir)
            if checkpoints_dir.exists():
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

    def _attach_domain_classifier(self, router) -> None:
        """Phase 4 — load the Stage-1 domain classifier and bind it to ``router``.

        No-op when ``self.config.domain_classifier_path`` is None. Failures
        (missing path, malformed checkpoint) downgrade to a warning so eval
        can still run with pre-Phase-4 routing — the FR/ESR numbers will
        just reflect that NF-1 is unfixed for that run.
        """
        path = self.config.domain_classifier_path
        if not path:
            return
        if not Path(path).exists():
            logger.warning(
                f"--domain_classifier_path {path} does not exist; skipping Stage-1."
            )
            return
        try:
            from src.routing.domain_classifier import DomainClassifier
            classifier = DomainClassifier.load(path, device="auto")
        except Exception as e:
            logger.warning(
                f"Failed to load domain classifier from {path}: {e}. "
                "Continuing without Stage-1 — NF-1 mitigation is OFF for this run."
            )
            return
        router._domain_classifier = classifier
        router._domain_confidence_threshold = self.config.domain_confidence_threshold
        router._domain_fallback_threshold = self.config.domain_fallback_threshold
        logger.info(
            f"Stage-1 domain classifier attached from {path} "
            f"(conf_thr={self.config.domain_confidence_threshold}, "
            f"fallback_thr={self.config.domain_fallback_threshold})"
        )

    def _build_parallel_pipeline(self):
        """Build a ParallelOrchestrator pipeline.

        Returns:
            ParallelOrchestrator instance for multi-adapter parallel inference.
        """
        import torch
        from src.eval.metrics import DEFAULT_SHORT_ANSWER_BOUNDARIES
        from src.inference import GenerationConfig
        from src.models.core import FrozenFoundationConfig, PatchAndRouteLLM, QuantizationType
        from src.routing import CentroidRouter, ParallelOrchestrator

        quant_map = {"none": QuantizationType.NONE, "int8": QuantizationType.INT8, "int4": QuantizationType.INT4}
        quantization = quant_map.get(self.config.quantization, QuantizationType.INT4)
        use_gpu = self.config.use_gpu and torch.cuda.is_available()

        # Build CentroidRouter (reuse embedding/manifest infrastructure)
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

        self._attach_domain_classifier(router)

        # Build PatchAndRouteLLM
        llm_config = FrozenFoundationConfig(
            model_id=self.config.model_id,
            quantization=quantization,
            use_cache=True,
        )
        llm = PatchAndRouteLLM(foundation_config=llm_config)
        llm.load_frozen_foundation()

        # Explicit `stop_sequences` so this builder doesn't quietly diverge
        # from `GenerationConfig`'s default if the dataclass default ever
        # changes (Change 6 — defensive plumbing).
        gen_config = GenerationConfig(
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            do_sample=self.config.do_sample,
            stop_sequences=DEFAULT_SHORT_ANSWER_BOUNDARIES,
        )

        return ParallelOrchestrator(
            centroid_router=router,
            llm=llm,
            generation_config=gen_config,
            query_planner_mode=self.config.parallel_query_planner,
            max_adapters=self.config.parallel_max_adapters,
            synthesis_max_new_tokens=self.config.parallel_synthesis_tokens,
            use_gpu=use_gpu,
            warm_context=self.config.warm_context,
        )

    def _build_pipeline(self):
        """Build the inference pipeline (PnR, X-LoRA, MORPHEUS, or Parallel).

        Returns:
            Configured inference pipeline instance.
        """
        if self.config.parallel_orchestrator:
            return self._build_parallel_pipeline()
        if self.config.morpheus:
            return self._build_morpheus_pipeline()
        if self.config.recipe_official_checkpoint:
            return self._build_recipe_official_pipeline()
        if self.config.lora_rag_adapter:
            return self._build_lora_rag_pipeline()
        if self.config.xlora_checkpoint:
            return self._build_xlora_pipeline()
        import torch
        from src.eval.metrics import DEFAULT_SHORT_ANSWER_BOUNDARIES
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

        # Build router — skip embedding model when routing is bypassed (monolithic/no_adapter)
        bypass_routing = bool(self.config.monolithic_adapter or self.config.no_adapter)
        embedding_model_path = None if bypass_routing else self.config.embedding_model
        if self.config.router_state_path and Path(self.config.router_state_path).exists():
            router = CentroidRouter.load(
                path=self.config.router_state_path,
                embedding_model_path=embedding_model_path,
                similarity_threshold=self.config.similarity_threshold,
                use_gpu=use_gpu,
            )
        else:
            router = CentroidRouter(
                embedding_model_path=embedding_model_path,
                similarity_threshold=self.config.similarity_threshold,
                use_gpu=use_gpu,
            )
            if not bypass_routing:
                checkpoints_dir = Path(self.config.checkpoints_dir)
                if checkpoints_dir.exists():
                    n_registered = router.register_from_checkpoints(str(checkpoints_dir))
                    logger.info(f"Registered {n_registered} adapters from {checkpoints_dir}")
                else:
                    logger.warning(f"Checkpoints dir not found: {checkpoints_dir}")

        self._attach_domain_classifier(router)

        # Build generation config (Change 6: explicit stop_sequences)
        gen_config = GenerationConfig(
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            do_sample=self.config.do_sample,
            stop_sequences=DEFAULT_SHORT_ANSWER_BOUNDARIES,
        )

        # Build inference pipeline
        pipeline = PatchAndRouteInference(
            model_id=self.config.model_id,
            router=router,
            quantization=quantization,
            generation_config=gen_config,
            use_gpu=use_gpu,
            warm_context=self.config.warm_context,
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

        # Long-form splits (AIT QM document QA) need a longer token budget and
        # no sentence-boundary stop sequences — the short-answer defaults would
        # truncate a multi-paragraph answer after its first clause. Only the
        # PnR pipeline (`PatchAndRouteInference`) honours a per-call generation
        # config; orchestrator / baseline backends ignore it (warn instead).
        long_form_cfg = None
        if sample.split in self.config.long_form_splits:
            if (self.config.parallel_orchestrator
                    or self.config.lora_rag_adapter
                    or self.config.xlora_checkpoint):
                logger.warning(
                    "Long-form split %r on a backend that ignores per-call "
                    "generation config — answer will use the short config.",
                    sample.split,
                )
            else:
                from src.inference import GenerationConfig
                long_form_cfg = GenerationConfig(
                    max_new_tokens=self.config.long_form_max_new_tokens,
                    temperature=self.config.temperature,
                    do_sample=self.config.do_sample,
                    stop_sequences=(),
                )

        # Time the inference
        t_start = time.perf_counter()

        if self.config.parallel_orchestrator:
            result = pipeline.generate(query=sample.question)
        elif self.config.recipe_official_checkpoint:
            result = pipeline.generate(
                query=sample.question,
                generation_config=long_form_cfg,
            )
        elif self.config.lora_rag_adapter:
            result = pipeline.generate(query=sample.question)
        elif self.config.xlora_checkpoint:
            result = pipeline.generate(query=sample.question)
        elif self.config.no_adapter:
            # Frozen base model only — skip routing and load no adapter.
            # This is Pass 1 of the CFR protocol: measures what the foundation
            # already knows, providing the true pre-patch baseline.
            result = pipeline.generate(
                query=sample.question,
                skip_routing=True,
                generation_config=long_form_cfg,
            )
        elif self.config.monolithic_adapter:
            result = pipeline.generate(
                query=sample.question,
                force_adapter=self.config.monolithic_adapter,
                generation_config=long_form_cfg,
            )
        else:
            result = pipeline.generate(
                query=sample.question,
                generation_config=long_form_cfg,
            )

        t_end = time.perf_counter()
        latency_ms = (t_end - t_start) * 1000.0

        # VRAM measurement
        if torch.cuda.is_available():
            vram_mb = torch.cuda.max_memory_allocated() / 1e6

        # Parse answer — long-form splits keep the full untruncated text;
        # truncating at the first sentence boundary would discard everything
        # after a multi-paragraph answer's opening clause.
        is_long_form = sample.split in self.config.long_form_splits
        parsed = parse_model_output(
            result.response,
            truncate_to_short_answer=not is_long_form,
        )

        # Compute metrics
        f1 = token_f1(parsed, sample.gold_answers)
        if sample.split == "qm_conflict":
            # QM ESR mirrors CounterFact's atomic-edit success criterion: the
            # edit landed iff the short `new_value` surfaces in the generated
            # answer. `gold_answers` stays the full `answer_new`, so `f1` above
            # still measures full-answer overlap. `old_value` presence is a
            # backward-interference diagnostic, not part of ESR.
            norm_pred = normalize_answer(parsed)
            new_value = normalize_answer((sample.metadata or {}).get("new_value") or "")
            old_value = normalize_answer((sample.metadata or {}).get("old_value") or "")
            is_em = _normalised_value_present(new_value, norm_pred)
            if sample.metadata is not None:
                sample.metadata["old_value_present"] = _normalised_value_present(
                    old_value, norm_pred
                )
        else:
            is_em = exact_match(parsed, sample.gold_answers)

        # Routing correctness
        adapter_used = result.adapter_loaded
        if self.config.parallel_orchestrator:
            # Parallel orchestrator may use multiple adapters (comma-separated)
            if sample.expected_adapter is None:
                routing_correct = True
            else:
                # Check if expected adapter is among those queried
                used_set = set(adapter_used.split(",")) if adapter_used else set()
                routing_correct = sample.expected_adapter in used_set
            routing_result = result.routing_result
            winner_sim = routing_result.winner_similarity if routing_result else None
            has_conflict = routing_result.has_conflict if routing_result else False
        elif self.config.morpheus:
            if sample.expected_adapter is None:
                routing_correct = True
            else:
                # Knowledge store override IS correct routing for MORPHEUS —
                # CF knowledge lives in System 5, not in patch_cf_main.
                # A hard_override hit on a CF query means the architecture
                # correctly handled the edit via its non-parametric store.
                morpheus_knowledge_override_check = bool(
                    getattr(result, "knowledge_override", False)
                )
                routing_correct = (
                    morpheus_knowledge_override_check
                    or adapter_used == sample.expected_adapter
                )
            routing_result = result.routing_result
            winner_sim = routing_result.winner_similarity if routing_result else None
            has_conflict = routing_result.has_conflict if routing_result else False
            # Capture graduated-factuality diagnostics so we can aggregate
            # zone distributions and hard_override hit-rate in the report.
            fd = getattr(result, "factuality_decision", None)
            if fd is not None:
                morpheus_zone = fd.zone
                morpheus_factuality_score = fd.factuality_score
                morpheus_max_sim = fd.confidence
            else:
                morpheus_zone = None
                morpheus_factuality_score = None
                morpheus_max_sim = None
            morpheus_knowledge_override = bool(getattr(result, "knowledge_override", False))
        elif self.config.recipe_official_checkpoint:
            # RECIPE retrieves continuous prompts — no discrete adapter routing
            routing_correct = True
            winner_sim = None
            has_conflict = False
        elif self.config.lora_rag_adapter:
            # LoRA+RAG uses fixed monolithic adapter + retrieval — no discrete routing
            routing_correct = True
            winner_sim = None
            has_conflict = False
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

        # MORPHEUS diagnostics default to None for non-MORPHEUS runs.
        morpheus_fields: dict[str, Any] = {}
        if self.config.morpheus:
            morpheus_fields = {
                "morpheus_zone": morpheus_zone,
                "morpheus_factuality_score": morpheus_factuality_score,
                "morpheus_max_sim": morpheus_max_sim,
                "morpheus_knowledge_override": morpheus_knowledge_override,
            }

        logprob_fields: dict[str, Any] = {}
        if self.config.compute_logprob:
            logprob_fields = self._compute_logprob_fields(sample, pipeline)

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
            **morpheus_fields,
            **logprob_fields,
        )

    # -------------------------------------------------------------------------
    # Log-probability scoring helper
    # -------------------------------------------------------------------------

    def _compute_logprob_fields(
        self,
        sample: EvalSample,
        pipeline: Any,
    ) -> dict[str, Any]:
        """Run ROME / MEMIT-style teacher-forced scoring on a single sample.

        Returns a dict suitable for the ``logprob_*`` fields on
        ``EvalResult``. Silently degrades to all-None when the active
        pipeline doesn't expose ``score_targets`` so older / experimental
        backends keep working without modification.

        Per-split target conventions:

        - ``cf_conflict``: scores ``target_new`` (gold) and ``target_true``
          (parametric prior). ``is_logprob_match = logp(new) > logp(true)``,
          i.e. the ROME / MEMIT edit-success criterion.
        - All other splits: scores every gold alias, takes the max, and
          records ``is_logprob_match = True`` when at least one alias has
          finite log-prob (i.e. tokenises cleanly). Without a paired
          distractor we can't define edit success, so this serves as a
          sanity / coverage signal only.
        """
        out: dict[str, Any] = {
            "logprob_gold": None,
            "logprob_target_new": None,
            "logprob_target_true": None,
            "is_logprob_match": None,
        }

        score_fn = getattr(pipeline, "score_targets", None)
        if score_fn is None:
            logger.debug(
                "Pipeline %s has no score_targets() — skipping log-prob scoring",
                type(pipeline).__name__,
            )
            return out

        if sample.split == "cf_conflict":
            target_new = sample.gold_answers[0] if sample.gold_answers else ""
            target_true = (sample.metadata or {}).get("target_true") or ""
            if not target_new or not target_true:
                return out
            try:
                kwargs: dict[str, Any] = {}
                # Skip-routing / force-adapter mirror what _run_single does
                # for generation, so the model state is identical.
                if self.config.no_adapter:
                    kwargs["skip_routing"] = True
                elif self.config.monolithic_adapter:
                    kwargs["force_adapter"] = self.config.monolithic_adapter
                scores = score_fn(
                    sample.question,
                    [target_new, target_true],
                    **kwargs,
                )
            except TypeError:
                scores = score_fn(sample.question, [target_new, target_true])
            except Exception as exc:
                logger.warning(
                    "score_targets failed for cf_conflict sample (case_id=%s): %s",
                    (sample.metadata or {}).get("case_id"),
                    exc,
                )
                return out

            lp_new = scores.get(target_new)
            lp_true = scores.get(target_true)
            out["logprob_target_new"] = lp_new
            out["logprob_target_true"] = lp_true
            out["logprob_gold"] = lp_new
            if lp_new is not None and lp_true is not None:
                out["is_logprob_match"] = bool(lp_new > lp_true)
            return out

        if sample.split == "qm_conflict":
            # Score the changed value (`new_value` vs `old_value`) at the edit
            # position, mirroring the CF `target_new`/`target_true` convention.
            # Scoring the full `answer_new`/`answer_old` documents was diluting
            # the single changed token across ~1000 tokens — and under
            # always-on Source-Replay both documents become near-perfectly
            # copyable from the in-context retrieved chunks, collapsing the
            # signal further (May 19 diagnosis).
            meta = sample.metadata or {}
            target_new = (meta.get("new_value") or "").strip()
            target_true = (meta.get("old_value") or "").strip()
            if not target_new or not target_true:
                return out
            try:
                kwargs: dict[str, Any] = {}
                if self.config.no_adapter:
                    kwargs["skip_routing"] = True
                elif self.config.monolithic_adapter:
                    kwargs["force_adapter"] = self.config.monolithic_adapter
                scores = score_fn(
                    sample.question,
                    [target_new, target_true],
                    **kwargs,
                )
            except TypeError:
                scores = score_fn(sample.question, [target_new, target_true])
            except Exception as exc:
                logger.warning(
                    "score_targets failed for qm_conflict sample (id=%s): %s",
                    meta.get("id"),
                    exc,
                )
                return out

            lp_new = scores.get(target_new)
            lp_true = scores.get(target_true)
            out["logprob_target_new"] = lp_new
            out["logprob_target_true"] = lp_true
            out["logprob_gold"] = lp_new
            if lp_new is not None and lp_true is not None:
                out["is_logprob_match"] = bool(lp_new > lp_true)
            return out

        # Other splits — score every gold alias, take the max.
        targets = [g for g in sample.gold_answers if g]
        if not targets:
            return out
        try:
            kwargs = {}
            if self.config.no_adapter:
                kwargs["skip_routing"] = True
            elif self.config.monolithic_adapter:
                kwargs["force_adapter"] = self.config.monolithic_adapter
            scores = score_fn(sample.question, targets, **kwargs)
        except TypeError:
            scores = score_fn(sample.question, targets)
        except Exception as exc:
            logger.warning(
                "score_targets failed for split=%s sample: %s",
                sample.split, exc,
            )
            return out

        finite = [v for v in scores.values() if v is not None and v != float("-inf")]
        if not finite:
            return out
        out["logprob_gold"] = max(finite)
        out["is_logprob_match"] = True
        return out

    # -------------------------------------------------------------------------
    # Split-Level Evaluation
    # -------------------------------------------------------------------------

    def _run_split(
        self,
        samples: list[EvalSample],
        pipeline,
        split_name: str,
        output_dir: Path | None = None,
        partial_every: int = 50,
    ) -> list[EvalResult]:
        """Evaluate all samples in a split.

        Args:
            samples: List of EvalSample for this split.
            pipeline: PatchAndRouteInference instance.
            split_name: Name of the split (for logging).
            output_dir: If provided, dump partial results every ``partial_every``
                successful samples to ``output_dir/results_<split>.partial.json``.
                Survives SIGTERM mid-split (e.g. SLURM wall-time kill).
            partial_every: Number of successful samples between partial dumps.

        Returns:
            List of EvalResult objects.
        """
        results: list[EvalResult] = []
        consecutive_failures = 0
        FAIL_FAST_THRESHOLD = 10

        partial_path = (
            output_dir / f"results_{split_name}.partial.json" if output_dir else None
        )

        def _dump_partial() -> None:
            if partial_path is None:
                return
            tmp = partial_path.with_suffix(partial_path.suffix + ".tmp")
            with open(tmp, "w") as f:
                json.dump([r.to_dict() for r in results], f, indent=2, default=str)
            tmp.replace(partial_path)  # atomic on POSIX

        for sample in tqdm(samples, desc=f"Eval [{split_name}]", unit="sample"):
            try:
                result = self._run_single(sample, pipeline)
                results.append(result)
                consecutive_failures = 0
                if partial_path is not None and len(results) % partial_every == 0:
                    _dump_partial()
            except Exception as e:
                consecutive_failures += 1
                logger.warning(f"Failed on sample (split={split_name}): {e}")
                if consecutive_failures >= FAIL_FAST_THRESHOLD and not results:
                    raise RuntimeError(
                        f"Aborting: first {FAIL_FAST_THRESHOLD} samples in split={split_name!r} "
                        f"all failed. Last error: {e}"
                    ) from e
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

        # MORPHEUS zone / override aggregation: only populated when at least
        # one sample carries a factuality decision. Lets us see at a glance
        # whether the hierarchy fires or collapses to "always boundary".
        if any(r.morpheus_zone is not None for r in all_results):
            summary["morpheus"] = _summarise_morpheus(all_results)

        # ROME / MEMIT-style log-prob ESR: shown alongside generation-based
        # ESR so the thesis can present both in a single comparison table.
        # Large divergences flag a parsing issue or a generation
        # distribution-mismatch in one of the systems.
        if any(r.is_logprob_match is not None for r in all_results):
            # logprob ESR is per conflict split: cf_conflict → "logprob_esr",
            # qm_conflict → "qm_logprob_esr" (a run carries at most one).
            for conflict_split in ("cf_conflict", "qm_conflict"):
                lp_esr = compute_logprob_esr(
                    all_results, split_filter=conflict_split
                )
                if lp_esr is not None:
                    key = ("logprob_esr" if conflict_split == "cf_conflict"
                           else "qm_logprob_esr")
                    summary[key] = lp_esr
            lp_em = compute_logprob_em(all_results)
            if lp_em is not None:
                summary["logprob_em"] = lp_em

        # Strict (decisive-override) ESR for AIT QM: the primary generation
        # ESR also counts answers that still emit the obsolete old_value; the
        # strict variant requires old_value absent. ESR - qm_strict_esr is the
        # hedging rate. None for any run without qm_conflict data.
        strict_esr = compute_strict_esr(all_results, split_filter="qm_conflict")
        if strict_esr is not None:
            summary["qm_strict_esr"] = strict_esr

        # D_control forgetting rate (no baseline needed — pre-filtered to 100%
        # base acc). Counts every `*_control` split, so it covers both
        # cf_control and qm_control transparently.
        fr = compute_dcontrol_forgetting_rate(all_results)
        if fr is not None:
            summary["dcontrol_forgetting_rate"] = fr
            summary["dcontrol_accuracy"] = round(1.0 - fr, 4)

        # CFR (requires baseline, for SituatedQA splits)
        if baseline_results:
            summary["cfr"] = compute_cfr(all_results, baseline_results)
            if any(r.sample.split == "cf_control" for r in all_results):
                summary["cfr_control"] = compute_cfr(
                    all_results, baseline_results, split_filter="cf_control"
                )

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
            if any(r.morpheus_zone is not None for r in split_results):
                splits[split_name]["morpheus"] = _summarise_morpheus(split_results)
            if any(r.is_logprob_match is not None for r in split_results):
                if split_name in ("cf_conflict", "qm_conflict"):
                    lp_esr_split = compute_logprob_esr(
                        split_results, split_filter=split_name
                    )
                    if lp_esr_split is not None:
                        splits[split_name]["logprob_esr"] = round(lp_esr_split, 4)
                lp_em_split = compute_logprob_em(split_results)
                if lp_em_split is not None:
                    splits[split_name]["logprob_match_rate"] = round(lp_em_split, 4)
            if split_name == "qm_conflict":
                # Lenient ESR is `exact_match` above; `strict_esr` additionally
                # requires the obsolete old_value to be absent from the answer.
                strict_esr_split = compute_strict_esr(
                    split_results, split_filter=split_name
                )
                if strict_esr_split is not None:
                    splits[split_name]["strict_esr"] = round(strict_esr_split, 4)

        # Round optional floats in summary
        for key in ("routing_accuracy", "esr", "stability_score", "cfr", "cfr_control",
                    "dcontrol_forgetting_rate", "dcontrol_accuracy",
                    "logprob_esr", "qm_logprob_esr", "qm_strict_esr", "logprob_em"):
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

            # Ensure output_dir exists up front so per-split checkpoints survive
            # even if a later split crashes or hits a SLURM time limit.
            output_dir = Path(self.config.output_dir) / run_name
            output_dir.mkdir(parents=True, exist_ok=True)

            for split_name in self.config.eval_sets:
                logger.info(f"Building dataset for split={split_name!r}...")

                try:
                    if split_name == "local":
                        samples = build_local_json_dataset(
                            data_paths=self.config.local_data_paths,
                            n_samples=self.config.n_samples,
                        )
                    elif split_name == "cf_conflict":
                        if not self.config.counterfact_eval_path:
                            raise ValueError(
                                "'cf_conflict' requires --counterfact_eval_path"
                            )
                        samples = build_counterfact_conflict_dataset(
                            counterfact_path=self.config.counterfact_eval_path,
                            n_samples=self.config.n_samples,
                            cf_adapter_name=self.config.cf_adapter_name,
                            cf_split_name=self.config.cf_split_name,
                        )
                    elif split_name == "cf_control":
                        if not self.config.triviaqa_dcontrol_path:
                            raise ValueError(
                                "'cf_control' requires --triviaqa_dcontrol_path"
                            )
                        samples = build_triviaqa_control_dataset(
                            triviaqa_path=self.config.triviaqa_dcontrol_path,
                            n_samples=self.config.n_samples,
                        )
                    elif split_name == "qm_stable":
                        if not self.config.qm_stable_path:
                            raise ValueError(
                                "'qm_stable' requires --qm_stable_path"
                            )
                        samples = build_qm_stable_dataset(
                            qm_stable_path=self.config.qm_stable_path,
                            n_samples=self.config.n_samples,
                            qm_adapter_name=self.config.qm_base_adapter_name,
                        )
                    elif split_name == "qm_conflict":
                        if not self.config.qm_conflict_path:
                            raise ValueError(
                                "'qm_conflict' requires --qm_conflict_path"
                            )
                        samples = build_qm_conflict_dataset(
                            qm_conflict_path=self.config.qm_conflict_path,
                            n_samples=self.config.n_samples,
                            qm_adapter_name=self.config.qm_adapter_name,
                        )
                    elif split_name == "qm_control":
                        if not self.config.triviaqa_dcontrol_path:
                            raise ValueError(
                                "'qm_control' requires --triviaqa_dcontrol_path"
                            )
                        samples = build_triviaqa_control_dataset(
                            triviaqa_path=self.config.triviaqa_dcontrol_path,
                            n_samples=self.config.n_samples,
                            split_name="qm_control",
                        )
                    elif split_name == "sqa_train":
                        if not self.config.sqa_deval_path:
                            raise ValueError(
                                "'sqa_train' requires --sqa_deval_path"
                            )
                        samples = build_sqa_train_dataset(
                            sqa_deval_path=self.config.sqa_deval_path,
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

                split_results = self._run_split(
                    samples, pipeline, split_name, output_dir=output_dir
                )
                all_results.extend(split_results)

                # Per-split checkpoint: survives SLURM timeouts and lets
                # single-split jobs be merged later via scripts/merge_eval_splits.py.
                # `_run_split` also writes results_<split>.partial.json every N
                # samples, so a mid-split kill still leaves usable results.
                split_path = output_dir / f"results_{split_name}.json"
                with open(split_path, "w") as f:
                    json.dump(
                        [r.to_dict() for r in split_results],
                        f,
                        indent=2,
                        default=str,
                    )
                partial_path = output_dir / f"results_{split_name}.partial.json"
                if partial_path.exists():
                    partial_path.unlink()
                logger.info(f"  checkpoint: {split_path} ({len(split_results)} samples)")

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

            # Save results to disk (output_dir already created above for
            # per-split checkpoints).
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
