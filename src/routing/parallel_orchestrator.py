"""
Parallel-Orchestrator Architecture
====================================

Implements the Parallel-Orchestrator routing/inference strategy for cooperative
scenarios where multiple adapters hold complementary knowledge.

Instead of picking a single winner adapter (like the CentroidRouter), the
orchestrator:
1. Plans which adapters to query based on query analysis
2. Generates answers from each selected adapter independently
3. Synthesizes all outputs into a unified response via the base model

Architecture (from Section 4.4.2 of the Master's Thesis Expose):
- Intelligent Router (Query Planner): Classifies query intent
- Parallel Execution Engine: Sequential hot-swap generation through adapters
- Context Synthesis Agent (The Resolver): Merges adapter outputs

Reference: "Parallel-Orchestrator Architecture (Ensemble & Synthesis)"
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np
import torch

from .base import AdapterMatch, RoutingResult, RoutingStrategy
from .centroid_router import CentroidRouter
from .source_replay import RetrievedChunk, SourceReplayStore

logger = logging.getLogger(__name__)


# =============================================================================
# Data Types
# =============================================================================

class QueryPlanType(Enum):
    """Classification of query intent for adapter selection."""

    SINGLE_LATEST = "single_latest"
    """Pick only the top-1 adapter (fast path, same as centroid behavior)."""

    MULTI_TEMPORAL = "multi_temporal"
    """Pick 2-3 adapters from different time periods for temporal queries."""

    BROAD_COMPOSITION = "broad_composition"
    """Pick a broad range of adapters for comprehensive/overview queries."""


@dataclass
class QueryPlan:
    """Result of the Query Planner analysis.

    Attributes:
        plan_type: Classification of the query intent.
        selected_adapters: Adapters chosen for parallel execution.
        reasoning: Explanation of why this plan was chosen (for logging).
    """

    plan_type: QueryPlanType
    selected_adapters: list[AdapterMatch]
    reasoning: str = ""


@dataclass
class OrchestratorResult:
    """Result of the Parallel-Orchestrator inference.

    Duck-types with PatchAndRouteInference's InferenceResult and
    MorpheusInferenceResult for eval compatibility (.response,
    .adapter_loaded, .routing_result).

    Attributes:
        response: The final synthesized (or single-adapter) response.
        adapter_loaded: Comma-joined IDs of all adapters queried.
        routing_result: A RoutingResult for eval compatibility.
        adapter_outputs: Per-adapter raw outputs.
        query_plan: The query plan that drove adapter selection.
        synthesis_prompt: Full prompt sent to the Resolver (for debugging).
        per_adapter_latency_ms: Generation latency per adapter.
        synthesis_latency_ms: Latency of the synthesis pass.
    """

    response: str
    adapter_loaded: str | None
    routing_result: RoutingResult | None
    adapter_outputs: dict[str, str] = field(default_factory=dict)
    query_plan: QueryPlan | None = None
    synthesis_prompt: str = ""
    per_adapter_latency_ms: dict[str, float] = field(default_factory=dict)
    synthesis_latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "response": self.response,
            "adapter_loaded": self.adapter_loaded,
            "routing": self.routing_result.to_dict() if self.routing_result else None,
            "plan_type": self.query_plan.plan_type.value if self.query_plan else None,
            "num_adapters_queried": len(self.adapter_outputs),
            "adapter_outputs": self.adapter_outputs,
            "per_adapter_latency_ms": self.per_adapter_latency_ms,
            "synthesis_latency_ms": self.synthesis_latency_ms,
        }


# =============================================================================
# Keyword Patterns for Heuristic Query Planner
# =============================================================================

_MULTI_TEMPORAL_PATTERNS = re.compile(
    r"\b("
    r"chang(?:e[ds]?|ing)"
    r"|evolv(?:e[ds]?|ing)"
    r"|differ(?:s|ed|ent|ence)?"
    r"|compar(?:e[ds]?|ing|ison)"
    r"|over\s+time"
    r"|history|historical"
    r"|transition(?:ed|s|ing)?"
    r"|before\s+and\s+after"
    r"|used\s+to\s+be"
    r"|previously"
    r"|originally"
    r"|update[ds]?"
    r"|shift(?:ed|s|ing)?"
    r"|trend(?:s|ed|ing)?"
    r")\b",
    re.IGNORECASE,
)

_BROAD_COMPOSITION_PATTERNS = re.compile(
    r"\b("
    r"explain"
    r"|describ(?:e[ds]?|ing)"
    r"|overview"
    r"|summar(?:y|ize[ds]?|izing)"
    r"|everything\s+about"
    r"|tell\s+me\s+(?:all\s+)?about"
    r"|comprehensive"
    r"|detail(?:s|ed)?"
    r"|full\s+picture"
    r"|all\s+(?:the\s+)?(?:information|facts|details)"
    r")\b",
    re.IGNORECASE,
)


# =============================================================================
# Synthesis Prompt Template
# =============================================================================

SYNTHESIS_PROMPT_TEMPLATE = """\
You are an expert knowledge resolver. You have received candidate answers \
from multiple specialized knowledge sources, each trained on a different \
context or time period.

Question: {query}

--- Source Answers ---
{source_answers}
--- End of Source Answers ---

--- Retrieved Evidence ---
{retrieved_evidence}
--- End of Retrieved Evidence ---

Rules:
1. Output ONLY the final answer. No explanation, no preamble, no source attribution.
2. If sources conflict, prefer the answer from the source with the LATEST training date.
3. Match the format of the question: a single fact ⇒ a single short phrase or word.
4. If retrieved evidence contradicts a source's answer, prefer the evidence.
5. Do NOT use phrases like "Previously X, but...", "As of [date]...", or "Both X and Y".

Final answer:"""

SYNTHESIS_PROMPT_TEMPLATE_LONG_FORM = """\
You are an expert knowledge resolver. You have received candidate answers \
from multiple specialized knowledge sources, each trained on a different \
context or time period.

Question: {query}

--- Source Answers ---
{source_answers}
--- End of Source Answers ---

--- Retrieved Evidence ---
{retrieved_evidence}
--- End of Retrieved Evidence ---

Rules:
1. Output the answer IN FULL. Do not summarize. Do not point to where the answer lives — restate the content directly.
2. Preserve markdown structure: headers, bullet lists, numbered steps, quoted policy text, and section references must appear verbatim from the chosen source.
3. If sources conflict, prefer the source with the LATEST training date. Copy that source's content verbatim; do not blend with older sources.
4. If retrieved evidence contradicts a source's answer, prefer the evidence and quote the relevant passage in full.
5. Do NOT use phrases like "Previously X, but...", "As of [date]...", or "Both X and Y".

Final answer:"""

# Sentinel used when a synthesis call has no retrieved evidence (e.g.
# Source-Replay disabled at the router level or all chunks below
# `retrieval_threshold`). Kept short so the Resolver doesn't waste budget
# parsing the evidence section when it's empty.
NO_EVIDENCE_PLACEHOLDER = "(none)"


# =============================================================================
# Parallel Orchestrator
# =============================================================================

class ParallelOrchestrator:
    """Parallel-Orchestrator for cooperative multi-adapter inference.

    Composes with CentroidRouter (for embedding/similarity) and PatchAndRouteLLM
    (for adapter hot-swapping and generation). Does NOT inherit from BaseRouter
    because the orchestrator's contract differs fundamentally: it returns multiple
    outputs synthesized into one, rather than a single routing decision.

    Example:
        ```python
        from src.routing import CentroidRouter, ParallelOrchestrator
        from src.models.core import PatchAndRouteLLM
        from src.inference import GenerationConfig

        router = CentroidRouter(embedding_model_path="...")
        router.register_from_checkpoints("checkpoints/")

        llm = PatchAndRouteLLM()
        llm.load_frozen_foundation()

        orchestrator = ParallelOrchestrator(
            centroid_router=router,
            llm=llm,
            generation_config=GenerationConfig(max_new_tokens=256),
        )

        result = orchestrator.generate("How has the CEO changed over time?")
        print(result.response)
        print(result.adapter_outputs)
        ```
    """

    # Keys accepted by `query_planner_mode`. "heuristic" is preserved as
    # an alias for "keyword" so legacy callers (and the older
    # `EvalConfig.parallel_query_planner` default) keep working.
    _PLANNER_MODES: tuple[str, ...] = ("similarity", "keyword", "heuristic", "llm")

    def __init__(
        self,
        centroid_router: CentroidRouter,
        llm: "PatchAndRouteLLM",
        generation_config: "GenerationConfig | None" = None,
        query_planner_mode: str = "similarity",
        max_adapters: int = 5,
        synthesis_max_new_tokens: int = 512,
        synthesis_max_new_tokens_long_form: int = 1536,
        use_gpu: bool = True,
        system_prompt: str | None = None,
        warm_context: bool = False,
        multi_gap_threshold: float = 0.15,
        broad_min_adapters: int = 3,
    ) -> None:
        """Initialize the Parallel Orchestrator.

        Args:
            centroid_router: Router for embedding/similarity computation.
            llm: PatchAndRouteLLM for adapter hot-swapping and generation.
            generation_config: Generation parameters for per-adapter inference.
            query_planner_mode: "similarity" (default, post-May-1) uses the
                per-adapter τ geometry; "keyword" / "heuristic" is the
                legacy regex planner (kept as ablation); "llm" classifies
                via the frozen base.
            max_adapters: Maximum adapters to query in parallel execution.
            synthesis_max_new_tokens: Token budget for the synthesis pass.
                Capped at runtime to ``generation_config.max_new_tokens``
                so factoid evals (CF max_new_tokens=32) don't have a 512-
                token Resolver budget that produces multi-sentence answers
                which then fail strict EM.
            use_gpu: Whether to use GPU for generation.
            system_prompt: Custom system prompt for per-adapter generation.
            warm_context: When True keep the previously loaded adapter
                attached on routing miss (legacy sticky-state). Default
                False — every "no winner" path explicitly detaches so
                Parallel and PnR have the same per-query state semantics.
            multi_gap_threshold: Maximum top-1 vs. top-N similarity gap
                under which the similarity planner fires MULTI / BROAD
                (default 0.15 — same value used by the conflict logic in
                ``CentroidRouter.route``).
            broad_min_adapters: Minimum number of above-τ adapters with a
                tight similarity spread that triggers BROAD instead of
                MULTI (default 3).
        """
        from src.inference import GenerationConfig, PromptBuilder

        if query_planner_mode not in self._PLANNER_MODES:
            raise ValueError(
                f"Unknown query_planner_mode={query_planner_mode!r}. "
                f"Expected one of {self._PLANNER_MODES}."
            )

        self.router = centroid_router
        self.llm = llm
        self.use_gpu = use_gpu
        self.query_planner_mode = query_planner_mode
        self.max_adapters = max_adapters
        self.warm_context = warm_context
        self.multi_gap_threshold = multi_gap_threshold
        self.broad_min_adapters = broad_min_adapters

        self.generation_config = generation_config or GenerationConfig()

        # Cap the synthesis budget at the per-call generation budget so a
        # CF eval (max_new_tokens=32) doesn't get a 512-token Resolver
        # pass that emits a paragraph (Change 4 fix). For long-form splits
        # the cap is lifted at call time inside `synthesize()` so the
        # Resolver can reproduce multi-paragraph QM answers in full.
        effective_synth_tokens = min(
            synthesis_max_new_tokens, self.generation_config.max_new_tokens
        )
        self.synthesis_max_new_tokens = effective_synth_tokens
        self.synthesis_max_new_tokens_long_form = synthesis_max_new_tokens_long_form

        self._synthesis_config = GenerationConfig(
            max_new_tokens=effective_synth_tokens,
            temperature=self.generation_config.temperature,
            top_p=self.generation_config.top_p,
            do_sample=self.generation_config.do_sample,
            repetition_penalty=self.generation_config.repetition_penalty,
            stop_sequences=self.generation_config.stop_sequences,
        )

        self._prompt_builder = PromptBuilder(
            tokenizer=llm.tokenizer,
            system_prompt=system_prompt,
        )

        logger.info("=" * 60)
        logger.info("PARALLEL ORCHESTRATOR INITIALIZED")
        logger.info("=" * 60)
        logger.info(f"  Query planner: {query_planner_mode}")
        logger.info(f"  Max adapters: {max_adapters}")
        logger.info(f"  Synthesis tokens: {effective_synth_tokens}")
        logger.info(f"  Warm context: {warm_context}")
        logger.info(f"  Multi gap threshold: {multi_gap_threshold}")
        logger.info("=" * 60)

    # -------------------------------------------------------------------------
    # Query Planner
    # -------------------------------------------------------------------------

    def plan_query(
        self,
        query: str,
        query_embedding: np.ndarray | None = None,
        allowed_adapter_ids: set[str] | None = None,
        fallback_threshold: float | None = None,
    ) -> QueryPlan:
        """Analyze a query to determine the adapter selection strategy.

        Args:
            query: User's input query.
            query_embedding: Optional pre-computed embedding (Change 7
                cache) — passed straight through to the similarity planner.
            allowed_adapter_ids: Optional Stage-1 mask (Phase 4); when
                supplied, the planner only sees the in-domain adapter
                subset, so MULTI/BROAD plans don't get inflated by
                cross-domain false positives.
            fallback_threshold: Optional override for the per-adapter τ
                fallback (Phase 4 lowered τ inside an active domain).

        Returns:
            QueryPlan with plan type and reasoning.
        """
        if self.query_planner_mode == "llm":
            return self._plan_query_llm(query)
        if self.query_planner_mode in ("keyword", "heuristic"):
            return self._plan_query_keyword(
                query,
                query_embedding=query_embedding,
                allowed_adapter_ids=allowed_adapter_ids,
                fallback_threshold=fallback_threshold,
            )
        return self._plan_query_similarity(
            query,
            query_embedding=query_embedding,
            allowed_adapter_ids=allowed_adapter_ids,
            fallback_threshold=fallback_threshold,
        )

    def _plan_query_similarity(
        self,
        query: str,
        query_embedding: np.ndarray | None = None,
        allowed_adapter_ids: set[str] | None = None,
        fallback_threshold: float | None = None,
    ) -> QueryPlan:
        """Default planner — pure similarity geometry (no keyword regex).

        Fires MULTI / BROAD whenever the query lights up multiple adapters
        within a tight similarity gap (the "overlap" signal that the
        TODO 7 calibration table quantifies as ``calibration_quality<0``).
        Empirically the keyword regex never fires on factoid evals — see
        ``tasks/fix_parallel_orchestrator.md`` Change 3.

        Strategy:
        1. Score every adapter via the router's shared cluster-flat path.
        2. Keep adapters whose similarity passes their own calibrated τ
           (or the global fallback for adapters with no calibration).
        3. ``≥broad_min_adapters`` above-τ AND tight gap ⇒ BROAD_COMPOSITION.
        4. ``≥2`` above-τ AND tight gap ⇒ MULTI_TEMPORAL.
        5. otherwise ⇒ SINGLE_LATEST.

        ``tight gap`` means the top-1 vs. top-N similarity range is
        ≤ ``self.multi_gap_threshold``.
        """
        try:
            _, sims, taus = self.router._score_adapters(
                query,
                query_embedding=query_embedding,
                allowed_adapter_ids=allowed_adapter_ids,
                fallback_threshold=fallback_threshold,
            )
        except (ValueError, RuntimeError):
            return QueryPlan(
                plan_type=QueryPlanType.SINGLE_LATEST,
                selected_adapters=[],
                reasoning="No adapters scored (empty manifest or embedding error)",
            )

        above_tau = sorted(
            (s for aid, s in sims.items() if s >= taus.get(aid, self.router.similarity_threshold)),
            reverse=True,
        )

        if not above_tau:
            return QueryPlan(
                plan_type=QueryPlanType.SINGLE_LATEST,
                selected_adapters=[],
                reasoning="No adapters above per-adapter τ — falls through to base",
            )

        if len(above_tau) == 1:
            return QueryPlan(
                plan_type=QueryPlanType.SINGLE_LATEST,
                selected_adapters=[],
                reasoning=f"Single adapter above τ (sim={above_tau[0]:.3f})",
            )

        gap = above_tau[0] - above_tau[-1]
        tight = gap <= self.multi_gap_threshold

        if tight and len(above_tau) >= self.broad_min_adapters:
            return QueryPlan(
                plan_type=QueryPlanType.BROAD_COMPOSITION,
                selected_adapters=[],
                reasoning=(
                    f"{len(above_tau)} adapters above τ within "
                    f"gap={gap:.3f} ≤ {self.multi_gap_threshold:.3f}"
                ),
            )
        if tight:
            return QueryPlan(
                plan_type=QueryPlanType.MULTI_TEMPORAL,
                selected_adapters=[],
                reasoning=(
                    f"{len(above_tau)} adapters above τ within "
                    f"gap={gap:.3f} ≤ {self.multi_gap_threshold:.3f}"
                ),
            )
        return QueryPlan(
            plan_type=QueryPlanType.SINGLE_LATEST,
            selected_adapters=[],
            reasoning=(
                f"{len(above_tau)} adapters above τ but gap={gap:.3f} > "
                f"{self.multi_gap_threshold:.3f} (top-1 dominates)"
            ),
        )

    def _plan_query_keyword(
        self,
        query: str,
        query_embedding: np.ndarray | None = None,
        allowed_adapter_ids: set[str] | None = None,
        fallback_threshold: float | None = None,
    ) -> QueryPlan:
        """Legacy keyword + similarity-distribution planner (kept for ablation).

        Empirically classifies 98% of factoid queries as ``SINGLE_LATEST``
        (incl. 97% of the SQA temporal split) — see ``tasks/
        fix_parallel_orchestrator.md`` for the empirical breakdown. Wired
        only when ``query_planner_mode`` is ``"keyword"`` or the legacy
        alias ``"heuristic"``.
        """
        has_temporal = bool(_MULTI_TEMPORAL_PATTERNS.search(query))
        has_broad = bool(_BROAD_COMPOSITION_PATTERNS.search(query))

        try:
            _, sims, taus = self.router._score_adapters(
                query,
                query_embedding=query_embedding,
                allowed_adapter_ids=allowed_adapter_ids,
                fallback_threshold=fallback_threshold,
            )
        except (ValueError, RuntimeError):
            sims, taus = {}, {}

        above_tau_sims = sorted(
            (s for aid, s in sims.items() if s >= taus.get(aid, self.router.similarity_threshold)),
            reverse=True,
        )
        above_threshold = len(above_tau_sims)
        has_cluster = (
            above_threshold >= self.broad_min_adapters
            and (above_tau_sims[0] - above_tau_sims[-1]) <= self.multi_gap_threshold
        )

        if has_temporal and above_threshold >= 2:
            plan_type = QueryPlanType.MULTI_TEMPORAL
            reasoning = (
                f"Temporal keywords detected, {above_threshold} adapters above τ"
            )
        elif has_broad or has_cluster:
            if above_threshold >= 2:
                plan_type = QueryPlanType.BROAD_COMPOSITION
                reasoning = (
                    f"Broad keywords={has_broad}, cluster={has_cluster}, "
                    f"{above_threshold} adapters above τ"
                )
            else:
                plan_type = QueryPlanType.SINGLE_LATEST
                reasoning = (
                    f"Broad/cluster detected but only {above_threshold} adapter(s) match"
                )
        else:
            plan_type = QueryPlanType.SINGLE_LATEST
            reasoning = "No temporal/broad signals detected, using fast path"

        return QueryPlan(plan_type=plan_type, selected_adapters=[], reasoning=reasoning)

    def _plan_query_llm(self, query: str) -> QueryPlan:
        """LLM-based query planner using the base model to classify intent.

        Slower but more accurate for ambiguous queries. Used only as an
        ablation knob (``query_planner_mode="llm"``); the post-May-1
        primary planner is the unsupervised similarity-distribution
        rule (``_plan_query_similarity``).
        """
        from src.inference import generate_text

        classification_prompt = self._prompt_builder.build(
            query=(
                "Classify the question into exactly ONE category. Output the "
                "category name only — no explanation, no preamble, no "
                "punctuation.\n\n"
                "Categories:\n"
                "- SINGLE_LATEST: asks about a single current fact or state.\n"
                "- MULTI_TEMPORAL: asks about changes over time or "
                "comparisons across periods.\n"
                "- BROAD_COMPOSITION: asks for a comprehensive overview "
                "combining multiple knowledge areas.\n\n"
                "Examples:\n"
                "Q: Who is the prime minister of India?\nA: SINGLE_LATEST\n"
                "Q: How did the US president change between 2020 and 2024?\n"
                "A: MULTI_TEMPORAL\n"
                "Q: Give an overview of European foreign policy.\n"
                "A: BROAD_COMPOSITION\n\n"
                f"Q: {query}\nA:"
            ),
            include_system=False,
        )

        if self.llm.has_expert_attached:
            self.llm.detach_expert()

        from src.inference import GenerationConfig

        # Tight stop_sequences so the classifier halts after one token
        # (avoids "The category is SINGLE_LATEST." → falls-through cases).
        classify_config = GenerationConfig(
            max_new_tokens=8,
            temperature=0.0,
            do_sample=False,
            stop_sequences=("\n", "."),
        )

        model, tokenizer = self.llm.get_inference_components()
        raw = generate_text(model, tokenizer, classification_prompt, classify_config, self.use_gpu)
        raw_upper = raw.strip().upper()

        if "MULTI_TEMPORAL" in raw_upper:
            plan_type = QueryPlanType.MULTI_TEMPORAL
        elif "BROAD_COMPOSITION" in raw_upper:
            plan_type = QueryPlanType.BROAD_COMPOSITION
        else:
            plan_type = QueryPlanType.SINGLE_LATEST

        return QueryPlan(
            plan_type=plan_type,
            selected_adapters=[],
            reasoning=f"LLM classified as {plan_type.value} (raw: {raw.strip()!r})",
        )

    # -------------------------------------------------------------------------
    # Adapter Selection
    # -------------------------------------------------------------------------

    def select_adapters(
        self,
        query: str,
        plan: QueryPlan,
        query_embedding: np.ndarray | None = None,
        allowed_adapter_ids: set[str] | None = None,
        fallback_threshold: float | None = None,
    ) -> list[AdapterMatch]:
        """Select adapters based on query similarity and plan type.

        Routes through ``CentroidRouter._score_adapters`` so that
        per-chunk anchors and per-adapter calibrated τ are honoured
        identically to ``CentroidRouter.route()`` (Change 1 — single
        source of truth for the routing core).

        Args:
            query: User's input query.
            plan: The query plan from ``plan_query``.
            query_embedding: Optional pre-computed embedding (Change 7).
            allowed_adapter_ids: Optional Stage-1 mask (Phase 4) — keeps
                Parallel from picking SQA-domain adapters for a CF
                query (and vice versa) when the domain class is known.
            fallback_threshold: Optional override for the per-adapter τ
                fallback (Phase 4 lowered τ inside an active domain).

        Returns:
            List of AdapterMatch objects for parallel execution.
        """
        try:
            _, sims, taus = self.router._score_adapters(
                query,
                query_embedding=query_embedding,
                allowed_adapter_ids=allowed_adapter_ids,
                fallback_threshold=fallback_threshold,
            )
        except (ValueError, RuntimeError):
            logger.warning("No adapters with centroids available")
            return []

        manifest = self.router._manifest
        candidates = [
            AdapterMatch(
                adapter_id=aid,
                similarity=sims[aid],
                timestamp=manifest[aid].timestamp,
            )
            for aid in sims
            if sims[aid] >= taus.get(aid, self.router.similarity_threshold)
        ]

        if not candidates:
            return []

        # Apply selection strategy. The legacy "lowered threshold for
        # broad queries" fallback is gone — per-adapter calibrated τ
        # already encodes each adapter's correct operating point.
        if plan.plan_type == QueryPlanType.SINGLE_LATEST:
            candidates.sort(key=lambda m: m.similarity, reverse=True)
            selected = candidates[:1]
        elif plan.plan_type == QueryPlanType.MULTI_TEMPORAL:
            # Top-N by similarity, then re-order by timestamp so the
            # most-recent winner is the conflict resolver (matches
            # ``CentroidRouter.route``'s "newest wins on conflict" rule).
            candidates.sort(key=lambda m: m.similarity, reverse=True)
            selected = candidates[: self.max_adapters]
            selected.sort(key=lambda m: m.timestamp)
        elif plan.plan_type == QueryPlanType.BROAD_COMPOSITION:
            candidates.sort(key=lambda m: m.similarity, reverse=True)
            selected = candidates[: self.max_adapters]
        else:
            selected = candidates[:1]

        logger.info(
            f"Selected {len(selected)} adapter(s) for {plan.plan_type.value}: "
            f"{[m.adapter_id for m in selected]}"
        )
        return selected

    # -------------------------------------------------------------------------
    # Parallel Execution Engine
    # -------------------------------------------------------------------------

    def execute_parallel(
        self,
        query: str,
        selected_adapters: list[AdapterMatch],
        query_embedding: np.ndarray | None = None,
        long_form: bool = False,
    ) -> tuple[dict[str, str], dict[str, float], dict[str, list[RetrievedChunk]]]:
        """Execute generation through each selected adapter sequentially.

        Hot-swaps adapters on the single GPU, generating one response per
        adapter with **always-on Source-Replay** (Change 2). Each adapter
        sees the top-K retrieved chunks from its own ``DataIndices_i`` so
        the LoRA can bias the distribution and the retrieved tokens
        provide the gold sequence — matches PnR's v5 behaviour.

        Args:
            query: User's input query.
            selected_adapters: Adapters to generate with.
            query_embedding: Optional pre-computed embedding (Change 7).
                When None we compute it once and reuse across the per-
                adapter retrieval calls.

        Returns:
            Tuple of ``(adapter_outputs, per_adapter_latency_ms,
            per_adapter_chunks)``. The chunks dict carries the retrieved
            evidence per adapter so the Resolver pass can include it in
            the synthesis prompt without re-running retrieval.
        """
        from src.eval.metrics import parse_model_output
        from src.inference import generate_text

        adapter_outputs: dict[str, str] = {}
        latencies: dict[str, float] = {}
        per_adapter_chunks: dict[str, list[RetrievedChunk]] = {}

        if query_embedding is None:
            query_embedding = self.router.compute_embedding(query)

        for match in selected_adapters:
            adapter_id = match.adapter_id
            entry = self.router._manifest.get(adapter_id)
            if entry is None:
                logger.warning(f"Adapter {adapter_id} not in manifest, skipping")
                continue

            # Hot-swap adapter
            if self.llm.has_expert_attached:
                self.llm.detach_expert()
            self.llm.load_expert(entry.adapter_path)

            # Source-Replay context for THIS adapter (Change 2). Filter
            # by the same retrieval threshold the centroid router uses,
            # so single-adapter Parallel emits prompts that are
            # byte-comparable to single-adapter PnR.
            chunks = self._retrieve_for_adapter(
                adapter_id=adapter_id,
                query_embedding=query_embedding,
                top_k=self.router.winner_replay_top_k,
            )
            per_adapter_chunks[adapter_id] = chunks
            retrieved_context = SourceReplayStore.build_context(
                chunks,
                max_context_length=self.router.max_context_length,
            )

            prompt = self._prompt_builder.build(
                query=query, retrieved_context=retrieved_context,
            )

            # Generate. Long-form splits widen the per-adapter budget so
            # multi-paragraph QM answers reach the Resolver intact, and
            # clear the short-answer stop sequences ("\n", ".", "!", "?")
            # — otherwise the first newline after a markdown header kills
            # generation, leaving the Resolver with a stub like
            # "**Responsible Persons:**" instead of the full document.
            per_adapter_gen_config = self.generation_config
            if long_form:
                from src.inference import GenerationConfig
                per_adapter_gen_config = GenerationConfig(
                    max_new_tokens=max(
                        self.generation_config.max_new_tokens,
                        self.synthesis_max_new_tokens_long_form // 2,
                    ),
                    temperature=self.generation_config.temperature,
                    top_p=self.generation_config.top_p,
                    do_sample=self.generation_config.do_sample,
                    repetition_penalty=self.generation_config.repetition_penalty,
                    stop_sequences=(),
                )
            model, tokenizer = self.llm.get_inference_components()
            t_start = time.perf_counter()
            raw_output = generate_text(
                model, tokenizer, prompt, per_adapter_gen_config, self.use_gpu
            )
            t_end = time.perf_counter()

            # Strip <think> blocks. For short-answer splits, also collapse
            # to first sentence so the Resolver sees the same string EM
            # will. For long-form splits, keep the full multi-paragraph
            # answer — that IS the answer the Resolver must reproduce.
            parsed = parse_model_output(
                raw_output,
                truncate_to_short_answer=not long_form,
            )
            adapter_outputs[adapter_id] = parsed
            latencies[adapter_id] = (t_end - t_start) * 1000.0

            logger.info(
                f"  [{adapter_id}] generated {len(parsed)} chars "
                f"in {latencies[adapter_id]:.0f}ms "
                f"(replay_chunks={len(chunks)})"
            )

        # NOTE: do NOT unconditionally detach here. Change 5 wires the
        # synthesis pass to run with the most-recent winner adapter still
        # loaded. ``generate()`` is responsible for the final detach so
        # the stateless / warm-context policy is enforced in one place.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return adapter_outputs, latencies, per_adapter_chunks

    def _retrieve_for_adapter(
        self,
        adapter_id: str,
        query_embedding: np.ndarray,
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Run Source-Replay against a single adapter's index.

        Returns an empty list (rather than raising) when the router has
        no Source-Replay store loaded — keeps Parallel running on
        manifests built before the v5 FAISS sidecar work.
        """
        store = self.router._source_replay
        if store is None:
            return []
        try:
            chunks = store.retrieve(
                query_embedding=query_embedding,
                adapter_id=adapter_id,
                top_k=top_k,
            )
        except (KeyError, ValueError) as exc:
            logger.warning(f"Source-Replay retrieve({adapter_id}) failed: {exc}")
            return []
        return [c for c in chunks if c.similarity >= self.router.retrieval_threshold]

    # -------------------------------------------------------------------------
    # Context Synthesis Agent (The Resolver)
    # -------------------------------------------------------------------------

    def synthesize(
        self,
        query: str,
        adapter_outputs: dict[str, str],
        per_adapter_chunks: dict[str, list[RetrievedChunk]] | None = None,
        resolver_adapter_id: str | None = None,
        long_form: bool = False,
    ) -> tuple[str, str, float]:
        """Synthesize multiple adapter outputs into a unified response.

        Implements Changes 4 + 5:

        - The Resolver runs with the **most-recent winner adapter loaded**
          (passed as ``resolver_adapter_id``) instead of the bare frozen
          base. This realises the thesis principle that the newest source
          wins on conflict (`docs/main.tex` §"Time-Aware Centroid
          Routing with Source-Replay") and gives the Resolver a LoRA
          bias that aligns with the recency rule in the prompt.
        - The synthesis prompt uses ``SYNTHESIS_PROMPT_TEMPLATE`` (Change 4):
          short-answer rule + recency-priority + retrieved-evidence
          section. ``stop_sequences`` and a tightened budget (capped at
          the per-call generation budget) keep output strict-EM-friendly.

        Args:
            query: User's original query.
            adapter_outputs: ``adapter_id → generated text`` per adapter.
            per_adapter_chunks: Retrieved evidence carried over from
                ``execute_parallel`` (Change 2). When None we fall back
                to running retrieval inline; the inline path exists so
                ``score_targets`` can synthesise without recomputing.
            resolver_adapter_id: Which adapter to leave attached during
                the Resolver pass. ``None`` ⇒ use the frozen base
                (legacy behaviour, kept for ablation).

        Returns:
            Tuple of ``(synthesized_response, synthesis_prompt, latency_ms)``.
        """
        from src.inference import generate_text

        manifest = self.router._manifest

        # Per-adapter source answers, ordered by training timestamp so
        # the prompt's "prefer the LATEST source" rule has the recent
        # answer at the bottom (closer to the answer cursor — empirically
        # less likely to be ignored under instruction following).
        ordered_ids = sorted(
            adapter_outputs.keys(),
            key=lambda aid: getattr(manifest.get(aid), "timestamp", 0.0) or 0.0,
        )
        source_parts: list[str] = []
        for adapter_id in ordered_ids:
            entry = manifest.get(adapter_id)
            ts_desc = (
                datetime.fromtimestamp(entry.timestamp).strftime("%Y-%m-%d")
                if entry and entry.timestamp
                else "unknown date"
            )
            source_parts.append(
                f"[Source: {adapter_id} (trained {ts_desc})]\n"
                f"{adapter_outputs[adapter_id]}"
            )
        source_answers = "\n\n".join(source_parts)

        # Retrieved-evidence block. Deduplicate by chunk text so the
        # Resolver doesn't see the same fact thrice when overlapping
        # adapters all surface the same training row.
        evidence_chunks: list[RetrievedChunk] = []
        seen_texts: set[str] = set()
        if per_adapter_chunks:
            for adapter_id in ordered_ids:
                for c in per_adapter_chunks.get(adapter_id, []):
                    if c.text in seen_texts:
                        continue
                    seen_texts.add(c.text)
                    evidence_chunks.append(c)
        evidence_chunks.sort(key=lambda c: c.similarity, reverse=True)
        retrieved_evidence = (
            SourceReplayStore.build_context(
                evidence_chunks,
                max_context_length=self.router.max_context_length,
            )
            or NO_EVIDENCE_PLACEHOLDER
        )

        template = (
            SYNTHESIS_PROMPT_TEMPLATE_LONG_FORM if long_form else SYNTHESIS_PROMPT_TEMPLATE
        )
        synthesis_text = template.format(
            query=query,
            source_answers=source_answers,
            retrieved_evidence=retrieved_evidence,
        )

        synthesis_prompt = self._prompt_builder.build(
            query=synthesis_text,
            include_system=False,
        )

        # Change 5 — load the most-recent winner adapter for the Resolver
        # pass (or the configured one). The thesis principle "newest wins
        # on conflict" is honoured by both the prompt rule (Change 4) and
        # the LoRA distribution shift, so they pull in the same direction.
        if resolver_adapter_id is not None:
            entry = manifest.get(resolver_adapter_id)
            if entry is not None and entry.adapter_path:
                if self.llm.has_expert_attached:
                    self.llm.detach_expert()
                self.llm.load_expert(entry.adapter_path)
                logger.info(
                    f"Resolver pass running with adapter loaded: {resolver_adapter_id}"
                )
        else:
            if self.llm.has_expert_attached:
                self.llm.detach_expert()

        # Long-form synthesis lifts the construction-time cap (which is
        # clamped to the short-factoid generation budget) so multi-paragraph
        # QM answers fit; CF/SQA keep the tight strict-EM-friendly budget.
        # Stop sequences are cleared for the same reason — the short-answer
        # boundaries cut the Resolver off after the first heading newline.
        synth_config = self._synthesis_config
        if long_form:
            from src.inference import GenerationConfig
            synth_config = GenerationConfig(
                max_new_tokens=self.synthesis_max_new_tokens_long_form,
                temperature=self._synthesis_config.temperature,
                top_p=self._synthesis_config.top_p,
                do_sample=self._synthesis_config.do_sample,
                repetition_penalty=self._synthesis_config.repetition_penalty,
                stop_sequences=(),
            )

        model, tokenizer = self.llm.get_inference_components()
        t_start = time.perf_counter()
        synthesized = generate_text(
            model, tokenizer, synthesis_prompt, synth_config, self.use_gpu
        )
        t_end = time.perf_counter()
        latency_ms = (t_end - t_start) * 1000.0

        logger.info(
            f"Synthesis completed in {latency_ms:.0f}ms "
            f"(resolver_adapter={resolver_adapter_id or '<none>'}, "
            f"evidence_chunks={len(evidence_chunks)})"
        )
        return synthesized, synthesis_prompt, latency_ms

    # -------------------------------------------------------------------------
    # Main Entry Point
    # -------------------------------------------------------------------------

    def generate(
        self,
        query: str,
        long_form: bool = False,
        **kwargs: Any,
    ) -> OrchestratorResult:
        """Run the full Parallel-Orchestrator pipeline.

        1. Plan query → determine adapter selection strategy
        2. Select adapters → choose which adapters to query
        3. Execute parallel → generate per-adapter responses with
           always-on Source-Replay (Change 2)
        4. Synthesize → merge outputs via the Resolver running with the
           most-recent winner adapter loaded (Change 5)

        Short-circuits to single-adapter mode when only one adapter
        matches, avoiding the synthesis overhead.

        Args:
            query: User's input query.
            **kwargs: Reserved for future extension.

        Returns:
            OrchestratorResult with synthesized response and metadata.
        """
        logger.info("=" * 60)
        logger.info("PARALLEL ORCHESTRATOR: Processing query")
        logger.info("=" * 60)
        logger.info(f"  Query: {query[:80]}...")

        # Embed once, reuse across plan/select/execute/build (Change 7).
        query_embedding = self.router.compute_embedding(query)

        # Stage-1 domain gate (Phase 4 / NF-1). The orchestrator must apply
        # the same mask the underlying CentroidRouter would, otherwise its
        # Parallel path re-introduces the cross-domain false-positive
        # selection that gives SQA queries CF adapters as winners.
        allowed_adapter_ids: set[str] | None = None
        fallback_threshold: float | None = None
        stage1 = self.router._classify_domain(query)
        if stage1 is not None:
            top_class, top_prob, _all_probs = stage1
            logger.info(
                f"  Stage-1 domain={top_class} prob={top_prob:.3f} "
                f"(thr={self.router._domain_confidence_threshold})"
            )
            if top_prob >= self.router._domain_confidence_threshold:
                if top_class == "ood_trivia":
                    logger.info(
                        f"  Stage-1 ood_trivia (prob={top_prob:.3f}) → base-only"
                    )
                    return self._generate_base_only(
                        query,
                        QueryPlan(
                            plan_type=QueryPlanType.SINGLE_LATEST,
                            selected_adapters=[],
                            reasoning=(
                                f"Stage-1 ood_trivia "
                                f"(prob={top_prob:.3f} ≥ "
                                f"{self.router._domain_confidence_threshold})"
                            ),
                        ),
                        query_embedding,
                    )
                allowed_adapter_ids = self.router._allowed_adapters_for_domain(top_class)
                fallback_threshold = self.router._domain_fallback_threshold

        plan = self.plan_query(
            query,
            query_embedding=query_embedding,
            allowed_adapter_ids=allowed_adapter_ids,
            fallback_threshold=fallback_threshold,
        )
        logger.info(f"  Plan: {plan.plan_type.value} — {plan.reasoning}")

        selected = self.select_adapters(
            query,
            plan,
            query_embedding=query_embedding,
            allowed_adapter_ids=allowed_adapter_ids,
            fallback_threshold=fallback_threshold,
        )
        plan.selected_adapters = selected

        if not selected:
            logger.info("No adapters selected, using base model only")
            return self._generate_base_only(query, plan, query_embedding)

        adapter_outputs, latencies, per_adapter_chunks = self.execute_parallel(
            query, selected, query_embedding=query_embedding, long_form=long_form,
        )

        if not adapter_outputs:
            logger.warning("No adapter outputs produced, falling back to base model")
            return self._generate_base_only(query, plan, query_embedding)

        # Short-circuit: single adapter, skip synthesis. The adapter
        # stays attached for the duration of the call; ``_finalise_state``
        # handles the stateless / warm-context detach at the end.
        if len(adapter_outputs) == 1:
            adapter_id = next(iter(adapter_outputs))
            response = next(iter(adapter_outputs.values()))
            logger.info(f"Single adapter ({adapter_id}), skipping synthesis")
            result = OrchestratorResult(
                response=response,
                adapter_loaded=adapter_id,
                routing_result=self._build_routing_result(
                    selected, query_embedding=query_embedding,
                ),
                adapter_outputs=adapter_outputs,
                query_plan=plan,
                synthesis_prompt="",
                per_adapter_latency_ms=latencies,
                synthesis_latency_ms=0.0,
            )
            self._finalise_state()
            return result

        # Step 4: Synthesize. The Resolver runs with the most-recent
        # winner adapter loaded (Change 5).
        most_recent = max(selected, key=lambda m: m.timestamp or 0.0)
        synthesized, synthesis_prompt, synth_latency = self.synthesize(
            query,
            adapter_outputs,
            per_adapter_chunks=per_adapter_chunks,
            resolver_adapter_id=most_recent.adapter_id,
            long_form=long_form,
        )

        # Audit string is the comma-joined per-adapter set so the eval
        # runner's ``set(adapter_used.split(","))`` membership check still
        # works. Synthesis vs. short-circuit is recoverable from
        # ``len(result.adapter_outputs) >= 2`` (already logged by the
        # runner) and the Resolver's adapter is on ``routing_result.
        # winner_adapter`` (Change 5 — most-recent winner).
        adapter_ids_str = ",".join(adapter_outputs.keys())

        result = OrchestratorResult(
            response=synthesized,
            adapter_loaded=adapter_ids_str,
            routing_result=self._build_routing_result(
                selected,
                query_embedding=query_embedding,
                resolver_adapter_id=most_recent.adapter_id,
            ),
            adapter_outputs=adapter_outputs,
            query_plan=plan,
            synthesis_prompt=synthesis_prompt,
            per_adapter_latency_ms=latencies,
            synthesis_latency_ms=synth_latency,
        )
        self._finalise_state()
        return result

    def _finalise_state(self) -> None:
        """Detach any expert left over by ``generate`` per the stateless
        policy (Change 0). With ``warm_context=True`` we leave whatever
        adapter happens to be loaded — that's the legacy "sticky" path
        kept for ablation and for parity with ``warm_context=True`` PnR.
        """
        if self.warm_context:
            return
        if self.llm.has_expert_attached:
            self.llm.detach_expert()

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _generate_base_only(
        self,
        query: str,
        plan: QueryPlan,
        query_embedding: np.ndarray | None = None,
    ) -> OrchestratorResult:
        """Fallback: generate using base model with no adapter."""
        from src.inference import generate_text

        # Stateless detach (matches PnR's `warm_context=False` semantics —
        # see Change 0). With `warm_context=True` we keep whatever adapter
        # was attached, mirroring PnR's legacy sticky behaviour for
        # apples-to-apples ablation.
        if not self.warm_context and self.llm.has_expert_attached:
            self.llm.detach_expert()

        prompt = self._prompt_builder.build(query=query)
        model, tokenizer = self.llm.get_inference_components()
        response = generate_text(
            model, tokenizer, prompt, self.generation_config, self.use_gpu
        )

        return OrchestratorResult(
            response=response,
            adapter_loaded=None,
            routing_result=self._empty_routing_result(query, query_embedding),
            query_plan=plan,
        )

    def _empty_routing_result(
        self,
        query: str,
        query_embedding: np.ndarray | None = None,
    ) -> RoutingResult:
        """Routing result for the no-winner / base-only path.

        Returns a populated ``RoutingResult`` (instead of ``None``) so the
        eval runner can read ``.has_conflict`` / ``.winner_similarity``
        without a second null check — fixes the latent
        ``AttributeError`` that would surface once Change 1 increases the
        rate at which the runner reaches this branch.
        """
        embedded = (
            query_embedding
            if query_embedding is not None
            else self.router.compute_embedding(query)
        )
        return RoutingResult(
            winner_adapter=None,
            winner_path=None,
            retrieved_context="",
            all_matches=[],
            query_embedding=embedded,
            has_conflict=False,
            routing_strategy=RoutingStrategy.PARALLEL,
        )

    def _build_routing_result(
        self,
        selected: list[AdapterMatch],
        query_embedding: np.ndarray,
        resolver_adapter_id: str | None = None,
    ) -> RoutingResult:
        """Build a RoutingResult for eval compatibility.

        ``winner_similarity`` is now populated via the
        ``RoutingResult.winner_similarity`` property — which iterates the
        ``all_matches`` list and returns the ``is_winner=True`` row's
        ``similarity``. We mark the *most-recent* selected adapter as
        the winner (Change 5) so the eval-time ``adapter_used`` audit
        agrees with which adapter actually drove the Resolver pass.
        """
        manifest = self.router._manifest

        winner_id = (
            resolver_adapter_id
            if resolver_adapter_id is not None
            else (selected[0].adapter_id if selected else None)
        )

        matches = [
            AdapterMatch(
                adapter_id=m.adapter_id,
                similarity=m.similarity,
                timestamp=m.timestamp,
                is_winner=(m.adapter_id == winner_id),
            )
            for m in selected
        ]

        winner_entry = manifest.get(winner_id) if winner_id else None

        return RoutingResult(
            winner_adapter=winner_id,
            winner_path=winner_entry.adapter_path if winner_entry else None,
            retrieved_context="",
            all_matches=matches,
            query_embedding=query_embedding,
            has_conflict=len(matches) > 1,
            routing_strategy=RoutingStrategy.PARALLEL,
        )

    def get_router(self) -> CentroidRouter:
        """Get the underlying CentroidRouter."""
        return self.router

    # -------------------------------------------------------------------------
    # Log-probability scoring (ROME / MEMIT-style ESR)
    # -------------------------------------------------------------------------

    def score_targets(self, query: str, targets: list[str]) -> dict[str, float]:
        """Compute log P(target | prompt) under the Resolver / single-adapter state.

        The Parallel Orchestrator's final answer is produced either by the
        single matching adapter (short-circuit path) or by the Resolver
        with the most-recent winner adapter attached (Change 5). For
        log-prob scoring we mirror the same decision tree, including
        Source-Replay context (Change 2) and the recency-priority
        Resolver state (Change 5), so that generation-ESR and log-prob-
        ESR aren't desynchronised by retrieval or by adapter state.
        """
        from src.inference import score_target_logprob

        manifest = self.router._manifest
        query_embedding = self.router.compute_embedding(query)
        plan = self.plan_query(query, query_embedding=query_embedding)
        selected = self.select_adapters(query, plan, query_embedding=query_embedding)

        def _score(prompt: str) -> dict[str, float]:
            model, tokenizer = self.llm.get_inference_components()
            return {
                t: score_target_logprob(
                    model=model, tokenizer=tokenizer, prompt=prompt,
                    target=t, use_gpu=self.use_gpu,
                )
                for t in targets
            }

        try:
            if not selected:
                if self.llm.has_expert_attached and not self.warm_context:
                    self.llm.detach_expert()
                prompt = self._prompt_builder.build(query=query)
                return _score(prompt)

            if len(selected) == 1:
                only = selected[0]
                entry = manifest.get(only.adapter_id)
                if self.llm.has_expert_attached:
                    self.llm.detach_expert()
                if entry is not None:
                    self.llm.load_expert(entry.adapter_path)
                chunks = self._retrieve_for_adapter(
                    only.adapter_id, query_embedding,
                    self.router.winner_replay_top_k,
                )
                retrieved_context = SourceReplayStore.build_context(
                    chunks, max_context_length=self.router.max_context_length,
                )
                prompt = self._prompt_builder.build(
                    query=query, retrieved_context=retrieved_context,
                )
                return _score(prompt)

            # Multi-adapter: drive the full execute_parallel + synthesis
            # state, then score against the Resolver prompt with the
            # most-recent adapter loaded.
            adapter_outputs, _, per_adapter_chunks = self.execute_parallel(
                query, selected, query_embedding=query_embedding,
            )
            if not adapter_outputs:
                if self.llm.has_expert_attached and not self.warm_context:
                    self.llm.detach_expert()
                prompt = self._prompt_builder.build(query=query)
                return _score(prompt)

            most_recent = max(selected, key=lambda m: m.timestamp or 0.0)

            ordered_ids = sorted(
                adapter_outputs.keys(),
                key=lambda aid: getattr(manifest.get(aid), "timestamp", 0.0) or 0.0,
            )
            source_parts: list[str] = []
            for adapter_id in ordered_ids:
                entry = manifest.get(adapter_id)
                ts_desc = (
                    datetime.fromtimestamp(entry.timestamp).strftime("%Y-%m-%d")
                    if entry and entry.timestamp else "unknown date"
                )
                source_parts.append(
                    f"[Source: {adapter_id} (trained {ts_desc})]\n"
                    f"{adapter_outputs[adapter_id]}"
                )
            evidence_chunks: list[RetrievedChunk] = []
            seen: set[str] = set()
            for adapter_id in ordered_ids:
                for c in per_adapter_chunks.get(adapter_id, []):
                    if c.text in seen:
                        continue
                    seen.add(c.text)
                    evidence_chunks.append(c)
            evidence_chunks.sort(key=lambda c: c.similarity, reverse=True)
            retrieved_evidence = (
                SourceReplayStore.build_context(
                    evidence_chunks,
                    max_context_length=self.router.max_context_length,
                )
                or NO_EVIDENCE_PLACEHOLDER
            )
            synthesis_text = SYNTHESIS_PROMPT_TEMPLATE.format(
                query=query,
                source_answers="\n\n".join(source_parts),
                retrieved_evidence=retrieved_evidence,
            )
            synthesis_prompt = self._prompt_builder.build(
                query=synthesis_text, include_system=False,
            )

            entry = manifest.get(most_recent.adapter_id)
            if entry is not None and entry.adapter_path:
                if self.llm.has_expert_attached:
                    self.llm.detach_expert()
                self.llm.load_expert(entry.adapter_path)

            return _score(synthesis_prompt)
        finally:
            # Stateless guarantee for downstream samples — same policy as
            # ``generate``'s ``_finalise_state``.
            if not self.warm_context and self.llm.has_expert_attached:
                self.llm.detach_expert()

    def summary(self) -> str:
        """Get a formatted summary of the orchestrator."""
        lines = [
            "=" * 60,
            "PARALLEL ORCHESTRATOR",
            "=" * 60,
            f"Query planner: {self.query_planner_mode}",
            f"Max adapters: {self.max_adapters}",
            f"Synthesis tokens: {self.synthesis_max_new_tokens}",
            f"Warm context: {self.warm_context}",
            f"Multi gap threshold: {self.multi_gap_threshold}",
            f"Broad min adapters: {self.broad_min_adapters}",
            f"Generation config: {self.generation_config.to_dict()}",
            "-" * 60,
            "ROUTER:",
            self.router.summary() if hasattr(self.router, "summary") else "N/A",
            "=" * 60,
        ]
        return "\n".join(lines)
