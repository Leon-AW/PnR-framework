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
You are an expert knowledge synthesizer. You have received answers to the same \
question from multiple specialized knowledge sources, each covering different \
contexts or time periods.

Question: {query}

--- Source Answers ---
{source_answers}
--- End of Source Answers ---

Instructions:
- Synthesize a single, comprehensive answer combining insights from all sources.
- If sources conflict, explain the change (e.g., "Previously X, but as of [date] Y").
- If sources provide complementary information, merge them coherently.
- Do not mention "sources" or "adapters" -- present the information naturally.
- Be concise and factual."""


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

    def __init__(
        self,
        centroid_router: CentroidRouter,
        llm: "PatchAndRouteLLM",
        generation_config: "GenerationConfig | None" = None,
        query_planner_mode: str = "heuristic",
        max_adapters: int = 5,
        synthesis_max_new_tokens: int = 512,
        use_gpu: bool = True,
        system_prompt: str | None = None,
    ) -> None:
        """Initialize the Parallel Orchestrator.

        Args:
            centroid_router: Router for embedding/similarity computation.
            llm: PatchAndRouteLLM for adapter hot-swapping and generation.
            generation_config: Generation parameters for per-adapter inference.
            query_planner_mode: "heuristic" (keyword + similarity) or "llm".
            max_adapters: Maximum adapters to query in parallel execution.
            synthesis_max_new_tokens: Token budget for the synthesis pass.
            use_gpu: Whether to use GPU for generation.
            system_prompt: Custom system prompt for per-adapter generation.
        """
        from src.inference import GenerationConfig, PromptBuilder

        self.router = centroid_router
        self.llm = llm
        self.use_gpu = use_gpu
        self.query_planner_mode = query_planner_mode
        self.max_adapters = max_adapters
        self.synthesis_max_new_tokens = synthesis_max_new_tokens

        self.generation_config = generation_config or GenerationConfig()
        self._synthesis_config = GenerationConfig(
            max_new_tokens=synthesis_max_new_tokens,
            temperature=self.generation_config.temperature,
            top_p=self.generation_config.top_p,
            do_sample=self.generation_config.do_sample,
            repetition_penalty=self.generation_config.repetition_penalty,
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
        logger.info(f"  Synthesis tokens: {synthesis_max_new_tokens}")
        logger.info("=" * 60)

    # -------------------------------------------------------------------------
    # Query Planner
    # -------------------------------------------------------------------------

    def plan_query(self, query: str) -> QueryPlan:
        """Analyze a query to determine the adapter selection strategy.

        Args:
            query: User's input query.

        Returns:
            QueryPlan with plan type and reasoning.
        """
        if self.query_planner_mode == "llm":
            return self._plan_query_llm(query)
        return self._plan_query_heuristic(query)

    def _plan_query_heuristic(self, query: str) -> QueryPlan:
        """Heuristic query planner using keyword detection + similarity distribution.

        Strategy:
        1. Check for temporal/comparative keywords → MULTI_TEMPORAL
        2. Check for broad/overview keywords → BROAD_COMPOSITION
        3. Check similarity distribution (many high matches → BROAD_COMPOSITION)
        4. Default → SINGLE_LATEST
        """
        # Keyword detection
        has_temporal = bool(_MULTI_TEMPORAL_PATTERNS.search(query))
        has_broad = bool(_BROAD_COMPOSITION_PATTERNS.search(query))

        # Similarity distribution analysis
        try:
            query_embedding = self.router.compute_embedding(query)
            centroids, adapter_ids = self.router._manifest.get_centroids_matrix()
            query_norm = query_embedding / np.linalg.norm(query_embedding)
            similarities = np.dot(centroids, query_norm)

            # Count matches above threshold
            above_threshold = sum(
                1 for s in similarities if s >= self.router.similarity_threshold
            )
            # Check if many adapters score similarly high (cluster)
            high_sims = sorted(
                [float(s) for s in similarities if s >= self.router.similarity_threshold],
                reverse=True,
            )
            has_cluster = (
                len(high_sims) >= 3
                and (high_sims[0] - high_sims[-1]) <= 0.15
            )
        except (ValueError, RuntimeError):
            above_threshold = 0
            has_cluster = False

        # Decision logic
        if has_temporal and above_threshold >= 2:
            plan_type = QueryPlanType.MULTI_TEMPORAL
            reasoning = (
                f"Temporal keywords detected, {above_threshold} adapters above threshold"
            )
        elif has_broad or has_cluster:
            if above_threshold >= 2:
                plan_type = QueryPlanType.BROAD_COMPOSITION
                reasoning = (
                    f"Broad keywords={has_broad}, cluster={has_cluster}, "
                    f"{above_threshold} adapters above threshold"
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

        Slower but more accurate for ambiguous queries.
        """
        from src.inference import generate_text

        classification_prompt = self._prompt_builder.build(
            query=(
                "Classify the following question into exactly one category.\n\n"
                f"Question: {query}\n\n"
                "Categories:\n"
                "- SINGLE_LATEST: The question asks about a single current fact or state.\n"
                "- MULTI_TEMPORAL: The question asks about changes over time or "
                "comparisons across periods.\n"
                "- BROAD_COMPOSITION: The question asks for a comprehensive overview "
                "combining multiple knowledge areas.\n\n"
                "Respond with ONLY the category name (e.g., SINGLE_LATEST)."
            ),
            include_system=False,
        )

        # Ensure no adapter is loaded for classification
        if self.llm.has_expert_attached:
            self.llm.detach_expert()

        from src.inference import GenerationConfig

        classify_config = GenerationConfig(
            max_new_tokens=20,
            temperature=0.0,
            do_sample=False,
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

    def select_adapters(self, query: str, plan: QueryPlan) -> list[AdapterMatch]:
        """Select adapters based on query similarity and plan type.

        Reuses CentroidRouter's embedding model and manifest for similarity
        computation, but applies different selection strategies per plan type.

        Args:
            query: User's input query.
            plan: The query plan from plan_query().

        Returns:
            List of AdapterMatch objects for parallel execution.
        """
        try:
            query_embedding = self.router.compute_embedding(query)
            centroids, adapter_ids = self.router._manifest.get_centroids_matrix()
        except (ValueError, RuntimeError):
            logger.warning("No adapters with centroids available")
            return []

        query_norm = query_embedding / np.linalg.norm(query_embedding)
        similarities = np.dot(centroids, query_norm)

        # Build candidate list above threshold
        candidates = []
        for adapter_id, sim in zip(adapter_ids, similarities):
            if sim >= self.router.similarity_threshold:
                entry = self.router._manifest[adapter_id]
                candidates.append(
                    AdapterMatch(
                        adapter_id=adapter_id,
                        similarity=float(sim),
                        timestamp=entry.timestamp,
                    )
                )

        if not candidates:
            # Fallback: use a lowered threshold for broad queries
            if plan.plan_type == QueryPlanType.BROAD_COMPOSITION:
                lowered = self.router.similarity_threshold * 0.85
                for adapter_id, sim in zip(adapter_ids, similarities):
                    if sim >= lowered:
                        entry = self.router._manifest[adapter_id]
                        candidates.append(
                            AdapterMatch(
                                adapter_id=adapter_id,
                                similarity=float(sim),
                                timestamp=entry.timestamp,
                            )
                        )

        if not candidates:
            return []

        # Apply selection strategy
        if plan.plan_type == QueryPlanType.SINGLE_LATEST:
            # Top-1 by similarity (same as centroid router)
            candidates.sort(key=lambda m: m.similarity, reverse=True)
            selected = candidates[:1]

        elif plan.plan_type == QueryPlanType.MULTI_TEMPORAL:
            # Sort by timestamp to ensure temporal diversity, then cap
            candidates.sort(key=lambda m: m.timestamp)
            selected = candidates[: self.max_adapters]

        elif plan.plan_type == QueryPlanType.BROAD_COMPOSITION:
            # All above threshold, sorted by similarity, capped
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
    ) -> tuple[dict[str, str], dict[str, float]]:
        """Execute generation through each selected adapter sequentially.

        Hot-swaps adapters on the single GPU, generating one response per adapter.

        Args:
            query: User's input query.
            selected_adapters: Adapters to generate with.

        Returns:
            Tuple of (adapter_outputs, per_adapter_latency_ms).
        """
        from src.eval.metrics import parse_model_output
        from src.inference import generate_text

        adapter_outputs: dict[str, str] = {}
        latencies: dict[str, float] = {}

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

            # Build prompt (no source-replay context -- each adapter answers independently)
            prompt = self._prompt_builder.build(query=query)

            # Generate
            model, tokenizer = self.llm.get_inference_components()
            t_start = time.perf_counter()
            raw_output = generate_text(
                model, tokenizer, prompt, self.generation_config, self.use_gpu
            )
            t_end = time.perf_counter()

            # Strip <think> blocks for cleaner synthesis input
            parsed = parse_model_output(raw_output)
            adapter_outputs[adapter_id] = parsed
            latencies[adapter_id] = (t_end - t_start) * 1000.0

            logger.info(
                f"  [{adapter_id}] generated {len(parsed)} chars "
                f"in {latencies[adapter_id]:.0f}ms"
            )

        # Detach last adapter to prepare for synthesis
        if self.llm.has_expert_attached:
            self.llm.detach_expert()

        # Free fragmented GPU memory between execution and synthesis
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return adapter_outputs, latencies

    # -------------------------------------------------------------------------
    # Context Synthesis Agent (The Resolver)
    # -------------------------------------------------------------------------

    def synthesize(
        self,
        query: str,
        adapter_outputs: dict[str, str],
    ) -> tuple[str, str, float]:
        """Synthesize multiple adapter outputs into a unified response.

        Uses the base model (no adapter) as the Resolver to merge all outputs.

        Args:
            query: User's original query.
            adapter_outputs: adapter_id -> generated text from each adapter.

        Returns:
            Tuple of (synthesized_response, synthesis_prompt, latency_ms).
        """
        from src.inference import generate_text

        # Build source answers section
        source_parts = []
        for adapter_id, output in adapter_outputs.items():
            entry = self.router._manifest.get(adapter_id)
            if entry and entry.timestamp:
                ts_desc = datetime.fromtimestamp(entry.timestamp).strftime("%Y-%m-%d")
            else:
                ts_desc = "unknown date"
            source_parts.append(
                f"[Source: {adapter_id} (trained {ts_desc})]\n{output}"
            )

        source_answers = "\n\n".join(source_parts)

        synthesis_text = SYNTHESIS_PROMPT_TEMPLATE.format(
            query=query,
            source_answers=source_answers,
        )

        # Apply chat template via PromptBuilder
        synthesis_prompt = self._prompt_builder.build(
            query=synthesis_text,
            include_system=False,
        )

        # Generate synthesis using base model (no adapter attached)
        model, tokenizer = self.llm.get_inference_components()
        t_start = time.perf_counter()
        synthesized = generate_text(
            model, tokenizer, synthesis_prompt, self._synthesis_config, self.use_gpu
        )
        t_end = time.perf_counter()
        latency_ms = (t_end - t_start) * 1000.0

        logger.info(f"Synthesis completed in {latency_ms:.0f}ms")
        return synthesized, synthesis_prompt, latency_ms

    # -------------------------------------------------------------------------
    # Main Entry Point
    # -------------------------------------------------------------------------

    def generate(
        self,
        query: str,
        **kwargs: Any,
    ) -> OrchestratorResult:
        """Run the full Parallel-Orchestrator pipeline.

        1. Plan query → determine adapter selection strategy
        2. Select adapters → choose which adapters to query
        3. Execute parallel → generate per-adapter responses
        4. Synthesize → merge outputs into unified answer

        Short-circuits to single-adapter mode when only one adapter matches,
        avoiding the synthesis overhead.

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

        # Step 1: Plan
        plan = self.plan_query(query)
        logger.info(f"  Plan: {plan.plan_type.value} — {plan.reasoning}")

        # Step 2: Select adapters
        selected = self.select_adapters(query, plan)
        plan.selected_adapters = selected

        # No adapters matched — fallback to base model
        if not selected:
            logger.info("No adapters selected, using base model only")
            return self._generate_base_only(query, plan)

        # Step 3: Execute parallel generation
        adapter_outputs, latencies = self.execute_parallel(query, selected)

        if not adapter_outputs:
            logger.warning("No adapter outputs produced, falling back to base model")
            return self._generate_base_only(query, plan)

        # Short-circuit: single adapter, skip synthesis
        if len(adapter_outputs) == 1:
            adapter_id = next(iter(adapter_outputs))
            response = next(iter(adapter_outputs.values()))
            logger.info(f"Single adapter ({adapter_id}), skipping synthesis")
            return OrchestratorResult(
                response=response,
                adapter_loaded=adapter_id,
                routing_result=self._build_routing_result(query, selected),
                adapter_outputs=adapter_outputs,
                query_plan=plan,
                synthesis_prompt="",
                per_adapter_latency_ms=latencies,
                synthesis_latency_ms=0.0,
            )

        # Step 4: Synthesize
        synthesized, synthesis_prompt, synth_latency = self.synthesize(
            query, adapter_outputs
        )

        adapter_ids_str = ",".join(adapter_outputs.keys())

        return OrchestratorResult(
            response=synthesized,
            adapter_loaded=adapter_ids_str,
            routing_result=self._build_routing_result(query, selected),
            adapter_outputs=adapter_outputs,
            query_plan=plan,
            synthesis_prompt=synthesis_prompt,
            per_adapter_latency_ms=latencies,
            synthesis_latency_ms=synth_latency,
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _generate_base_only(
        self,
        query: str,
        plan: QueryPlan,
    ) -> OrchestratorResult:
        """Fallback: generate using base model with no adapter."""
        from src.inference import generate_text

        if self.llm.has_expert_attached:
            self.llm.detach_expert()

        prompt = self._prompt_builder.build(query=query)
        model, tokenizer = self.llm.get_inference_components()
        response = generate_text(
            model, tokenizer, prompt, self.generation_config, self.use_gpu
        )

        return OrchestratorResult(
            response=response,
            adapter_loaded=None,
            routing_result=None,
            query_plan=plan,
        )

    def _build_routing_result(
        self,
        query: str,
        selected: list[AdapterMatch],
    ) -> RoutingResult:
        """Build a RoutingResult for eval compatibility.

        Marks the first selected adapter as the 'winner' to satisfy the
        RoutingResult contract, but all selected adapters are in all_matches.
        """
        query_embedding = self.router.compute_embedding(query)

        # Mark first adapter as winner for compatibility
        matches = []
        for i, m in enumerate(selected):
            matches.append(
                AdapterMatch(
                    adapter_id=m.adapter_id,
                    similarity=m.similarity,
                    timestamp=m.timestamp,
                    is_winner=(i == 0),
                )
            )

        winner = matches[0] if matches else None
        winner_entry = (
            self.router._manifest.get(winner.adapter_id) if winner else None
        )

        return RoutingResult(
            winner_adapter=winner.adapter_id if winner else None,
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

    def summary(self) -> str:
        """Get a formatted summary of the orchestrator."""
        lines = [
            "=" * 60,
            "PARALLEL ORCHESTRATOR",
            "=" * 60,
            f"Query planner: {self.query_planner_mode}",
            f"Max adapters: {self.max_adapters}",
            f"Synthesis tokens: {self.synthesis_max_new_tokens}",
            f"Generation config: {self.generation_config.to_dict()}",
            "-" * 60,
            "ROUTER:",
            self.router.summary() if hasattr(self.router, "summary") else "N/A",
            "=" * 60,
        ]
        return "\n".join(lines)
