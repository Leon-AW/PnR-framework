"""
MORPHEUS Inference Pipeline
=============================

Unified inference pipeline that integrates all six MORPHEUS subsystems
into a single generate() call compatible with the PnR evaluation runner.

Inference flow:
1. Query -> Meta-controller observes system state
2. Query -> Prototype router selects experts
3. Query -> Knowledge store checks for factual override
4. Selected experts loaded onto stable core
5. Prompt built with knowledge store context (if factual override)
6. Generation with active expert adapter
7. Meta-controller records routing statistics
8. Buffer absorbs query for future learning

The inference pipeline lazily initializes the stable core and router
to minimize startup cost. All subsystem state is preserved across calls.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from peft import PeftModel
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from src.routing.base import RoutingResult, RoutingStrategy

from .config import MorpheusConfig, ExpertState
from .stable_core import StableCore
from .expert_bank import ExpertBank
from .fast_buffer import FastBuffer
from .knowledge_store import KnowledgeStore, FactualityDecision
from .consolidation import ConsolidationEngine
from .meta_controller import MetaController, SystemState
from .router import PrototypeRouter

logger = logging.getLogger(__name__)


@dataclass
class MorpheusGenerationConfig:
    """Generation hyperparameters for MORPHEUS inference."""
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    repetition_penalty: float = 1.1

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature if self.do_sample else None,
            "top_p": self.top_p if self.do_sample else None,
            "top_k": self.top_k if self.do_sample else None,
            "do_sample": self.do_sample,
            "repetition_penalty": self.repetition_penalty,
        }


@dataclass
class MorpheusInferenceResult:
    """Result of a MORPHEUS inference call.

    Compatible with the PnR evaluation runner's expected interface:
    must have .response, .adapter_loaded, and .routing_result attributes.
    """
    response: str
    routing_result: RoutingResult | None
    adapter_loaded: str | None
    full_prompt: str = ""
    generation_config: MorpheusGenerationConfig | None = None

    # MORPHEUS-specific metadata
    factuality_decision: FactualityDecision | None = None
    knowledge_override: bool = False
    novelty_level: float = 0.0
    routing_confidence: float = 0.0
    active_experts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "response": self.response,
            "adapter_loaded": self.adapter_loaded,
            "routing": self.routing_result.to_dict() if self.routing_result else None,
            "knowledge_override": self.knowledge_override,
            "novelty_level": self.novelty_level,
            "routing_confidence": self.routing_confidence,
        }


class MorpheusPromptBuilder:
    """Builds prompts with integrated knowledge store context."""

    DEFAULT_SYSTEM_PROMPT = (
        "You are a helpful AI assistant with access to verified factual knowledge "
        "and specialized domain expertise. When verified facts are provided, use them "
        "as your primary source. Be concise and accurate."
    )

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        system_prompt: str | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT

    def build(
        self,
        query: str,
        knowledge_context: str = "",
        retrieved_context: str = "",
        uncertainty_signal: str = "",
    ) -> str:
        """Build prompt with graduated factuality integration.

        Args:
            query: User's query.
            knowledge_context: Verified facts from System 5 (hard override).
            retrieved_context: Context from source-replay (older experts).
            uncertainty_signal: Explicit uncertainty from boundary zone.
        """
        messages = []

        system_content = self.system_prompt
        if uncertainty_signal:
            system_content += f"\n\nNote: {uncertainty_signal}"

        messages.append({"role": "system", "content": system_content})

        user_content = query
        context_parts = []
        if knowledge_context:
            context_parts.append(knowledge_context)
        if retrieved_context:
            context_parts.append(retrieved_context)

        if context_parts:
            user_content = "\n\n".join(context_parts) + f"\n\n---\n\n{query}"

        messages.append({"role": "user", "content": user_content})

        try:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            parts = [f"{m['role'].capitalize()}: {m['content']}" for m in messages]
            parts.append("Assistant:")
            prompt = "\n\n".join(parts)

        return prompt


class MorpheusInference:
    """Unified MORPHEUS inference pipeline.

    Integrates all six subsystems into a single interface that is
    compatible with the PnR evaluation runner (EvalRunner).

    Usage:
        config = MorpheusConfig()
        pipeline = MorpheusInference(config=config)
        result = pipeline.generate("Who is the Chancellor of Germany?")
        print(result.response)
        print(result.adapter_loaded)
    """

    def __init__(
        self,
        config: MorpheusConfig | None = None,
        stable_core: StableCore | None = None,
        expert_bank: ExpertBank | None = None,
        router: PrototypeRouter | None = None,
        fast_buffer: FastBuffer | None = None,
        knowledge_store: KnowledgeStore | None = None,
        consolidation: ConsolidationEngine | None = None,
        meta_controller: MetaController | None = None,
        generation_config: MorpheusGenerationConfig | None = None,
        embedding_fn: Callable[[str], np.ndarray] | None = None,
    ) -> None:
        self.config = config or MorpheusConfig()
        self.generation_config = generation_config or MorpheusGenerationConfig()

        # Subsystems (lazily initialized if not provided)
        self._core = stable_core
        self._expert_bank = expert_bank or ExpertBank(self.config.expert_bank)
        self._router = router
        self._buffer = fast_buffer or FastBuffer(self.config.fast_buffer)
        self._knowledge_store = knowledge_store or KnowledgeStore(self.config.knowledge_store)
        self._consolidation = consolidation
        self._meta = meta_controller or MetaController(self.config.meta_controller)

        self._prompt_builder: MorpheusPromptBuilder | None = None
        self._current_adapter: str | None = None
        self._embedding_fn = embedding_fn

        # If no router provided, create one
        if self._router is None:
            self._router = PrototypeRouter(
                config=self.config.router,
                embedding_fn=embedding_fn,
            )

        logger.info("=" * 60)
        logger.info("MORPHEUS INFERENCE PIPELINE INITIALIZED")
        logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def _ensure_core_loaded(self) -> None:
        """Ensure the stable core model is loaded."""
        if self._core is None:
            self._core = StableCore(self.config.stable_core)
        if self._core._model is None:
            self._core.load()
        if self._prompt_builder is None:
            self._prompt_builder = MorpheusPromptBuilder(
                tokenizer=self._core.tokenizer,
            )

    def _load_expert_adapter(self, adapter_path: str) -> None:
        """Load a specific expert adapter onto the stable core."""
        self._ensure_core_loaded()

        if self._current_adapter == adapter_path:
            return

        # Detach current adapter
        if isinstance(self._core.model, PeftModel):
            self._core.detach_adapter()

        # Load new adapter
        if Path(adapter_path).exists():
            self._core.load_adapter(adapter_path)
            self._current_adapter = adapter_path
            logger.debug(f"Loaded expert adapter: {adapter_path}")

    # ------------------------------------------------------------------
    # Main generation
    # ------------------------------------------------------------------

    def generate(
        self,
        query: str,
        generation_config: MorpheusGenerationConfig | None = None,
        skip_routing: bool = False,
        force_adapter: str | None = None,
    ) -> MorpheusInferenceResult:
        """Generate a response using the full MORPHEUS pipeline.

        This is the main entry point, compatible with EvalRunner.

        Flow:
        1. Update meta-controller state
        2. Route query to experts (prototype matching)
        3. Check knowledge store (graduated factuality)
        4. Load winning expert adapter
        5. Build prompt with knowledge context
        6. Generate response
        7. Record statistics

        Args:
            query: User's input query.
            generation_config: Override generation parameters.
            skip_routing: Skip routing, use base model only.
            force_adapter: Force a specific adapter path.

        Returns:
            MorpheusInferenceResult with response and metadata.
        """
        self._ensure_core_loaded()

        gen_config = generation_config or self.generation_config
        routing_result = None
        adapter_loaded = None
        knowledge_context = ""
        uncertainty_signal = ""
        novelty_level = 0.0
        routing_confidence = 1.0
        factuality_decision = None
        knowledge_override = False

        # Step 1: Meta-controller observes state
        system_state = self._build_system_state()
        self._meta.observe(system_state)
        novelty_level = self._meta.get_novelty_level()

        # Step 2: Route query
        if not skip_routing and not force_adapter:
            routing_result = self._router.route(query)

            if routing_result.winner_path:
                self._load_expert_adapter(routing_result.winner_path)
                adapter_loaded = routing_result.winner_adapter

            routing_confidence = self._router.compute_routing_confidence(query)

        elif force_adapter:
            self._load_expert_adapter(force_adapter)
            adapter_loaded = Path(force_adapter).name

        # Step 3: Knowledge store check (graduated factuality)
        if self._embedding_fn and not skip_routing:
            query_emb = self._embedding_fn(query)

            factuality_decision = self._knowledge_store.assess_factuality(
                query_embedding=query_emb,
                factuality_score=0.5,  # Default; real classifier would provide this
                novelty_level=novelty_level,
            )

            if factuality_decision.zone == "hard_override":
                knowledge_context = self._knowledge_store.build_override_context(
                    factuality_decision.system5_records,
                )
                knowledge_override = True
            elif factuality_decision.zone == "boundary":
                knowledge_context = self._knowledge_store.build_override_context(
                    factuality_decision.system5_records,
                )
                uncertainty_signal = factuality_decision.uncertainty_signal

        # Step 4: Build prompt
        prompt = self._prompt_builder.build(
            query=query,
            knowledge_context=knowledge_context,
            retrieved_context=routing_result.retrieved_context if routing_result else "",
            uncertainty_signal=uncertainty_signal,
        )

        # Step 5: Generate
        response = self._generate_text(prompt, gen_config)

        # Step 6: Record buffer sample for future learning
        self._buffer.add_sample(
            text=query,
            domain_signal=adapter_loaded or "base",
        )

        return MorpheusInferenceResult(
            response=response,
            routing_result=routing_result,
            adapter_loaded=adapter_loaded,
            full_prompt=prompt,
            generation_config=gen_config,
            factuality_decision=factuality_decision,
            knowledge_override=knowledge_override,
            novelty_level=novelty_level,
            routing_confidence=routing_confidence,
        )

    def _generate_text(
        self,
        prompt: str,
        config: MorpheusGenerationConfig,
    ) -> str:
        """Generate text from the current model state."""
        model = self._core.model
        tokenizer = self._core.tokenizer

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                **config.to_dict(),
            )

        prompt_length = inputs["input_ids"].shape[1]
        response_tokens = outputs[0][prompt_length:]
        response = tokenizer.decode(response_tokens, skip_special_tokens=True)
        return response.strip()

    # ------------------------------------------------------------------
    # System state construction
    # ------------------------------------------------------------------

    def _build_system_state(self) -> SystemState:
        """Build the current system state for the meta-controller."""
        buffer_stats = self._buffer.get_loss_statistics()
        shift = self._buffer.detect_distribution_shift()

        return SystemState(
            buffer_fill_level=self._buffer.fill_level,
            buffer_loss_mean=buffer_stats["mean"],
            buffer_loss_trend=buffer_stats["trend"],
            distribution_shift_magnitude=shift,
            num_active_experts=len(self._expert_bank.active_experts),
            num_shadow_experts=len(self._expert_bank.shadow_experts),
            capacity_utilization=(
                self._expert_bank.num_experts / self.config.expert_bank.max_experts
            ),
            routing_confidence_mean=0.5,
            core_version=self._core.version if self._core else 0,
        )

    # ------------------------------------------------------------------
    # Consolidation interface
    # ------------------------------------------------------------------

    def trigger_consolidation(
        self,
        probe_texts: list[str] | None = None,
    ) -> dict[str, Any]:
        """Manually trigger a consolidation cycle.

        Typically called by the meta-controller when conditions are met.
        """
        if self._consolidation is None:
            self._consolidation = ConsolidationEngine(
                config=self.config.consolidation,
                stable_core=self._core,
                expert_bank=self._expert_bank,
                fast_buffer=self._buffer,
                knowledge_store=self._knowledge_store,
            )

        result = self._consolidation.run_cycle(
            probe_texts=probe_texts,
        )
        return result.to_dict()

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_router(self) -> PrototypeRouter:
        return self._router

    def get_core(self) -> StableCore:
        self._ensure_core_loaded()
        return self._core

    def get_expert_bank(self) -> ExpertBank:
        return self._expert_bank

    def get_meta_controller(self) -> MetaController:
        return self._meta

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "MORPHEUS INFERENCE PIPELINE",
            "=" * 60,
            f"Model: {self.config.stable_core.model_id}",
            f"Current adapter: {self._current_adapter or 'None'}",
            f"Core loaded: {self._core is not None and self._core._model is not None}",
            "-" * 60,
            "SUBSYSTEM STATUS:",
            f"  Core: v{self._core.version if self._core else 'N/A'}",
            f"  {self._expert_bank.summary()}",
            f"  {self._buffer.summary()}",
            f"  {self._knowledge_store.summary()}",
            f"  {self._router.summary()}",
            f"  {self._meta.summary()}",
            "-" * 60,
            "GENERATION CONFIG:",
            f"  max_new_tokens: {self.generation_config.max_new_tokens}",
            f"  temperature: {self.generation_config.temperature}",
            "=" * 60,
        ]
        return "\n".join(lines)
