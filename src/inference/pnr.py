"""
Unified Inference Pipeline
===========================

End-to-end inference combining the Time-Aware Centroid Router with LLM generation.

This module provides:
- PatchAndRouteInference: Main inference class tying router to LLM
- PromptBuilder: Constructs prompts with retrieved context
- GenerationConfig: Generation hyperparameters
- generate_text: Stateless generation utility (used by every system in the
  framework — PnR, Parallel Orchestrator, MORPHEUS via shared helper)
- score_target_logprob: ROME / MEMIT-style log-probability scorer

Flow:
1. User query → Router → (winner_adapter, retrieved_context)
2. Load winner adapter via PEFT
3. Build prompt with system message + context + query
4. Generate response with loaded adapter
5. Return response with routing metadata

Reference: Section 4.4 of the Master's Thesis Exposé - "The Intelligent Dispatcher"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import torch
from peft import PeftModel
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers import StoppingCriteria, StoppingCriteriaList

from src.eval.metrics import DEFAULT_SHORT_ANSWER_BOUNDARIES
from src.models.core import (
    PatchAndRouteLLM,
    FrozenFoundationConfig,
    QuantizationType,
)
from src.routing import CentroidRouter, RoutingResult

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class GenerationConfig:
    """Configuration for text generation.

    Attributes:
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature (higher = more random).
        top_p: Nucleus sampling probability.
        top_k: Top-K sampling (0 = disabled).
        do_sample: Whether to use sampling (False = greedy).
        repetition_penalty: Penalty for repeating tokens.
        num_beams: Beam search width (1 = no beam search).
        stop_sequences: Strings that, when emitted, terminate generation.
            Defaults to ``DEFAULT_SHORT_ANSWER_BOUNDARIES`` (newline +
            sentence-ending punctuation) so factoid eval halts as early
            as the answer is emitted — this matches industry practice
            (lm-evaluation-harness ``until``, ROME / MEMIT). Pass an
            empty tuple to disable.
    """
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    repetition_penalty: float = 1.1
    num_beams: int = 1
    stop_sequences: tuple[str, ...] = DEFAULT_SHORT_ANSWER_BOUNDARIES

    def to_dict(self) -> dict[str, Any]:
        """Convert to HuggingFace generate() kwargs.

        ``stop_sequences`` is *not* a HuggingFace-native arg; it is wired
        into ``generate_text`` separately via a ``StoppingCriteria``.
        """
        return {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature if self.do_sample else None,
            "top_p": self.top_p if self.do_sample else None,
            "top_k": self.top_k if self.do_sample else None,
            "do_sample": self.do_sample,
            "repetition_penalty": self.repetition_penalty,
            "num_beams": self.num_beams,
        }


# =============================================================================
# Stop Sequences (cross-system, dataset-agnostic)
# =============================================================================

class _StopOnSubstrings(StoppingCriteria):
    """HuggingFace ``StoppingCriteria`` that halts on textual substrings.

    Decoding-based check, not token-id-based: stop strings like ``"\\n"`` or
    ``"."`` can land at different token-id positions depending on
    BPE / SentencePiece context (e.g. ``" Berlin."`` is one token in many
    Mistral / Llama tokenizers, while a standalone ``"."`` is another). A
    pure token-id check therefore drops the most common factoid-EM path.
    Decoding the *generated* slice every step is O(generated_len) per call,
    which is negligible at the answer lengths we care about (<= 30 tokens).
    """

    __slots__ = ("_tokenizer", "_stop_strings", "_prompt_lens")

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        stop_strings: Sequence[str],
        prompt_lens: Sequence[int],
    ) -> None:
        self._tokenizer = tokenizer
        # Drop empty strings — they would short-circuit on every step.
        self._stop_strings = tuple(s for s in stop_strings if s)
        self._prompt_lens = tuple(int(p) for p in prompt_lens)

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        **kwargs: Any,
    ) -> bool:
        if not self._stop_strings:
            return False
        # Greedy / beam=1 path emits one row; treat any-row trigger as a stop
        # so the runner can rely on substring termination semantics.
        for row_idx in range(input_ids.shape[0]):
            prompt_len = (
                self._prompt_lens[row_idx]
                if row_idx < len(self._prompt_lens)
                else self._prompt_lens[-1]
            )
            new_ids = input_ids[row_idx, prompt_len:]
            if new_ids.numel() == 0:
                continue
            decoded = self._tokenizer.decode(
                new_ids, skip_special_tokens=True
            )
            if any(stop in decoded for stop in self._stop_strings):
                return True
        return False


@dataclass
class InferenceResult:
    """Result of an inference call.
    
    Attributes:
        response: Generated text response.
        routing_result: Full routing decision details.
        full_prompt: The complete prompt sent to the LLM.
        generation_config: Generation parameters used.
        adapter_loaded: Which adapter was loaded (if any).
    """
    response: str
    routing_result: RoutingResult
    full_prompt: str
    generation_config: GenerationConfig
    adapter_loaded: str | None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging/API responses."""
        return {
            "response": self.response,
            "adapter_loaded": self.adapter_loaded,
            "routing": self.routing_result.to_dict() if self.routing_result else None,
            "has_context": bool(self.routing_result and self.routing_result.retrieved_context),
        }


# =============================================================================
# Prompt Builder
# =============================================================================

class PromptBuilder:
    """Builds prompts for the Patch-and-Route pipeline.
    
    Handles:
    - System prompt injection
    - Retrieved context formatting (Source-Replay)
    - Chat template application
    """
    
    DEFAULT_SYSTEM_PROMPT = """You are a helpful AI assistant with access to specialized knowledge. 
When context is provided, use it to inform your answer, but also apply your general knowledge.
Be concise and accurate."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        system_prompt: str | None = None,
    ) -> None:
        """Initialize the prompt builder.
        
        Args:
            tokenizer: Tokenizer for applying chat template.
            system_prompt: Custom system prompt (uses default if None).
        """
        self.tokenizer = tokenizer
        # Default to no system message: the D_control pre-filter
        # (scripts/build_triviaqa_dcontrol.py) and all LoRA adapter training
        # used bare user/assistant pairs, so injecting a system prompt at eval
        # time changes the format and tanks frozen-base accuracy.
        self.system_prompt = system_prompt
    
    def build(
        self,
        query: str,
        retrieved_context: str = "",
        include_system: bool = True,
    ) -> str:
        """Build the full prompt.
        
        Args:
            query: User's query.
            retrieved_context: Context from Source-Replay (T_old adapters).
            include_system: Whether to include system prompt.
            
        Returns:
            Formatted prompt string.
        """
        messages = []
        
        # System message (if supported by chat template)
        if include_system and self.system_prompt:
            messages.append({
                "role": "system",
                "content": self.system_prompt,
            })
        
        # User message with optional context
        user_content = query
        if retrieved_context:
            user_content = f"{retrieved_context}\n\n---\n\n{query}"
        
        messages.append({
            "role": "user",
            "content": user_content,
        })
        
        # Apply chat template
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as e:
            # Fallback for models without chat template
            logger.warning(f"Chat template failed, using fallback: {e}")
            prompt = self._fallback_format(messages)
        
        return prompt
    
    def _fallback_format(self, messages: list[dict]) -> str:
        """Fallback formatting when chat template is unavailable."""
        parts = []
        for msg in messages:
            role = msg["role"].capitalize()
            content = msg["content"]
            parts.append(f"{role}: {content}")
        parts.append("Assistant:")
        return "\n\n".join(parts)


# =============================================================================
# Generation Utility
# =============================================================================

def generate_text(
    model: PreTrainedModel | PeftModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    config: GenerationConfig,
    use_gpu: bool = True,
) -> str:
    """Generate text from a prompt using a model and tokenizer.

    Stateless utility that can be used by any component needing text generation
    (PatchAndRouteInference, ParallelOrchestrator, MORPHEUS, etc.).

    Stop-sequence behaviour is governed by ``config.stop_sequences``: when
    non-empty, generation halts as soon as any of the listed substrings
    appears in the decoded continuation. This applies uniformly across all
    callers (and therefore all dataset splits) so factoid EM is consistent
    between PnR, the Parallel Orchestrator, MORPHEUS, etc.

    Args:
        model: The model to generate with (base or PEFT-wrapped).
        tokenizer: Tokenizer for encoding/decoding.
        prompt: Full formatted prompt.
        config: Generation parameters.
        use_gpu: Whether to move inputs to GPU.

    Returns:
        Generated text (response only, prompt stripped). When a stop
        sequence triggers, the trailing stop substring is *not* trimmed —
        ``parse_model_output`` re-applies the same boundary logic so the
        scoring path stays single-source-of-truth.
    """
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=4096,
    )

    if use_gpu:
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    stopping_criteria: StoppingCriteriaList | None = None
    if config.stop_sequences:
        prompt_lens = [int(inputs["input_ids"].shape[1])]
        stopping_criteria = StoppingCriteriaList([
            _StopOnSubstrings(
                tokenizer=tokenizer,
                stop_strings=config.stop_sequences,
                prompt_lens=prompt_lens,
            )
        ])

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            stopping_criteria=stopping_criteria,
            **config.to_dict(),
        )

    prompt_length = inputs["input_ids"].shape[1]
    response_tokens = outputs[0][prompt_length:]
    response = tokenizer.decode(response_tokens, skip_special_tokens=True)

    return response.strip()


# =============================================================================
# Log-probability scoring (ROME / MEMIT-style ESR)
# =============================================================================

@torch.no_grad()
def score_target_logprob(
    model: PreTrainedModel | PeftModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    target: str,
    use_gpu: bool = True,
    length_normalised: bool = False,
) -> float:
    """Compute the conditional log-probability of ``target`` given ``prompt``.

    Industry-standard knowledge-editing metric (ROME, MEMIT, GRACE, MEND):
    instead of decoding text and string-matching, we score whether the
    model assigns higher probability to the edit target than to the
    original fact. This sidesteps every parsing artefact and gives a
    smooth, differentiable signal — a sample with EM=0 but
    ``logp(target_new) > logp(target_true)`` is a *successful* edit by
    the field's standard definition.

    Implementation is teacher-forced: we tokenise ``prompt + " " + target``
    in one go, run a single forward pass, and sum the per-token
    log-probabilities for the target slice. The leading space is added
    only when the prompt does not already end in whitespace (matches
    SentencePiece / Mistral conventions where ``"_Berlin"`` is the
    natural continuation).

    Args:
        model: The model to score with (base or PEFT-wrapped). Must be in
            eval mode; this helper does *not* call ``model.eval()`` since
            that mutation is the caller's responsibility.
        tokenizer: Tokenizer paired with ``model``.
        prompt: Full formatted prompt (post chat-template).
        target: Target text to score (usually one or a few tokens).
        use_gpu: Whether to move tensors to ``model.device``.
        length_normalised: If True, return mean log-prob per target
            token instead of the sum. Useful when comparing targets of
            different lengths; default ``False`` matches ROME / MEMIT.

    Returns:
        Total (or mean) log-probability of ``target`` as a Python float.
        Returns ``-inf`` when ``target`` tokenises to an empty span — the
        caller can treat that as "untestable" without special-casing.
    """
    if not target:
        return float("-inf")

    needs_space = bool(prompt) and not prompt[-1].isspace()
    sep = " " if needs_space else ""
    prompt_ids = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True,
        truncation=True,
        max_length=4096,
    )["input_ids"]
    full_ids = tokenizer(
        prompt + sep + target,
        return_tensors="pt",
        add_special_tokens=True,
        truncation=True,
        max_length=4096,
    )["input_ids"]

    target_len = int(full_ids.shape[1] - prompt_ids.shape[1])
    if target_len <= 0:
        return float("-inf")

    if use_gpu:
        full_ids = full_ids.to(model.device)

    # Pass via keyword: the official RECIPE editor wraps `model.forward` with
    # `forward_recipe(**kargs)` (external/RECIPE/editors/recipe/recipe.py:119)
    # which rejects positional args. `input_ids=` is the canonical kwarg name
    # for every HuggingFace causal LM, so this stays generic.
    logits = model(input_ids=full_ids).logits  # (1, seq, vocab)
    # Shift: position t predicts token t+1; gather log-probs for the
    # target slice only (the last `target_len` tokens of `full_ids`).
    log_probs = torch.log_softmax(logits[0, :-1, :], dim=-1)
    target_token_ids = full_ids[0, 1:]
    target_slice = target_token_ids[-target_len:]
    target_logprobs = log_probs[-target_len:].gather(
        1, target_slice.unsqueeze(-1)
    ).squeeze(-1)

    total = float(target_logprobs.sum().item())
    if length_normalised:
        return total / target_len
    return total


# =============================================================================
# Main Inference Class
# =============================================================================

class PatchAndRouteInference:
    """Unified inference pipeline for Patch-and-Route.
    
    Combines:
    - CentroidRouter: Query routing and conflict resolution
    - PatchAndRouteLLM: Model loading and adapter management
    - PromptBuilder: Prompt construction with context
    
    Example:
        ```python
        # Initialize
        inference = PatchAndRouteInference(
            model_id="mistralai/Mistral-7B-Instruct-v0.3",
            router_path="router_state/",
            embedding_model_path="/path/to/KaLM-Embedding-Gemma3-12B",
        )
        
        # Run inference
        result = inference.generate("Who is the Chancellor of Germany in 2023?")
        
        print(result.response)
        print(f"Adapter used: {result.adapter_loaded}")
        print(f"Had conflict: {result.routing_result.has_conflict}")
        ```
    """
    
    def __init__(
        self,
        model_id: str = "mistralai/Mistral-7B-Instruct-v0.3",
        router: CentroidRouter | None = None,
        router_path: str | Path | None = None,
        embedding_model_path: str | None = None,
        quantization: QuantizationType = QuantizationType.INT4,
        system_prompt: str | None = None,
        generation_config: GenerationConfig | None = None,
        similarity_threshold: float = 0.65,
        use_gpu: bool = True,
    ) -> None:
        """Initialize the inference pipeline.
        
        Args:
            model_id: HuggingFace model ID for the base LLM.
            router: Pre-initialized CentroidRouter (optional).
            router_path: Path to load router state from (optional).
            embedding_model_path: Path to embedding model for router.
            quantization: Quantization type for LLM.
            system_prompt: Custom system prompt.
            generation_config: Generation hyperparameters.
            similarity_threshold: Router similarity threshold.
            use_gpu: Whether to use GPU.
        """
        self.model_id = model_id
        self.use_gpu = use_gpu
        self.generation_config = generation_config or GenerationConfig()
        
        # Initialize router
        if router:
            self.router = router
        elif router_path:
            self.router = CentroidRouter.load(
                path=router_path,
                embedding_model_path=embedding_model_path,
                similarity_threshold=similarity_threshold,
                use_gpu=use_gpu,
            )
        else:
            self.router = CentroidRouter(
                embedding_model_path=embedding_model_path,
                similarity_threshold=similarity_threshold,
                use_gpu=use_gpu,
            )
        
        # Initialize LLM (lazy loading)
        self._llm: PatchAndRouteLLM | None = None
        self._llm_config = FrozenFoundationConfig(
            model_id=model_id,
            quantization=quantization,
            use_cache=True,  # Enable KV cache for inference
        )
        
        # Initialize prompt builder (after LLM loads)
        self._prompt_builder: PromptBuilder | None = None
        self._system_prompt = system_prompt
        
        # Track current adapter
        self._current_adapter: str | None = None
        
        logger.info("=" * 60)
        logger.info("PATCH-AND-ROUTE INFERENCE INITIALIZED")
        logger.info("=" * 60)
        logger.info(f"  Model: {model_id}")
        logger.info(f"  Quantization: {quantization.value}")
        logger.info(f"  Router threshold: {similarity_threshold}")
        logger.info("=" * 60)
    
    # -------------------------------------------------------------------------
    # Lazy Loading
    # -------------------------------------------------------------------------
    
    def _ensure_llm_loaded(self) -> None:
        """Ensure the base LLM is loaded."""
        if self._llm is None:
            logger.info("Loading Frozen Foundation (base LLM)...")
            self._llm = PatchAndRouteLLM(foundation_config=self._llm_config)
            self._llm.load_frozen_foundation()
            
            # Initialize prompt builder
            self._prompt_builder = PromptBuilder(
                tokenizer=self._llm.tokenizer,
                system_prompt=self._system_prompt,
            )
            
            logger.info("✓ LLM loaded and ready")
    
    def _load_adapter(self, adapter_path: str) -> None:
        """Load a specific adapter.
        
        Args:
            adapter_path: Path to adapter checkpoint.
        """
        self._ensure_llm_loaded()
        
        if self._current_adapter == adapter_path:
            logger.debug(f"Adapter already loaded: {adapter_path}")
            return
        
        # Detach current adapter if any
        if self._llm.has_expert_attached:
            logger.debug("Detaching current adapter...")
            self._llm.detach_expert()
        
        # Load new adapter
        logger.info(f"Loading adapter: {adapter_path}")
        self._llm.load_expert(adapter_path)
        self._current_adapter = adapter_path
    
    # -------------------------------------------------------------------------
    # Generation
    # -------------------------------------------------------------------------
    
    def generate(
        self,
        query: str,
        generation_config: GenerationConfig | None = None,
        skip_routing: bool = False,
        force_adapter: str | None = None,
    ) -> InferenceResult:
        """Generate a response for a query.
        
        This is the main entry point for inference.
        
        Args:
            query: User's input query.
            generation_config: Override generation parameters.
            skip_routing: Skip routing, use base model only.
            force_adapter: Force a specific adapter (skip routing).
            
        Returns:
            InferenceResult with response and metadata.
        """
        self._ensure_llm_loaded()
        
        gen_config = generation_config or self.generation_config
        routing_result = None
        retrieved_context = ""
        adapter_loaded = None
        
        # Step 1: Route query (unless skipped)
        if not skip_routing and not force_adapter:
            routing_result = self.router.route(query)
            
            # Load winner adapter if found
            if routing_result.winner_path:
                self._load_adapter(routing_result.winner_path)
                adapter_loaded = routing_result.winner_adapter
                retrieved_context = routing_result.retrieved_context
                
                logger.info(f"Routed to: {adapter_loaded}")
                if routing_result.has_conflict:
                    logger.info(f"Conflict resolved, losers: {routing_result.loser_adapters}")
        
        elif force_adapter:
            # Force specific adapter
            self._load_adapter(force_adapter)
            adapter_loaded = Path(force_adapter).name
        
        # Step 2: Build prompt
        full_prompt = self._prompt_builder.build(
            query=query,
            retrieved_context=retrieved_context,
        )
        
        # Step 3: Generate
        response = self._generate_text(full_prompt, gen_config)
        
        return InferenceResult(
            response=response,
            routing_result=routing_result,
            full_prompt=full_prompt,
            generation_config=gen_config,
            adapter_loaded=adapter_loaded,
        )
    
    def _generate_text(
        self,
        prompt: str,
        config: GenerationConfig,
    ) -> str:
        """Generate text from a prompt.

        Args:
            prompt: Full formatted prompt.
            config: Generation parameters.

        Returns:
            Generated text (response only, prompt stripped).
        """
        model, tokenizer = self._llm.get_inference_components()
        return generate_text(model, tokenizer, prompt, config, use_gpu=self.use_gpu)

    # -------------------------------------------------------------------------
    # Log-probability scoring (ROME / MEMIT-style ESR)
    # -------------------------------------------------------------------------

    def score_targets(
        self,
        query: str,
        targets: list[str],
        skip_routing: bool = False,
        force_adapter: str | None = None,
    ) -> dict[str, float]:
        """Compute log P(target | prompt) for each target on the routed model.

        Sets up the *exact* same model state used for ``generate`` (router
        chooses the adapter, source-replay context is added to the prompt,
        etc.) and then scores each target via teacher-forced log-probability.
        Identical control flow to ``generate`` so the two metrics are
        comparable for a single sample without re-routing artefacts.

        Args:
            query: User's input query.
            targets: List of target strings (e.g. ``[target_new, target_true]``
                for CounterFact, or ``gold_aliases`` for TriviaQA).
            skip_routing: Same semantics as ``generate``.
            force_adapter: Same semantics as ``generate``.

        Returns:
            ``{target: logprob}`` mapping (sum-of-log-probs, *not*
            length-normalised — matches ROME / MEMIT convention).
        """
        self._ensure_llm_loaded()

        retrieved_context = ""

        if not skip_routing and not force_adapter:
            routing_result = self.router.route(query)
            if routing_result.winner_path:
                self._load_adapter(routing_result.winner_path)
                retrieved_context = routing_result.retrieved_context
        elif force_adapter:
            self._load_adapter(force_adapter)
        elif skip_routing and self._llm and self._llm.has_expert_attached:
            # CFR Pass 1 / frozen-base scoring must NOT see the previous
            # sample's adapter — detach if one is left over from generate().
            self._llm.detach_expert()
            self._current_adapter = None

        full_prompt = self._prompt_builder.build(
            query=query,
            retrieved_context=retrieved_context,
        )

        model, tokenizer = self._llm.get_inference_components()
        scores: dict[str, float] = {}
        for target in targets:
            scores[target] = score_target_logprob(
                model=model,
                tokenizer=tokenizer,
                prompt=full_prompt,
                target=target,
                use_gpu=self.use_gpu,
            )
        return scores
    
    # -------------------------------------------------------------------------
    # Batch Generation
    # -------------------------------------------------------------------------
    
    def generate_batch(
        self,
        queries: list[str],
        generation_config: GenerationConfig | None = None,
    ) -> list[InferenceResult]:
        """Generate responses for multiple queries.
        
        Note: Each query is routed independently, which may cause
        adapter switching overhead.
        
        Args:
            queries: List of user queries.
            generation_config: Override generation parameters.
            
        Returns:
            List of InferenceResult objects.
        """
        results = []
        
        for query in queries:
            result = self.generate(query, generation_config)
            results.append(result)
        
        return results
    
    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------
    
    def get_router(self) -> CentroidRouter:
        """Get the underlying router."""
        return self.router
    
    def get_llm(self) -> PatchAndRouteLLM:
        """Get the underlying LLM manager."""
        self._ensure_llm_loaded()
        return self._llm
    
    def summary(self) -> str:
        """Get a formatted summary of the inference pipeline."""
        lines = [
            "=" * 60,
            "PATCH-AND-ROUTE INFERENCE PIPELINE",
            "=" * 60,
            f"Model: {self.model_id}",
            f"Current adapter: {self._current_adapter or 'None'}",
            f"LLM loaded: {self._llm is not None}",
            "-" * 60,
            "ROUTER:",
            self.router.summary() if hasattr(self.router, 'summary') else "N/A",
            "-" * 60,
            "GENERATION CONFIG:",
            f"  max_new_tokens: {self.generation_config.max_new_tokens}",
            f"  temperature: {self.generation_config.temperature}",
            f"  top_p: {self.generation_config.top_p}",
            "=" * 60,
        ]
        
        return "\n".join(lines)


# =============================================================================
# Convenience Functions
# =============================================================================

def create_inference_pipeline(
    model_id: str = "mistralai/Mistral-7B-Instruct-v0.3",
    checkpoints_dir: str = "checkpoints",
    embedding_model_path: str | None = None,
    router_state_path: str | None = None,
    similarity_threshold: float = 0.65,
) -> PatchAndRouteInference:
    """Create a ready-to-use inference pipeline.
    
    Convenience function that:
    1. Creates router and auto-discovers adapters
    2. Initializes LLM with optimal settings
    3. Returns ready inference pipeline
    
    Args:
        model_id: Base LLM model ID.
        checkpoints_dir: Directory with adapter checkpoints.
        embedding_model_path: Path to embedding model.
        router_state_path: Path to saved router state (optional).
        similarity_threshold: Router similarity threshold.
        
    Returns:
        Configured PatchAndRouteInference instance.
    """
    # Create router
    if router_state_path and Path(router_state_path).exists():
        router = CentroidRouter.load(
            path=router_state_path,
            embedding_model_path=embedding_model_path,
            similarity_threshold=similarity_threshold,
        )
    else:
        router = CentroidRouter(
            embedding_model_path=embedding_model_path,
            similarity_threshold=similarity_threshold,
        )
        router.register_from_checkpoints(checkpoints_dir)
    
    # Create inference pipeline
    pipeline = PatchAndRouteInference(
        model_id=model_id,
        router=router,
    )
    
    return pipeline

