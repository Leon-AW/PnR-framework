"""
Unified Inference Pipeline
===========================

End-to-end inference combining the Time-Aware Centroid Router with LLM generation.

This module provides:
- PatchAndRouteInference: Main inference class tying router to LLM
- PromptBuilder: Constructs prompts with retrieved context
- GenerationConfig: Generation hyperparameters

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
from typing import Any

import torch
from peft import PeftModel
from transformers import PreTrainedModel, PreTrainedTokenizerBase

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
    """
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    repetition_penalty: float = 1.1
    num_beams: int = 1
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to HuggingFace generate() kwargs."""
        return {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature if self.do_sample else None,
            "top_p": self.top_p if self.do_sample else None,
            "top_k": self.top_k if self.do_sample else None,
            "do_sample": self.do_sample,
            "repetition_penalty": self.repetition_penalty,
            "num_beams": self.num_beams,
        }


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
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
    
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
    (PatchAndRouteInference, ParallelOrchestrator, etc.).

    Args:
        model: The model to generate with (base or PEFT-wrapped).
        tokenizer: Tokenizer for encoding/decoding.
        prompt: Full formatted prompt.
        config: Generation parameters.
        use_gpu: Whether to move inputs to GPU.

    Returns:
        Generated text (response only, prompt stripped).
    """
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=4096,
    )

    if use_gpu:
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

