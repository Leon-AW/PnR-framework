"""
X-LoRA Inference Wrapper
=========================

Thin wrapper around an X-LoRA model that provides the same interface as
PatchAndRouteInference.generate(), returning an object with .response,
.adapter_loaded, and .routing_result attributes.

This keeps eval_runner changes minimal: the runner calls .generate(query)
and reads the same fields regardless of which backend is active.

Reference: arXiv:2402.07148 — "X-LoRA: Mixture of Low-Rank Adapter Experts"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# Lightweight result type (mirrors InferenceResult fields used by eval runner)
# =============================================================================

@dataclass
class XLoRAInferenceResult:
    """Result from XLoRAInference.generate().

    Attributes:
        response: Generated text (prompt stripped).
        adapter_loaded: Always "xlora_blend" (no discrete adapter selection).
        routing_result: Always None (X-LoRA has no discrete routing).
    """
    response: str
    adapter_loaded: str = "xlora_blend"
    routing_result: Any = None  # None → eval runner treats routing_correct=True


# =============================================================================
# Main wrapper
# =============================================================================

class XLoRAInference:
    """Inference wrapper for an X-LoRA gating checkpoint.

    Loads the base model + X-LoRA gating weights and exposes a .generate()
    method compatible with PatchAndRouteInference so the eval runner can use
    either backend without special-casing.

    Example::

        wrapper = XLoRAInference(
            xlora_checkpoint="checkpoints/xlora_baseline",
            model_id="mistralai/Mistral-7B-Instruct-v0.3",
        )
        result = wrapper.generate("Who is the Chancellor of Germany?")
        print(result.response)
    """

    def __init__(
        self,
        xlora_checkpoint: str | Path,
        model_id: str = "mistralai/Mistral-7B-Instruct-v0.3",
        quantization: str = "int4",
        max_new_tokens: int = 256,
        temperature: float = 0.1,
        do_sample: bool = False,
        use_gpu: bool = True,
    ) -> None:
        """Initialise the X-LoRA inference wrapper.

        Args:
            xlora_checkpoint: Directory containing the X-LoRA gating checkpoint
                (xlora_config.json + gating weights saved by train_xlora_baseline).
            model_id: HuggingFace model ID for the frozen base LLM.
            quantization: "int4", "int8", or "none".
            max_new_tokens: Maximum tokens to generate per call.
            temperature: Sampling temperature.
            do_sample: Whether to use sampling (False = greedy).
            use_gpu: Whether to move tensors to CUDA.
        """
        try:
            import xlora  # noqa: F401
        except ImportError:
            raise ImportError(
                "xlora is not installed.\n"
                "Install with: pip install git+https://github.com/EricLBuehler/xlora.git"
            )

        self.xlora_checkpoint = Path(xlora_checkpoint)
        self.model_id = model_id
        self.quantization = quantization
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample
        self.use_gpu = use_gpu

        self._model = None
        self._tokenizer = None

    # -------------------------------------------------------------------------
    # Lazy loading
    # -------------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Load model + tokenizer on first call."""
        if self._model is not None:
            return

        import torch
        import xlora
        from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM

        logger.info("Loading X-LoRA model from %s ...", self.xlora_checkpoint)

        # Quantization config
        bnb_config = None
        if self.quantization == "int4":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif self.quantization == "int8":
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)

        device_map = "auto" if (self.use_gpu and torch.cuda.is_available()) else "cpu"

        # Load base model (use_cache must be False for xlora)
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if bnb_config is None else None,
        )
        base_model.config.use_cache = False

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Wrap with X-LoRA gating (loads xlora_config.json + xlora_classifier.pt)
        device_str = "cuda" if (self.use_gpu and torch.cuda.is_available()) else "cpu"

        # The xlora library has a bug: from_pretrained sets conf["adapters"] = None
        # before creating xLoRAConfig, overwriting the adapters loaded from the JSON.
        # Workaround: load the adapters dict explicitly and pass it in.
        import json as _json
        _config_path = self.xlora_checkpoint / "xlora_config.json"
        with open(_config_path) as _f:
            _adapters = _json.load(_f).get("adapters")

        self._model = xlora.from_pretrained(
            load_directory=str(self.xlora_checkpoint),
            model=base_model,
            device=device_str,
            from_safetensors=False,  # we save as .pt, not safetensors
            verbose=False,
            adapters=_adapters,  # explicit pass bypasses the library bug
        )
        self._model.eval()

        logger.info("X-LoRA model loaded successfully.")

    # -------------------------------------------------------------------------
    # Generation
    # -------------------------------------------------------------------------

    def generate(self, query: str) -> XLoRAInferenceResult:
        """Generate a response using the X-LoRA blended adapters.

        Args:
            query: User query string.

        Returns:
            XLoRAInferenceResult with .response, .adapter_loaded, .routing_result.
        """
        import torch

        self._ensure_loaded()

        # Build a minimal chat-formatted prompt
        messages = [{"role": "user", "content": query}]
        try:
            prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            prompt = f"User: {query}\nAssistant:"

        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        )

        device = next(self._model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # SituatedQA gold answers are short factoids (years, names, places).
        # The default max_new_tokens=256 lets the model loop into noisy repetition
        # ("Lionel Messi's Cristiano Ronal Lionel Lionel ..."). The right
        # answer is usually in the first few tokens but gets buried, killing
        # token-level F1. Cap at 30, then truncate at the first sentence/line.
        max_new = min(self.max_new_tokens, 30)
        # no_repeat_ngram_size=3 breaks the soft-blend repetition loop
        # ("Lionel Lionel Lionel ..."): with 14 adapters at near-uniform
        # softmax weights the LM head's top-1 token is often correct on the
        # first step but the blended hidden state then re-attracts the same
        # token, killing both EM and F1. repetition_penalty=1.3 alone is not
        # enough to break this; forbidding any 3-gram repeat is.
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new,
            "do_sample": self.do_sample,
            "pad_token_id": self._tokenizer.pad_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
            "repetition_penalty": 1.3,
            "no_repeat_ngram_size": 3,
        }
        if self.do_sample:
            gen_kwargs["temperature"] = self.temperature

        with torch.no_grad():
            outputs = self._model.generate(**inputs, **gen_kwargs)

        prompt_len = inputs["input_ids"].shape[1]
        response = self._tokenizer.decode(
            outputs[0][prompt_len:], skip_special_tokens=True
        ).strip()

        for sep in ("\n", "."):
            if sep in response:
                response = response[: response.index(sep)].strip()
                break

        return XLoRAInferenceResult(response=response)

    # -------------------------------------------------------------------------
    # Log-probability scoring (ROME / MEMIT-style ESR)
    # -------------------------------------------------------------------------

    def score_targets(self, query: str, targets: list[str]) -> dict[str, float]:
        """Score targets via teacher-forced log-probability on the X-LoRA model.

        The X-LoRA classifier dynamically blends adapters per layer and per
        token, so the post-edit distribution is fully encoded in
        ``self._model``; teacher-forcing the target through the same
        wrapped model gives a metric directly comparable to ROME / MEMIT.
        """
        from src.inference import score_target_logprob

        self._ensure_loaded()

        messages = [{"role": "user", "content": query}]
        try:
            prompt = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            prompt = f"User: {query}\nAssistant:"

        return {
            t: score_target_logprob(
                model=self._model,
                tokenizer=self._tokenizer,
                prompt=prompt,
                target=t,
                use_gpu=self.use_gpu,
            )
            for t in targets
        }
