"""
RLEdit Inference Wrapper
========================

Thin wrapper around the RLEdit hypernetwork-based lifelong editing method that
provides the same interface as PatchAndRouteInference.generate(), returning an
object with .response, .adapter_loaded, and .routing_result attributes.

RLEdit (Li et al., ICML 2025) treats lifelong editing as a Markov Decision
Process, training a hypernetwork H that maps fine-tuning gradient components
(δ, u) to weight updates (δ̃, ū) via an RL reward maximising target knowledge
update, locality preservation, and memory backtracking.

Architecture
------------
For each target linear layer L with weight W:
    1. Run 1-step parameter-frozen fine-tuning on (x_e, y_e).
    2. Capture activation u (input to L) and gradient δ (error at L) via hooks.
    3. Feed (δ, u) through hypernetwork H_L → (δ̃, ū).
    4. Compute parameter update: ΔW = outer(δ̃, ū).
    5. Register a forward hook on L so output_L(x) += ΔW_accumulated @ x.
       This is equivalent to a rank-1 LoRA addition on top of the frozen
       (possibly quantized) base weights — no in-place quantized-weight
       modification is needed.

Reference: arXiv:2502.05759 — "Reinforced Lifelong Editing for Language Models"
Repo:      github.com/zhrli324/RLEdit
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class RLEditConfig:
    """Hyperparameters for RLEdit.

    Attributes:
        k: Memory backtracking window — number of previous edits reviewed when
            computing the backtracking reward component (Eq. 9 in paper).
        gamma: Discount factor for total reward J = Σ γ^t r_t (Eq. 12).
        mu: Exponential decay factor for the backtracking loss weights (Eq. 9).
        eta: Regularisation coefficient for the magnitude penalty ||ΔW||² (Eq. 10).
        lambda_loc: Weight of locality KL divergence in the base reward (Eq. 8).
        target_modules: Substrings of module names to target for editing.
            Any module whose full name contains one of these substrings is
            included. Defaults to MLP projection layers common in Qwen/Mistral.
        hidden_dim: Hidden dimension of the 4-layer hypernetwork MLPs (per-layer).
        n_grad_steps: Number of frozen fine-tuning steps for gradient collection.
        grad_lr: Learning rate used during the 1-step gradient collection.
        max_seq_length: Maximum token length for edit inputs.
    """
    k: int = 5
    gamma: float = 1.0
    mu: float = 0.9
    eta: float = 0.1
    lambda_loc: float = 0.5
    target_modules: list[str] = field(
        default_factory=lambda: ["mlp.down_proj", "mlp.gate_proj"]
    )
    hidden_dim: int = 256
    n_grad_steps: int = 1
    grad_lr: float = 1e-3
    max_seq_length: int = 512

    def to_dict(self) -> dict:
        return {
            "k": self.k,
            "gamma": self.gamma,
            "mu": self.mu,
            "eta": self.eta,
            "lambda_loc": self.lambda_loc,
            "target_modules": self.target_modules,
            "hidden_dim": self.hidden_dim,
            "n_grad_steps": self.n_grad_steps,
            "grad_lr": self.grad_lr,
            "max_seq_length": self.max_seq_length,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RLEditConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# =============================================================================
# Hypernetwork
# =============================================================================

class RLEditHypernetwork(nn.Module):
    """Per-layer hypernetwork H mapping gradient components to update components.

    Implements the rank-1 decomposition:
        ∇W = δ u^T  →  ΔW = δ̃ ū^T
    where H: (δ, u) → (δ̃, ū) via two independent 4-layer MLPs.

    The gradient ∇W_{l} of a single-sample causal-LM loss w.r.t. a linear
    layer weight W (shape d_out × d_in) is rank-1 for batch_size=1:
        ∇W = δ_{l+1} u_l^T
    with δ_{l+1} ∈ ℝ^{d_out} the backprop error and u_l ∈ ℝ^{d_in} the input.

    Args:
        d_out: Output dimension of the target linear layer.
        d_in: Input dimension of the target linear layer.
        hidden_dim: Hidden size of the two 4-layer MLPs.
    """

    def __init__(self, d_out: int, d_in: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.d_out = d_out
        self.d_in = d_in

        # MLP for δ_{l+1}: d_out → d_out
        self.delta_net = nn.Sequential(
            nn.Linear(d_out, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, d_out),
        )
        # MLP for u_l: d_in → d_in
        self.u_net = nn.Sequential(
            nn.Linear(d_in, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, d_in),
        )

    def forward(
        self, delta: torch.Tensor, u: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Map gradient components to parameter update components.

        Args:
            delta: δ_{l+1} of shape (d_out,) — backprop error at layer l+1.
            u: u_l of shape (d_in,) — input activation to layer l.

        Returns:
            (δ̃, ū): update components so ΔW = outer(δ̃, ū).
        """
        return self.delta_net(delta), self.u_net(u)


# =============================================================================
# Result type (duck-typed to match InferenceResult / XLoRAInferenceResult)
# =============================================================================

@dataclass
class RLEditInferenceResult:
    """Result from RLEditInference.generate().

    Attributes:
        response: Generated text (prompt stripped).
        adapter_loaded: Always "rledit" — signals direct-weight editing mode.
        routing_result: Always None — RLEdit has no discrete routing.
        n_edits_applied: Number of edits baked into the model weights.
    """
    response: str
    adapter_loaded: str = "rledit"
    routing_result: Any = None  # None → eval runner treats routing_correct=True
    n_edits_applied: int = 0


# =============================================================================
# Main inference wrapper
# =============================================================================

class RLEditInference:
    """Inference wrapper for a trained RLEdit hypernetwork.

    Loads the base LLM and a per-layer trained hypernetwork, then applies edits
    by computing 1-step gradients and running them through the hypernetwork to
    produce additive weight deltas. The deltas are accumulated via forward hooks
    on the target linear layers (no in-place quantized-weight modification).

    Example::

        wrapper = RLEditInference(
            checkpoint_dir="checkpoints/rledit_baseline",
            model_id="mistralai/Mistral-7B-Instruct-v0.3",
        )
        # Apply edits from evaluation set
        wrapper.apply_edits([
            {"question": "Who is the CEO?", "answer": "Alice"},
        ])
        result = wrapper.generate("Who leads the company?")
        print(result.response)
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        model_id: str = "mistralai/Mistral-7B-Instruct-v0.3",
        quantization: str = "int4",
        max_new_tokens: int = 256,
        temperature: float = 0.1,
        do_sample: bool = False,
        use_gpu: bool = True,
    ) -> None:
        """Initialise the RLEdit inference wrapper.

        Args:
            checkpoint_dir: Directory produced by train_rledit_baseline.py,
                containing rledit_config.json and rledit_hypernetwork.pt.
            model_id: HuggingFace model ID for the frozen base LLM.
            quantization: "int4", "int8", or "none". Note: edits are applied
                via additive forward hooks, so all quantization modes work.
            max_new_tokens: Maximum tokens to generate per call.
            temperature: Sampling temperature.
            do_sample: Whether to use sampling (False = greedy).
            use_gpu: Whether to use CUDA if available.
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.model_id = model_id
        self.quantization = quantization
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample
        self.use_gpu = use_gpu

        self._model = None
        self._tokenizer = None
        self._config: RLEditConfig | None = None
        # module_name → RLEditHypernetwork
        self._hypernetworks: dict[str, RLEditHypernetwork] = {}
        # module_name → accumulated delta tensor (d_out, d_in)
        self._accumulated_deltas: dict[str, torch.Tensor] = {}
        # forward hook handles (for cleanup)
        self._hook_handles: list = []
        self._n_edits_applied: int = 0

    # -------------------------------------------------------------------------
    # Lazy loading
    # -------------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Load model, tokenizer, config, and hypernetwork on first call."""
        if self._model is not None:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        logger.info("Loading RLEdit model from %s ...", self.model_id)
        logger.info("Loading RLEdit hypernetwork from %s ...", self.checkpoint_dir)

        # --- Load config ---
        config_path = self.checkpoint_dir / "rledit_config.json"
        if not config_path.exists():
            raise FileNotFoundError(
                f"RLEdit config not found at {config_path}. "
                "Run train_rledit_baseline.py first."
            )
        with open(config_path) as f:
            self._config = RLEditConfig.from_dict(json.load(f))

        # --- Build quantization config ---
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

        device_map = (
            "auto" if (self.use_gpu and torch.cuda.is_available()) else "cpu"
        )

        # --- Load base model ---
        model_kwargs: dict[str, Any] = {
            "device_map": device_map,
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16 if bnb_config is None else None,
        }
        if bnb_config is not None:
            model_kwargs["quantization_config"] = bnb_config

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id, **model_kwargs
        )
        self._model.eval()

        # --- Load tokenizer ---
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # --- Load hypernetworks ---
        hn_path = self.checkpoint_dir / "rledit_hypernetwork.pt"
        if not hn_path.exists():
            raise FileNotFoundError(
                f"RLEdit hypernetwork weights not found at {hn_path}. "
                "Run train_rledit_baseline.py first."
            )
        hn_state = torch.load(hn_path, map_location="cpu")
        hidden_dim = self._config.hidden_dim

        device = torch.device(
            "cuda" if (self.use_gpu and torch.cuda.is_available()) else "cpu"
        )
        for module_name, state_dict in hn_state.items():
            # Infer d_out / d_in from saved weight shapes
            d_out = state_dict["delta_net.0.weight"].shape[1]
            d_in = state_dict["u_net.0.weight"].shape[1]
            hn = RLEditHypernetwork(d_out=d_out, d_in=d_in, hidden_dim=hidden_dim)
            hn.load_state_dict(state_dict)
            hn.to(device).eval()
            self._hypernetworks[module_name] = hn

        # Initialise accumulated delta buffers for each target module
        for name, module in self._iter_target_modules():
            if hasattr(module, "weight") and module.weight is not None:
                weight_shape = (
                    module.weight.shape
                    if not hasattr(module.weight, "quant_state")
                    else (module.out_features, module.in_features)
                )
                self._accumulated_deltas[name] = torch.zeros(
                    weight_shape, dtype=torch.float32, device=device
                )

        # Register forward hooks that add the accumulated delta to each layer
        self._register_edit_hooks()

        logger.info(
            "RLEdit loaded: %d hypernetworks covering %d target modules.",
            len(self._hypernetworks),
            len(self._accumulated_deltas),
        )

    # -------------------------------------------------------------------------
    # Module helpers
    # -------------------------------------------------------------------------

    def _iter_target_modules(self):
        """Yield (name, module) for all target linear layers."""
        assert self._model is not None
        assert self._config is not None
        for name, module in self._model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if any(tgt in name for tgt in self._config.target_modules):
                yield name, module

    def _get_module(self, name: str) -> nn.Module:
        """Retrieve a named sub-module from the model."""
        parts = name.split(".")
        module = self._model
        for part in parts:
            module = getattr(module, part)
        return module

    # -------------------------------------------------------------------------
    # Forward hooks for accumulated deltas
    # -------------------------------------------------------------------------

    def _register_edit_hooks(self) -> None:
        """Register output-patching hooks on all target linear layers.

        Each hook adds the accumulated delta (ΔW_acc) to the layer output:
            output += F.linear(input, ΔW_acc)
        This is equivalent to a trainable rank-k additive path on top of the
        frozen (possibly quantized) base weights, similar to LoRA.
        """
        for name, module in self._iter_target_modules():
            delta_ref = self._accumulated_deltas  # capture by ref

            def make_hook(mod_name: str):
                def hook(module, input, output):
                    delta = delta_ref.get(mod_name)
                    if delta is None or delta.abs().max() == 0:
                        return output
                    x = input[0]  # (batch, seq_len, d_in)
                    x_flat = x.to(delta.dtype)
                    # output += x_flat @ delta.T
                    addition = F.linear(x_flat, delta)
                    return output + addition.to(output.dtype)
                return hook

            handle = module.register_forward_hook(make_hook(name))
            self._hook_handles.append(handle)

    # -------------------------------------------------------------------------
    # Gradient collection
    # -------------------------------------------------------------------------

    def _collect_gradient_components(
        self, question: str, answer: str
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """Run 1-step frozen fine-tuning and capture (δ, u) per target layer.

        For a single-sample causal LM loss:
            ∇W_l = δ_{l+1} u_l^T   (exactly rank-1)
        where δ_{l+1} = ∂L/∂pre_activation and u_l = input to layer l.

        We capture these components via forward (u) and backward (δ) hooks,
        avoiding construction of the full d_out × d_in gradient matrix.

        Args:
            question: Edit input text x_e.
            answer: Target output text y_e.

        Returns:
            Dict mapping module_name → (delta_vec, u_vec), both float32 CPU.
        """
        assert self._model is not None
        assert self._tokenizer is not None
        assert self._config is not None

        device = next(self._model.parameters()).device

        # Build causal LM training pair: prompt + answer as one sequence,
        # with answer tokens used for loss computation only
        messages = [{"role": "user", "content": question}]
        try:
            prompt = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            prompt = f"User: {question}\nAssistant:"

        full_text = prompt + answer + self._tokenizer.eos_token

        enc = self._tokenizer(
            full_text,
            return_tensors="pt",
            max_length=self._config.max_seq_length,
            truncation=True,
        )
        prompt_enc = self._tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self._config.max_seq_length,
            truncation=True,
        )
        prompt_len = prompt_enc["input_ids"].shape[1]

        input_ids = enc["input_ids"].to(device)
        # Labels: -100 for prompt tokens (masked), real token ids for answer
        labels = input_ids.clone()
        labels[:, :prompt_len] = -100

        # Storage for captured components
        captured_u: dict[str, torch.Tensor] = {}
        captured_delta: dict[str, torch.Tensor] = {}

        hooks = []
        for name, module in self._iter_target_modules():
            def make_fwd_hook(mod_name):
                def fwd_hook(mod, input, output):
                    x = input[0]  # (1, seq_len, d_in)
                    # Average over sequence length → shape (d_in,)
                    captured_u[mod_name] = x.detach().float().mean(dim=(0, 1)).cpu()
                return fwd_hook

            def make_bwd_hook(mod_name):
                def bwd_hook(mod, grad_input, grad_output):
                    # grad_output[0]: (1, seq_len, d_out)
                    captured_delta[mod_name] = (
                        grad_output[0].detach().float().mean(dim=(0, 1)).cpu()
                    )
                return bwd_hook

            hooks.append(module.register_forward_hook(make_fwd_hook(name)))
            hooks.append(module.register_full_backward_hook(make_bwd_hook(name)))

        # Temporarily enable gradients on the model for backprop
        try:
            for param in self._model.parameters():
                param.requires_grad_(False)  # keep frozen

            with torch.enable_grad():
                outputs = self._model(
                    input_ids=input_ids, labels=labels, use_cache=False
                )
                loss = outputs.loss
                loss.backward()

        finally:
            for h in hooks:
                h.remove()
            self._model.zero_grad(set_to_none=True)

        return {
            name: (captured_delta[name], captured_u[name])
            for name in captured_delta
            if name in captured_u
        }

    # -------------------------------------------------------------------------
    # Editing
    # -------------------------------------------------------------------------

    def apply_edit(self, question: str, answer: str) -> None:
        """Apply a single RLEdit update.

        Runs the 1-step gradient collection on (question, answer), feeds the
        captured (δ, u) components through the trained hypernetwork H to
        produce (δ̃, ū), then accumulates ΔW = outer(δ̃, ū) into the
        corresponding target layer's delta buffer.

        The forward hooks registered at load time automatically add
        ΔW_accumulated @ x to each target layer's output.

        Args:
            question: Edit question x_e (the query describing the new fact).
            answer: Target answer y_e (the correct answer to inject).
        """
        self._ensure_loaded()
        assert self._config is not None

        device = torch.device(
            "cuda" if (self.use_gpu and torch.cuda.is_available()) else "cpu"
        )

        gradient_components = self._collect_gradient_components(question, answer)

        for mod_name, (delta_vec, u_vec) in gradient_components.items():
            if mod_name not in self._hypernetworks:
                logger.debug("No hypernetwork for %s, skipping.", mod_name)
                continue

            hn = self._hypernetworks[mod_name]
            delta_t = delta_vec.to(device)
            u_t = u_vec.to(device)

            with torch.no_grad():
                delta_hat, u_hat = hn(delta_t, u_t)
                delta_W = torch.outer(delta_hat, u_hat)  # (d_out, d_in)
                self._accumulated_deltas[mod_name] += delta_W.float()

        self._n_edits_applied += 1
        logger.debug("RLEdit: applied edit %d.", self._n_edits_applied)

    def apply_edits(self, edit_list: list[dict | list | tuple]) -> None:
        """Apply multiple edits sequentially.

        Args:
            edit_list: Each item is either:
                - A dict with "question" and "answer" keys.
                - A (question, answer) tuple or list.
        """
        self._ensure_loaded()
        for item in edit_list:
            if isinstance(item, dict):
                q = item.get("question") or item.get("x_e") or item.get("prompt", "")
                a = item.get("answer") or item.get("y_e") or item.get("target", "")
            else:
                q, a = item[0], item[1]
            self.apply_edit(q, a)
        logger.info("RLEdit: %d edits applied.", self._n_edits_applied)

    def reset_edits(self) -> None:
        """Clear all accumulated deltas, reverting to the unedited base model."""
        for name in self._accumulated_deltas:
            self._accumulated_deltas[name].zero_()
        self._n_edits_applied = 0
        logger.info("RLEdit: all edits cleared.")

    # -------------------------------------------------------------------------
    # Generation
    # -------------------------------------------------------------------------

    def generate(self, query: str) -> RLEditInferenceResult:
        """Generate a response using the RLEdit-edited model.

        Args:
            query: User query string.

        Returns:
            RLEditInferenceResult with .response, .adapter_loaded, .routing_result.
        """
        self._ensure_loaded()

        messages = [{"role": "user", "content": query}]
        try:
            prompt = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
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

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "pad_token_id": self._tokenizer.pad_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
            "use_cache": True,
        }
        if self.do_sample:
            gen_kwargs["temperature"] = self.temperature

        with torch.no_grad():
            outputs = self._model.generate(**inputs, **gen_kwargs)

        prompt_len = inputs["input_ids"].shape[1]
        response = self._tokenizer.decode(
            outputs[0][prompt_len:], skip_special_tokens=True
        ).strip()

        return RLEditInferenceResult(
            response=response,
            n_edits_applied=self._n_edits_applied,
        )

    def __del__(self):
        """Clean up hook handles."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
