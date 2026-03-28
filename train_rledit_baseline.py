#!/usr/bin/env python3
"""
Train RLEdit Baseline — Hypernetwork-Based Lifelong Editing
============================================================

Trains the RLEdit hypernetwork (Li et al., ICML 2025) for use as a baseline
in the Patch-and-Route evaluation framework.

RLEdit formulates lifelong model editing as a Markov Decision Process (MDP):
  - Agent:   Hypernetwork H (per target layer, 4-layer MLP)
  - State:   (current LLM params W, edit pair (x_e, y_e))
  - Action:  Parameter update ΔW = H(∇W)
  - Reward:  r_t = -(L_base_t + L_back_t + η·||ΔW_t||²)
             where L_base = L_e + λ_loc·L_loc  (target update + locality)
                   L_back = Σ_{i=t-k}^{t-1} μ^{t-i} L_base_i  (memory backtracking)

The hypernetwork maps rank-1 gradient components (δ, u) → (δ̃, ū):
    ∇W_l = δ_{l+1} u_l^T  →  ΔW_l = δ̃_{l+1} ū_l^T

Training follows offline policy optimisation (Algorithm 1):
    J = Σ_{t=1}^{n} γ^t r_t  →  θ' = argmax_θ J

Output
------
checkpoints/rledit_baseline/
├── rledit_hypernetwork.pt    # state_dict per target module
└── rledit_config.json        # RLEditConfig used for training

Usage
-----
    python train_rledit_baseline.py \\
        --data_path data/counterfact_edits.json \\
        --output_dir checkpoints/rledit_baseline \\
        --n_epochs 5 \\
        --edits_per_batch 20 \\
        --target_modules mlp.down_proj mlp.gate_proj

    # Minimal smoke-test (5 edits, 2 epochs):
    python train_rledit_baseline.py \\
        --data_path data/counterfact_edits.json \\
        --output_dir checkpoints/rledit_baseline \\
        --n_epochs 2 --edits_per_batch 5 --max_samples 100

Data Format
-----------
The --data_path JSON file should be a list of dicts with at least:
    {
        "question":      str,   # Edit query x_e
        "answer":        str,   # Target answer y_e (the new fact)
        "question_loc":  str,   # Locality probe x_loc (unrelated query)
        "answer_loc":    str    # Expected answer y_loc (should not change)
    }

Reference: arXiv:2502.05759 — "Reinforced Lifelong Editing for Language Models"
Repo:      github.com/zhrli324/RLEdit
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

sys.path.insert(0, str(Path(__file__).parent))

from src.inference.rledit_inference import RLEditConfig, RLEditHypernetwork
from src.utils.logging import setup_logger, configure_framework_logging

logger = logging.getLogger(__name__)


# =============================================================================
# Argument Parsing
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train RLEdit hypernetwork for lifelong model editing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help=(
            "Path to JSON file with editing samples. Each entry must have: "
            "question, answer, question_loc, answer_loc."
        ),
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit number of samples loaded (useful for smoke tests)",
    )

    # Model
    parser.add_argument(
        "--model_id",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        help="HuggingFace model ID for the base LLM",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        choices=["none", "int8"],
        default="none",
        help=(
            "Quantization for the base LLM during training. "
            "int4 is NOT supported since we need real float32 gradients. "
            "Use 'none' (BF16) or 'int8' to reduce memory."
        ),
    )

    # Target modules
    parser.add_argument(
        "--target_modules",
        type=str,
        nargs="+",
        default=["mlp.down_proj", "mlp.gate_proj"],
        help="Substrings of module names to target for editing",
    )
    parser.add_argument(
        "--target_layer_range",
        type=int,
        nargs=2,
        default=None,
        metavar=("START", "END"),
        help=(
            "Optional: restrict editing to transformer layers [START, END) "
            "by index. E.g. --target_layer_range 36 40 for the last 4 layers "
            "of a 40-layer model. If not set, all matching modules are used."
        ),
    )

    # RL hyperparameters
    parser.add_argument("--k", type=int, default=5,
                        help="Memory backtracking window size")
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="Discount factor for total reward J")
    parser.add_argument("--mu", type=float, default=0.9,
                        help="Decay factor for backtracking loss weights")
    parser.add_argument("--eta", type=float, default=0.1,
                        help="Regularisation coefficient for ||ΔW||²")
    parser.add_argument("--lambda_loc", type=float, default=0.5,
                        help="Weight of locality KL divergence in base reward")

    # Hypernetwork architecture
    parser.add_argument("--hidden_dim", type=int, default=256,
                        help="Hidden dimension of the 4-layer hypernetwork MLPs")

    # Training
    parser.add_argument("--n_epochs", type=int, default=5,
                        help="Number of training epochs over the dataset")
    parser.add_argument(
        "--edits_per_batch",
        type=int,
        default=20,
        help="Editing sequence length n per RL trajectory (batch of edits)",
    )
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Hypernetwork learning rate")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Gradient norm clipping for hypernetwork")
    parser.add_argument("--grad_lr", type=float, default=1e-3,
                        help="LR used for 1-step gradient collection on LLM")
    parser.add_argument("--max_seq_length", type=int, default=512,
                        help="Maximum token length for edit inputs")

    # Logging / output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/rledit_baseline",
        help="Directory to save the trained hypernetwork",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


# =============================================================================
# Data loading
# =============================================================================

@dataclass
class EditSample:
    question: str
    answer: str
    question_loc: str
    answer_loc: str


def load_dataset(path: str, max_samples: int | None = None) -> list[EditSample]:
    """Load editing samples from a JSON file."""
    with open(path) as f:
        raw = json.load(f)

    samples: list[EditSample] = []
    for item in raw:
        q = item.get("question") or item.get("prompt") or item.get("x_e", "")
        a = item.get("answer") or item.get("target") or item.get("y_e", "")
        q_loc = item.get("question_loc") or item.get("locality_prompt") or item.get("x_loc", q)
        a_loc = item.get("answer_loc") or item.get("locality_answer") or item.get("y_loc", a)
        if q and a:
            samples.append(EditSample(q, a, q_loc, a_loc))

    if max_samples is not None:
        samples = samples[:max_samples]

    logger.info("Loaded %d edit samples from %s", len(samples), path)
    return samples


# =============================================================================
# Training utilities
# =============================================================================

def build_causal_lm_batch(
    tokenizer,
    question: str,
    answer: str,
    max_seq_length: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Tokenise (question, answer) for causal LM loss on the answer tokens."""
    messages = [{"role": "user", "content": question}]
    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        prompt = f"User: {question}\nAssistant:"

    full_text = prompt + answer + tokenizer.eos_token
    enc = tokenizer(
        full_text, return_tensors="pt",
        max_length=max_seq_length, truncation=True,
    )
    prompt_len = tokenizer(
        prompt, return_tensors="pt", truncation=True,
    )["input_ids"].shape[1]

    input_ids = enc["input_ids"].to(device)
    labels = input_ids.clone()
    labels[:, :prompt_len] = -100  # mask prompt tokens from loss

    return {"input_ids": input_ids, "labels": labels}


def compute_causal_lm_loss(
    model,
    tokenizer,
    question: str,
    answer: str,
    max_seq_length: int,
    device: torch.device,
) -> torch.Tensor:
    """Return the causal LM loss for the given (question, answer) pair."""
    batch = build_causal_lm_batch(tokenizer, question, answer, max_seq_length, device)
    outputs = model(**batch, use_cache=False)
    return outputs.loss


def collect_gradient_components(
    model,
    tokenizer,
    question: str,
    answer: str,
    target_module_substrings: list[str],
    layer_range: tuple[int, int] | None,
    max_seq_length: int,
    device: torch.device,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Capture (δ, u) gradient components per target linear layer.

    Uses forward and backward hooks to capture:
        u   = mean input activation (d_in,)   → passed through u_net
        δ   = mean output gradient (d_out,)   → passed through delta_net

    Returns:
        Dict mapping module_name → (delta_vec, u_vec) on CPU float32.
    """
    captured_u: dict[str, torch.Tensor] = {}
    captured_delta: dict[str, torch.Tensor] = {}

    hooks = []

    def _in_target_layer(name: str) -> bool:
        if not any(sub in name for sub in target_module_substrings):
            return False
        if layer_range is not None:
            # Extract layer index from name like "model.layers.36.mlp.down_proj"
            parts = name.split(".")
            for i, p in enumerate(parts):
                if p == "layers" and i + 1 < len(parts):
                    try:
                        idx = int(parts[i + 1])
                        return layer_range[0] <= idx < layer_range[1]
                    except ValueError:
                        pass
            return False
        return True

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not _in_target_layer(name):
            continue

        def make_fwd(mod_name):
            def fwd(mod, inp, out):
                captured_u[mod_name] = inp[0].detach().float().mean(dim=(0, 1)).cpu()
            return fwd

        def make_bwd(mod_name):
            def bwd(mod, grad_inp, grad_out):
                if grad_out[0] is not None:
                    captured_delta[mod_name] = (
                        grad_out[0].detach().float().mean(dim=(0, 1)).cpu()
                    )
            return bwd

        hooks.append(module.register_forward_hook(make_fwd(name)))
        hooks.append(module.register_full_backward_hook(make_bwd(name)))

    try:
        batch = build_causal_lm_batch(
            tokenizer, question, answer, max_seq_length, device
        )
        # Temporarily enable grad on all params for backward pass
        for p in model.parameters():
            p.requires_grad_(False)
        with torch.enable_grad():
            loss = model(**batch, use_cache=False).loss
            loss.backward()
        model.zero_grad(set_to_none=True)
    finally:
        for h in hooks:
            h.remove()

    return {
        name: (captured_delta[name], captured_u[name])
        for name in captured_delta
        if name in captured_u
    }


# =============================================================================
# RLEdit Training Loop (Algorithm 1)
# =============================================================================

class RLEditTrainer:
    """Trains the RLEdit hypernetwork using offline RL policy optimisation.

    Implements Algorithm 1 from the paper:
        for each trajectory of n edit pairs:
            for t = 1..n:
                collect ∇W_{t-1} via 1-step gradient  (frozen fine-tune)
                ΔW_t = H(∇W_{t-1})
                temporarily apply W_t = W_{t-1} + ΔW_t
                compute r_t = -(L_base_t + L_back_t + η||ΔW_t||²)
            J = Σ γ^t r_t
            back-propagate J w.r.t. H.params  (offline update)
    """

    def __init__(
        self,
        model,
        tokenizer,
        hypernetworks: dict[str, RLEditHypernetwork],
        config: RLEditConfig,
        layer_range: tuple[int, int] | None,
        lr: float,
        grad_clip: float,
        device: torch.device,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.hypernetworks = hypernetworks
        self.config = config
        self.layer_range = layer_range
        self.device = device
        self.grad_clip = grad_clip

        # Single optimizer across all hypernetworks
        all_params = [p for hn in hypernetworks.values() for p in hn.parameters()]
        self.optimizer = AdamW(all_params, lr=lr)

    def _target_modules(self) -> list[str]:
        return self.config.target_modules

    def _in_target_layer(self, name: str) -> bool:
        if not any(sub in name for sub in self._target_modules()):
            return False
        if self.layer_range is not None:
            parts = name.split(".")
            for i, p in enumerate(parts):
                if p == "layers" and i + 1 < len(parts):
                    try:
                        idx = int(parts[i + 1])
                        return self.layer_range[0] <= idx < self.layer_range[1]
                    except ValueError:
                        pass
            return False
        return True

    def _get_module(self, name: str) -> nn.Module:
        module = self.model
        for part in name.split("."):
            module = getattr(module, part)
        return module

    def _apply_deltas_via_hooks(
        self, accumulated: dict[str, torch.Tensor]
    ) -> list:
        """Register temporary hooks adding accumulated deltas to layer outputs."""
        handles = []
        for name, module in self.model.named_modules():
            if not isinstance(module, nn.Linear) or not self._in_target_layer(name):
                continue
            delta = accumulated.get(name)
            if delta is None:
                continue

            def make_hook(d, dv):
                def hook(mod, inp, out):
                    x = inp[0].to(d.dtype)
                    return out + F.linear(x, dv).to(out.dtype)
                return hook

            handles.append(module.register_forward_hook(make_hook(delta.device, delta)))
        return handles

    def _compute_loss(
        self,
        question: str,
        answer: str,
        accumulated: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Forward pass with current accumulated deltas, return loss."""
        handles = self._apply_deltas_via_hooks(accumulated)
        try:
            batch = build_causal_lm_batch(
                self.tokenizer, question, answer,
                self.config.max_seq_length, self.device,
            )
            with torch.enable_grad():
                loss = self.model(**batch, use_cache=False).loss
        finally:
            for h in handles:
                h.remove()
        return loss

    def _compute_locality_kl(
        self,
        question_loc: str,
        answer_loc: str,
        accumulated: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """KL(p_W0(·|x_loc) || p_Wt(·|x_loc)) approximated via cross-entropy diff."""
        # KL ≈ L(Wt, x_loc, y_loc) - L(W0, x_loc, y_loc)
        # Since W0 has no hooks, we compute L(Wt) and compare to L(W0)
        batch = build_causal_lm_batch(
            self.tokenizer, question_loc, answer_loc,
            self.config.max_seq_length, self.device,
        )

        # Loss under W0 (no hooks)
        with torch.no_grad():
            loss_w0 = self.model(**batch, use_cache=False).loss.detach()

        # Loss under Wt (with hooks)
        handles = self._apply_deltas_via_hooks(accumulated)
        try:
            with torch.enable_grad():
                loss_wt = self.model(**batch, use_cache=False).loss
        finally:
            for h in handles:
                h.remove()

        # KL proxy: max(0, L_wt - L_w0) — positive means locality degraded
        kl_approx = F.relu(loss_wt - loss_w0)
        return kl_approx

    def run_trajectory(
        self, batch: list[EditSample]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Run one RL trajectory over a batch of edit samples.

        Returns:
            (J, metrics_dict) where J = -total_reward (for minimisation).
        """
        n = len(batch)
        # accumulated[module_name] = ΔW_accumulated (tracked tensor for autograd)
        accumulated: dict[str, torch.Tensor] = {}
        # initialise as zero tracked tensors
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and self._in_target_layer(name):
                d_out, d_in = module.out_features, module.in_features
                accumulated[name] = torch.zeros(
                    d_out, d_in, device=self.device, dtype=torch.float32,
                    requires_grad=False,
                )

        rewards = []
        # history of (L_base, ΔW_norm²) tuples for backtracking
        history: deque[dict[str, Any]] = deque(maxlen=self.config.k)
        base_losses_history: deque[torch.Tensor] = deque(maxlen=self.config.k)

        for t, sample in enumerate(batch):
            # --- Step 1: collect gradient components (no autograd over LLM) ---
            grad_components = collect_gradient_components(
                model=self.model,
                tokenizer=self.tokenizer,
                question=sample.question,
                answer=sample.answer,
                target_module_substrings=self._target_modules(),
                layer_range=self.layer_range,
                max_seq_length=self.config.max_seq_length,
                device=self.device,
            )

            # --- Step 2: hypernetwork forward → ΔW_t (tracked) ---
            delta_W_t: dict[str, torch.Tensor] = {}
            delta_norms_sq: list[torch.Tensor] = []

            for mod_name, (delta_vec, u_vec) in grad_components.items():
                if mod_name not in self.hypernetworks:
                    continue
                hn = self.hypernetworks[mod_name]
                dv = delta_vec.to(self.device)
                uv = u_vec.to(self.device)
                delta_hat, u_hat = hn(dv, uv)  # tracked through hn params
                dW = torch.outer(delta_hat, u_hat)
                delta_W_t[mod_name] = dW
                delta_norms_sq.append((dW ** 2).sum())

            # --- Step 3: accumulate delta (detached for subsequent gradient steps) ---
            for mod_name, dW in delta_W_t.items():
                accumulated[mod_name] = accumulated[mod_name] + dW.detach()

            # --- Step 4: compute reward components ---
            # L_e: target knowledge update loss
            l_e = self._compute_loss(
                sample.question, sample.answer, accumulated
            )

            # L_loc: locality KL divergence
            l_loc = self._compute_locality_kl(
                sample.question_loc, sample.answer_loc, accumulated
            )

            l_base = l_e + self.config.lambda_loc * l_loc

            # L_back: memory backtracking (Eq. 9)
            l_back = torch.tensor(0.0, device=self.device)
            for i, past_l_base in enumerate(reversed(list(base_losses_history))):
                decay = self.config.mu ** (i + 1)
                l_back = l_back + decay * past_l_base

            # Regularisation: η·||ΔW_t||²
            reg = torch.tensor(0.0, device=self.device)
            if delta_norms_sq:
                reg = self.config.eta * torch.stack(delta_norms_sq).sum()

            # Reward: r_t = -(L_base + L_back + η||ΔW||²)
            r_t = -(l_base + l_back + reg)
            rewards.append(self.config.gamma ** t * r_t)
            base_losses_history.append(l_base.detach())

        if not rewards:
            return torch.tensor(0.0, device=self.device, requires_grad=True), {}

        # --- Total reward J = Σ γ^t r_t (offline, Eq. 12) ---
        J = torch.stack(rewards).sum()

        metrics = {
            "J": J.item(),
            "n_edits": n,
            "avg_reward": J.item() / max(n, 1),
        }
        return J, metrics

    def train_epoch(
        self,
        samples: list[EditSample],
        edits_per_batch: int,
    ) -> dict[str, float]:
        """Run one epoch: iterate over samples in trajectory batches."""
        random.shuffle(samples)
        epoch_rewards = []
        n_batches = max(1, len(samples) // edits_per_batch)

        for i in range(n_batches):
            batch = samples[i * edits_per_batch: (i + 1) * edits_per_batch]
            if not batch:
                continue

            self.optimizer.zero_grad()

            J, metrics = self.run_trajectory(batch)

            # Maximise J ≡ minimise -J
            loss = -J
            if loss.requires_grad:
                loss.backward()
                # Clip gradients
                for hn in self.hypernetworks.values():
                    nn.utils.clip_grad_norm_(hn.parameters(), self.grad_clip)
                self.optimizer.step()

            epoch_rewards.append(metrics.get("J", 0.0))

            if (i + 1) % max(1, n_batches // 5) == 0:
                avg_J = sum(epoch_rewards) / len(epoch_rewards)
                logger.info(
                    "  Batch %d/%d | avg J=%.4f", i + 1, n_batches, avg_J
                )

        return {
            "avg_J": sum(epoch_rewards) / max(len(epoch_rewards), 1),
            "n_batches": n_batches,
        }


# =============================================================================
# Discover target module names + shapes from model
# =============================================================================

def discover_target_modules(
    model,
    target_module_substrings: list[str],
    layer_range: tuple[int, int] | None,
) -> dict[str, tuple[int, int]]:
    """Return {module_name: (d_out, d_in)} for all matching linear layers."""
    result: dict[str, tuple[int, int]] = {}

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not any(sub in name for sub in target_module_substrings):
            continue
        if layer_range is not None:
            parts = name.split(".")
            in_range = False
            for i, p in enumerate(parts):
                if p == "layers" and i + 1 < len(parts):
                    try:
                        idx = int(parts[i + 1])
                        in_range = layer_range[0] <= idx < layer_range[1]
                    except ValueError:
                        pass
            if not in_range:
                continue

        result[name] = (module.out_features, module.in_features)

    return result


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()
    configure_framework_logging(level=args.log_level)
    logger.info("=" * 70)
    logger.info("RLEDIT HYPERNETWORK TRAINING")
    logger.info("=" * 70)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    logger.info("Device: %s", device)

    # -------------------------------------------------------------------------
    # Load dataset
    # -------------------------------------------------------------------------
    samples = load_dataset(args.data_path, args.max_samples)
    if not samples:
        logger.error("No samples loaded. Check --data_path.")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # Load base model (float precision — we need real gradients)
    # -------------------------------------------------------------------------
    logger.info("\n[1/4] Loading base model: %s ...", args.model_id)
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_kwargs: dict[str, Any] = {
        "device_map": "auto" if str(device) != "cpu" else "cpu",
        "trust_remote_code": True,
    }
    if args.quantization == "int8":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        logger.info("  Using int8 quantization (BF16 compute)")
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16
        logger.info("  Using BF16 (no quantization)")

    model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)
    model.eval()
    # Freeze all LLM parameters — only H is trained
    for param in model.parameters():
        param.requires_grad_(False)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # -------------------------------------------------------------------------
    # Discover target modules
    # -------------------------------------------------------------------------
    layer_range = tuple(args.target_layer_range) if args.target_layer_range else None

    logger.info("\n[2/4] Discovering target modules ...")
    module_shapes = discover_target_modules(
        model, args.target_modules, layer_range
    )
    if not module_shapes:
        logger.error(
            "No matching modules found for target_modules=%s. "
            "Check --target_modules and --target_layer_range.",
            args.target_modules,
        )
        sys.exit(1)
    logger.info("  Found %d target modules:", len(module_shapes))
    for name, (d_out, d_in) in sorted(module_shapes.items()):
        logger.info("    %s  (d_out=%d, d_in=%d)", name, d_out, d_in)

    # -------------------------------------------------------------------------
    # Build hypernetworks
    # -------------------------------------------------------------------------
    logger.info("\n[3/4] Building hypernetworks (hidden_dim=%d) ...", args.hidden_dim)
    hypernetworks: dict[str, RLEditHypernetwork] = {}
    for mod_name, (d_out, d_in) in module_shapes.items():
        hn = RLEditHypernetwork(d_out=d_out, d_in=d_in, hidden_dim=args.hidden_dim)
        hn.to(device).train()
        hypernetworks[mod_name] = hn

    total_params = sum(
        p.numel() for hn in hypernetworks.values() for p in hn.parameters()
    )
    logger.info(
        "  Total hypernetwork parameters: %s",
        f"{total_params:,}",
    )

    # -------------------------------------------------------------------------
    # Build config
    # -------------------------------------------------------------------------
    config = RLEditConfig(
        k=args.k,
        gamma=args.gamma,
        mu=args.mu,
        eta=args.eta,
        lambda_loc=args.lambda_loc,
        target_modules=args.target_modules,
        hidden_dim=args.hidden_dim,
        n_grad_steps=1,
        grad_lr=args.grad_lr,
        max_seq_length=args.max_seq_length,
    )

    # -------------------------------------------------------------------------
    # Train
    # -------------------------------------------------------------------------
    logger.info("\n[4/4] Training (%d epochs, %d edits/batch) ...",
                args.n_epochs, args.edits_per_batch)

    trainer = RLEditTrainer(
        model=model,
        tokenizer=tokenizer,
        hypernetworks=hypernetworks,
        config=config,
        layer_range=layer_range,
        lr=args.lr,
        grad_clip=args.grad_clip,
        device=device,
    )

    for epoch in range(1, args.n_epochs + 1):
        t_epoch = time.time()
        metrics = trainer.train_epoch(samples, args.edits_per_batch)
        elapsed = time.time() - t_epoch
        logger.info(
            "Epoch %d/%d | avg_J=%.4f | n_batches=%d | %.1fs",
            epoch, args.n_epochs,
            metrics["avg_J"],
            metrics["n_batches"],
            elapsed,
        )

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save hypernetwork state dicts keyed by module name
    hn_state = {
        name: hn.state_dict()
        for name, hn in hypernetworks.items()
    }
    hn_path = output_dir / "rledit_hypernetwork.pt"
    torch.save(hn_state, hn_path)
    logger.info("Saved hypernetwork weights → %s", hn_path)

    # Save config
    cfg_path = output_dir / "rledit_config.json"
    with open(cfg_path, "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    logger.info("Saved RLEdit config → %s", cfg_path)

    # Save training provenance
    provenance = {
        "model_id": args.model_id,
        "quantization": args.quantization,
        "n_epochs": args.n_epochs,
        "edits_per_batch": args.edits_per_batch,
        "n_samples": len(samples),
        "target_modules": args.target_modules,
        "target_layer_range": args.target_layer_range,
        "n_hypernetworks": len(hypernetworks),
        "total_hypernetwork_params": total_params,
        "lr": args.lr,
        "seed": args.seed,
    }
    prov_path = output_dir / "training_config.json"
    with open(prov_path, "w") as f:
        json.dump(provenance, f, indent=2)
    logger.info("Saved training provenance → %s", prov_path)

    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("  Output: %s", output_dir)
    logger.info(
        "\nTo evaluate with RLEdit baseline:\n"
        "    python eval_pnr.py \\\n"
        "        --rledit %s \\\n"
        "        --rledit_edits data/eval_edits.json \\\n"
        "        --eval_sets conflict\n",
        output_dir,
    )
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
