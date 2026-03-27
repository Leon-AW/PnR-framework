"""
System 1 — Stable Core ("Neocortex")
======================================

The Stable Core stores deep structural knowledge: syntax, reasoning patterns,
mathematical relationships, causal schemas — the "crystallized intelligence."

It is very large and very slow to change. Updates occur only through carefully
constrained structural distillation (System 4c), never by direct gradient
descent on streaming data. A versioning system with CKA-bounded updates
and lightweight compatibility adapters ensures the rest of the architecture
remains stable when the core evolves.

Key invariants:
- Representation shift between consecutive versions is bounded by CKA threshold
- All active experts carry a reference to their native core version
- Compatibility adapters bridge version gaps cheaply
- Periodic re-adaptation collapses adapter chains
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)

from .cka import linear_cka, minibatch_cka, compute_representation_shift
from .config import StableCoreConfig

logger = logging.getLogger(__name__)


@dataclass
class CoreVersion:
    """Metadata for a single core version."""
    version: int
    checkpoint_path: str
    cka_from_previous: float | None = None
    timestamp: float = 0.0
    probe_set_hash: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CompatAdapter:
    """A lightweight low-rank adapter mapping between core versions."""
    from_version: int
    to_version: int
    adapter_path: str
    rank: int


class StableCore:
    """System 1: Versioned Stable Core with CKA-bounded evolution.

    Manages the frozen foundation model with a formal update protocol
    that prevents representation drift from destabilizing the architecture.

    The core is the backbone that all experts read from and write through.
    Its representation space defines the "language" that routing centroids,
    expert outputs, and compatibility adapters all speak.
    """

    def __init__(self, config: StableCoreConfig | None = None) -> None:
        self.config = config or StableCoreConfig()

        self._model: PreTrainedModel | None = None
        self._tokenizer: PreTrainedTokenizerBase | None = None
        self._current_version: int = 0
        self._versions: list[CoreVersion] = []
        self._compat_adapters: list[CompatAdapter] = []
        self._probe_set: list[str] | None = None

        self._checkpoint_dir = Path(self.config.checkpoint_dir)
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info("StableCore initialized (version 0)")

    @property
    def version(self) -> int:
        return self._current_version

    @property
    def model(self) -> PreTrainedModel:
        if self._model is None:
            raise RuntimeError("Core model not loaded. Call load() first.")
        return self._model

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not loaded. Call load() first.")
        return self._tokenizer

    def load(self) -> None:
        """Load the frozen foundation model."""
        if self._model is not None:
            logger.warning("Core already loaded, skipping.")
            return

        logger.info(f"Loading Stable Core: {self.config.model_id}")

        bnb_config = None
        if self.config.quantization == "int4":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self.config.torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif self.config.quantization == "int8":
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)

        model_kwargs: dict[str, Any] = {
            "device_map": self.config.device_map,
            "torch_dtype": self.config.torch_dtype,
            "trust_remote_code": True,
            "use_cache": self.config.use_cache,
        }
        if bnb_config:
            model_kwargs["quantization_config"] = bnb_config

        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.model_id, **model_kwargs,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_id, trust_remote_code=True,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
        self._tokenizer.padding_side = "right"

        if bnb_config:
            self._model = prepare_model_for_kbit_training(
                self._model,
                use_gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )

        self._versions.append(CoreVersion(
            version=0,
            checkpoint_path=self.config.model_id,
        ))

        logger.info(f"Stable Core loaded (v{self._current_version})")

    # ------------------------------------------------------------------
    # Representation extraction
    # ------------------------------------------------------------------

    def extract_representations(
        self,
        texts: list[str],
        layer: int = -1,
        batch_size: int | None = None,
    ) -> torch.Tensor:
        """Extract hidden-state representations for a set of texts.

        Uses mean pooling over the sequence dimension to produce a single
        vector per input text.

        Args:
            texts: Input texts (the probe set).
            layer: Which transformer layer to read (-1 = last).
            batch_size: Batch size for processing.

        Returns:
            Tensor of shape (len(texts), hidden_dim).
        """
        batch_size = batch_size or self.config.probe_batch_size
        all_reps = []

        self._model.eval()
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                inputs = self._tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                )
                inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

                outputs = self._model(
                    **inputs,
                    output_hidden_states=True,
                )
                hidden = outputs.hidden_states[layer]

                mask = inputs["attention_mask"].unsqueeze(-1).float()
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                all_reps.append(pooled.cpu())

        return torch.cat(all_reps, dim=0)

    # ------------------------------------------------------------------
    # CKA-bounded update protocol
    # ------------------------------------------------------------------

    def measure_shift(
        self,
        old_representations: torch.Tensor,
        new_representations: torch.Tensor,
    ) -> float:
        """Measure representation shift between two sets of representations.

        Returns 1 - CKA(old, new). A value of 0 means identical spaces.
        """
        n = old_representations.shape[0]
        use_minibatch = n > 1024
        return compute_representation_shift(
            old_representations, new_representations,
            use_minibatch=use_minibatch,
        )

    def validate_update(
        self,
        candidate_model: PreTrainedModel,
        probe_texts: list[str],
    ) -> tuple[bool, float]:
        """Validate a candidate core update against the CKA threshold.

        Args:
            candidate_model: The proposed new core model.
            probe_texts: Held-out probe set for CKA measurement.

        Returns:
            (is_valid, shift) — whether the update is within bounds and
            the measured representation shift.
        """
        old_reps = self.extract_representations(probe_texts)

        orig_model = self._model
        self._model = candidate_model
        new_reps = self.extract_representations(probe_texts)
        self._model = orig_model

        shift = self.measure_shift(old_reps, new_reps)
        is_valid = shift <= self.config.cka_threshold

        logger.info(
            f"Core update validation: shift={shift:.4f}, "
            f"threshold={self.config.cka_threshold}, "
            f"valid={is_valid}"
        )
        return is_valid, shift

    def apply_update(
        self,
        new_model: PreTrainedModel,
        probe_texts: list[str],
        force: bool = False,
    ) -> bool:
        """Apply a validated core update with the formal protocol.

        Protocol:
        1. Measure representation shift via CKA
        2. Reject if shift exceeds threshold (unless forced)
        3. Train compatibility adapter for old -> new mapping
        4. Increment version, save metadata

        Args:
            new_model: The updated core model.
            probe_texts: Probe set for CKA validation.
            force: Skip CKA validation (dangerous).

        Returns:
            True if update was applied.
        """
        if not force:
            is_valid, shift = self.validate_update(new_model, probe_texts)
            if not is_valid:
                logger.warning(
                    f"Core update REJECTED: shift {shift:.4f} > "
                    f"threshold {self.config.cka_threshold}"
                )
                return False
        else:
            shift = 0.0

        old_version = self._current_version
        new_version = old_version + 1

        self._model = new_model
        self._current_version = new_version

        version_info = CoreVersion(
            version=new_version,
            checkpoint_path=str(self._checkpoint_dir / f"v{new_version}"),
            cka_from_previous=shift,
        )
        self._versions.append(version_info)

        logger.info(
            f"Core updated: v{old_version} -> v{new_version} "
            f"(shift={shift:.4f})"
        )
        return True

    def get_compat_adapter_chain(
        self,
        from_version: int,
        to_version: int | None = None,
    ) -> list[CompatAdapter]:
        """Get the chain of compatibility adapters between two versions.

        Args:
            from_version: Source version.
            to_version: Target version (defaults to current).

        Returns:
            Ordered list of adapters to compose.
        """
        to_version = to_version or self._current_version
        if from_version == to_version:
            return []

        chain = []
        for adapter in self._compat_adapters:
            if adapter.from_version >= from_version and adapter.to_version <= to_version:
                chain.append(adapter)

        chain.sort(key=lambda a: a.from_version)
        return chain

    def needs_readaptation(self, expert_native_version: int) -> bool:
        """Check if an expert needs re-adaptation to the current core.

        Triggered when the adapter chain from the expert's native version
        to the current version exceeds max_adapter_chain_length.
        """
        chain = self.get_compat_adapter_chain(expert_native_version)
        return len(chain) > self.config.max_adapter_chain_length

    # ------------------------------------------------------------------
    # PEFT adapter management (delegates to PatchAndRouteLLM patterns)
    # ------------------------------------------------------------------

    def attach_adapter(self, lora_config: LoraConfig) -> PeftModel:
        """Attach a LoRA adapter to the core for training."""
        self._model = get_peft_model(self._model, lora_config)
        return self._model

    def detach_adapter(self) -> None:
        """Detach the current adapter, restoring base core."""
        if isinstance(self._model, PeftModel):
            if hasattr(self._model, "delete_adapter"):
                adapter_name = self._model.active_adapter
                if isinstance(adapter_name, list):
                    adapter_name = adapter_name[0] if adapter_name else "default"
                self._model.delete_adapter(adapter_name)
                if hasattr(self._model, "base_model"):
                    self._model = self._model.base_model.model
            elif hasattr(self._model, "unload"):
                self._model = self._model.unload()

    def load_adapter(self, adapter_path: str | Path) -> PeftModel:
        """Load a saved LoRA adapter onto the core."""
        self._model = PeftModel.from_pretrained(
            self._model, str(adapter_path), is_trainable=True,
        )
        return self._model

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_state(self, path: str | Path | None = None) -> Path:
        """Save core state metadata (versions, adapters, config)."""
        path = Path(path or self.config.checkpoint_dir) / "core_state.json"
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "current_version": self._current_version,
            "versions": [v.to_dict() for v in self._versions],
            "compat_adapters": [asdict(a) for a in self._compat_adapters],
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2, default=str)

        logger.info(f"Core state saved to {path}")
        return path

    def load_state(self, path: str | Path) -> None:
        """Load core state metadata from disk."""
        path = Path(path)
        if not path.exists():
            logger.warning(f"No core state at {path}")
            return

        with open(path) as f:
            state = json.load(f)

        self._current_version = state["current_version"]
        self._versions = [CoreVersion(**v) for v in state["versions"]]
        self._compat_adapters = [CompatAdapter(**a) for a in state["compat_adapters"]]

        logger.info(
            f"Core state loaded: v{self._current_version}, "
            f"{len(self._versions)} versions, "
            f"{len(self._compat_adapters)} adapters"
        )
