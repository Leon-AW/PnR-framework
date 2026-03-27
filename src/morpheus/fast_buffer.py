"""
System 3 — Fast Adaptation Buffer ("Hippocampus")
===================================================

A small, highly plastic LoRA adapter that absorbs the incoming data stream
immediately without touching the rest of the system. This is the "scratch
space" where new information lands before consolidation.

Key properties:
- Very high learning rate (10-100x expert rate)
- Designed to be overwritten during consolidation cycles
- Pattern-separated representations (high rank, noise injection)
  to minimize internal interference within the buffer
- Capacity-limited: triggers consolidation when full

The buffer answers "how do you learn from a stream in real-time without
corrupting anything?" — you don't learn into the main system, you learn
into a scratch space.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from peft import LoraConfig, TaskType

from .config import FastBufferConfig

logger = logging.getLogger(__name__)


@dataclass
class BufferSample:
    """A sample stored in the buffer's memory."""
    text: str
    embedding: np.ndarray | None = None
    timestamp: float = field(default_factory=time.time)
    loss: float = 0.0
    domain_signal: str = ""


class FastBuffer:
    """System 3: Hippocampal fast adaptation buffer.

    The buffer sits between the incoming data stream and the rest of the
    MORPHEUS architecture. It absorbs new information at high plasticity,
    stores a window of recent samples, and signals the consolidation engine
    when it's time to integrate buffer knowledge into the expert bank.

    During training:
    - A dedicated high-learning-rate LoRA adapter trains on streaming data
    - Pattern separation noise is injected into embeddings
    - The buffer tracks sample statistics for distribution shift detection

    During consolidation:
    - Buffer contents are interleaved with self-rehearsal data
    - The buffer adapter is reset (overwritten) after consolidation
    """

    def __init__(self, config: FastBufferConfig | None = None) -> None:
        self.config = config or FastBufferConfig()

        self._samples: deque[BufferSample] = deque(
            maxlen=self.config.max_capacity_steps * 2,
        )
        self._steps_since_consolidation: int = 0
        self._total_steps: int = 0
        self._loss_history: deque[float] = deque(maxlen=500)
        self._is_full: bool = False

        self._checkpoint_dir = Path(self.config.checkpoint_dir)
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"FastBuffer initialized (capacity={self.config.max_capacity_steps}, "
            f"lr={self.config.learning_rate})"
        )

    @property
    def is_full(self) -> bool:
        """Check if buffer has reached capacity and needs consolidation."""
        return self._steps_since_consolidation >= self.config.max_capacity_steps

    @property
    def fill_level(self) -> float:
        """Current fill level as fraction of capacity."""
        return min(
            self._steps_since_consolidation / self.config.max_capacity_steps,
            1.0,
        )

    @property
    def num_samples(self) -> int:
        return len(self._samples)

    def get_lora_config(self) -> LoraConfig:
        """Get the high-plasticity LoRA config for the buffer adapter."""
        return LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

    # ------------------------------------------------------------------
    # Sample management
    # ------------------------------------------------------------------

    def add_sample(
        self,
        text: str,
        embedding: np.ndarray | None = None,
        loss: float = 0.0,
        domain_signal: str = "",
    ) -> None:
        """Add a sample to the buffer."""
        sample = BufferSample(
            text=text,
            embedding=embedding,
            loss=loss,
            domain_signal=domain_signal,
        )
        self._samples.append(sample)

    def get_samples(self, n: int | None = None) -> list[BufferSample]:
        """Get recent samples from the buffer."""
        samples = list(self._samples)
        if n is not None:
            samples = samples[-n:]
        return samples

    def get_training_texts(self, n: int | None = None) -> list[str]:
        """Get texts from buffer samples for training."""
        samples = self.get_samples(n)
        return [s.text for s in samples]

    # ------------------------------------------------------------------
    # Training step tracking
    # ------------------------------------------------------------------

    def record_step(self, loss: float) -> None:
        """Record a training step on the buffer adapter."""
        self._steps_since_consolidation += 1
        self._total_steps += 1
        self._loss_history.append(loss)

        if self.is_full:
            self._is_full = True
            logger.info(
                f"Buffer FULL ({self._steps_since_consolidation} steps). "
                "Consolidation recommended."
            )

    def reset(self) -> None:
        """Reset the buffer after consolidation.

        Clears samples and step counter. The buffer adapter should
        also be re-initialized externally.
        """
        n_samples = len(self._samples)
        self._samples.clear()
        self._steps_since_consolidation = 0
        self._is_full = False

        logger.info(
            f"Buffer reset: cleared {n_samples} samples, "
            f"step counter -> 0"
        )

    # ------------------------------------------------------------------
    # Distribution statistics
    # ------------------------------------------------------------------

    def get_loss_statistics(self) -> dict[str, float]:
        """Get loss statistics for meta-controller state."""
        if not self._loss_history:
            return {"mean": 0.0, "std": 0.0, "recent": 0.0, "trend": 0.0}

        losses = list(self._loss_history)
        recent = losses[-20:] if len(losses) >= 20 else losses
        older = losses[:-20] if len(losses) > 20 else losses

        return {
            "mean": float(np.mean(losses)),
            "std": float(np.std(losses)),
            "recent": float(np.mean(recent)),
            "trend": float(np.mean(recent) - np.mean(older)),
        }

    def detect_distribution_shift(self, window: int = 50) -> float:
        """Detect distribution shift from loss dynamics.

        Returns a shift magnitude (0 = no shift, higher = more shift).
        Used by the meta-controller (System 6) for change-point detection.
        """
        if len(self._loss_history) < window * 2:
            return 0.0

        losses = list(self._loss_history)
        old_window = losses[-(window * 2):-window]
        new_window = losses[-window:]

        old_mean = np.mean(old_window)
        new_mean = np.mean(new_window)
        old_std = np.std(old_window) + 1e-8

        shift = abs(new_mean - old_mean) / old_std
        return float(shift)

    # ------------------------------------------------------------------
    # Pattern separation noise
    # ------------------------------------------------------------------

    def apply_pattern_separation(
        self,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Apply pattern separation noise to buffer embeddings.

        Sparse, high-dimensional perturbations that minimize internal
        interference within the buffer. This is analogous to the
        hippocampal dentate gyrus pattern separation mechanism.
        """
        if self.config.pattern_separation_noise <= 0:
            return embeddings

        noise = torch.randn_like(embeddings) * self.config.pattern_separation_noise
        mask = (torch.rand_like(embeddings) < 0.1).float()
        return embeddings + noise * mask

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_state(self, path: str | Path | None = None) -> Path:
        """Save buffer metadata (samples are ephemeral)."""
        path = Path(path or self.config.checkpoint_dir) / "buffer_state.json"
        path.parent.mkdir(parents=True, exist_ok=True)

        import json
        state = {
            "steps_since_consolidation": self._steps_since_consolidation,
            "total_steps": self._total_steps,
            "num_samples": len(self._samples),
            "is_full": self._is_full,
            "loss_stats": self.get_loss_statistics(),
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

        return path

    def summary(self) -> str:
        stats = self.get_loss_statistics()
        return (
            f"FastBuffer: {self.num_samples} samples, "
            f"{self.fill_level:.0%} full, "
            f"loss={stats['recent']:.4f}, "
            f"shift={self.detect_distribution_shift():.2f}"
        )
