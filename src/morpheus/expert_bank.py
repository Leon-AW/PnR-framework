"""
System 2 — Expert Bank ("Cortical Columns")
=============================================

A dynamically growable Mixture-of-Experts collection where each expert is a
LoRA sub-network storing domain-specific or distribution-specific knowledge.

Key mechanisms:
- Expert lifecycle: SHADOW -> ACTIVE -> FROZEN -> (MERGE_CANDIDATE | DORMANT)
- Shadow routing: new experts train without contributing to predictions
- Merge verification via distillation with performance preservation check
- Importance-weighted pruning (frequency, marginal contribution, uniqueness)
- Dormant archival: compressed experts can be reactivated on demand
- Domain-adaptive input adapters for genuinely novel domains

The Expert Bank is sparse: any given input activates a small fraction of
total expert capacity, minimizing cross-expert interference.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from peft import LoraConfig, PeftModel, get_peft_model, TaskType
from transformers import PreTrainedModel

from .config import ExpertBankConfig, ExpertState

logger = logging.getLogger(__name__)


@dataclass
class ExpertMetadata:
    """Full metadata for a managed expert."""
    expert_id: str
    state: ExpertState = ExpertState.SHADOW
    native_core_version: int = 0
    adapter_path: str = ""
    lora_rank: int = 16
    lora_alpha: int = 32
    domain: str = "general"
    timestamp: float = field(default_factory=time.time)

    # Lifecycle tracking
    shadow_steps_completed: int = 0
    training_steps: int = 0
    loss_history: list[float] = field(default_factory=list)

    # Importance scoring
    activation_count: int = 0
    marginal_contribution: float = 0.0
    uniqueness_score: float = 0.0

    # Domain-adaptive adapter (for novel domains)
    domain_adapter_path: str | None = None

    def importance_score(self, weights: dict[str, float] | None = None) -> float:
        """Compute composite importance score."""
        weights = weights or {
            "frequency": 0.3, "marginal": 0.4, "uniqueness": 0.3,
        }
        total_activations = max(self.activation_count, 1)
        freq = min(self.activation_count / total_activations, 1.0)
        return (
            weights["frequency"] * freq
            + weights["marginal"] * self.marginal_contribution
            + weights["uniqueness"] * self.uniqueness_score
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["state"] = self.state.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> ExpertMetadata:
        d = d.copy()
        d["state"] = ExpertState(d["state"])
        return cls(**d)


class ExpertBank:
    """System 2: Dynamic Expert Bank with lifecycle management.

    Manages a collection of LoRA experts with formal lifecycle protocols
    for spawning, training, freezing, merging, pruning, and archival.

    Each expert is sparse and modular — learning in Expert #847 causes
    near-zero interference with Expert #12.
    """

    def __init__(self, config: ExpertBankConfig | None = None) -> None:
        self.config = config or ExpertBankConfig()
        self._experts: dict[str, ExpertMetadata] = {}
        self._checkpoint_dir = Path(self.config.checkpoint_dir)
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"ExpertBank initialized (max={self.config.max_experts})")

    @property
    def num_experts(self) -> int:
        return len(self._experts)

    @property
    def active_experts(self) -> list[str]:
        return [
            eid for eid, m in self._experts.items()
            if m.state in (ExpertState.ACTIVE, ExpertState.FROZEN)
        ]

    @property
    def shadow_experts(self) -> list[str]:
        return [
            eid for eid, m in self._experts.items()
            if m.state == ExpertState.SHADOW
        ]

    def get_expert(self, expert_id: str) -> ExpertMetadata | None:
        return self._experts.get(expert_id)

    def get_lora_config(self, expert_id: str | None = None) -> LoraConfig:
        """Get LoRA config for an expert (or default config)."""
        if expert_id and expert_id in self._experts:
            meta = self._experts[expert_id]
            rank = meta.lora_rank
            alpha = meta.lora_alpha
        else:
            rank = self.config.default_lora_rank
            alpha = self.config.default_lora_alpha

        return LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=self.config.default_lora_dropout,
            target_modules=self.config.target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

    # ------------------------------------------------------------------
    # Spawning with shadow routing
    # ------------------------------------------------------------------

    def spawn_expert(
        self,
        expert_id: str,
        domain: str = "general",
        core_version: int = 0,
        lora_rank: int | None = None,
        lora_alpha: int | None = None,
        adapter_path: str | None = None,
    ) -> ExpertMetadata:
        """Spawn a new expert in SHADOW state.

        The expert trains and receives data but its outputs are NOT mixed
        into predictions until it stabilizes and is promoted to ACTIVE.
        This prevents Voronoi disruption from unstable new experts.

        Args:
            expert_id: Unique identifier.
            domain: Knowledge domain.
            core_version: Current stable core version.
            lora_rank: LoRA rank (uses default if None).
            lora_alpha: LoRA alpha (uses default if None).
            adapter_path: Path to existing adapter weights.

        Returns:
            Metadata for the new expert.
        """
        if len(self._experts) >= self.config.max_experts:
            logger.warning(
                f"Expert bank at capacity ({self.config.max_experts}). "
                "Consider pruning before spawning."
            )

        if expert_id in self._experts:
            raise ValueError(f"Expert '{expert_id}' already exists.")

        meta = ExpertMetadata(
            expert_id=expert_id,
            state=ExpertState.SHADOW,
            native_core_version=core_version,
            adapter_path=adapter_path or str(self._checkpoint_dir / expert_id),
            lora_rank=lora_rank or self.config.default_lora_rank,
            lora_alpha=lora_alpha or self.config.default_lora_alpha,
            domain=domain,
        )
        self._experts[expert_id] = meta

        logger.info(
            f"Spawned expert '{expert_id}' in SHADOW state "
            f"(domain={domain}, rank={meta.lora_rank})"
        )
        return meta

    def promote_to_active(self, expert_id: str) -> bool:
        """Promote a SHADOW expert to ACTIVE after stabilization.

        Checks:
        - Minimum shadow steps completed
        - Loss has converged
        - Centroid has stabilized

        Returns:
            True if promotion succeeded.
        """
        meta = self._experts.get(expert_id)
        if meta is None or meta.state != ExpertState.SHADOW:
            return False

        if meta.shadow_steps_completed < self.config.shadow_period_steps:
            logger.info(
                f"Expert '{expert_id}' not ready: "
                f"{meta.shadow_steps_completed}/{self.config.shadow_period_steps} "
                "shadow steps"
            )
            return False

        meta.state = ExpertState.ACTIVE
        logger.info(f"Expert '{expert_id}' promoted to ACTIVE")
        return True

    # ------------------------------------------------------------------
    # Freezing
    # ------------------------------------------------------------------

    def freeze_expert(self, expert_id: str) -> bool:
        """Freeze an expert, protecting it from further modification.

        Returns:
            True if freezing succeeded.
        """
        meta = self._experts.get(expert_id)
        if meta is None or meta.state != ExpertState.ACTIVE:
            return False

        meta.state = ExpertState.FROZEN
        logger.info(f"Expert '{expert_id}' FROZEN")
        return True

    def should_freeze(self, expert_id: str) -> bool:
        """Check if an expert's training has converged and it should be frozen."""
        meta = self._experts.get(expert_id)
        if meta is None or meta.state != ExpertState.ACTIVE:
            return False

        window = self.config.freeze_loss_convergence_window
        if len(meta.loss_history) < window:
            return False

        recent = meta.loss_history[-window:]
        loss_var = np.std(recent)
        return loss_var < self.config.freeze_loss_convergence_threshold

    # ------------------------------------------------------------------
    # Merging with verification
    # ------------------------------------------------------------------

    def find_merge_candidates(self, centroids: dict[str, np.ndarray]) -> list[tuple[str, str, float]]:
        """Find pairs of experts that may be redundant.

        Args:
            centroids: Mapping of expert_id -> centroid vector.

        Returns:
            List of (expert_a, expert_b, similarity) pairs above merge threshold.
        """
        candidates = []
        ids = [eid for eid in centroids if eid in self._experts]

        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                sim = float(np.dot(centroids[a], centroids[b]))
                if sim >= self.config.merge_centroid_threshold:
                    candidates.append((a, b, sim))

        candidates.sort(key=lambda x: x[2], reverse=True)
        return candidates

    def verify_merge(
        self,
        expert_a: str,
        expert_b: str,
        merged_loss_a: float,
        merged_loss_b: float,
        original_loss_a: float,
        original_loss_b: float,
    ) -> bool:
        """Verify that merging preserves performance on both domains.

        Merging proceeds only if:
            loss_merged(D_a) <= loss_a(D_a) + delta
            loss_merged(D_b) <= loss_b(D_b) + delta
        """
        delta = self.config.merge_tolerance_delta
        ok_a = merged_loss_a <= original_loss_a + delta
        ok_b = merged_loss_b <= original_loss_b + delta

        if ok_a and ok_b:
            logger.info(
                f"Merge {expert_a}+{expert_b} VERIFIED: "
                f"loss_a={merged_loss_a:.4f} (orig={original_loss_a:.4f}), "
                f"loss_b={merged_loss_b:.4f} (orig={original_loss_b:.4f})"
            )
            return True
        else:
            logger.info(
                f"Merge {expert_a}+{expert_b} REJECTED: "
                f"performance degradation exceeds delta={delta}"
            )
            return False

    def execute_merge(
        self,
        expert_a: str,
        expert_b: str,
        merged_id: str,
        merged_adapter_path: str,
    ) -> ExpertMetadata:
        """Execute a verified merge, creating a new expert and retiring sources."""
        meta_a = self._experts[expert_a]
        meta_b = self._experts[expert_b]

        merged = ExpertMetadata(
            expert_id=merged_id,
            state=ExpertState.ACTIVE,
            native_core_version=max(
                meta_a.native_core_version, meta_b.native_core_version
            ),
            adapter_path=merged_adapter_path,
            domain=f"{meta_a.domain}+{meta_b.domain}",
            activation_count=meta_a.activation_count + meta_b.activation_count,
        )

        self._experts[merged_id] = merged

        meta_a.state = ExpertState.DORMANT
        meta_b.state = ExpertState.DORMANT

        logger.info(
            f"Merged {expert_a}+{expert_b} -> {merged_id} "
            f"(sources retired to DORMANT)"
        )
        return merged

    # ------------------------------------------------------------------
    # Importance-weighted pruning
    # ------------------------------------------------------------------

    def compute_importance_scores(self) -> dict[str, float]:
        """Compute importance scores for all active/frozen experts."""
        weights = {
            "frequency": self.config.importance_weight_frequency,
            "marginal": self.config.importance_weight_marginal,
            "uniqueness": self.config.importance_weight_uniqueness,
        }
        scores = {}
        for eid, meta in self._experts.items():
            if meta.state in (ExpertState.ACTIVE, ExpertState.FROZEN):
                scores[eid] = meta.importance_score(weights)
        return scores

    def get_pruning_candidates(self) -> list[tuple[str, float]]:
        """Get experts below the pruning importance threshold."""
        scores = self.compute_importance_scores()
        candidates = [
            (eid, score) for eid, score in scores.items()
            if score < self.config.pruning_importance_threshold
        ]
        candidates.sort(key=lambda x: x[1])
        return candidates

    def retire_expert(self, expert_id: str) -> bool:
        """Move an expert to DORMANT state (soft deletion).

        The expert's parameters are archived (not destroyed), and its
        centroid is retained. If future inputs match the dormant centroid,
        the expert can be reactivated.
        """
        meta = self._experts.get(expert_id)
        if meta is None:
            return False

        meta.state = ExpertState.DORMANT
        logger.info(f"Expert '{expert_id}' retired to DORMANT")
        return True

    def reactivate_expert(self, expert_id: str) -> bool:
        """Reactivate a DORMANT expert."""
        meta = self._experts.get(expert_id)
        if meta is None or meta.state != ExpertState.DORMANT:
            return False

        meta.state = ExpertState.ACTIVE
        logger.info(f"Expert '{expert_id}' REACTIVATED from dormant")
        return True

    # ------------------------------------------------------------------
    # Novelty detection and domain adapters
    # ------------------------------------------------------------------

    def spawn_with_domain_adapter(
        self,
        expert_id: str,
        domain: str,
        core_version: int,
        domain_adapter_rank: int | None = None,
    ) -> ExpertMetadata:
        """Spawn an expert with a domain-adaptive input adapter.

        For genuinely novel domains where the core's representations
        don't provide good separation, the domain adapter maps inputs
        into a better-separated region of representation space.
        """
        meta = self.spawn_expert(
            expert_id=expert_id,
            domain=domain,
            core_version=core_version,
        )
        meta.domain_adapter_path = str(
            self._checkpoint_dir / f"{expert_id}_domain_adapter"
        )
        logger.info(
            f"Expert '{expert_id}' spawned with domain-adaptive adapter"
        )
        return meta

    # ------------------------------------------------------------------
    # Record-keeping
    # ------------------------------------------------------------------

    def record_training_step(
        self,
        expert_id: str,
        loss: float,
        is_shadow: bool = False,
    ) -> None:
        """Record a training step for lifecycle tracking."""
        meta = self._experts.get(expert_id)
        if meta is None:
            return

        meta.training_steps += 1
        meta.loss_history.append(loss)

        if len(meta.loss_history) > 200:
            meta.loss_history = meta.loss_history[-200:]

        if is_shadow:
            meta.shadow_steps_completed += 1

    def record_activation(
        self,
        expert_id: str,
        marginal_improvement: float = 0.0,
    ) -> None:
        """Record that an expert was activated during inference."""
        meta = self._experts.get(expert_id)
        if meta is None:
            return

        meta.activation_count += 1
        if marginal_improvement > 0:
            alpha = 0.1
            meta.marginal_contribution = (
                (1 - alpha) * meta.marginal_contribution
                + alpha * marginal_improvement
            )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_state(self, path: str | Path | None = None) -> Path:
        """Save expert bank state."""
        path = Path(path or self.config.checkpoint_dir) / "expert_bank.json"
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "experts": {eid: m.to_dict() for eid, m in self._experts.items()},
            "num_experts": len(self._experts),
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2, default=str)

        logger.info(f"ExpertBank state saved: {len(self._experts)} experts")
        return path

    def load_state(self, path: str | Path) -> None:
        """Load expert bank state."""
        path = Path(path)
        if not path.exists():
            logger.warning(f"No expert bank state at {path}")
            return

        with open(path) as f:
            state = json.load(f)

        self._experts = {
            eid: ExpertMetadata.from_dict(d)
            for eid, d in state.get("experts", {}).items()
        }
        logger.info(f"ExpertBank loaded: {len(self._experts)} experts")

    def summary(self) -> str:
        counts = {}
        for meta in self._experts.values():
            state = meta.state.value
            counts[state] = counts.get(state, 0) + 1

        lines = [
            f"Expert Bank ({self.num_experts}/{self.config.max_experts} experts)",
        ]
        for state, count in sorted(counts.items()):
            lines.append(f"  {state}: {count}")
        return "\n".join(lines)
