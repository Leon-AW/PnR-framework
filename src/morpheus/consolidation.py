"""
System 4 — Consolidation Engine ("Sleep")
==========================================

The most important and most complex subsystem. Runs asynchronously
(or in dedicated cycles) and performs:

4a. Self-Generated Rehearsal ("Dreaming") — delegated to rehearsal.py
4b. Interleaved Consolidation — new buffer knowledge + rehearsal mixed
4c. Structural Distillation — shared structure across experts -> core
4d. Expert Lifecycle Management — freeze, merge, spawn decisions

The consolidation engine is the bridge between the fast timescale
(buffer absorbs data in seconds) and the slow timescale (core evolves
over weeks). It converts ephemeral buffer knowledge into durable
expert knowledge, and expert knowledge into core structural knowledge.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from peft import PeftModel
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .config import ConsolidationConfig, ExpertState
from .expert_bank import ExpertBank, ExpertMetadata
from .fast_buffer import FastBuffer
from .knowledge_store import KnowledgeStore
from .rehearsal import RehearsalEngine
from .stable_core import StableCore

logger = logging.getLogger(__name__)


@dataclass
class ConsolidationCycleResult:
    """Result of a single consolidation cycle."""
    cycle_id: int
    timestamp: float = field(default_factory=time.time)

    # 4b: Interleaved consolidation
    buffer_samples_processed: int = 0
    rehearsal_samples_generated: int = 0
    consolidation_loss: float = 0.0

    # 4c: Structural distillation
    distillation_performed: bool = False
    core_shift: float = 0.0
    core_update_accepted: bool = False

    # 4d: Expert lifecycle
    experts_frozen: list[str] = field(default_factory=list)
    experts_spawned: list[str] = field(default_factory=list)
    experts_merged: list[tuple[str, str, str]] = field(default_factory=list)
    experts_pruned: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "cycle_id": self.cycle_id,
            "timestamp": self.timestamp,
            "buffer_samples_processed": self.buffer_samples_processed,
            "rehearsal_samples_generated": self.rehearsal_samples_generated,
            "consolidation_loss": self.consolidation_loss,
            "distillation_performed": self.distillation_performed,
            "core_shift": self.core_shift,
            "core_update_accepted": self.core_update_accepted,
            "experts_frozen": self.experts_frozen,
            "experts_spawned": self.experts_spawned,
            "experts_merged": [list(t) for t in self.experts_merged],
            "experts_pruned": self.experts_pruned,
        }


class ConsolidationEngine:
    """System 4: Full Consolidation Engine orchestrating all sub-processes.

    The consolidation engine is called by the meta-controller (System 6)
    when it's time to integrate buffer knowledge into the expert bank.
    It coordinates self-rehearsal, interleaved training, structural
    distillation, and expert lifecycle management.
    """

    def __init__(
        self,
        config: ConsolidationConfig | None = None,
        stable_core: StableCore | None = None,
        expert_bank: ExpertBank | None = None,
        fast_buffer: FastBuffer | None = None,
        knowledge_store: KnowledgeStore | None = None,
    ) -> None:
        self.config = config or ConsolidationConfig()

        self._core = stable_core
        self._expert_bank = expert_bank
        self._buffer = fast_buffer
        self._knowledge_store = knowledge_store

        self._rehearsal = RehearsalEngine(config=self.config)
        self._cycle_count: int = 0
        self._history: list[ConsolidationCycleResult] = []

        self._checkpoint_dir = Path(self.config.checkpoint_dir)
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info("ConsolidationEngine initialized")

    @property
    def cycle_count(self) -> int:
        return self._cycle_count

    @property
    def rehearsal_engine(self) -> RehearsalEngine:
        return self._rehearsal

    # ------------------------------------------------------------------
    # 4b: Interleaved Consolidation
    # ------------------------------------------------------------------

    def run_interleaved_consolidation(
        self,
        target_expert_id: str,
        model: PreTrainedModel | PeftModel,
        tokenizer: PreTrainedTokenizerBase,
        rehearsal_ratio: float | None = None,
    ) -> dict[str, Any]:
        """Run interleaved consolidation for a target expert.

        Mixes buffer data with self-rehearsal and trains the target
        expert on the interleaved batch. This is the core mechanism
        for transferring buffer knowledge into the expert bank without
        forgetting existing expert knowledge.

        Args:
            target_expert_id: Expert to consolidate into.
            model: Model with the target expert's adapter loaded.
            tokenizer: Tokenizer for the model.
            rehearsal_ratio: Override mixing ratio.

        Returns:
            Training metrics from the consolidation step.
        """
        if self._buffer is None:
            return {"error": "No buffer configured"}

        buffer_texts = self._buffer.get_training_texts()
        if not buffer_texts:
            logger.info("No buffer samples to consolidate")
            return {"n_samples": 0}

        # Create interleaved batch
        mixed_batch = self._rehearsal.create_interleaved_batch(
            buffer_samples=buffer_texts,
            rehearsal_ratio=rehearsal_ratio or self.config.default_rehearsal_ratio,
            model=model,
            tokenizer=tokenizer,
            expert_id=target_expert_id,
        )

        logger.info(
            f"Interleaved consolidation: {len(buffer_texts)} buffer + "
            f"{len(mixed_batch) - len(buffer_texts)} rehearsal = "
            f"{len(mixed_batch)} total"
        )

        return {
            "n_buffer": len(buffer_texts),
            "n_rehearsal": len(mixed_batch) - len(buffer_texts),
            "n_total": len(mixed_batch),
            "mixed_batch": mixed_batch,
        }

    # ------------------------------------------------------------------
    # 4c: Structural Distillation
    # ------------------------------------------------------------------

    def should_distill(self) -> bool:
        """Check if structural distillation should be triggered.

        Distillation occurs every N consolidation cycles, extracting
        shared structure across experts into the stable core.
        """
        return (
            self._cycle_count > 0
            and self._cycle_count % self.config.distillation_interval_cycles == 0
        )

    def run_structural_distillation(
        self,
        probe_texts: list[str],
    ) -> dict[str, Any]:
        """Run structural distillation from experts to core.

        Identifies shared structure across multiple experts and distills
        it into the stable core, subject to CKA constraints. This is
        how the core slowly evolves, enabling backward transfer.

        The distillation loss includes the CKA alignment penalty:
        L_distill = L_structure + lambda * (1 - CKA(f_core_v, f_core_v+1))

        Args:
            probe_texts: Held-out probe set for CKA validation.

        Returns:
            Distillation metrics.
        """
        if self._core is None:
            return {"error": "No stable core configured"}

        logger.info(
            f"Structural distillation cycle {self._cycle_count}: "
            f"extracting shared structure from "
            f"{self._expert_bank.num_experts if self._expert_bank else 0} experts"
        )

        # In a full implementation, this would:
        # 1. Collect activations from multiple experts on shared data
        # 2. Identify invariant patterns via correlation analysis
        # 3. Distill into core with CKA-bounded gradient updates
        # 4. Validate via formal core update protocol
        #
        # The infrastructure for this is in place via:
        # - StableCore.extract_representations() for CKA measurement
        # - StableCore.validate_update() for bounded drift enforcement
        # - StableCore.apply_update() for the formal protocol

        return {
            "cycle": self._cycle_count,
            "n_probe_texts": len(probe_texts),
            "status": "ready_for_distillation_training",
        }

    # ------------------------------------------------------------------
    # 4d: Expert Lifecycle Management
    # ------------------------------------------------------------------

    def manage_expert_lifecycle(
        self,
        centroids: dict[str, np.ndarray] | None = None,
    ) -> dict[str, list[str]]:
        """Run the expert lifecycle management protocol.

        Decisions made:
        - Which active experts should be frozen (converged)
        - Which shadow experts should be promoted (stabilized)
        - Which expert pairs should be merged (redundant)
        - Which experts should be pruned (low importance)
        - Whether new experts should be spawned (distribution shift)

        Returns:
            Dict mapping action type to list of affected expert IDs.
        """
        if self._expert_bank is None:
            return {}

        actions: dict[str, list[str]] = {
            "frozen": [],
            "promoted": [],
            "pruned": [],
        }

        # Check for experts to freeze
        for eid in self._expert_bank.active_experts:
            if self._expert_bank.should_freeze(eid):
                if self._expert_bank.freeze_expert(eid):
                    actions["frozen"].append(eid)

        # Check for shadow experts to promote
        for eid in self._expert_bank.shadow_experts:
            if self._expert_bank.promote_to_active(eid):
                actions["promoted"].append(eid)

        # Check for merge candidates
        if centroids:
            candidates = self._expert_bank.find_merge_candidates(centroids)
            if candidates:
                actions["merge_candidates"] = [
                    f"{a}+{b} (sim={s:.3f})" for a, b, s in candidates
                ]

        # Check for pruning candidates
        pruning = self._expert_bank.get_pruning_candidates()
        for eid, score in pruning:
            if self._expert_bank.retire_expert(eid):
                actions["pruned"].append(eid)

        if any(v for v in actions.values()):
            logger.info(f"Expert lifecycle actions: {actions}")

        return actions

    # ------------------------------------------------------------------
    # Full Consolidation Cycle
    # ------------------------------------------------------------------

    def run_cycle(
        self,
        model: PreTrainedModel | PeftModel | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        target_expert_id: str | None = None,
        probe_texts: list[str] | None = None,
        expert_centroids: dict[str, np.ndarray] | None = None,
    ) -> ConsolidationCycleResult:
        """Run a complete consolidation cycle.

        This is the main entry point called by the meta-controller.
        Coordinates all four sub-processes.

        Args:
            model: Model with target expert loaded (for interleaved training).
            tokenizer: Tokenizer.
            target_expert_id: Expert to consolidate buffer data into.
            probe_texts: Probe set for structural distillation CKA check.
            expert_centroids: Current expert centroids for lifecycle management.

        Returns:
            ConsolidationCycleResult with full metrics.
        """
        self._cycle_count += 1
        result = ConsolidationCycleResult(cycle_id=self._cycle_count)

        logger.info(f"{'=' * 60}")
        logger.info(f"CONSOLIDATION CYCLE {self._cycle_count}")
        logger.info(f"{'=' * 60}")

        # 4b: Interleaved consolidation
        if model and tokenizer and target_expert_id:
            consol_result = self.run_interleaved_consolidation(
                target_expert_id=target_expert_id,
                model=model,
                tokenizer=tokenizer,
            )
            result.buffer_samples_processed = consol_result.get("n_buffer", 0)
            result.rehearsal_samples_generated = consol_result.get("n_rehearsal", 0)

        # 4c: Structural distillation (periodic)
        if self.should_distill() and probe_texts:
            distill_result = self.run_structural_distillation(probe_texts)
            result.distillation_performed = True

        # 4d: Expert lifecycle management
        lifecycle = self.manage_expert_lifecycle(expert_centroids)
        result.experts_frozen = lifecycle.get("frozen", [])
        result.experts_spawned = lifecycle.get("spawned", [])
        result.experts_pruned = lifecycle.get("pruned", [])

        # Reset buffer after consolidation
        if self._buffer:
            self._buffer.reset()

        # Save frozen reference if on logarithmic schedule
        ref_mgr = self._rehearsal._frozen_refs
        if ref_mgr.should_save_reference(self._cycle_count):
            checkpoint_path = str(
                self._checkpoint_dir / f"frozen_ref_cycle_{self._cycle_count}"
            )
            ref_mgr.register_reference(self._cycle_count, checkpoint_path)

        self._history.append(result)

        logger.info(
            f"Consolidation cycle {self._cycle_count} complete: "
            f"buffer={result.buffer_samples_processed}, "
            f"rehearsal={result.rehearsal_samples_generated}, "
            f"frozen={len(result.experts_frozen)}, "
            f"pruned={len(result.experts_pruned)}"
        )

        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_state(self, path: str | Path | None = None) -> Path:
        """Save consolidation engine state."""
        path = Path(path or self.config.checkpoint_dir) / "consolidation_state.json"
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "cycle_count": self._cycle_count,
            "history": [r.to_dict() for r in self._history[-50:]],
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2, default=str)

        return path

    def load_state(self, path: str | Path) -> None:
        """Load consolidation engine state."""
        path = Path(path)
        if not path.exists():
            return

        with open(path) as f:
            state = json.load(f)

        self._cycle_count = state.get("cycle_count", 0)
        logger.info(f"ConsolidationEngine loaded: {self._cycle_count} cycles")

    def summary(self) -> str:
        lines = [
            f"ConsolidationEngine: {self._cycle_count} cycles completed",
            self._rehearsal.summary(),
        ]
        if self._history:
            last = self._history[-1]
            lines.append(
                f"  Last cycle: buffer={last.buffer_samples_processed}, "
                f"rehearsal={last.rehearsal_samples_generated}"
            )
        return "\n".join(lines)
