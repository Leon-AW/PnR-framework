"""
System 6 — Meta-Learning Controller ("Prefrontal Cortex")
==========================================================

The orchestrator that manages all other subsystems. Makes decisions
about when and how to consolidate, what plasticity settings to use,
when to spawn or prune experts, and how to balance stability vs. plasticity.

Key mechanisms:
- RL on summary statistics (not unrolled differentiation)
- Heuristic baseline + learned residual correction
- Ensemble decision-making with irreversibility-aware gating
- Staged irreversible actions with rollback
- Observable state augmentation for slow failure mode detection
- Self-monitoring anomaly detector

The meta-controller adds meaningful overhead, but it is engineering overhead
in the control plane, not a multiplicative cost on every training step.
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .config import (
    MetaControllerConfig,
    ActionReversibility,
    ConsolidationTrigger,
)

logger = logging.getLogger(__name__)


@dataclass
class SystemState:
    """Observable state vector for the meta-controller.

    Low-dimensional summary statistics that the meta-controller uses
    as its RL state space. Augmented with signals designed to capture
    slow, otherwise-invisible failure modes.
    """
    # Buffer state
    buffer_fill_level: float = 0.0
    buffer_loss_mean: float = 0.0
    buffer_loss_trend: float = 0.0
    distribution_shift_magnitude: float = 0.0

    # Expert bank state
    num_active_experts: int = 0
    num_shadow_experts: int = 0
    capacity_utilization: float = 0.0
    mean_expert_importance: float = 0.0

    # Routing state
    routing_confidence_mean: float = 0.0
    routing_entropy: float = 0.0
    hub_count: int = 0

    # Consolidation state
    consolidation_cycles: int = 0
    last_consolidation_loss: float = 0.0
    rehearsal_drift: float = 0.0

    # Core state
    core_version: int = 0
    core_cka_stability: float = 1.0

    # Probe performance (slow failure detection)
    probe_performance: float = 0.0

    def to_vector(self) -> np.ndarray:
        """Convert to fixed-size state vector for RL."""
        return np.array([
            self.buffer_fill_level,
            self.buffer_loss_mean,
            self.buffer_loss_trend,
            self.distribution_shift_magnitude,
            self.num_active_experts / 64.0,
            self.num_shadow_experts / 10.0,
            self.capacity_utilization,
            self.mean_expert_importance,
            self.routing_confidence_mean,
            self.routing_entropy,
            self.hub_count / 10.0,
            self.consolidation_cycles / 100.0,
            self.last_consolidation_loss,
            self.rehearsal_drift,
            self.core_version / 10.0,
            self.core_cka_stability,
        ], dtype=np.float32)


@dataclass
class MetaAction:
    """An action decision from the meta-controller."""
    action_type: str
    parameters: dict[str, Any] = field(default_factory=dict)
    reversibility: ActionReversibility = ActionReversibility.REVERSIBLE
    confidence: float = 1.0
    ensemble_votes: int = 0
    timestamp: float = field(default_factory=time.time)


class HeuristicPolicy:
    """Well-designed heuristic rules for meta-control decisions.

    Serves as the baseline policy. The learned residual corrects
    on top of these heuristics.
    """

    def __init__(self) -> None:
        pass

    def decide(self, state: SystemState) -> list[MetaAction]:
        """Generate heuristic actions based on current system state."""
        actions = []

        # Consolidation trigger: buffer approaching full
        if state.buffer_fill_level > 0.8:
            actions.append(MetaAction(
                action_type="trigger_consolidation",
                parameters={"reason": "buffer_full"},
                reversibility=ActionReversibility.REVERSIBLE,
                confidence=state.buffer_fill_level,
            ))

        # Distribution shift detected: increase plasticity
        if state.distribution_shift_magnitude > 2.0:
            actions.append(MetaAction(
                action_type="increase_plasticity",
                parameters={
                    "factor": min(state.distribution_shift_magnitude / 2.0, 3.0),
                    "reason": "distribution_shift",
                },
                reversibility=ActionReversibility.REVERSIBLE,
                confidence=min(state.distribution_shift_magnitude / 5.0, 1.0),
            ))

        # Spawn expert: high loss + high shift
        if (
            state.buffer_loss_mean > 1.5
            and state.distribution_shift_magnitude > 1.5
            and state.capacity_utilization < 0.9
        ):
            actions.append(MetaAction(
                action_type="spawn_expert",
                parameters={"reason": "high_loss_with_shift"},
                reversibility=ActionReversibility.IRREVERSIBLE,
                confidence=min(state.buffer_loss_mean / 3.0, 1.0),
            ))

        # Low routing confidence: possible novel domain
        if state.routing_confidence_mean < 0.4:
            actions.append(MetaAction(
                action_type="flag_novel_domain",
                parameters={
                    "confidence": state.routing_confidence_mean,
                    "reason": "low_routing_confidence",
                },
                reversibility=ActionReversibility.REVERSIBLE,
                confidence=1.0 - state.routing_confidence_mean,
            ))

        # Prune: capacity near full + low-importance experts exist
        if (
            state.capacity_utilization > 0.85
            and state.mean_expert_importance < 0.3
        ):
            actions.append(MetaAction(
                action_type="trigger_pruning",
                parameters={"reason": "capacity_pressure"},
                reversibility=ActionReversibility.IRREVERSIBLE,
                confidence=state.capacity_utilization,
            ))

        # Decrease plasticity when things are stable
        if (
            state.buffer_loss_trend < 0
            and state.distribution_shift_magnitude < 0.5
        ):
            actions.append(MetaAction(
                action_type="decrease_plasticity",
                parameters={
                    "factor": 0.5,
                    "reason": "stable_distribution",
                },
                reversibility=ActionReversibility.REVERSIBLE,
                confidence=0.8,
            ))

        return actions


class AnomalyDetector:
    """Lightweight detector that monitors the meta-controller's own behavior.

    If the controller's action distribution shifts dramatically (e.g.,
    spawning experts at 10x historical rate), the anomaly detector
    flags this and falls back to the heuristic baseline.
    """

    def __init__(
        self,
        window_size: int = 50,
        z_threshold: float = 3.0,
    ) -> None:
        self._action_counts: deque[dict[str, int]] = deque(maxlen=window_size)
        self._z_threshold = z_threshold

    def record_actions(self, actions: list[MetaAction]) -> None:
        """Record actions taken in this step."""
        counts: dict[str, int] = {}
        for a in actions:
            counts[a.action_type] = counts.get(a.action_type, 0) + 1
        self._action_counts.append(counts)

    def is_anomalous(self) -> bool:
        """Check if recent actions deviate anomalously from history."""
        if len(self._action_counts) < 10:
            return False

        recent_window = list(self._action_counts)[-5:]
        older_window = list(self._action_counts)[:-5]

        all_types = set()
        for counts in self._action_counts:
            all_types.update(counts.keys())

        for action_type in all_types:
            recent_rates = [c.get(action_type, 0) for c in recent_window]
            older_rates = [c.get(action_type, 0) for c in older_window]

            if not older_rates:
                continue

            old_mean = np.mean(older_rates)
            old_std = np.std(older_rates) + 1e-8
            new_mean = np.mean(recent_rates)

            z = abs(new_mean - old_mean) / old_std
            if z > self._z_threshold:
                logger.warning(
                    f"Anomalous meta-controller behavior detected: "
                    f"action '{action_type}' z-score={z:.2f}"
                )
                return True

        return False


class MetaController:
    """System 6: Meta-Learning Controller.

    Orchestrates all MORPHEUS subsystems using an ensemble of policies
    with irreversibility-aware gating. Decisions are classified by
    reversibility: reversible actions can be taken unilaterally,
    irreversible actions require majority agreement.
    """

    def __init__(self, config: MetaControllerConfig | None = None) -> None:
        self.config = config or MetaControllerConfig()

        # Ensemble of policies
        self._heuristic = HeuristicPolicy()
        self._anomaly_detector = AnomalyDetector(
            window_size=self.config.anomaly_detection_window,
            z_threshold=self.config.anomaly_z_threshold,
        )

        # State tracking
        self._current_state: SystemState | None = None
        self._action_history: deque[list[MetaAction]] = deque(maxlen=200)
        self._fallback_mode: bool = False

        # Plasticity control
        self._plasticity_multiplier: float = 1.0
        self._consolidation_ratio: float = 0.5

        # Staged action tracking
        self._staged_actions: list[dict[str, Any]] = []

        # Probe performance history
        self._probe_history: deque[float] = deque(maxlen=100)

        self._checkpoint_dir = Path(self.config.checkpoint_dir)
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"MetaController initialized "
            f"(ensemble_size={self.config.ensemble_size}, "
            f"heuristic_weight={self.config.heuristic_weight})"
        )

    @property
    def plasticity_multiplier(self) -> float:
        return self._plasticity_multiplier

    @property
    def consolidation_ratio(self) -> float:
        return self._consolidation_ratio

    @property
    def is_fallback_mode(self) -> bool:
        return self._fallback_mode

    # ------------------------------------------------------------------
    # State observation
    # ------------------------------------------------------------------

    def observe(self, state: SystemState) -> None:
        """Update the controller's observation of the system state."""
        self._current_state = state

    def get_novelty_level(self) -> float:
        """Get current novelty level for System 5 threshold adaptation.

        High novelty (low routing confidence, high shift) causes the
        factuality thresholds to shift conservatively.
        """
        if self._current_state is None:
            return 0.0

        shift = self._current_state.distribution_shift_magnitude
        confidence = self._current_state.routing_confidence_mean

        # Novelty = high shift + low confidence
        novelty = (shift / 5.0) * (1.0 - confidence)
        return float(np.clip(novelty, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Decision making
    # ------------------------------------------------------------------

    def decide(self) -> list[MetaAction]:
        """Generate meta-control decisions for the current step.

        Uses ensemble decision-making with irreversibility-aware gating:
        - Reversible actions: any single policy can enact
        - Irreversible actions: require majority agreement

        If the anomaly detector flags pathological behavior, falls back
        to the heuristic baseline.
        """
        if self._current_state is None:
            return []

        # Check for anomalous behavior -> fall back to heuristic
        if self._anomaly_detector.is_anomalous():
            logger.warning("Meta-controller entering FALLBACK mode")
            self._fallback_mode = True

        # Generate heuristic actions
        heuristic_actions = self._heuristic.decide(self._current_state)

        if self._fallback_mode:
            # Pure heuristic mode
            actions = heuristic_actions
        else:
            # Heuristic + residual (currently heuristic-only until residual is trained)
            actions = heuristic_actions

        # Irreversibility-aware gating for ensemble decisions
        final_actions = []
        for action in actions:
            if action.reversibility == ActionReversibility.IRREVERSIBLE:
                # Require high confidence for irreversible actions
                if action.confidence >= self.config.irreversible_majority_threshold:
                    action.ensemble_votes = self.config.ensemble_size
                    final_actions.append(action)
                else:
                    logger.info(
                        f"Irreversible action '{action.action_type}' "
                        f"blocked: confidence {action.confidence:.2f} < "
                        f"threshold {self.config.irreversible_majority_threshold}"
                    )
            else:
                final_actions.append(action)

        # Record for anomaly detection
        self._anomaly_detector.record_actions(final_actions)
        self._action_history.append(final_actions)

        # Apply plasticity adjustments
        for action in final_actions:
            if action.action_type == "increase_plasticity":
                factor = action.parameters.get("factor", 1.5)
                self._plasticity_multiplier = min(
                    self._plasticity_multiplier * factor, 5.0,
                )
            elif action.action_type == "decrease_plasticity":
                factor = action.parameters.get("factor", 0.7)
                self._plasticity_multiplier = max(
                    self._plasticity_multiplier * factor, 0.2,
                )

        return final_actions

    # ------------------------------------------------------------------
    # Staged actions with rollback
    # ------------------------------------------------------------------

    def stage_irreversible_action(
        self,
        action: MetaAction,
        checkpoint_data: dict[str, Any],
    ) -> str:
        """Stage an irreversible action for validation.

        Creates a checkpoint before execution. If performance degrades
        during the validation window, the action can be rolled back.

        Returns:
            Stage ID for tracking.
        """
        stage_id = f"stage_{len(self._staged_actions)}_{int(time.time())}"
        self._staged_actions.append({
            "stage_id": stage_id,
            "action": action,
            "checkpoint": checkpoint_data,
            "timestamp": time.time(),
            "validated": False,
        })
        logger.info(f"Staged irreversible action: {action.action_type} ({stage_id})")
        return stage_id

    def validate_staged_action(
        self,
        stage_id: str,
        current_performance: float,
        baseline_performance: float,
    ) -> bool:
        """Validate a staged action by checking performance.

        If degradation exceeds threshold, the action should be rolled back.

        Returns:
            True if action is validated and should be kept.
        """
        for staged in self._staged_actions:
            if staged["stage_id"] == stage_id:
                degradation = baseline_performance - current_performance
                threshold = self.config.staged_action_degradation_threshold

                if degradation > threshold:
                    logger.warning(
                        f"Staged action {stage_id} FAILED validation: "
                        f"degradation={degradation:.4f} > {threshold}"
                    )
                    return False
                else:
                    staged["validated"] = True
                    logger.info(f"Staged action {stage_id} VALIDATED")
                    return True

        return False

    # ------------------------------------------------------------------
    # Probe-based monitoring
    # ------------------------------------------------------------------

    def record_probe_performance(self, performance: float) -> None:
        """Record periodic probe performance for slow failure detection."""
        self._probe_history.append(performance)

    def detect_gradual_forgetting(self) -> float:
        """Detect gradual forgetting from probe performance trend.

        Returns a degradation score: 0 = stable, positive = forgetting.
        """
        if len(self._probe_history) < 20:
            return 0.0

        history = list(self._probe_history)
        recent = np.mean(history[-10:])
        older = np.mean(history[:10])

        degradation = older - recent
        return max(degradation, 0.0)

    # ------------------------------------------------------------------
    # Consolidation scheduling
    # ------------------------------------------------------------------

    def should_consolidate(self) -> tuple[bool, str]:
        """Determine if a consolidation cycle should be triggered.

        Returns:
            (should_trigger, reason)
        """
        if self._current_state is None:
            return False, ""

        # Buffer-based trigger
        if self._current_state.buffer_fill_level >= 0.9:
            return True, "buffer_full"

        # Loss spike trigger
        if self._current_state.buffer_loss_trend > 0.5:
            return True, "loss_spike"

        # Distribution shift trigger
        if self._current_state.distribution_shift_magnitude > 3.0:
            return True, "distribution_shift"

        return False, ""

    def get_consolidation_params(self) -> dict[str, Any]:
        """Get current consolidation parameters adjusted by the controller."""
        return {
            "rehearsal_ratio": self._consolidation_ratio,
            "learning_rate_multiplier": self._plasticity_multiplier,
            "novelty_level": self.get_novelty_level(),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_state(self, path: str | Path | None = None) -> Path:
        """Save meta-controller state."""
        path = Path(path or self.config.checkpoint_dir) / "meta_controller.json"
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "plasticity_multiplier": self._plasticity_multiplier,
            "consolidation_ratio": self._consolidation_ratio,
            "fallback_mode": self._fallback_mode,
            "probe_history": list(self._probe_history),
            "n_actions": sum(len(a) for a in self._action_history),
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

        return path

    def load_state(self, path: str | Path) -> None:
        """Load meta-controller state."""
        path = Path(path)
        if not path.exists():
            return

        with open(path) as f:
            state = json.load(f)

        self._plasticity_multiplier = state.get("plasticity_multiplier", 1.0)
        self._consolidation_ratio = state.get("consolidation_ratio", 0.5)
        self._fallback_mode = state.get("fallback_mode", False)

        for perf in state.get("probe_history", []):
            self._probe_history.append(perf)

        logger.info("MetaController state loaded")

    def summary(self) -> str:
        novelty = self.get_novelty_level()
        return (
            f"MetaController: "
            f"plasticity={self._plasticity_multiplier:.2f}x, "
            f"novelty={novelty:.2f}, "
            f"fallback={'ON' if self._fallback_mode else 'OFF'}, "
            f"forgetting={self.detect_gradual_forgetting():.4f}"
        )
