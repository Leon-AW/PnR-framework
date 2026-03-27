"""
Unit Tests — Meta-Controller (System 6)
=========================================

Tests for the meta-learning controller:
- Heuristic policy decision rules
- Anomaly detector for pathological behavior
- Ensemble irreversibility-aware gating
- Staged actions with rollback validation
- Novelty level computation
- Consolidation scheduling
- Plasticity control
- Probe-based gradual forgetting detection
"""

import pytest
import numpy as np

from src.morpheus.meta_controller import (
    MetaController,
    SystemState,
    HeuristicPolicy,
    AnomalyDetector,
    MetaAction,
)
from src.morpheus.config import MetaControllerConfig, ActionReversibility


class TestSystemState:
    """Tests for the SystemState observation dataclass."""

    def test_to_vector_shape(self):
        state = SystemState()
        vec = state.to_vector()
        assert vec.shape == (16,)
        assert vec.dtype == np.float32

    def test_to_vector_values(self):
        state = SystemState(
            buffer_fill_level=0.5,
            num_active_experts=10,
            core_version=3,
        )
        vec = state.to_vector()
        assert vec[0] == pytest.approx(0.5)
        assert vec[4] == pytest.approx(10 / 64.0)


class TestHeuristicPolicy:
    """Tests for the well-designed heuristic baseline."""

    def test_buffer_full_triggers_consolidation(self):
        policy = HeuristicPolicy()
        state = SystemState(buffer_fill_level=0.9)
        actions = policy.decide(state)
        types = [a.action_type for a in actions]
        assert "trigger_consolidation" in types

    def test_distribution_shift_increases_plasticity(self):
        policy = HeuristicPolicy()
        state = SystemState(distribution_shift_magnitude=3.0)
        actions = policy.decide(state)
        types = [a.action_type for a in actions]
        assert "increase_plasticity" in types

    def test_high_loss_with_shift_spawns_expert(self):
        policy = HeuristicPolicy()
        state = SystemState(
            buffer_loss_mean=2.0,
            distribution_shift_magnitude=2.0,
            capacity_utilization=0.5,
        )
        actions = policy.decide(state)
        types = [a.action_type for a in actions]
        assert "spawn_expert" in types

    def test_spawn_expert_is_irreversible(self):
        policy = HeuristicPolicy()
        state = SystemState(
            buffer_loss_mean=2.0,
            distribution_shift_magnitude=2.0,
            capacity_utilization=0.5,
        )
        actions = policy.decide(state)
        spawn = [a for a in actions if a.action_type == "spawn_expert"]
        assert spawn[0].reversibility == ActionReversibility.IRREVERSIBLE

    def test_low_routing_confidence_flags_novel_domain(self):
        policy = HeuristicPolicy()
        state = SystemState(routing_confidence_mean=0.2)
        actions = policy.decide(state)
        types = [a.action_type for a in actions]
        assert "flag_novel_domain" in types

    def test_capacity_pressure_triggers_pruning(self):
        policy = HeuristicPolicy()
        state = SystemState(
            capacity_utilization=0.9,
            mean_expert_importance=0.1,
        )
        actions = policy.decide(state)
        types = [a.action_type for a in actions]
        assert "trigger_pruning" in types

    def test_stable_distribution_decreases_plasticity(self):
        policy = HeuristicPolicy()
        state = SystemState(
            buffer_loss_trend=-0.1,
            distribution_shift_magnitude=0.1,
        )
        actions = policy.decide(state)
        types = [a.action_type for a in actions]
        assert "decrease_plasticity" in types

    def test_neutral_state_no_actions(self):
        policy = HeuristicPolicy()
        state = SystemState(
            buffer_fill_level=0.3,
            buffer_loss_trend=0.01,
            distribution_shift_magnitude=0.5,
            routing_confidence_mean=0.7,
            capacity_utilization=0.4,
        )
        actions = policy.decide(state)
        assert len(actions) == 0


class TestAnomalyDetector:
    """Tests for the self-monitoring anomaly detector."""

    def test_no_anomaly_with_few_samples(self):
        detector = AnomalyDetector(window_size=50, z_threshold=3.0)
        for _ in range(5):
            detector.record_actions([MetaAction(action_type="trigger_consolidation")])
        assert not detector.is_anomalous()

    def test_anomaly_detected_on_sudden_spike(self):
        detector = AnomalyDetector(window_size=50, z_threshold=2.0)

        for _ in range(20):
            detector.record_actions([MetaAction(action_type="trigger_consolidation")])

        for _ in range(5):
            detector.record_actions([
                MetaAction(action_type="spawn_expert"),
                MetaAction(action_type="spawn_expert"),
                MetaAction(action_type="spawn_expert"),
                MetaAction(action_type="spawn_expert"),
                MetaAction(action_type="spawn_expert"),
            ])

        assert detector.is_anomalous()

    def test_stable_behavior_no_anomaly(self):
        detector = AnomalyDetector(window_size=50, z_threshold=3.0)
        for _ in range(30):
            detector.record_actions([MetaAction(action_type="decrease_plasticity")])
        assert not detector.is_anomalous()


class TestMetaController:
    """Tests for the full meta-controller."""

    def test_observe_and_decide(self):
        mc = MetaController(MetaControllerConfig(checkpoint_dir="/tmp/test_mc"))
        state = SystemState(buffer_fill_level=0.95)
        mc.observe(state)
        actions = mc.decide()
        types = [a.action_type for a in actions]
        assert "trigger_consolidation" in types

    def test_irreversible_action_blocked_below_threshold(self):
        config = MetaControllerConfig(
            irreversible_majority_threshold=0.99,
            checkpoint_dir="/tmp/test_mc",
        )
        mc = MetaController(config)
        state = SystemState(
            buffer_loss_mean=2.0,
            distribution_shift_magnitude=2.0,
            capacity_utilization=0.5,
        )
        mc.observe(state)
        actions = mc.decide()
        spawn_actions = [a for a in actions if a.action_type == "spawn_expert"]
        # The heuristic gives confidence = min(2.0/3.0, 1.0) ≈ 0.67 < 0.99
        assert len(spawn_actions) == 0

    def test_plasticity_increases_on_shift(self):
        mc = MetaController(MetaControllerConfig(checkpoint_dir="/tmp/test_mc"))
        initial = mc.plasticity_multiplier
        state = SystemState(distribution_shift_magnitude=3.0)
        mc.observe(state)
        mc.decide()
        assert mc.plasticity_multiplier > initial

    def test_plasticity_decreases_on_stability(self):
        mc = MetaController(MetaControllerConfig(checkpoint_dir="/tmp/test_mc"))
        initial = mc.plasticity_multiplier
        state = SystemState(
            buffer_loss_trend=-0.1,
            distribution_shift_magnitude=0.1,
        )
        mc.observe(state)
        mc.decide()
        assert mc.plasticity_multiplier < initial

    def test_plasticity_bounded(self):
        mc = MetaController(MetaControllerConfig(checkpoint_dir="/tmp/test_mc"))
        for _ in range(50):
            state = SystemState(distribution_shift_magnitude=5.0)
            mc.observe(state)
            mc.decide()
        assert mc.plasticity_multiplier <= 5.0

    def test_fallback_mode_on_anomaly(self):
        mc = MetaController(MetaControllerConfig(
            anomaly_detection_window=50,
            anomaly_z_threshold=2.0,
            checkpoint_dir="/tmp/test_mc",
        ))
        for _ in range(20):
            state = SystemState(buffer_fill_level=0.5)
            mc.observe(state)
            mc.decide()

        for _ in range(6):
            state = SystemState(
                buffer_loss_mean=10.0,
                distribution_shift_magnitude=10.0,
                capacity_utilization=0.1,
                buffer_fill_level=1.0,
            )
            mc.observe(state)
            mc.decide()

        assert mc.is_fallback_mode


class TestNoveltyLevel:
    """Tests for novelty level computation."""

    def test_no_novelty_when_stable(self):
        mc = MetaController(MetaControllerConfig(checkpoint_dir="/tmp/test_mc"))
        state = SystemState(
            distribution_shift_magnitude=0.0,
            routing_confidence_mean=0.9,
        )
        mc.observe(state)
        assert mc.get_novelty_level() == pytest.approx(0.0, abs=0.01)

    def test_high_novelty_on_shift_and_low_confidence(self):
        mc = MetaController(MetaControllerConfig(checkpoint_dir="/tmp/test_mc"))
        state = SystemState(
            distribution_shift_magnitude=5.0,
            routing_confidence_mean=0.0,
        )
        mc.observe(state)
        assert mc.get_novelty_level() > 0.5

    def test_novelty_without_observation_is_zero(self):
        mc = MetaController(MetaControllerConfig(checkpoint_dir="/tmp/test_mc"))
        assert mc.get_novelty_level() == 0.0


class TestConsolidationScheduling:
    """Tests for consolidation decision logic."""

    def test_should_consolidate_buffer_full(self):
        mc = MetaController(MetaControllerConfig(checkpoint_dir="/tmp/test_mc"))
        mc.observe(SystemState(buffer_fill_level=0.95))
        should, reason = mc.should_consolidate()
        assert should
        assert reason == "buffer_full"

    def test_should_consolidate_loss_spike(self):
        mc = MetaController(MetaControllerConfig(checkpoint_dir="/tmp/test_mc"))
        mc.observe(SystemState(buffer_loss_trend=1.0))
        should, reason = mc.should_consolidate()
        assert should
        assert reason == "loss_spike"

    def test_should_not_consolidate_stable(self):
        mc = MetaController(MetaControllerConfig(checkpoint_dir="/tmp/test_mc"))
        mc.observe(SystemState(buffer_fill_level=0.3, buffer_loss_trend=0.0))
        should, _ = mc.should_consolidate()
        assert not should


class TestStagedActions:
    """Tests for staged irreversible actions with rollback."""

    def test_stage_and_validate_success(self):
        mc = MetaController(MetaControllerConfig(checkpoint_dir="/tmp/test_mc"))
        action = MetaAction(action_type="prune_expert")
        stage_id = mc.stage_irreversible_action(action, checkpoint_data={"expert": "exp_1"})

        result = mc.validate_staged_action(stage_id, current_performance=0.85, baseline_performance=0.86)
        assert result is True

    def test_stage_and_validate_failure(self):
        config = MetaControllerConfig(
            staged_action_degradation_threshold=0.01,
            checkpoint_dir="/tmp/test_mc",
        )
        mc = MetaController(config)
        action = MetaAction(action_type="prune_expert")
        stage_id = mc.stage_irreversible_action(action, checkpoint_data={})

        result = mc.validate_staged_action(stage_id, current_performance=0.50, baseline_performance=0.90)
        assert result is False


class TestGradualForgettingDetection:
    """Tests for probe-based slow failure mode detection."""

    def test_no_forgetting_with_stable_probes(self):
        mc = MetaController(MetaControllerConfig(checkpoint_dir="/tmp/test_mc"))
        for _ in range(30):
            mc.record_probe_performance(0.85)
        assert mc.detect_gradual_forgetting() < 0.01

    def test_forgetting_detected_on_declining_probes(self):
        mc = MetaController(MetaControllerConfig(checkpoint_dir="/tmp/test_mc"))
        for perf in np.linspace(0.9, 0.6, 30):
            mc.record_probe_performance(float(perf))
        degradation = mc.detect_gradual_forgetting()
        assert degradation > 0.1

    def test_insufficient_history_returns_zero(self):
        mc = MetaController(MetaControllerConfig(checkpoint_dir="/tmp/test_mc"))
        mc.record_probe_performance(0.8)
        assert mc.detect_gradual_forgetting() == 0.0


class TestPersistence:
    """Tests for save/load of meta-controller state."""

    def test_save_and_load(self, tmp_path):
        mc = MetaController(MetaControllerConfig(checkpoint_dir=str(tmp_path / "mc")))
        mc._plasticity_multiplier = 2.5
        mc._consolidation_ratio = 0.7
        for perf in [0.8, 0.85, 0.9]:
            mc.record_probe_performance(perf)

        mc.save_state(tmp_path / "mc")

        mc2 = MetaController(MetaControllerConfig(checkpoint_dir=str(tmp_path / "mc")))
        mc2.load_state(tmp_path / "mc" / "meta_controller.json")

        assert mc2._plasticity_multiplier == pytest.approx(2.5)
        assert mc2._consolidation_ratio == pytest.approx(0.7)
