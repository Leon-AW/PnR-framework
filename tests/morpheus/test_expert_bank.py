"""
Unit Tests — Expert Bank (System 2)
=====================================

Tests for expert lifecycle management:
- Spawning in SHADOW state
- Promotion to ACTIVE
- Freezing converged experts
- Merge candidate detection and verification
- Importance-weighted pruning
- Dormant retirement and reactivation
"""

import pytest
import numpy as np

from src.morpheus.expert_bank import ExpertBank, ExpertMetadata
from src.morpheus.config import ExpertBankConfig, ExpertState


class TestSpawning:
    """Tests for expert spawning with shadow routing."""

    def test_spawn_creates_shadow_expert(self):
        bank = ExpertBank(ExpertBankConfig(checkpoint_dir="/tmp/test_eb"))
        meta = bank.spawn_expert("exp_1", domain="medical")
        assert meta.state == ExpertState.SHADOW
        assert meta.domain == "medical"

    def test_spawn_duplicate_raises(self):
        bank = ExpertBank(ExpertBankConfig(checkpoint_dir="/tmp/test_eb"))
        bank.spawn_expert("exp_1")
        with pytest.raises(ValueError, match="already exists"):
            bank.spawn_expert("exp_1")

    def test_spawn_uses_custom_lora_config(self):
        bank = ExpertBank(ExpertBankConfig(checkpoint_dir="/tmp/test_eb"))
        meta = bank.spawn_expert("exp_1", lora_rank=32, lora_alpha=64)
        assert meta.lora_rank == 32
        assert meta.lora_alpha == 64

    def test_spawn_records_core_version(self):
        bank = ExpertBank(ExpertBankConfig(checkpoint_dir="/tmp/test_eb"))
        meta = bank.spawn_expert("exp_1", core_version=3)
        assert meta.native_core_version == 3

    def test_spawn_with_domain_adapter(self):
        bank = ExpertBank(ExpertBankConfig(checkpoint_dir="/tmp/test_eb"))
        meta = bank.spawn_with_domain_adapter("novel_exp", domain="biology", core_version=1)
        assert meta.state == ExpertState.SHADOW
        assert meta.domain_adapter_path is not None


class TestPromotion:
    """Tests for promoting SHADOW -> ACTIVE."""

    def test_promote_after_shadow_period(self):
        config = ExpertBankConfig(shadow_period_steps=10, checkpoint_dir="/tmp/test_eb")
        bank = ExpertBank(config)
        bank.spawn_expert("exp_1")

        for i in range(10):
            bank.record_training_step("exp_1", loss=1.0 - i * 0.05, is_shadow=True)

        assert bank.promote_to_active("exp_1")
        assert bank.get_expert("exp_1").state == ExpertState.ACTIVE

    def test_promote_too_early_fails(self):
        config = ExpertBankConfig(shadow_period_steps=100, checkpoint_dir="/tmp/test_eb")
        bank = ExpertBank(config)
        bank.spawn_expert("exp_1")

        for i in range(5):
            bank.record_training_step("exp_1", loss=1.0, is_shadow=True)

        assert not bank.promote_to_active("exp_1")
        assert bank.get_expert("exp_1").state == ExpertState.SHADOW

    def test_promote_nonexistent_returns_false(self):
        bank = ExpertBank(ExpertBankConfig(checkpoint_dir="/tmp/test_eb"))
        assert not bank.promote_to_active("ghost")

    def test_promote_non_shadow_returns_false(self):
        config = ExpertBankConfig(shadow_period_steps=5, checkpoint_dir="/tmp/test_eb")
        bank = ExpertBank(config)
        bank.spawn_expert("exp_1")
        for _ in range(5):
            bank.record_training_step("exp_1", loss=0.5, is_shadow=True)
        bank.promote_to_active("exp_1")
        assert not bank.promote_to_active("exp_1")


class TestFreezing:
    """Tests for freezing converged experts."""

    def _make_active_expert(self, bank, expert_id="exp_1"):
        config = bank.config
        bank.spawn_expert(expert_id)
        for _ in range(config.shadow_period_steps):
            bank.record_training_step(expert_id, loss=0.5, is_shadow=True)
        bank.promote_to_active(expert_id)
        return bank.get_expert(expert_id)

    def test_freeze_active_expert(self):
        bank = ExpertBank(ExpertBankConfig(
            shadow_period_steps=5,
            checkpoint_dir="/tmp/test_eb",
        ))
        self._make_active_expert(bank)
        assert bank.freeze_expert("exp_1")
        assert bank.get_expert("exp_1").state == ExpertState.FROZEN

    def test_freeze_non_active_returns_false(self):
        bank = ExpertBank(ExpertBankConfig(checkpoint_dir="/tmp/test_eb"))
        bank.spawn_expert("exp_1")
        assert not bank.freeze_expert("exp_1")

    def test_should_freeze_detects_convergence(self):
        config = ExpertBankConfig(
            shadow_period_steps=5,
            freeze_loss_convergence_window=10,
            freeze_loss_convergence_threshold=0.01,
            checkpoint_dir="/tmp/test_eb",
        )
        bank = ExpertBank(config)
        self._make_active_expert(bank)

        for _ in range(20):
            bank.record_training_step("exp_1", loss=0.1)

        assert bank.should_freeze("exp_1")

    def test_should_freeze_not_converged(self):
        config = ExpertBankConfig(
            shadow_period_steps=5,
            freeze_loss_convergence_window=10,
            freeze_loss_convergence_threshold=0.001,
            checkpoint_dir="/tmp/test_eb",
        )
        bank = ExpertBank(config)
        self._make_active_expert(bank)

        rng = np.random.RandomState(42)
        for _ in range(20):
            bank.record_training_step("exp_1", loss=rng.uniform(0.1, 1.0))

        assert not bank.should_freeze("exp_1")


class TestMerging:
    """Tests for merge candidate detection and verification."""

    def test_find_merge_candidates(self):
        config = ExpertBankConfig(
            merge_centroid_threshold=0.95,
            checkpoint_dir="/tmp/test_eb",
        )
        bank = ExpertBank(config)
        bank.spawn_expert("exp_1")
        bank.spawn_expert("exp_2")

        v = np.random.randn(64).astype(np.float32)
        v /= np.linalg.norm(v)
        centroids = {
            "exp_1": v,
            "exp_2": v + 0.01 * np.random.randn(64).astype(np.float32),
        }
        centroids["exp_2"] /= np.linalg.norm(centroids["exp_2"])

        candidates = bank.find_merge_candidates(centroids)
        assert len(candidates) >= 1
        assert candidates[0][0] in ("exp_1", "exp_2")

    def test_verify_merge_passes(self):
        bank = ExpertBank(ExpertBankConfig(
            merge_tolerance_delta=0.1,
            checkpoint_dir="/tmp/test_eb",
        ))
        assert bank.verify_merge("a", "b", 0.5, 0.6, 0.5, 0.55)

    def test_verify_merge_fails_on_degradation(self):
        bank = ExpertBank(ExpertBankConfig(
            merge_tolerance_delta=0.01,
            checkpoint_dir="/tmp/test_eb",
        ))
        assert not bank.verify_merge("a", "b", 1.0, 1.0, 0.5, 0.5)

    def test_execute_merge(self):
        config = ExpertBankConfig(shadow_period_steps=1, checkpoint_dir="/tmp/test_eb")
        bank = ExpertBank(config)
        bank.spawn_expert("exp_a", domain="medical")
        bank.spawn_expert("exp_b", domain="legal")

        merged = bank.execute_merge("exp_a", "exp_b", "exp_merged", "/tmp/merged")
        assert merged.state == ExpertState.ACTIVE
        assert bank.get_expert("exp_a").state == ExpertState.DORMANT
        assert bank.get_expert("exp_b").state == ExpertState.DORMANT
        assert "exp_merged" in [e for e in bank._experts]


class TestImportancePruning:
    """Tests for importance scoring and pruning."""

    def test_importance_score_computation(self):
        meta = ExpertMetadata(
            expert_id="test",
            state=ExpertState.ACTIVE,
            activation_count=100,
            marginal_contribution=0.8,
            uniqueness_score=0.9,
        )
        score = meta.importance_score()
        assert score > 0.0

    def test_pruning_candidates_below_threshold(self):
        config = ExpertBankConfig(
            pruning_importance_threshold=0.5,
            shadow_period_steps=1,
            checkpoint_dir="/tmp/test_eb",
        )
        bank = ExpertBank(config)
        bank.spawn_expert("high_imp")
        bank.record_training_step("high_imp", 0.1, is_shadow=True)
        bank.promote_to_active("high_imp")
        bank._experts["high_imp"].marginal_contribution = 0.9
        bank._experts["high_imp"].uniqueness_score = 0.9

        bank.spawn_expert("low_imp")
        bank.record_training_step("low_imp", 0.1, is_shadow=True)
        bank.promote_to_active("low_imp")
        bank._experts["low_imp"].marginal_contribution = 0.0
        bank._experts["low_imp"].uniqueness_score = 0.0

        candidates = bank.get_pruning_candidates()
        low_ids = [eid for eid, _ in candidates]
        assert "low_imp" in low_ids


class TestRetirementReactivation:
    """Tests for expert dormant archival and reactivation."""

    def test_retire_expert(self):
        bank = ExpertBank(ExpertBankConfig(checkpoint_dir="/tmp/test_eb"))
        bank.spawn_expert("exp_1")
        assert bank.retire_expert("exp_1")
        assert bank.get_expert("exp_1").state == ExpertState.DORMANT

    def test_reactivate_dormant_expert(self):
        bank = ExpertBank(ExpertBankConfig(checkpoint_dir="/tmp/test_eb"))
        bank.spawn_expert("exp_1")
        bank.retire_expert("exp_1")
        assert bank.reactivate_expert("exp_1")
        assert bank.get_expert("exp_1").state == ExpertState.ACTIVE

    def test_reactivate_non_dormant_fails(self):
        bank = ExpertBank(ExpertBankConfig(checkpoint_dir="/tmp/test_eb"))
        bank.spawn_expert("exp_1")
        assert not bank.reactivate_expert("exp_1")

    def test_reactivate_nonexistent_fails(self):
        bank = ExpertBank(ExpertBankConfig(checkpoint_dir="/tmp/test_eb"))
        assert not bank.reactivate_expert("ghost")


class TestProperties:
    """Tests for bank-level property accessors."""

    def test_active_experts_property(self):
        config = ExpertBankConfig(shadow_period_steps=1, checkpoint_dir="/tmp/test_eb")
        bank = ExpertBank(config)
        bank.spawn_expert("exp_1")
        bank.record_training_step("exp_1", 0.1, is_shadow=True)
        bank.promote_to_active("exp_1")

        bank.spawn_expert("exp_2")

        assert "exp_1" in bank.active_experts
        assert "exp_2" not in bank.active_experts

    def test_shadow_experts_property(self):
        bank = ExpertBank(ExpertBankConfig(checkpoint_dir="/tmp/test_eb"))
        bank.spawn_expert("s1")
        bank.spawn_expert("s2")
        assert len(bank.shadow_experts) == 2

    def test_num_experts(self):
        bank = ExpertBank(ExpertBankConfig(checkpoint_dir="/tmp/test_eb"))
        assert bank.num_experts == 0
        bank.spawn_expert("exp_1")
        assert bank.num_experts == 1


class TestPersistence:
    """Tests for save/load of expert bank state."""

    def test_save_and_load(self, tmp_path):
        bank = ExpertBank(ExpertBankConfig(checkpoint_dir=str(tmp_path / "eb")))
        bank.spawn_expert("exp_1", domain="medical")
        bank.spawn_expert("exp_2", domain="legal")

        save_path = bank.save_state(tmp_path / "eb")

        bank2 = ExpertBank(ExpertBankConfig(checkpoint_dir=str(tmp_path / "eb")))
        bank2.load_state(save_path)

        assert bank2.num_experts == 2
        assert bank2.get_expert("exp_1").domain == "medical"
        assert bank2.get_expert("exp_2").state == ExpertState.SHADOW
