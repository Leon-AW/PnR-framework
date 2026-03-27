"""
Unit Tests — MORPHEUS Configuration
=====================================

Tests for the configuration dataclasses and validation logic.
"""

import pytest
from src.morpheus.config import (
    MorpheusConfig,
    StableCoreConfig,
    ExpertBankConfig,
    FastBufferConfig,
    ConsolidationConfig,
    KnowledgeStoreConfig,
    MetaControllerConfig,
    PrototypeRouterConfig,
    ExpertState,
    ConsolidationTrigger,
    ActionReversibility,
)


class TestEnums:
    """Tests for configuration enums."""

    def test_expert_state_values(self):
        assert ExpertState.SHADOW.value == "shadow"
        assert ExpertState.ACTIVE.value == "active"
        assert ExpertState.FROZEN.value == "frozen"
        assert ExpertState.MERGE_CANDIDATE.value == "merge"
        assert ExpertState.DORMANT.value == "dormant"

    def test_consolidation_trigger_values(self):
        assert ConsolidationTrigger.BUFFER_FULL.value == "buffer_full"
        assert ConsolidationTrigger.META_DECISION.value == "meta_decision"

    def test_action_reversibility_values(self):
        assert ActionReversibility.REVERSIBLE.value == "reversible"
        assert ActionReversibility.IRREVERSIBLE.value == "irreversible"


class TestMorpheusConfig:
    """Tests for the top-level config aggregation and validation."""

    def test_default_config_creates(self):
        config = MorpheusConfig()
        assert config.stable_core.model_id != ""
        assert config.expert_bank.max_experts > 0
        assert config.fast_buffer.learning_rate > 0

    def test_validation_catches_lr_inversion(self):
        config = MorpheusConfig()
        config.fast_buffer.learning_rate = 1e-6
        config.consolidation.consolidation_learning_rate = 1e-4
        warnings = config.validate()
        assert any("Buffer learning rate" in w for w in warnings)

    def test_validation_catches_low_rehearsal_ratio(self):
        config = MorpheusConfig()
        config.consolidation.default_rehearsal_ratio = 0.1
        warnings = config.validate()
        assert any("rehearsal ratio" in w for w in warnings)

    def test_validation_catches_small_ensemble(self):
        config = MorpheusConfig()
        config.meta_controller.ensemble_size = 1
        warnings = config.validate()
        assert any("Ensemble size" in w for w in warnings)

    def test_validation_catches_high_cka_threshold(self):
        config = MorpheusConfig()
        config.stable_core.cka_threshold = 0.5
        warnings = config.validate()
        assert any("CKA threshold" in w for w in warnings)

    def test_valid_config_no_warnings(self):
        config = MorpheusConfig()
        warnings = config.validate()
        assert len(warnings) == 0


class TestStableCoreConfig:
    """Tests for Stable Core configuration defaults."""

    def test_defaults(self):
        config = StableCoreConfig()
        assert config.cka_lambda == 0.5
        assert config.cka_threshold == 0.05
        assert config.probe_set_size == 512
        assert config.readaptation_interval == 5
        assert config.max_adapter_chain_length == 3


class TestExpertBankConfig:
    """Tests for Expert Bank configuration defaults."""

    def test_defaults(self):
        config = ExpertBankConfig()
        assert config.max_experts == 64
        assert config.shadow_period_steps == 200
        assert config.merge_centroid_threshold == 0.92
        assert config.pruning_importance_threshold == 0.1

    def test_importance_weights_sum_to_one(self):
        config = ExpertBankConfig()
        total = (
            config.importance_weight_frequency
            + config.importance_weight_marginal
            + config.importance_weight_uniqueness
        )
        assert total == pytest.approx(1.0)


class TestPrototypeRouterConfig:
    """Tests for Prototype Router configuration."""

    def test_defaults(self):
        config = PrototypeRouterConfig()
        assert config.projection_dim == 256
        assert config.similarity_threshold == 0.55
        assert config.hierarchical_routing is True
        assert config.hub_detection_threshold == 3.0
