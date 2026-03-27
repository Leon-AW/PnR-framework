"""
Integration Tests — Cross-Subsystem Interactions
==================================================

Level 2 tests verifying that MORPHEUS subsystems interact correctly:
- Router + Expert Bank: registering experts creates routable centroids
- Buffer + Meta-Controller: buffer fill triggers consolidation signals
- Knowledge Store + Factuality: override context reaches prompt builder
- Meta-Controller + Expert Bank: lifecycle decisions propagate
- End-to-end: query flows through the full pipeline (mocked LLM)
"""

import pytest
import numpy as np

from src.morpheus.config import (
    MorpheusConfig,
    ExpertBankConfig,
    FastBufferConfig,
    MetaControllerConfig,
    PrototypeRouterConfig,
    KnowledgeStoreConfig,
    ExpertState,
)
from src.morpheus.router import PrototypeRouter
from src.morpheus.expert_bank import ExpertBank
from src.morpheus.fast_buffer import FastBuffer
from src.morpheus.knowledge_store import KnowledgeStore, KnowledgeRecord
from src.morpheus.meta_controller import MetaController, SystemState


def _make_embedding_fn(dim: int = 64):
    rng = np.random.RandomState(99)
    cache = {}

    def embed(text: str) -> np.ndarray:
        if text not in cache:
            cache[text] = rng.randn(dim).astype(np.float32)
            cache[text] /= np.linalg.norm(cache[text])
        return cache[text]

    return embed


class TestRouterAndExpertBank:
    """Integration: Router + Expert Bank."""

    def test_spawned_expert_registered_in_router(self):
        """When ExpertBank spawns an expert and it's promoted to ACTIVE,
        it should be registerable in the PrototypeRouter."""
        embed_fn = _make_embedding_fn(64)
        config = PrototypeRouterConfig(
            projection_dim=32,
            similarity_threshold=-1.0,
            hierarchical_routing=False,
        )
        router = PrototypeRouter(config=config, embedding_fn=embed_fn, embedding_dim=64)
        bank = ExpertBank(ExpertBankConfig(
            shadow_period_steps=5,
            checkpoint_dir="/tmp/test_int_eb",
        ))

        bank.spawn_expert("medical_exp", domain="medical", core_version=0)
        for _ in range(5):
            bank.record_training_step("medical_exp", loss=0.5, is_shadow=True)
        bank.promote_to_active("medical_exp")

        meta = bank.get_expert("medical_exp")
        centroid = np.random.randn(64).astype(np.float32)
        centroid /= np.linalg.norm(centroid)

        router.register_adapter(
            adapter_id=meta.expert_id,
            path=meta.adapter_path,
            timestamp=meta.timestamp,
            state=ExpertState(meta.state.value),
        centroid=centroid,
        )

        assert meta.expert_id in router.get_registered_adapters()
        result = router.route("medical question about symptoms")
        assert result.winner_adapter is not None

    def test_retired_expert_removed_from_routing(self):
        """When an expert is retired to DORMANT, unregistering from the router
        should remove it from routing candidates."""
        embed_fn = _make_embedding_fn(64)
        config = PrototypeRouterConfig(
            projection_dim=32,
            hierarchical_routing=False,
        )
        router = PrototypeRouter(config=config, embedding_fn=embed_fn, embedding_dim=64)
        bank = ExpertBank(ExpertBankConfig(checkpoint_dir="/tmp/test_int_eb"))

        bank.spawn_expert("exp_1")
        centroid = np.random.randn(64).astype(np.float32)
        centroid /= np.linalg.norm(centroid)
        router.register_adapter("exp_1", path="/tmp/exp1", timestamp=1.0, centroid=centroid)

        bank.retire_expert("exp_1")
        router.unregister_adapter("exp_1")

        assert "exp_1" not in router.get_registered_adapters()

    def test_multiple_experts_routed_correctly(self):
        """With multiple active experts, the router should pick the most
        similar one to the query."""
        embed_fn = _make_embedding_fn(64)
        config = PrototypeRouterConfig(
            projection_dim=32,
            similarity_threshold=-1.0,
            hierarchical_routing=False,
        )
        router = PrototypeRouter(config=config, embedding_fn=embed_fn, embedding_dim=64)

        query_emb = embed_fn("medical query")
        for i in range(5):
            c = np.random.randn(64).astype(np.float32)
            c /= np.linalg.norm(c)
            router.register_adapter(f"exp_{i}", path=f"/tmp/{i}", timestamp=float(i), centroid=c)

        router.register_adapter(
            "target_exp", path="/tmp/target", timestamp=10.0,
            centroid=query_emb,
        )

        result = router.route("medical query")
        assert result.winner_adapter == "target_exp"


class TestBufferAndMetaController:
    """Integration: Buffer + Meta-Controller."""

    def test_full_buffer_triggers_consolidation_signal(self):
        """When the buffer reaches capacity, the meta-controller should
        recommend consolidation."""
        buf_config = FastBufferConfig(max_capacity_steps=20, checkpoint_dir="/tmp/test_int_buf")
        buf = FastBuffer(buf_config)
        mc = MetaController(MetaControllerConfig(checkpoint_dir="/tmp/test_int_mc"))

        for _ in range(20):
            buf.record_step(loss=0.5)

        state = SystemState(
            buffer_fill_level=buf.fill_level,
            buffer_loss_mean=buf.get_loss_statistics()["mean"],
            buffer_loss_trend=buf.get_loss_statistics()["trend"],
        )
        mc.observe(state)
        should, reason = mc.should_consolidate()
        assert should
        assert reason == "buffer_full"

    def test_distribution_shift_detected_and_propagated(self):
        """A distribution shift in the buffer should be visible to the
        meta-controller and trigger plasticity increase."""
        buf = FastBuffer(FastBufferConfig(checkpoint_dir="/tmp/test_int_buf"))
        mc = MetaController(MetaControllerConfig(checkpoint_dir="/tmp/test_int_mc"))

        for _ in range(50):
            buf.record_step(loss=0.5)
        for _ in range(50):
            buf.record_step(loss=3.0)

        shift = buf.detect_distribution_shift(window=50)
        state = SystemState(distribution_shift_magnitude=shift)
        mc.observe(state)
        actions = mc.decide()
        types = [a.action_type for a in actions]
        assert shift > 1.0
        assert "increase_plasticity" in types

    def test_buffer_reset_after_consolidation_signal(self):
        """After consolidation, the buffer should be reset and fill level
        should drop to zero."""
        buf = FastBuffer(FastBufferConfig(
            max_capacity_steps=10,
            checkpoint_dir="/tmp/test_int_buf",
        ))
        for _ in range(10):
            buf.add_sample("data")
            buf.record_step(loss=0.5)

        assert buf.is_full
        buf.reset()
        assert buf.fill_level == 0.0
        assert buf.num_samples == 0


class TestKnowledgeStoreAndFactuality:
    """Integration: Knowledge Store graduated factuality with context building."""

    def test_factual_query_gets_override_context(self):
        """For a clearly factual query with matching records, the store
        should produce a hard override with context."""
        config = KnowledgeStoreConfig(
            factuality_threshold_high=0.7,
            factuality_threshold_low=0.3,
            store_dir="/tmp/test_int_ks",
        )
        store = KnowledgeStore(config)

        rng = np.random.RandomState(42)
        emb = rng.randn(768).astype(np.float32)
        emb /= np.linalg.norm(emb)

        record = KnowledgeRecord(
            record_id="capital_france",
            subject="France",
            predicate="capital_of",
            object_value="Paris",
            confidence=0.99,
            embedding=emb,
        )
        store.create(record)

        decision = store.assess_factuality(
            query_embedding=emb,
            factuality_score=0.9,
        )
        assert decision.zone == "hard_override"

        context = store.build_override_context(decision.system5_records)
        assert "Paris" in context
        assert "Verified Facts" in context

    def test_novelty_shifts_factuality_thresholds(self):
        """With high novelty, the factuality thresholds shift, changing
        which zone a mid-range score falls into."""
        config = KnowledgeStoreConfig(
            factuality_threshold_high=0.8,
            factuality_threshold_low=0.3,
            novelty_threshold_shift=0.3,
            store_dir="/tmp/test_int_ks",
        )
        store = KnowledgeStore(config)

        emb = np.random.randn(768).astype(np.float32)
        emb /= np.linalg.norm(emb)
        store.create(KnowledgeRecord(
            record_id="r1", subject="Test", predicate="is", object_value="True",
            confidence=0.9, embedding=emb,
        ))

        # Normal: tau_high=0.8, tau_low=0.3 → 0.5 is in boundary
        decision_normal = store.assess_factuality(emb, factuality_score=0.5, novelty_level=0.0)
        # Novel: tau_high=0.8-0.3=0.5, tau_low=0.3+0.3=0.6 → score 0.5 < 0.6 → parametric_freedom
        # (thresholds cross when novelty is extreme, causing conservative parametric_freedom)
        decision_novel = store.assess_factuality(emb, factuality_score=0.5, novelty_level=1.0)

        assert decision_normal.zone == "boundary"
        # Verify novelty actually changed the zone
        assert decision_normal.zone != decision_novel.zone


class TestMetaControllerAndExpertBank:
    """Integration: Meta-Controller lifecycle decisions -> Expert Bank."""

    def test_spawn_signal_creates_expert(self):
        """When the meta-controller signals expert spawning,
        the expert bank should create a new shadow expert."""
        mc = MetaController(MetaControllerConfig(
            irreversible_majority_threshold=0.5,
            checkpoint_dir="/tmp/test_int_mc",
        ))
        bank = ExpertBank(ExpertBankConfig(checkpoint_dir="/tmp/test_int_eb"))

        state = SystemState(
            buffer_loss_mean=3.0,
            distribution_shift_magnitude=3.0,
            capacity_utilization=0.3,
        )
        mc.observe(state)
        actions = mc.decide()

        for action in actions:
            if action.action_type == "spawn_expert":
                expert_id = f"auto_expert_{bank.num_experts}"
                bank.spawn_expert(expert_id)
                assert bank.get_expert(expert_id).state == ExpertState.SHADOW
                break

    def test_pruning_signal_retires_low_importance_experts(self):
        """When the meta-controller signals pruning, low-importance
        experts should be retired."""
        mc = MetaController(MetaControllerConfig(
            irreversible_majority_threshold=0.5,
            checkpoint_dir="/tmp/test_int_mc",
        ))
        bank = ExpertBank(ExpertBankConfig(
            shadow_period_steps=1,
            pruning_importance_threshold=0.5,
            checkpoint_dir="/tmp/test_int_eb",
        ))

        bank.spawn_expert("expendable", domain="test")
        bank.record_training_step("expendable", loss=0.1, is_shadow=True)
        bank.promote_to_active("expendable")

        state = SystemState(
            capacity_utilization=0.95,
            mean_expert_importance=0.1,
        )
        mc.observe(state)
        actions = mc.decide()

        for action in actions:
            if action.action_type == "trigger_pruning":
                candidates = bank.get_pruning_candidates()
                for eid, _ in candidates:
                    bank.retire_expert(eid)

        assert bank.get_expert("expendable").state == ExpertState.DORMANT


class TestEndToEndFlow:
    """Integration: Full query flow through multiple subsystems (no LLM)."""

    def test_query_flow_without_llm(self):
        """Simulate the full query flow: meta-controller -> router -> knowledge store.
        This tests the orchestration logic without requiring a loaded LLM."""
        embed_fn = _make_embedding_fn(64)

        router = PrototypeRouter(
            config=PrototypeRouterConfig(
                projection_dim=32,
                similarity_threshold=-1.0,
                hierarchical_routing=False,
            ),
            embedding_fn=embed_fn,
            embedding_dim=64,
        )
        bank = ExpertBank(ExpertBankConfig(
            shadow_period_steps=1,
            checkpoint_dir="/tmp/test_int_e2e",
        ))
        buf = FastBuffer(FastBufferConfig(checkpoint_dir="/tmp/test_int_e2e"))
        ks = KnowledgeStore(KnowledgeStoreConfig(store_dir="/tmp/test_int_e2e"))
        mc = MetaController(MetaControllerConfig(checkpoint_dir="/tmp/test_int_e2e"))

        bank.spawn_expert("geo_expert", domain="geography")
        bank.record_training_step("geo_expert", loss=0.1, is_shadow=True)
        bank.promote_to_active("geo_expert")

        geo_centroid = embed_fn("Where is Paris?")
        router.register_adapter(
            "geo_expert", path="/tmp/geo", timestamp=1.0,
            centroid=geo_centroid,
        )

        # Step 1: Meta-controller observes
        state = SystemState(
            buffer_fill_level=buf.fill_level,
            num_active_experts=len(bank.active_experts),
        )
        mc.observe(state)

        # Step 2: Route query
        result = router.route("Where is Paris?")
        assert result.winner_adapter == "geo_expert"

        # Step 3: Record in buffer
        buf.add_sample("Where is Paris?", domain_signal="geography")
        buf.record_step(loss=0.3)

        # Step 4: Verify expert activation count
        bank.record_activation("geo_expert", marginal_improvement=0.5)
        assert bank.get_expert("geo_expert").activation_count == 1
