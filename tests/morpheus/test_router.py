"""
Unit Tests — Prototype Router
==============================

Tests for non-parametric prototype-based routing with:
- Random projection (JL lemma) distance preservation
- Hub detection and correction
- EMA centroid updates
- Hierarchical routing pre-filter
- Expert registration / unregistration
- Routing result compatibility with EvalRunner
"""

import pytest
import numpy as np

from src.morpheus.router import PrototypeRouter, ExpertPrototype
from src.morpheus.config import PrototypeRouterConfig, ExpertState
from src.routing.base import RoutingResult, RoutingStrategy


def _make_embedding_fn(dim: int = 768):
    """Create a deterministic embedding function for testing."""
    rng = np.random.RandomState(99)
    cache = {}

    def embed(text: str) -> np.ndarray:
        if text not in cache:
            cache[text] = rng.randn(dim).astype(np.float32)
            cache[text] /= np.linalg.norm(cache[text])
        return cache[text]

    return embed


class TestRandomProjection:
    """Tests that the JL random projection preserves pairwise distances."""

    def test_projection_reduces_dimension(self):
        config = PrototypeRouterConfig(projection_dim=128)
        router = PrototypeRouter(config=config, embedding_dim=768)
        v = np.random.randn(768).astype(np.float32)
        projected = router._project(v)
        assert projected.shape == (128,)

    def test_pairwise_distances_approximately_preserved(self):
        """JL lemma: pairwise distances preserved up to bounded distortion."""
        config = PrototypeRouterConfig(projection_dim=256)
        router = PrototypeRouter(config=config, embedding_dim=768)

        rng = np.random.RandomState(42)
        vectors = [rng.randn(768).astype(np.float32) for _ in range(20)]
        for i in range(len(vectors)):
            vectors[i] /= np.linalg.norm(vectors[i])

        original_dists = []
        projected_dists = []
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                original_dists.append(np.linalg.norm(vectors[i] - vectors[j]))
                pi = router._project(vectors[i])
                pj = router._project(vectors[j])
                projected_dists.append(np.linalg.norm(pi - pj))

        original_dists = np.array(original_dists)
        projected_dists = np.array(projected_dists)

        ratios = projected_dists / (original_dists + 1e-9)
        assert np.mean(ratios) == pytest.approx(1.0, abs=0.3)

    def test_projection_is_deterministic(self):
        config = PrototypeRouterConfig(projection_dim=128)
        r1 = PrototypeRouter(config=config, embedding_dim=768)
        r2 = PrototypeRouter(config=config, embedding_dim=768)
        v = np.random.randn(768).astype(np.float32)
        np.testing.assert_array_equal(r1._project(v), r2._project(v))

    def test_projected_vector_is_normalized(self):
        config = PrototypeRouterConfig(projection_dim=128)
        router = PrototypeRouter(config=config, embedding_dim=768)
        v = np.random.randn(768).astype(np.float32)
        proj = router._project(v)
        assert np.linalg.norm(proj) == pytest.approx(1.0, abs=1e-5)


class TestExpertRegistration:
    """Tests for adding/removing expert prototypes."""

    def test_register_and_list(self):
        router = PrototypeRouter(
            config=PrototypeRouterConfig(hierarchical_routing=False),
            embedding_dim=64,
        )
        centroid = np.random.randn(64).astype(np.float32)
        centroid /= np.linalg.norm(centroid)

        router.register_adapter("exp_1", path="/tmp/exp1", timestamp=1.0, centroid=centroid)
        assert "exp_1" in router.get_registered_adapters()

    def test_unregister(self):
        router = PrototypeRouter(
            config=PrototypeRouterConfig(hierarchical_routing=False),
            embedding_dim=64,
        )
        centroid = np.random.randn(64).astype(np.float32)
        centroid /= np.linalg.norm(centroid)

        router.register_adapter("exp_1", path="/tmp/exp1", timestamp=1.0, centroid=centroid)
        assert router.unregister_adapter("exp_1")
        assert "exp_1" not in router.get_registered_adapters()

    def test_unregister_nonexistent_returns_false(self):
        router = PrototypeRouter(
            config=PrototypeRouterConfig(hierarchical_routing=False),
            embedding_dim=64,
        )
        assert not router.unregister_adapter("ghost")

    def test_shadow_experts_not_routable(self):
        router = PrototypeRouter(
            config=PrototypeRouterConfig(hierarchical_routing=False),
            embedding_dim=64,
        )
        centroid = np.random.randn(64).astype(np.float32)
        centroid /= np.linalg.norm(centroid)

        router.register_adapter(
            "shadow_exp", path="/tmp/shadow", timestamp=1.0,
            centroid=centroid, state=ExpertState.SHADOW,
        )
        assert len(router._get_routable_prototypes()) == 0

    def test_dormant_experts_not_routable(self):
        router = PrototypeRouter(
            config=PrototypeRouterConfig(hierarchical_routing=False),
            embedding_dim=64,
        )
        centroid = np.random.randn(64).astype(np.float32)
        centroid /= np.linalg.norm(centroid)

        router.register_adapter(
            "dormant_exp", path="/tmp/dormant", timestamp=1.0,
            centroid=centroid, state=ExpertState.DORMANT,
        )
        assert len(router._get_routable_prototypes()) == 0


class TestRouting:
    """Tests for the core routing logic."""

    def _setup_router_with_experts(self, n_experts=5, dim=64):
        embed_fn = _make_embedding_fn(dim)
        config = PrototypeRouterConfig(
            projection_dim=32,
            similarity_threshold=-1.0,
            hierarchical_routing=False,
        )
        router = PrototypeRouter(config=config, embedding_fn=embed_fn, embedding_dim=dim)

        rng = np.random.RandomState(7)
        for i in range(n_experts):
            c = rng.randn(dim).astype(np.float32)
            c /= np.linalg.norm(c)
            router.register_adapter(
                f"expert_{i}", path=f"/tmp/expert_{i}", timestamp=float(i),
                centroid=c,
            )
        return router

    def test_route_returns_routing_result(self):
        router = self._setup_router_with_experts()
        result = router.route("test query")
        assert isinstance(result, RoutingResult)

    def test_route_picks_a_winner(self):
        router = self._setup_router_with_experts()
        result = router.route("test query")
        assert result.winner_adapter is not None

    def test_empty_router_returns_empty_result(self):
        embed_fn = _make_embedding_fn(64)
        config = PrototypeRouterConfig(hierarchical_routing=False)
        router = PrototypeRouter(config=config, embedding_fn=embed_fn, embedding_dim=64)
        result = router.route("test")
        assert result.winner_adapter is None
        assert result.all_matches == []

    def test_routing_strategy_is_centroid(self):
        router = self._setup_router_with_experts()
        result = router.route("test query")
        assert result.routing_strategy == RoutingStrategy.CENTROID

    def test_activation_counts_increment(self):
        router = self._setup_router_with_experts()
        router.route("query 1")
        router.route("query 2")
        total = sum(p.activation_count for p in router._prototypes.values())
        assert total > 0

    def test_similarity_threshold_filters(self):
        embed_fn = _make_embedding_fn(64)
        config = PrototypeRouterConfig(
            projection_dim=32,
            similarity_threshold=0.99,
            hierarchical_routing=False,
        )
        router = PrototypeRouter(config=config, embedding_fn=embed_fn, embedding_dim=64)

        c = np.random.randn(64).astype(np.float32)
        c /= np.linalg.norm(c)
        router.register_adapter("exp", path="/tmp/exp", timestamp=1.0, centroid=c)

        result = router.route("unrelated query")
        assert result.winner_adapter is None or result.winner_similarity > 0.99


class TestEMACentroidUpdate:
    """Tests for EMA-based centroid updates."""

    def test_ema_moves_toward_new_embedding(self):
        config = PrototypeRouterConfig(ema_decay=0.9, hierarchical_routing=False)
        router = PrototypeRouter(config=config, embedding_dim=64)

        old_centroid = np.ones(64, dtype=np.float32)
        old_centroid /= np.linalg.norm(old_centroid)
        router.register_adapter("exp", path="/tmp/exp", timestamp=1.0, centroid=old_centroid.copy())

        new_emb = -np.ones(64, dtype=np.float32)
        new_emb /= np.linalg.norm(new_emb)
        router.update_centroid_ema("exp", new_emb)

        updated = router._prototypes["exp"].centroid
        old_sim = np.dot(updated, old_centroid)
        new_sim = np.dot(updated, new_emb)
        # Should be closer to old (decay=0.9 means 90% old)
        assert old_sim > new_sim

    def test_ema_nonexistent_expert_noop(self):
        router = PrototypeRouter(
            config=PrototypeRouterConfig(hierarchical_routing=False),
            embedding_dim=64,
        )
        router.update_centroid_ema("ghost", np.zeros(64, dtype=np.float32))


class TestHubDetection:
    """Tests for hub detection and correction."""

    def test_no_hubs_with_few_routes(self):
        router = PrototypeRouter(
            config=PrototypeRouterConfig(hierarchical_routing=False),
            embedding_dim=64,
        )
        c = np.random.randn(64).astype(np.float32)
        c /= np.linalg.norm(c)
        router.register_adapter("exp", path="/tmp", timestamp=1.0, centroid=c)
        router._total_routes = 10
        assert router._detect_hubs() == []

    def test_hub_detected_when_frequency_extreme(self):
        config = PrototypeRouterConfig(
            hub_detection_threshold=2.0,
            hierarchical_routing=False,
        )
        router = PrototypeRouter(config=config, embedding_dim=64)

        for i in range(5):
            c = np.random.randn(64).astype(np.float32)
            c /= np.linalg.norm(c)
            router.register_adapter(f"exp_{i}", path=f"/tmp/{i}", timestamp=1.0, centroid=c)

        router._total_routes = 500
        router._prototypes["exp_0"].activation_count = 400
        for i in range(1, 5):
            router._prototypes[f"exp_{i}"].activation_count = 25

        hubs = router._detect_hubs()
        assert "exp_0" in hubs


class TestRoutingConfidence:
    """Tests for routing confidence (novelty detection)."""

    def test_confidence_returns_float(self):
        embed_fn = _make_embedding_fn(64)
        config = PrototypeRouterConfig(hierarchical_routing=False)
        router = PrototypeRouter(config=config, embedding_fn=embed_fn, embedding_dim=64)
        c = np.random.randn(64).astype(np.float32)
        c /= np.linalg.norm(c)
        router.register_adapter("exp", path="/tmp", timestamp=1.0, centroid=c)

        conf = router.compute_routing_confidence("test")
        assert isinstance(conf, float)

    def test_empty_router_confidence_is_zero(self):
        embed_fn = _make_embedding_fn(64)
        config = PrototypeRouterConfig(hierarchical_routing=False)
        router = PrototypeRouter(config=config, embedding_fn=embed_fn, embedding_dim=64)
        assert router.compute_routing_confidence("test") == 0.0


class TestRouterPersistence:
    """Tests for save/load of router state."""

    def test_save_and_load(self, tmp_path):
        embed_fn = _make_embedding_fn(64)
        config = PrototypeRouterConfig(projection_dim=32, hierarchical_routing=False)
        router = PrototypeRouter(config=config, embedding_fn=embed_fn, embedding_dim=64)

        c = np.random.randn(64).astype(np.float32)
        c /= np.linalg.norm(c)
        router.register_adapter("exp_1", path="/tmp/exp1", timestamp=1.0, centroid=c)
        router._total_routes = 42

        router.save(tmp_path / "router_state")
        loaded = PrototypeRouter.load(
            tmp_path / "router_state",
            embedding_fn=embed_fn,
            config=config,
            embedding_dim=64,
        )

        assert "exp_1" in loaded.get_registered_adapters()
        assert loaded._total_routes == 42
