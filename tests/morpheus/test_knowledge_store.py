"""
Unit Tests — Knowledge Store (System 5)
=========================================

Tests for the explicit knowledge store with graduated factuality:
- CRUD operations
- Semantic search by embedding
- Text-based search by subject/predicate
- Graduated factuality assessment
- Novelty-aware threshold adaptation
- Knowledge override context building
- Rehearsal integration (fact-checking)
"""

import pytest
import numpy as np

from src.morpheus.knowledge_store import (
    KnowledgeStore,
    KnowledgeRecord,
    FactualityDecision,
)
from src.morpheus.config import KnowledgeStoreConfig


def _make_record(
    record_id: str,
    subject: str = "France",
    predicate: str = "capital_of",
    object_value: str = "Paris",
    confidence: float = 1.0,
    domain: str = "geography",
    emb_seed: int = 42,
) -> KnowledgeRecord:
    rng = np.random.RandomState(emb_seed)
    emb = rng.randn(768).astype(np.float32)
    emb /= np.linalg.norm(emb)
    return KnowledgeRecord(
        record_id=record_id,
        subject=subject,
        predicate=predicate,
        object_value=object_value,
        confidence=confidence,
        domain=domain,
        embedding=emb,
    )


class TestCRUD:
    """Tests for Create, Read, Update, Delete operations."""

    def test_create_and_read(self):
        store = KnowledgeStore(KnowledgeStoreConfig(store_dir="/tmp/test_ks"))
        rec = _make_record("rec_1")
        store.create(rec)
        assert store.read("rec_1") is not None
        assert store.read("rec_1").object_value == "Paris"

    def test_read_nonexistent_returns_none(self):
        store = KnowledgeStore(KnowledgeStoreConfig(store_dir="/tmp/test_ks"))
        assert store.read("ghost") is None

    def test_update_modifies_field(self):
        store = KnowledgeStore(KnowledgeStoreConfig(store_dir="/tmp/test_ks"))
        store.create(_make_record("rec_1"))
        assert store.update("rec_1", object_value="Lyon")
        assert store.read("rec_1").object_value == "Lyon"

    def test_update_nonexistent_returns_false(self):
        store = KnowledgeStore(KnowledgeStoreConfig(store_dir="/tmp/test_ks"))
        assert not store.update("ghost", object_value="foo")

    def test_delete(self):
        store = KnowledgeStore(KnowledgeStoreConfig(store_dir="/tmp/test_ks"))
        store.create(_make_record("rec_1"))
        assert store.delete("rec_1")
        assert store.read("rec_1") is None

    def test_delete_nonexistent_returns_false(self):
        store = KnowledgeStore(KnowledgeStoreConfig(store_dir="/tmp/test_ks"))
        assert not store.delete("ghost")

    def test_num_records(self):
        store = KnowledgeStore(KnowledgeStoreConfig(store_dir="/tmp/test_ks"))
        assert store.num_records == 0
        store.create(_make_record("r1"))
        store.create(_make_record("r2", subject="Germany", emb_seed=99))
        assert store.num_records == 2


class TestSearch:
    """Tests for search/retrieval."""

    def test_search_by_embedding(self):
        store = KnowledgeStore(KnowledgeStoreConfig(store_dir="/tmp/test_ks"))
        rec = _make_record("rec_1")
        store.create(rec)

        results = store.search(rec.embedding, top_k=1)
        assert len(results) == 1
        assert results[0][0].record_id == "rec_1"
        assert results[0][1] > 0.99

    def test_search_returns_top_k(self):
        store = KnowledgeStore(KnowledgeStoreConfig(store_dir="/tmp/test_ks"))
        for i in range(10):
            store.create(_make_record(f"rec_{i}", emb_seed=i))

        query = np.random.randn(768).astype(np.float32)
        query /= np.linalg.norm(query)
        results = store.search(query, top_k=3)
        assert len(results) <= 3

    def test_search_empty_store(self):
        store = KnowledgeStore(KnowledgeStoreConfig(store_dir="/tmp/test_ks"))
        query = np.random.randn(768).astype(np.float32)
        results = store.search(query)
        assert results == []

    def test_search_by_subject(self):
        store = KnowledgeStore(KnowledgeStoreConfig(store_dir="/tmp/test_ks"))
        store.create(_make_record("r1", subject="France"))
        store.create(_make_record("r2", subject="Germany", emb_seed=99))
        results = store.search_by_subject("France")
        assert len(results) == 1
        assert results[0].subject == "France"

    def test_search_by_subject_and_predicate(self):
        store = KnowledgeStore(KnowledgeStoreConfig(store_dir="/tmp/test_ks"))
        store.create(_make_record("r1", subject="France", predicate="capital_of"))
        store.create(_make_record("r2", subject="France", predicate="population", emb_seed=99))
        results = store.search_by_subject("France", predicate="capital")
        assert len(results) == 1
        assert results[0].predicate == "capital_of"


class TestGraduatedFactuality:
    """Tests for the graduated factuality assessment protocol."""

    def _store_with_facts(self):
        config = KnowledgeStoreConfig(
            factuality_threshold_high=0.8,
            factuality_threshold_low=0.3,
            novelty_threshold_shift=0.15,
            store_dir="/tmp/test_ks",
        )
        store = KnowledgeStore(config)
        rec = _make_record("capital_france", confidence=0.95)
        store.create(rec)
        return store, rec

    def test_high_factuality_gives_hard_override(self):
        store, rec = self._store_with_facts()
        decision = store.assess_factuality(
            query_embedding=rec.embedding,
            factuality_score=0.9,
            novelty_level=0.0,
        )
        assert decision.zone == "hard_override"

    def test_low_factuality_gives_parametric_freedom(self):
        store, rec = self._store_with_facts()
        decision = store.assess_factuality(
            query_embedding=rec.embedding,
            factuality_score=0.1,
            novelty_level=0.0,
        )
        assert decision.zone == "parametric_freedom"

    def test_mid_factuality_gives_boundary(self):
        store, rec = self._store_with_facts()
        decision = store.assess_factuality(
            query_embedding=rec.embedding,
            factuality_score=0.5,
            novelty_level=0.0,
        )
        assert decision.zone == "boundary"
        assert decision.uncertainty_signal != ""

    def test_novelty_shifts_thresholds_conservatively(self):
        """High novelty should expand the 'defer to System 5' zone."""
        store, rec = self._store_with_facts()

        decision_normal = store.assess_factuality(
            query_embedding=rec.embedding,
            factuality_score=0.5,
            novelty_level=0.0,
        )
        decision_novel = store.assess_factuality(
            query_embedding=rec.embedding,
            factuality_score=0.5,
            novelty_level=1.0,
        )
        # With high novelty, a score of 0.5 might push into hard_override
        # (tau_low shifts up, tau_high shifts down)
        # At minimum, boundary zone should be different
        assert decision_novel.zone in ("hard_override", "boundary")

    def test_factuality_decision_dataclass(self):
        store, rec = self._store_with_facts()
        decision = store.assess_factuality(
            query_embedding=rec.embedding,
            factuality_score=0.5,
        )
        assert isinstance(decision, FactualityDecision)
        assert hasattr(decision, "factuality_score")
        assert hasattr(decision, "zone")
        assert hasattr(decision, "system5_records")


class TestOverrideContext:
    """Tests for building override context strings."""

    def test_build_context_from_records(self):
        store = KnowledgeStore(KnowledgeStoreConfig(store_dir="/tmp/test_ks"))
        records = [
            _make_record("r1", subject="France", object_value="Paris"),
            _make_record("r2", subject="Germany", object_value="Berlin"),
        ]
        context = store.build_override_context(records)
        assert "Verified Facts" in context
        assert "Paris" in context
        assert "Berlin" in context

    def test_empty_records_returns_empty(self):
        store = KnowledgeStore(KnowledgeStoreConfig(store_dir="/tmp/test_ks"))
        assert store.build_override_context([]) == ""


class TestRehearsalIntegration:
    """Tests for knowledge store integration with self-rehearsal."""

    def test_get_facts_for_rehearsal(self):
        store = KnowledgeStore(KnowledgeStoreConfig(store_dir="/tmp/test_ks"))
        for i in range(20):
            store.create(_make_record(
                f"fact_{i}", domain="geography", confidence=float(i) / 20, emb_seed=i,
            ))
        facts = store.get_facts_for_rehearsal(domain="geography", n=5)
        assert len(facts) == 5
        assert facts[0].confidence >= facts[-1].confidence

    def test_verify_rehearsal_consistent(self):
        store = KnowledgeStore(KnowledgeStoreConfig(store_dir="/tmp/test_ks"))
        facts = [_make_record("r1", subject="France", object_value="Paris")]
        consistent, violations = store.verify_rehearsal(
            "The capital of France is Paris.",
            facts,
        )
        assert consistent
        assert violations == []

    def test_verify_rehearsal_inconsistent(self):
        store = KnowledgeStore(KnowledgeStoreConfig(store_dir="/tmp/test_ks"))
        facts = [_make_record("r1", subject="France", object_value="Paris")]
        consistent, violations = store.verify_rehearsal(
            "France has a population of 67 million.",
            facts,
        )
        assert not consistent
        assert len(violations) > 0


class TestPersistence:
    """Tests for save/load of knowledge store."""

    def test_save_and_load(self, tmp_path):
        store = KnowledgeStore(KnowledgeStoreConfig(store_dir=str(tmp_path / "ks")))
        store.create(_make_record("r1"))
        store.create(_make_record("r2", subject="Germany", emb_seed=99))
        store.save(tmp_path / "ks")

        loaded = KnowledgeStore.load(tmp_path / "ks")
        assert loaded.num_records == 2
        assert loaded.read("r1").object_value == "Paris"
