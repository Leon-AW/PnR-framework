"""
System 5 — Explicit Knowledge Store ("Episodic/Declarative Memory")
====================================================================

A non-parametric store for discrete facts, events, entities, and relationships.
Unlike parametric knowledge in neural network weights, entries here support
CRUD operations: facts can be added, corrected, or removed without touching
any model parameters.

Key mechanisms:
- Graduated factuality scoring: continuous s in [0, 1] instead of binary
  fact/reasoning classification
- Hard architectural override: factual claims with high System 5 confidence
  MUST come from the store, not parametric memory
- Soft middle zone: boundary cases get weighted combination with explicit
  uncertainty signaling
- Novelty-aware threshold adaptation: thresholds shift conservatively when
  the system encounters novel domains (coupled with System 6)
- Self-consistency verification for borderline cases
- Deeply integrated with rehearsal (facts anchor generated training data)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np

from .config import KnowledgeStoreConfig

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeRecord:
    """A single record in the knowledge store."""
    record_id: str
    subject: str
    predicate: str
    object_value: str
    source: str = ""
    timestamp: float = field(default_factory=time.time)
    confidence: float = 1.0
    domain: str = "general"
    embedding: np.ndarray | None = field(default=None, repr=False)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def fact_text(self) -> str:
        return f"{self.subject} {self.predicate} {self.object_value}"

    def to_dict(self) -> dict:
        d = asdict(self)
        if d.get("embedding") is not None:
            d["embedding"] = d["embedding"].tolist()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> KnowledgeRecord:
        d = d.copy()
        if d.get("embedding") is not None:
            d["embedding"] = np.array(d["embedding"], dtype=np.float32)
        return cls(**d)


@dataclass
class FactualityDecision:
    """Result of the graduated factuality assessment."""
    factuality_score: float
    zone: str  # "hard_override", "parametric_freedom", "boundary"
    system5_records: list[KnowledgeRecord]
    confidence: float
    uncertainty_signal: str = ""


class KnowledgeStore:
    """System 5: Explicit Knowledge Store with graduated factuality.

    Stores discrete facts in a searchable, non-parametric database.
    At inference time, retrieved facts override parametric knowledge
    for clearly factual claims (graduated factuality protocol).

    Integration points:
    - Inference: facts retrieved and injected into context
    - Rehearsal: facts constrain generated training examples
    - Consolidation: decides what's a "fact" vs. a "skill"
    - Meta-controller: signals for novel domain detection
    """

    def __init__(self, config: KnowledgeStoreConfig | None = None) -> None:
        self.config = config or KnowledgeStoreConfig()

        self._records: dict[str, KnowledgeRecord] = {}
        self._embeddings: np.ndarray | None = None
        self._record_ids_index: list[str] = []

        # Graduated factuality thresholds (adaptable by System 6)
        self._tau_high = self.config.factuality_threshold_high
        self._tau_low = self.config.factuality_threshold_low

        self._store_dir = Path(self.config.store_dir)
        self._store_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"KnowledgeStore initialized "
            f"(tau_high={self._tau_high}, tau_low={self._tau_low})"
        )

    @property
    def num_records(self) -> int:
        return len(self._records)

    # ------------------------------------------------------------------
    # CRUD operations
    # ------------------------------------------------------------------

    def create(self, record: KnowledgeRecord) -> str:
        """Add a new fact to the store."""
        self._records[record.record_id] = record
        self._invalidate_index()
        logger.debug(f"Created record: {record.record_id}")
        return record.record_id

    def read(self, record_id: str) -> KnowledgeRecord | None:
        """Read a fact by ID."""
        return self._records.get(record_id)

    def update(self, record_id: str, **updates) -> bool:
        """Update fields of an existing record."""
        record = self._records.get(record_id)
        if record is None:
            return False

        for key, value in updates.items():
            if hasattr(record, key):
                setattr(record, key, value)

        record.timestamp = time.time()
        self._invalidate_index()
        logger.debug(f"Updated record: {record_id}")
        return True

    def delete(self, record_id: str) -> bool:
        """Remove a fact from the store."""
        if record_id in self._records:
            del self._records[record_id]
            self._invalidate_index()
            logger.debug(f"Deleted record: {record_id}")
            return True
        return False

    def _invalidate_index(self) -> None:
        """Mark the search index as needing rebuild."""
        self._embeddings = None
        self._record_ids_index = []

    # ------------------------------------------------------------------
    # Search / Retrieval
    # ------------------------------------------------------------------

    def _build_index(self) -> None:
        """Build the embedding index for search."""
        records_with_emb = [
            (rid, r) for rid, r in self._records.items()
            if r.embedding is not None
        ]
        if not records_with_emb:
            return

        self._record_ids_index = [rid for rid, _ in records_with_emb]
        self._embeddings = np.vstack(
            [r.embedding for _, r in records_with_emb]
        )

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        min_confidence: float = 0.3,
    ) -> list[tuple[KnowledgeRecord, float]]:
        """Search for relevant facts by embedding similarity.

        Args:
            query_embedding: Query vector.
            top_k: Number of results.
            min_confidence: Minimum record confidence.

        Returns:
            List of (record, similarity) tuples, sorted by similarity.
        """
        if self._embeddings is None:
            self._build_index()

        if self._embeddings is None or len(self._embeddings) == 0:
            return []

        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)
        similarities = self._embeddings @ query_norm
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            rid = self._record_ids_index[idx]
            record = self._records[rid]
            sim = float(similarities[idx])
            if record.confidence >= min_confidence and sim > 0:
                results.append((record, sim))

        return results

    def search_by_subject(
        self,
        subject: str,
        predicate: str | None = None,
    ) -> list[KnowledgeRecord]:
        """Search by subject (and optionally predicate) text matching."""
        subject_lower = subject.lower()
        results = []
        for record in self._records.values():
            if subject_lower in record.subject.lower():
                if predicate is None or predicate.lower() in record.predicate.lower():
                    results.append(record)
        results.sort(key=lambda r: r.timestamp, reverse=True)
        return results

    # ------------------------------------------------------------------
    # Graduated factuality assessment
    # ------------------------------------------------------------------

    def assess_factuality(
        self,
        query_embedding: np.ndarray,
        factuality_score: float | None = None,
        novelty_level: float = 0.0,
    ) -> FactualityDecision:
        """Assess whether System 5 should override parametric output.

        Implements the graduated factuality protocol:
        - s > tau_high: Hard override (System 5 controls factual content)
        - s < tau_low: Parametric freedom (experts/core control content)
        - tau_low <= s <= tau_high: Weighted combination with uncertainty

        The thresholds adapt based on novelty_level from System 6:
        higher novelty shifts toward more conservative (System 5 dominant)
        posture.

        Args:
            query_embedding: Embedded query vector.
            factuality_score: Continuous score in [0, 1] from a learned
                classifier (arch doc §249). When ``None`` (the default), the
                score is derived from retrieval evidence: ``max_sim`` of the
                top-k KB matches serves as the proxy. A tight KB hit is
                strong evidence the query concerns a stored fact, which is
                precisely what the classifier would score high anyway. Pass
                an explicit value once a trained classifier is available.
            novelty_level: Distribution novelty from meta-controller [0, 1].

        Returns:
            FactualityDecision with zone classification and relevant records.
        """
        # Novelty-aware threshold adaptation (coupling with System 6)
        shift = novelty_level * self.config.novelty_threshold_shift
        tau_high = self._tau_high - shift
        tau_low = self._tau_low + shift

        records_result = self.search(query_embedding, top_k=3)
        records = [r for r, _ in records_result]
        max_sim = max((s for _, s in records_result), default=0.0)

        # Retrieval-driven default: use KB similarity as the factuality proxy.
        # Without this, a hardcoded mid-range score (e.g. 0.5) never exceeds
        # tau_high=0.8, so hard_override is unreachable and the entire
        # graduated-factuality hierarchy collapses to "always boundary".
        if factuality_score is None:
            factuality_score = max_sim

        if factuality_score > tau_high and records and max_sim > 0.5:
            zone = "hard_override"
            uncertainty = ""
        elif factuality_score < tau_low:
            zone = "parametric_freedom"
            uncertainty = ""
        else:
            zone = "boundary"
            uncertainty = (
                f"Factuality score {factuality_score:.2f} is in the boundary "
                f"zone [{tau_low:.2f}, {tau_high:.2f}]. Both retrieved evidence "
                f"and parametric reasoning may be relevant."
            )

        return FactualityDecision(
            factuality_score=factuality_score,
            zone=zone,
            system5_records=records,
            confidence=max_sim,
            uncertainty_signal=uncertainty,
        )

    def build_override_context(
        self,
        records: list[KnowledgeRecord],
        max_length: int = 1000,
    ) -> str:
        """Build context string from retrieved knowledge records.

        Used to inject factual knowledge into the prompt when the
        graduated factuality protocol determines System 5 should override.
        """
        if not records:
            return ""

        parts = ["[Verified Facts from Knowledge Store]"]
        total_len = len(parts[0])

        for record in records:
            fact_line = f"- {record.fact_text} (confidence: {record.confidence:.2f})"
            if total_len + len(fact_line) > max_length:
                break
            parts.append(fact_line)
            total_len += len(fact_line)

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Rehearsal integration
    # ------------------------------------------------------------------

    def get_facts_for_rehearsal(
        self,
        domain: str | None = None,
        n: int = 50,
    ) -> list[KnowledgeRecord]:
        """Get facts for constraining self-rehearsal generation.

        During "dreaming" (System 4a), the knowledge store acts as a
        fact-checker to prevent model collapse and factual drift.
        """
        records = list(self._records.values())
        if domain:
            records = [r for r in records if r.domain == domain]

        records.sort(key=lambda r: r.confidence, reverse=True)
        return records[:n]

    def verify_rehearsal(
        self,
        generated_text: str,
        relevant_facts: list[KnowledgeRecord],
    ) -> tuple[bool, list[str]]:
        """Verify that generated rehearsal doesn't contradict stored facts.

        Returns (is_consistent, list_of_violations).
        This is a lightweight string-matching check; the full verification
        uses the discriminator network in System 4.
        """
        violations = []
        text_lower = generated_text.lower()

        for fact in relevant_facts:
            subject_lower = fact.subject.lower()
            if subject_lower in text_lower:
                if fact.object_value.lower() not in text_lower:
                    violations.append(
                        f"Generated text mentions '{fact.subject}' but may "
                        f"contradict: expected '{fact.object_value}'"
                    )

        return len(violations) == 0, violations

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path | None = None) -> Path:
        """Save the knowledge store to disk."""
        path = Path(path or self.config.store_dir)
        path.mkdir(parents=True, exist_ok=True)

        records_path = path / "records.json"
        with open(records_path, "w") as f:
            records_data = {
                rid: r.to_dict() for rid, r in self._records.items()
            }
            json.dump(records_data, f, indent=2, default=str)

        logger.info(f"KnowledgeStore saved: {self.num_records} records")
        return path

    @classmethod
    def load(cls, path: str | Path, config: KnowledgeStoreConfig | None = None) -> KnowledgeStore:
        """Load a knowledge store from disk."""
        path = Path(path)
        store = cls(config=config)

        records_path = path / "records.json"
        if records_path.exists():
            with open(records_path) as f:
                records_data = json.load(f)
            store._records = {
                rid: KnowledgeRecord.from_dict(d)
                for rid, d in records_data.items()
            }
            logger.info(f"KnowledgeStore loaded: {store.num_records} records")

        return store

    def summary(self) -> str:
        domains = {}
        for r in self._records.values():
            domains[r.domain] = domains.get(r.domain, 0) + 1

        lines = [f"KnowledgeStore ({self.num_records} records)"]
        for domain, count in sorted(domains.items()):
            lines.append(f"  {domain}: {count}")
        return "\n".join(lines)
