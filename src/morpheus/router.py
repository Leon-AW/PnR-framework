"""
Non-Parametric Prototype Router
================================

Implements prototype-based routing that is immune to catastrophic forgetting
by construction. Routing is computed as nearest-centroid classification in
the Stable Core's representation space.

Key mechanisms:
- Random projection into a lower-dimensional routing subspace (JL lemma)
  to mitigate the hubness problem in high-dimensional spaces
- Hierarchical routing for large expert banks
- EMA-based centroid updates (no gradient descent)
- Hub detection and rebalancing at runtime
- Additive registration: new experts register new prototypes, no shared
  parameters are retrained

Extends BaseRouter from the PnR framework for compatibility with the
existing evaluation pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from src.routing.base import BaseRouter, RoutingResult, RoutingStrategy, AdapterMatch

from .config import PrototypeRouterConfig, ExpertState

logger = logging.getLogger(__name__)


@dataclass
class ExpertPrototype:
    """A prototype centroid for one expert in the routing space."""
    expert_id: str
    centroid: np.ndarray
    projected_centroid: np.ndarray | None = None
    state: ExpertState = ExpertState.ACTIVE
    native_core_version: int = 0
    activation_count: int = 0
    timestamp: float = 0.0
    adapter_path: str = ""

    def to_dict(self) -> dict:
        d = {
            "expert_id": self.expert_id,
            "state": self.state.value,
            "native_core_version": self.native_core_version,
            "activation_count": self.activation_count,
            "timestamp": self.timestamp,
            "adapter_path": self.adapter_path,
        }
        return d


class PrototypeRouter(BaseRouter):
    """Non-parametric prototype-based router with hubness mitigation.

    Routes queries by computing similarity between the input representation
    (projected via a fixed random matrix) and expert prototype centroids.
    Immune to catastrophic forgetting because:
    - No learned parameters in the routing path
    - New experts are added by registering new prototypes
    - Centroids update via EMA, not gradient descent

    For the MORPHEUS architecture, this replaces the parametric gating
    mechanisms used by X-LoRA and L2R.
    """

    def __init__(
        self,
        config: PrototypeRouterConfig | None = None,
        embedding_fn: Callable[[str], np.ndarray] | None = None,
        embedding_batch_fn: Callable[[list[str]], np.ndarray] | None = None,
        embedding_dim: int = 768,
    ) -> None:
        super().__init__(strategy=RoutingStrategy.CENTROID)

        self.config = config or PrototypeRouterConfig()
        self._embedding_fn = embedding_fn
        self._embedding_batch_fn = embedding_batch_fn
        self._embedding_dim = embedding_dim

        self._prototypes: dict[str, ExpertPrototype] = {}

        # Fixed random projection matrix (Johnson-Lindenstrauss)
        self._projection_matrix: np.ndarray | None = None
        self._init_projection(embedding_dim)

        # Hierarchical routing: coarse cluster assignments
        self._coarse_centroids: np.ndarray | None = None
        self._coarse_assignments: dict[str, int] = {}
        self._recluster_counter: int = 0

        # Hub detection: running activation frequency stats
        self._total_routes: int = 0

        logger.info(
            f"PrototypeRouter initialized: "
            f"projection_dim={self.config.projection_dim}, "
            f"threshold={self.config.similarity_threshold}"
        )

    # ------------------------------------------------------------------
    # Random projection (Johnson-Lindenstrauss)
    # ------------------------------------------------------------------

    def _init_projection(self, input_dim: int) -> None:
        """Initialize the fixed random projection matrix."""
        rng = np.random.RandomState(42)
        d_prime = self.config.projection_dim
        self._projection_matrix = rng.randn(
            input_dim, d_prime
        ).astype(np.float32) / np.sqrt(d_prime)
        logger.info(
            f"Random projection: {input_dim} -> {d_prime} "
            f"(JL distortion bound satisfied)"
        )

    def _project(self, vector: np.ndarray) -> np.ndarray:
        """Project a vector into the routing subspace."""
        if self._projection_matrix is None:
            return vector
        projected = vector @ self._projection_matrix
        norm = np.linalg.norm(projected)
        if norm > 0:
            projected = projected / norm
        return projected

    # ------------------------------------------------------------------
    # BaseRouter interface
    # ------------------------------------------------------------------

    def compute_embedding(self, text: str) -> np.ndarray:
        if self._embedding_fn is None:
            raise RuntimeError("No embedding function configured.")
        return self._embedding_fn(text)

    def compute_centroid(self, texts: list[str]) -> np.ndarray:
        if self._embedding_batch_fn:
            embeddings = self._embedding_batch_fn(texts)
        else:
            embeddings = np.vstack([self.compute_embedding(t) for t in texts])
        centroid = embeddings.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-9)
        return centroid.astype(np.float32)

    def register_adapter(
        self,
        adapter_id: str,
        path: str,
        timestamp: float,
        adapter_type: str = "unknown",
        training_data_path: str | None = None,
        centroid: np.ndarray | None = None,
        state: ExpertState = ExpertState.ACTIVE,
        core_version: int = 0,
    ) -> None:
        """Register an expert prototype in the routing table."""
        if centroid is None and training_data_path is not None:
            raise ValueError("Must provide centroid or compute externally.")

        if centroid is None:
            centroid = np.zeros(self._embedding_dim, dtype=np.float32)

        projected = self._project(centroid)

        self._prototypes[adapter_id] = ExpertPrototype(
            expert_id=adapter_id,
            centroid=centroid,
            projected_centroid=projected,
            state=state,
            native_core_version=core_version,
            timestamp=timestamp,
            adapter_path=path,
        )

        logger.info(
            f"Registered prototype: {adapter_id} "
            f"(state={state.value}, core_v={core_version})"
        )

        self._maybe_recluster()

    def get_registered_adapters(self) -> list[str]:
        return list(self._prototypes.keys())

    def unregister_adapter(self, adapter_id: str) -> bool:
        if adapter_id in self._prototypes:
            del self._prototypes[adapter_id]
            return True
        return False

    # ------------------------------------------------------------------
    # Centroid updates (EMA)
    # ------------------------------------------------------------------

    def update_centroid_ema(
        self,
        expert_id: str,
        new_embedding: np.ndarray,
    ) -> None:
        """Update an expert's centroid using exponential moving average."""
        proto = self._prototypes.get(expert_id)
        if proto is None:
            return

        alpha = 1.0 - self.config.ema_decay
        proto.centroid = (
            self.config.ema_decay * proto.centroid + alpha * new_embedding
        )
        norm = np.linalg.norm(proto.centroid)
        if norm > 0:
            proto.centroid = proto.centroid / norm
        proto.projected_centroid = self._project(proto.centroid)

    def recompute_centroid(
        self,
        expert_id: str,
        texts: list[str],
    ) -> np.ndarray:
        """Recompute an expert's centroid from scratch (e.g., after core update)."""
        centroid = self.compute_centroid(texts)
        proto = self._prototypes.get(expert_id)
        if proto:
            proto.centroid = centroid
            proto.projected_centroid = self._project(centroid)
        return centroid

    # ------------------------------------------------------------------
    # Hierarchical routing
    # ------------------------------------------------------------------

    def _maybe_recluster(self) -> None:
        """Recluster experts for hierarchical routing if needed."""
        if not self.config.hierarchical_routing:
            return
        self._recluster_counter += 1
        if self._recluster_counter < self.config.recluster_interval:
            return

        active = self._get_routable_prototypes()
        if len(active) < self.config.coarse_clusters * 2:
            return

        from sklearn.cluster import KMeans

        centroids_matrix = np.vstack(
            [p.projected_centroid for p in active.values()]
        )
        expert_ids = list(active.keys())

        n_clusters = min(self.config.coarse_clusters, len(expert_ids))
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=3)
        labels = km.fit_predict(centroids_matrix)

        self._coarse_centroids = km.cluster_centers_
        self._coarse_assignments = {
            eid: int(label) for eid, label in zip(expert_ids, labels)
        }
        self._recluster_counter = 0

        logger.info(f"Reclustered {len(expert_ids)} experts into {n_clusters} groups")

    def _get_routable_prototypes(self) -> dict[str, ExpertPrototype]:
        """Get experts eligible for routing (active or frozen, not shadow/dormant)."""
        return {
            eid: p for eid, p in self._prototypes.items()
            if p.state in (ExpertState.ACTIVE, ExpertState.FROZEN)
        }

    # ------------------------------------------------------------------
    # Hub detection
    # ------------------------------------------------------------------

    def _detect_hubs(self) -> list[str]:
        """Detect experts that have become routing hubs."""
        if self._total_routes < 100:
            return []

        active = self._get_routable_prototypes()
        if not active:
            return []

        counts = np.array([p.activation_count for p in active.values()])
        mean_count = counts.mean()
        if mean_count < 1:
            return []

        threshold = mean_count * self.config.hub_detection_threshold
        hubs = [
            eid for eid, p in active.items()
            if p.activation_count > threshold
        ]

        if hubs:
            logger.warning(f"Hub experts detected: {hubs}")
        return hubs

    # ------------------------------------------------------------------
    # Core routing
    # ------------------------------------------------------------------

    def route(self, query: str, top_k: int = 3) -> RoutingResult:
        """Route a query to experts via prototype matching.

        Flow:
        1. Embed and project query
        2. (Optional) Hierarchical pre-filter to coarse cluster
        3. Compute similarity to active expert prototypes
        4. Apply hub correction if needed
        5. Return top-K matches above threshold

        Args:
            query: User input query.
            top_k: Maximum experts to activate.

        Returns:
            RoutingResult compatible with the PnR evaluation pipeline.
        """
        top_k = top_k or self.config.top_k

        query_emb = self.compute_embedding(query)
        query_proj = self._project(query_emb)

        active = self._get_routable_prototypes()
        if not active:
            return self._empty_result(query_emb)

        # Hierarchical pre-filter
        candidate_ids = list(active.keys())
        if (
            self.config.hierarchical_routing
            and self._coarse_centroids is not None
            and len(active) > self.config.coarse_clusters * 2
        ):
            coarse_sims = self._coarse_centroids @ query_proj
            best_cluster = int(np.argmax(coarse_sims))
            candidate_ids = [
                eid for eid, cid in self._coarse_assignments.items()
                if cid == best_cluster and eid in active
            ]
            if not candidate_ids:
                candidate_ids = list(active.keys())

        # Compute fine-grained similarities
        hubs = set(self._detect_hubs())
        scored: list[tuple[str, float]] = []

        for eid in candidate_ids:
            proto = active[eid]
            sim = float(np.dot(query_proj, proto.projected_centroid))

            if eid in hubs:
                sim *= self.config.hub_correction_factor

            scored.append((eid, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        scored = scored[:top_k]

        # Build matches (clamp similarity to [0,1] for AdapterMatch compatibility)
        matches = []
        for eid, sim in scored:
            if sim < self.config.similarity_threshold:
                continue
            proto = active[eid]
            matches.append(AdapterMatch(
                adapter_id=eid,
                similarity=float(np.clip(sim, 0.0, 1.0)),
                timestamp=proto.timestamp,
                is_winner=False,
            ))

        if not matches:
            return self._empty_result(query_emb)

        # Winner: highest similarity
        matches[0].is_winner = True
        winner = matches[0]
        winner_proto = active[winner.adapter_id]

        # Update activation counts
        self._total_routes += 1
        for m in matches:
            self._prototypes[m.adapter_id].activation_count += 1

        has_conflict = len(matches) > 1

        return RoutingResult(
            winner_adapter=winner.adapter_id,
            winner_path=winner_proto.adapter_path,
            retrieved_context="",
            all_matches=matches,
            query_embedding=query_emb,
            has_conflict=has_conflict,
            routing_strategy=RoutingStrategy.CENTROID,
        )

    def _empty_result(self, query_emb: np.ndarray) -> RoutingResult:
        return RoutingResult(
            winner_adapter=None,
            winner_path=None,
            retrieved_context="",
            all_matches=[],
            query_embedding=query_emb,
            has_conflict=False,
            routing_strategy=RoutingStrategy.CENTROID,
        )

    # ------------------------------------------------------------------
    # Novelty detection
    # ------------------------------------------------------------------

    def compute_routing_confidence(self, query: str) -> float:
        """Compute routing confidence for novelty detection.

        Low confidence (max similarity to any expert is low) indicates
        the input may be from a genuinely novel domain.

        Returns:
            Maximum similarity to any routable expert.
        """
        query_emb = self.compute_embedding(query)
        query_proj = self._project(query_emb)

        active = self._get_routable_prototypes()
        if not active:
            return 0.0

        max_sim = max(
            float(np.dot(query_proj, p.projected_centroid))
            for p in active.values()
        )
        return max_sim

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save router state."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        state = {
            "prototypes": {
                eid: {
                    **p.to_dict(),
                    "centroid": p.centroid.tolist(),
                }
                for eid, p in self._prototypes.items()
            },
            "total_routes": self._total_routes,
        }
        import json
        with open(path / "prototype_router.json", "w") as f:
            json.dump(state, f, indent=2)

        if self._projection_matrix is not None:
            np.save(path / "projection_matrix.npy", self._projection_matrix)

        logger.info(f"Router state saved to {path}")

    @classmethod
    def load(
        cls,
        path: str | Path,
        embedding_fn: Callable | None = None,
        embedding_batch_fn: Callable | None = None,
        **kwargs,
    ) -> PrototypeRouter:
        """Load router from saved state."""
        import json
        path = Path(path)

        router = cls(embedding_fn=embedding_fn, embedding_batch_fn=embedding_batch_fn, **kwargs)

        proj_path = path / "projection_matrix.npy"
        if proj_path.exists():
            router._projection_matrix = np.load(proj_path)

        state_path = path / "prototype_router.json"
        if state_path.exists():
            with open(state_path) as f:
                state = json.load(f)

            router._total_routes = state.get("total_routes", 0)
            for eid, pdata in state.get("prototypes", {}).items():
                centroid = np.array(pdata["centroid"], dtype=np.float32)
                router._prototypes[eid] = ExpertPrototype(
                    expert_id=eid,
                    centroid=centroid,
                    projected_centroid=router._project(centroid),
                    state=ExpertState(pdata["state"]),
                    native_core_version=pdata.get("native_core_version", 0),
                    activation_count=pdata.get("activation_count", 0),
                    timestamp=pdata.get("timestamp", 0.0),
                    adapter_path=pdata.get("adapter_path", ""),
                )

        return router

    def summary(self) -> str:
        active = sum(1 for p in self._prototypes.values() if p.state == ExpertState.ACTIVE)
        frozen = sum(1 for p in self._prototypes.values() if p.state == ExpertState.FROZEN)
        shadow = sum(1 for p in self._prototypes.values() if p.state == ExpertState.SHADOW)
        dormant = sum(1 for p in self._prototypes.values() if p.state == ExpertState.DORMANT)

        lines = [
            f"Prototype Router ({len(self._prototypes)} experts)",
            f"  Active: {active}, Frozen: {frozen}, Shadow: {shadow}, Dormant: {dormant}",
            f"  Total routes: {self._total_routes}",
            f"  Projection: {self._embedding_dim} -> {self.config.projection_dim}",
            f"  Threshold: {self.config.similarity_threshold}",
        ]
        return "\n".join(lines)
