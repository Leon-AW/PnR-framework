"""
Base Router Interface
=====================

Defines the abstract interface for routing strategies in the Patch-and-Route framework.

This module implements the Strategy Pattern, allowing different routing strategies
to be swapped without changing the inference pipeline.

Current Strategy:
- CentroidRouter: Time-Aware Centroid matching with Source-Replay

Future Strategies (planned):
- ParallelOrchestrator: Multi-adapter parallel inference with MoA aggregation

Reference: Section 4.4 of the Master's Thesis Exposé
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .manifest import AdapterEntry


class RoutingStrategy(Enum):
    """Available routing strategies."""
    CENTROID = "centroid"           # Time-Aware Centroid Router (Section 4.4.1)
    PARALLEL = "parallel"           # Parallel Orchestrator (Section 4.4.2) - Future
    ENSEMBLE = "ensemble"           # Ensemble voting - Future


@dataclass
class AdapterMatch:
    """Represents a matched adapter during routing.
    
    Attributes:
        adapter_id: Unique identifier for the adapter.
        similarity: Cosine similarity score (0-1).
        timestamp: Training timestamp for conflict resolution.
        is_winner: True if this adapter should be weight-loaded.
        retrieved_context: If loser, contains retrieved text chunks.
    """
    adapter_id: str
    similarity: float
    timestamp: float
    is_winner: bool = False
    retrieved_context: list[str] = field(default_factory=list)
    
    def __post_init__(self) -> None:
        """Validate similarity score."""
        if not 0.0 <= self.similarity <= 1.0:
            raise ValueError(f"Similarity must be in [0, 1], got {self.similarity}")


@dataclass
class RoutingResult:
    """Result of the routing decision.
    
    Contains all information needed for the inference pipeline:
    - Which adapter to load (weight loading)
    - What context to prepend (from source-replay)
    - Confidence metrics
    
    Attributes:
        winner_adapter: Adapter ID to load for weight loading (T_new).
        winner_path: Filesystem path to the adapter checkpoint.
        retrieved_context: Aggregated context from loser adapters (T_old).
        all_matches: All adapters that matched above threshold.
        query_embedding: The embedded query vector.
        has_conflict: True if multiple adapters matched (conflict detected).
        routing_strategy: The strategy that produced this result.
    """
    winner_adapter: str | None
    winner_path: str | None
    retrieved_context: str
    all_matches: list[AdapterMatch]
    query_embedding: np.ndarray
    has_conflict: bool = False
    routing_strategy: RoutingStrategy = RoutingStrategy.CENTROID
    
    @property
    def num_matches(self) -> int:
        """Number of adapters that matched."""
        return len(self.all_matches)
    
    @property
    def winner_similarity(self) -> float | None:
        """Similarity score of the winning adapter."""
        for match in self.all_matches:
            if match.is_winner:
                return match.similarity
        return None
    
    @property
    def loser_adapters(self) -> list[str]:
        """List of adapter IDs that lost (source-replay only)."""
        return [m.adapter_id for m in self.all_matches if not m.is_winner]
    
    def to_dict(self) -> dict:
        """Convert to dictionary for logging/serialization."""
        return {
            "winner_adapter": self.winner_adapter,
            "winner_path": self.winner_path,
            "has_conflict": self.has_conflict,
            "num_matches": self.num_matches,
            "winner_similarity": self.winner_similarity,
            "loser_adapters": self.loser_adapters,
            "routing_strategy": self.routing_strategy.value,
            "context_length": len(self.retrieved_context),
        }


class BaseRouter(ABC):
    """Abstract base class for routing strategies.
    
    Implements the Strategy Pattern to allow swapping routing logic
    without changing the inference pipeline.
    
    Subclasses must implement:
    - route(): Main routing logic
    - register_adapter(): Add adapter to the routing pool
    - compute_embedding(): Generate query embedding
    
    Example:
        ```python
        # Using CentroidRouter (Section 4.4.1)
        router = CentroidRouter(
            embedding_model_path="/path/to/KaLM-Embedding",
            similarity_threshold=0.7,
        )
        
        # Register adapters
        router.register_adapter(
            adapter_id="base_v1",
            path="checkpoints/base_v1",
            timestamp=1609459200.0,  # 2021-01-01
            training_data_path="data/base_training.jsonl",
        )
        
        # Route a query
        result = router.route("Who is the CEO of Google in 2023?")
        
        # Result contains:
        # - winner_adapter: The adapter to load
        # - retrieved_context: Context from older conflicting adapters
        ```
    """
    
    def __init__(self, strategy: RoutingStrategy = RoutingStrategy.CENTROID) -> None:
        """Initialize the router.
        
        Args:
            strategy: The routing strategy this router implements.
        """
        self._strategy = strategy
    
    @property
    def strategy(self) -> RoutingStrategy:
        """Get the routing strategy."""
        return self._strategy
    
    @abstractmethod
    def route(self, query: str, top_k: int = 3) -> RoutingResult:
        """Route a query to the appropriate adapter(s).
        
        This is the main entry point for the routing logic.
        
        Args:
            query: The user's input query.
            top_k: Maximum number of matching adapters to consider.
            
        Returns:
            RoutingResult containing the winner adapter and any retrieved context.
        """
        pass
    
    @abstractmethod
    def register_adapter(
        self,
        adapter_id: str,
        path: str,
        timestamp: float,
        training_data_path: str | None = None,
        centroid: np.ndarray | None = None,
    ) -> None:
        """Register an adapter with the router.
        
        Adapters must be registered before they can be considered for routing.
        If centroid is not provided, it will be computed from training_data_path.
        
        Args:
            adapter_id: Unique identifier for the adapter.
            path: Filesystem path to the adapter checkpoint.
            timestamp: Training timestamp (epoch seconds) for conflict resolution.
            training_data_path: Path to training data JSONL for centroid computation.
            centroid: Pre-computed centroid vector (optional).
        """
        pass
    
    @abstractmethod
    def compute_embedding(self, text: str) -> np.ndarray:
        """Compute embedding vector for a text.
        
        Uses the configured embedding model.
        
        Args:
            text: Input text to embed.
            
        Returns:
            Embedding vector as numpy array.
        """
        pass
    
    @abstractmethod
    def compute_centroid(self, texts: list[str]) -> np.ndarray:
        """Compute centroid (mean vector) of multiple texts.
        
        Used during adapter registration to create the adapter's signature.
        
        Args:
            texts: List of training texts.
            
        Returns:
            Mean embedding vector.
        """
        pass
    
    def get_registered_adapters(self) -> list[str]:
        """Get list of registered adapter IDs.
        
        Returns:
            List of adapter identifiers.
        """
        raise NotImplementedError("Subclass must implement get_registered_adapters()")
    
    def unregister_adapter(self, adapter_id: str) -> bool:
        """Remove an adapter from the routing pool.
        
        Args:
            adapter_id: Adapter to remove.
            
        Returns:
            True if adapter was removed, False if not found.
        """
        raise NotImplementedError("Subclass must implement unregister_adapter()")

