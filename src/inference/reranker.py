"""
Cross-Encoder Reranker
======================

Reranks retrieval candidates using a cross-encoder model
for improved precision.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from src.inference.vector_store import SearchResult

logger = logging.getLogger(__name__)


@dataclass
class RerankerConfig:
    """Configuration for the cross-encoder reranker.

    Attributes:
        model_name: Cross-encoder model name (HuggingFace)
        device: Device to run on ('cuda', 'cpu', or None for auto)
        batch_size: Batch size for scoring
        max_length: Maximum input sequence length
    """
    model_name: str = "BAAI/bge-reranker-v2-m3"
    device: Optional[str] = None
    batch_size: int = 16
    max_length: int = 512


class Reranker:
    """Cross-encoder reranker for retrieval candidates.

    Uses a cross-encoder model to score query-document pairs,
    providing more accurate relevance scores than bi-encoder retrieval.

    Example:
        ```python
        reranker = Reranker(RerankerConfig())
        reranked = reranker.rerank(
            query="Was ist Akkreditierung?",
            candidates=retrieval_results,
            top_k=5,
        )
        ```
    """

    def __init__(self, config: Optional[RerankerConfig] = None):
        self.config = config or RerankerConfig()
        self._model = None

    def _load_model(self):
        """Lazy-load the cross-encoder model."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for reranking. "
                "Install with: pip install sentence-transformers"
            )

        logger.info(f"Loading reranker model: {self.config.model_name}")
        self._model = CrossEncoder(
            self.config.model_name,
            max_length=self.config.max_length,
            device=self.config.device,
        )
        logger.info("Reranker model loaded")

    def warmup(self):
        """Eagerly load the model and run a dummy prediction to initialize CUDA/cuBLAS."""
        self._load_model()
        self._model.predict([("warmup", "warmup")], show_progress_bar=False)
        logger.info("Reranker warmup complete")

    def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
    ) -> list[SearchResult]:
        """Rerank candidates using cross-encoder scores.

        Args:
            query: The search query
            candidates: List of retrieval candidates
            top_k: Number of results to return (default: all)
            min_score: Minimum relevance score threshold (default: no filtering)

        Returns:
            Reranked list of SearchResult objects with updated scores
        """
        if not candidates:
            return []

        self._load_model()

        # Build query-document pairs
        pairs = [(query, c.content) for c in candidates]

        # Score in batches
        scores = self._model.predict(
            pairs,
            batch_size=self.config.batch_size,
            show_progress_bar=False,
        )

        # Create results with cross-encoder scores
        scored = []
        for candidate, score in zip(candidates, scores):
            scored.append(SearchResult(
                id=candidate.id,
                score=float(score),
                content=candidate.content,
                metadata=candidate.metadata,
            ))

        # Sort by score descending
        scored.sort(key=lambda x: x.score, reverse=True)

        # Apply minimum score threshold before top_k truncation
        if min_score is not None:
            scored = [r for r in scored if r.score >= min_score]

        if top_k is not None:
            scored = scored[:top_k]

        return scored
