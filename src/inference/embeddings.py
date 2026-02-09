"""
Embedding Model Wrapper
=======================

Provides embedding functionality for RAG retrieval using
sentence-transformers models.

Features:
- Multilingual support (important for German QM documents)
- Batch embedding for efficiency
- Configurable model selection
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class EmbeddingConfig:
    """Configuration for embedding model.

    Attributes:
        model_name: HuggingFace model name or local path
        device: Device to run on ('cuda', 'cpu', or None for auto)
        batch_size: Batch size for encoding
        normalize_embeddings: Whether to L2-normalize embeddings
        show_progress_bar: Show progress bar during encoding
    """
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    device: Optional[str] = None
    batch_size: int = 32
    normalize_embeddings: bool = True
    show_progress_bar: bool = True


# =============================================================================
# Embedding Model
# =============================================================================

class EmbeddingModel:
    """Wrapper for sentence-transformers embedding models.

    Example:
        ```python
        config = EmbeddingConfig(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        model = EmbeddingModel(config)

        # Single text
        embedding = model.encode("Was ist Mikrohärteprüfung?")

        # Batch encoding
        embeddings = model.encode_batch([
            "Prüfverfahren in der Metallografie",
            "Korngrößenbestimmung laut ASTM E112",
        ])
        ```
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """Initialize the embedding model.

        Args:
            config: Embedding configuration (uses defaults if None)
        """
        self.config = config or EmbeddingConfig()
        self._model = None
        self._dimension: Optional[int] = None

    def _load_model(self):
        """Lazy-load the sentence-transformers model."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for embeddings. "
                "Install with: pip install sentence-transformers"
            )

        logger.info(f"Loading embedding model: {self.config.model_name}")
        self._model = SentenceTransformer(
            self.config.model_name,
            device=self.config.device,
        )
        self._dimension = self._model.get_sentence_embedding_dimension()
        logger.info(f"Loaded model with dimension: {self._dimension}")

    @property
    def model(self):
        """Get the underlying SentenceTransformer model."""
        self._load_model()
        return self._model

    @property
    def dimension(self) -> int:
        """Get the embedding dimension."""
        self._load_model()
        return self._dimension

    def encode(self, text: str) -> np.ndarray:
        """Encode a single text to embedding.

        Args:
            text: Text to encode

        Returns:
            Embedding vector as numpy array
        """
        self._load_model()
        return self._model.encode(
            text,
            normalize_embeddings=self.config.normalize_embeddings,
            show_progress_bar=False,
        )

    def encode_batch(
        self,
        texts: list[str],
        show_progress_bar: Optional[bool] = None,
    ) -> np.ndarray:
        """Encode multiple texts to embeddings.

        Args:
            texts: List of texts to encode
            show_progress_bar: Override config setting for progress bar

        Returns:
            Embedding matrix as numpy array (num_texts x dimension)
        """
        self._load_model()

        if show_progress_bar is None:
            show_progress_bar = self.config.show_progress_bar

        return self._model.encode(
            texts,
            batch_size=self.config.batch_size,
            normalize_embeddings=self.config.normalize_embeddings,
            show_progress_bar=show_progress_bar,
        )

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        """Encode queries for retrieval.

        Some models use different encoding for queries vs documents.
        This method handles that distinction.

        Args:
            queries: List of query texts

        Returns:
            Query embeddings
        """
        # Most models don't distinguish, but this provides a hook
        return self.encode_batch(queries, show_progress_bar=False)

    def encode_documents(
        self,
        documents: list[str],
        show_progress_bar: Optional[bool] = None,
    ) -> np.ndarray:
        """Encode documents for indexing.

        Args:
            documents: List of document texts
            show_progress_bar: Override config setting

        Returns:
            Document embeddings
        """
        return self.encode_batch(documents, show_progress_bar=show_progress_bar)

    def similarity(self, query_embedding: np.ndarray, doc_embeddings: np.ndarray) -> np.ndarray:
        """Compute similarity between query and documents.

        Args:
            query_embedding: Single query embedding (dimension,)
            doc_embeddings: Document embeddings (num_docs x dimension)

        Returns:
            Similarity scores (num_docs,)
        """
        # Cosine similarity (embeddings are normalized)
        if self.config.normalize_embeddings:
            return np.dot(doc_embeddings, query_embedding)
        else:
            # Normalize on the fly
            query_norm = query_embedding / np.linalg.norm(query_embedding)
            doc_norms = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
            return np.dot(doc_norms, query_norm)

    def save(self, path: Union[str, Path]):
        """Save the model to a directory.

        Args:
            path: Directory to save to
        """
        self._load_model()
        self._model.save(str(path))
        logger.info(f"Saved embedding model to: {path}")

    @classmethod
    def load(cls, path: Union[str, Path], config: Optional[EmbeddingConfig] = None) -> "EmbeddingModel":
        """Load a saved model from directory.

        Args:
            path: Directory containing saved model
            config: Optional config (model_name will be overridden)

        Returns:
            EmbeddingModel instance
        """
        config = config or EmbeddingConfig()
        config.model_name = str(path)
        return cls(config)
