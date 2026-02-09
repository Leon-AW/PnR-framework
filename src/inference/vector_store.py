"""
Vector Store Abstractions
=========================

Provides vector storage and retrieval backends for RAG.

Supported backends:
- FAISS: Fast in-memory search, good for local usage
- ChromaDB: Persistent storage with metadata filtering
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SearchResult:
    """A single search result.

    Attributes:
        id: Document/chunk ID
        score: Similarity score
        content: Original text content
        metadata: Additional metadata
    """
    id: str
    score: float
    content: str
    metadata: dict = field(default_factory=dict)


# =============================================================================
# Base Vector Store
# =============================================================================

class BaseVectorStore(ABC):
    """Abstract base class for vector stores."""

    @abstractmethod
    def add(
        self,
        ids: list[str],
        embeddings: np.ndarray,
        contents: list[str],
        metadatas: Optional[list[dict]] = None,
    ) -> None:
        """Add documents to the store.

        Args:
            ids: Document IDs
            embeddings: Embedding vectors (num_docs x dimension)
            contents: Document texts
            metadatas: Optional metadata dictionaries
        """
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        filter_metadata: Optional[dict] = None,
    ) -> list[SearchResult]:
        """Search for similar documents.

        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            filter_metadata: Optional metadata filter

        Returns:
            List of SearchResult objects
        """
        pass

    @abstractmethod
    def delete(self, ids: list[str]) -> None:
        """Delete documents by ID.

        Args:
            ids: Document IDs to delete
        """
        pass

    @abstractmethod
    def save(self, path: Union[str, Path]) -> None:
        """Save the vector store to disk.

        Args:
            path: Directory to save to
        """
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: Union[str, Path]) -> "BaseVectorStore":
        """Load a vector store from disk.

        Args:
            path: Directory to load from

        Returns:
            Vector store instance
        """
        pass

    @property
    @abstractmethod
    def count(self) -> int:
        """Get the number of documents in the store."""
        pass


# =============================================================================
# FAISS Vector Store
# =============================================================================

@dataclass
class FAISSConfig:
    """Configuration for FAISS vector store.

    Attributes:
        dimension: Embedding dimension (auto-detected if None)
        index_type: FAISS index type ('flat', 'ivf', 'hnsw')
        nlist: Number of clusters for IVF index
        nprobe: Number of clusters to search for IVF
        metric: Distance metric ('cosine', 'l2', 'ip')
    """
    dimension: Optional[int] = None
    index_type: str = "flat"
    nlist: int = 100
    nprobe: int = 10
    metric: str = "cosine"


class FAISSVectorStore(BaseVectorStore):
    """FAISS-based vector store.

    Fast in-memory vector search using Facebook AI Similarity Search.

    Example:
        ```python
        store = FAISSVectorStore(FAISSConfig(dimension=384))

        # Add documents
        store.add(
            ids=["doc1", "doc2"],
            embeddings=np.array([[0.1, 0.2, ...], [0.3, 0.4, ...]]),
            contents=["First document", "Second document"],
            metadatas=[{"source": "file1.md"}, {"source": "file2.md"}],
        )

        # Search
        results = store.search(query_embedding, k=5)
        ```
    """

    def __init__(self, config: Optional[FAISSConfig] = None):
        """Initialize FAISS vector store.

        Args:
            config: FAISS configuration
        """
        self.config = config or FAISSConfig()
        self._index = None
        self._id_to_idx: dict[str, int] = {}
        self._idx_to_id: dict[int, str] = {}
        self._contents: dict[str, str] = {}
        self._metadatas: dict[str, dict] = {}
        self._dimension = self.config.dimension
        self._faiss = None

    def _import_faiss(self):
        """Lazy import FAISS."""
        if self._faiss is not None:
            return

        try:
            import faiss
            self._faiss = faiss
        except ImportError:
            raise ImportError(
                "faiss is required for FAISSVectorStore. "
                "Install with: pip install faiss-cpu (or faiss-gpu)"
            )

    def _create_index(self, dimension: int):
        """Create FAISS index.

        Args:
            dimension: Embedding dimension
        """
        self._import_faiss()
        self._dimension = dimension

        if self.config.index_type == "flat":
            if self.config.metric == "cosine" or self.config.metric == "ip":
                self._index = self._faiss.IndexFlatIP(dimension)
            else:
                self._index = self._faiss.IndexFlatL2(dimension)
        elif self.config.index_type == "ivf":
            quantizer = self._faiss.IndexFlatIP(dimension)
            self._index = self._faiss.IndexIVFFlat(
                quantizer, dimension, self.config.nlist
            )
            self._index.nprobe = self.config.nprobe
        elif self.config.index_type == "hnsw":
            self._index = self._faiss.IndexHNSWFlat(dimension, 32)
        else:
            raise ValueError(f"Unknown index type: {self.config.index_type}")

    def add(
        self,
        ids: list[str],
        embeddings: np.ndarray,
        contents: list[str],
        metadatas: Optional[list[dict]] = None,
    ) -> None:
        """Add documents to the store."""
        if len(ids) == 0:
            return

        embeddings = np.asarray(embeddings, dtype=np.float32)

        # Create index if needed
        if self._index is None:
            self._create_index(embeddings.shape[1])

        # Normalize for cosine similarity
        if self.config.metric == "cosine":
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.maximum(norms, 1e-10)

        # Train IVF index if needed
        if self.config.index_type == "ivf" and not self._index.is_trained:
            self._index.train(embeddings)

        # Add to index
        start_idx = len(self._id_to_idx)
        self._index.add(embeddings)

        # Store mappings
        if metadatas is None:
            metadatas = [{} for _ in ids]

        for i, (doc_id, content, metadata) in enumerate(zip(ids, contents, metadatas)):
            idx = start_idx + i
            self._id_to_idx[doc_id] = idx
            self._idx_to_id[idx] = doc_id
            self._contents[doc_id] = content
            self._metadatas[doc_id] = metadata

        logger.debug(f"Added {len(ids)} documents to FAISS index")

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        filter_metadata: Optional[dict] = None,
    ) -> list[SearchResult]:
        """Search for similar documents."""
        if self._index is None or self._index.ntotal == 0:
            return []

        query_embedding = np.asarray(query_embedding, dtype=np.float32)

        # Normalize for cosine similarity
        if self.config.metric == "cosine":
            norm = np.linalg.norm(query_embedding)
            if norm > 0:
                query_embedding = query_embedding / norm

        # Reshape for FAISS
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Search more if we need to filter
        search_k = k * 3 if filter_metadata else k

        scores, indices = self._index.search(query_embedding, min(search_k, self._index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue

            doc_id = self._idx_to_id.get(idx)
            if doc_id is None:
                continue

            metadata = self._metadatas.get(doc_id, {})

            # Apply filter
            if filter_metadata:
                match = all(
                    metadata.get(key) == value
                    for key, value in filter_metadata.items()
                )
                if not match:
                    continue

            results.append(SearchResult(
                id=doc_id,
                score=float(score),
                content=self._contents.get(doc_id, ""),
                metadata=metadata,
            ))

            if len(results) >= k:
                break

        return results

    def delete(self, ids: list[str]) -> None:
        """Delete documents by ID.

        Note: FAISS doesn't support deletion efficiently.
        This marks them as deleted but doesn't free memory.
        """
        for doc_id in ids:
            if doc_id in self._id_to_idx:
                idx = self._id_to_idx[doc_id]
                del self._id_to_idx[doc_id]
                del self._idx_to_id[idx]
                del self._contents[doc_id]
                del self._metadatas[doc_id]

        logger.warning(
            "FAISS deletion doesn't free memory. "
            "Rebuild index if memory is a concern."
        )

    def save(self, path: Union[str, Path]) -> None:
        """Save the vector store to disk."""
        self._import_faiss()
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        if self._index is not None:
            self._faiss.write_index(self._index, str(path / "index.faiss"))

        # Save metadata
        metadata = {
            "config": {
                "dimension": self._dimension,
                "index_type": self.config.index_type,
                "nlist": self.config.nlist,
                "nprobe": self.config.nprobe,
                "metric": self.config.metric,
            },
            "id_to_idx": self._id_to_idx,
            "contents": self._contents,
            "metadatas": self._metadatas,
        }
        with open(path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved FAISS vector store to: {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "FAISSVectorStore":
        """Load a vector store from disk."""
        path = Path(path)

        # Load metadata
        with open(path / "metadata.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)

        config = FAISSConfig(**metadata["config"])
        store = cls(config)

        # Load FAISS index
        store._import_faiss()
        index_path = path / "index.faiss"
        if index_path.exists():
            store._index = store._faiss.read_index(str(index_path))

        # Restore mappings
        store._id_to_idx = metadata["id_to_idx"]
        store._idx_to_id = {int(v): k for k, v in store._id_to_idx.items()}
        store._contents = metadata["contents"]
        store._metadatas = metadata["metadatas"]
        store._dimension = config.dimension

        logger.info(f"Loaded FAISS vector store from: {path}")
        return store

    @property
    def count(self) -> int:
        """Get the number of documents in the store."""
        return len(self._id_to_idx)


# =============================================================================
# ChromaDB Vector Store
# =============================================================================

@dataclass
class ChromaConfig:
    """Configuration for ChromaDB vector store.

    Attributes:
        collection_name: Name of the collection
        persist_directory: Directory for persistence (None for in-memory)
        distance_fn: Distance function ('cosine', 'l2', 'ip')
    """
    collection_name: str = "qm_documents"
    persist_directory: Optional[str] = None
    distance_fn: str = "cosine"


class ChromaVectorStore(BaseVectorStore):
    """ChromaDB-based vector store.

    Persistent vector storage with rich metadata filtering.

    Example:
        ```python
        store = ChromaVectorStore(ChromaConfig(
            collection_name="qm_docs",
            persist_directory="./chroma_db",
        ))

        # Add documents
        store.add(
            ids=["doc1", "doc2"],
            embeddings=np.array([[0.1, 0.2, ...], [0.3, 0.4, ...]]),
            contents=["First document", "Second document"],
            metadatas=[{"source": "file1.md", "type": "table"}, {"source": "file2.md", "type": "paragraph"}],
        )

        # Search with filter
        results = store.search(
            query_embedding,
            k=5,
            filter_metadata={"type": "table"},
        )
        ```
    """

    def __init__(self, config: Optional[ChromaConfig] = None):
        """Initialize ChromaDB vector store.

        Args:
            config: ChromaDB configuration
        """
        self.config = config or ChromaConfig()
        self._client = None
        self._collection = None

    def _init_client(self):
        """Initialize ChromaDB client."""
        if self._client is not None:
            return

        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            raise ImportError(
                "chromadb is required for ChromaVectorStore. "
                "Install with: pip install chromadb"
            )

        if self.config.persist_directory:
            self._client = chromadb.PersistentClient(
                path=self.config.persist_directory,
            )
        else:
            self._client = chromadb.Client()

        # Map distance function
        distance_map = {
            "cosine": "cosine",
            "l2": "l2",
            "ip": "ip",
        }

        self._collection = self._client.get_or_create_collection(
            name=self.config.collection_name,
            metadata={"hnsw:space": distance_map.get(self.config.distance_fn, "cosine")},
        )

        logger.info(f"Initialized ChromaDB collection: {self.config.collection_name}")

    def add(
        self,
        ids: list[str],
        embeddings: np.ndarray,
        contents: list[str],
        metadatas: Optional[list[dict]] = None,
    ) -> None:
        """Add documents to the store."""
        self._init_client()

        if len(ids) == 0:
            return

        embeddings = np.asarray(embeddings, dtype=np.float32)

        if metadatas is None:
            metadatas = [{} for _ in ids]

        # ChromaDB requires metadata values to be str, int, float, or bool
        clean_metadatas = []
        for m in metadatas:
            clean = {}
            for k, v in m.items():
                if v is None:
                    continue
                elif isinstance(v, (str, int, float, bool)):
                    clean[k] = v
                else:
                    clean[k] = str(v)
            clean_metadatas.append(clean)

        self._collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=contents,
            metadatas=clean_metadatas,
        )

        logger.debug(f"Added {len(ids)} documents to ChromaDB")

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        filter_metadata: Optional[dict] = None,
    ) -> list[SearchResult]:
        """Search for similar documents."""
        self._init_client()

        query_embedding = np.asarray(query_embedding, dtype=np.float32)
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Build where clause for filtering
        where = None
        if filter_metadata:
            where = {k: v for k, v in filter_metadata.items()}

        results = self._collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        search_results = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                # Convert distance to similarity score
                distance = results["distances"][0][i] if results["distances"] else 0
                # For cosine, distance is 1 - similarity
                score = 1 - distance if self.config.distance_fn == "cosine" else -distance

                search_results.append(SearchResult(
                    id=doc_id,
                    score=float(score),
                    content=results["documents"][0][i] if results["documents"] else "",
                    metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                ))

        return search_results

    def delete(self, ids: list[str]) -> None:
        """Delete documents by ID."""
        self._init_client()
        self._collection.delete(ids=ids)
        logger.debug(f"Deleted {len(ids)} documents from ChromaDB")

    def save(self, path: Union[str, Path]) -> None:
        """Save the vector store.

        Note: ChromaDB with persist_directory auto-saves.
        This method is a no-op for persistent stores.
        """
        if self.config.persist_directory:
            logger.info("ChromaDB auto-persists to disk")
        else:
            logger.warning("In-memory ChromaDB cannot be saved. Use persist_directory.")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ChromaVectorStore":
        """Load a ChromaDB vector store."""
        config = ChromaConfig(persist_directory=str(path))
        return cls(config)

    @property
    def count(self) -> int:
        """Get the number of documents in the store."""
        self._init_client()
        return self._collection.count()
