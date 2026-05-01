"""
Source-Replay Store
====================

FAISS-based retrieval system for the Source-Replay mechanism.

When an older adapter loses the conflict resolution (T_old), we don't load its
weights. Instead, we retrieve relevant chunks from its training data and inject
them as context in the prompt (RAG-style).

This module provides:
- Per-adapter FAISS indices for efficient similarity search
- Batch embedding and indexing of training data
- Scoped retrieval for specific adapters

Key Design Decisions:
1. Uses FAISS for fast similarity search (GPU-accelerated if available)
2. Stores training data chunks alongside embeddings for retrieval
3. Indices are adapter-specific (one index per adapter)
4. Supports lazy loading to minimize memory footprint

Reference: Section 4.4.1 of the Master's Thesis Exposé - "Source-Replay"
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

try:
    import faiss
    HAS_FAISS = True
    # Check if GPU FAISS is available (faiss-gpu vs faiss-cpu)
    HAS_FAISS_GPU = hasattr(faiss, 'StandardGpuResources')
except ImportError:
    HAS_FAISS = False
    HAS_FAISS_GPU = False

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """A retrieved text chunk from Source-Replay.
    
    Attributes:
        text: The retrieved text content.
        adapter_id: Which adapter's training data this came from.
        similarity: Cosine similarity score (0-1).
        metadata: Additional chunk metadata (e.g., question, date, location).
    """
    text: str
    adapter_id: str
    similarity: float
    metadata: dict[str, Any] | None = None
    
    def __repr__(self) -> str:
        preview = self.text[:50] + "..." if len(self.text) > 50 else self.text
        return f"RetrievedChunk(adapter={self.adapter_id}, sim={self.similarity:.3f}, text='{preview}')"


class AdapterIndex:
    """FAISS index for a single adapter's training data.
    
    Stores embeddings and corresponding text chunks for retrieval.
    
    Attributes:
        adapter_id: The adapter this index belongs to.
        index: FAISS index for similarity search.
        chunks: List of text chunks (aligned with index).
        metadata: List of metadata dicts (aligned with index).
    """
    
    def __init__(
        self,
        adapter_id: str,
        embedding_dim: int,
        use_gpu: bool = False,
    ) -> None:
        """Initialize the adapter index.
        
        Args:
            adapter_id: Unique identifier for the adapter.
            embedding_dim: Dimension of embedding vectors.
            use_gpu: Whether to use GPU-accelerated FAISS.
        """
        if not HAS_FAISS:
            raise ImportError(
                "FAISS is required for Source-Replay. "
                "Install with: pip install faiss-cpu (or faiss-gpu)"
            )
        
        self.adapter_id = adapter_id
        self.embedding_dim = embedding_dim
        # `use_gpu` ends up reflecting whether GPU FAISS *actually* succeeded,
        # not what the caller requested. The save path keys off this flag
        # to decide whether `faiss.index_gpu_to_cpu` is needed; if we left
        # it at the requested value when running against `faiss-cpu`
        # (HAS_FAISS_GPU=False), `save()` would crash with
        # `module 'faiss' has no attribute 'index_gpu_to_cpu'`.
        self.use_gpu = False

        # Create FAISS index (Inner Product for cosine similarity on normalized vectors)
        self.index = faiss.IndexFlatIP(embedding_dim)

        # GPU acceleration if available (requires faiss-gpu, not faiss-cpu)
        if use_gpu and HAS_FAISS_GPU:
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                self.use_gpu = True
                logger.info(f"Using GPU-accelerated FAISS for {adapter_id}")
            except Exception as e:
                logger.debug(f"GPU FAISS initialization failed: {e}")  # Debug level, not warning
        
        # Storage for chunks and metadata
        self.chunks: list[str] = []
        self.metadata: list[dict[str, Any]] = []
        
        logger.debug(f"Initialized AdapterIndex for {adapter_id} (dim={embedding_dim})")
    
    @property
    def num_chunks(self) -> int:
        """Number of indexed chunks."""
        return len(self.chunks)
    
    def add(
        self,
        embeddings: np.ndarray,
        chunks: list[str],
        metadata: list[dict[str, Any]] | None = None,
    ) -> None:
        """Add embeddings and chunks to the index.
        
        Args:
            embeddings: Embedding vectors, shape (n, embedding_dim).
            chunks: Corresponding text chunks.
            metadata: Optional metadata for each chunk.
        """
        if len(embeddings) != len(chunks):
            raise ValueError(
                f"Mismatch: {len(embeddings)} embeddings vs {len(chunks)} chunks"
            )
        
        # Normalize for cosine similarity
        embeddings = embeddings.astype(np.float32)
        faiss.normalize_L2(embeddings)
        
        # Add to index
        self.index.add(embeddings)
        
        # Store chunks and metadata
        self.chunks.extend(chunks)
        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{}] * len(chunks))
        
        logger.debug(f"Added {len(chunks)} chunks to {self.adapter_id} index")
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 3,
    ) -> list[RetrievedChunk]:
        """Search for similar chunks.
        
        Args:
            query_embedding: Query vector, shape (embedding_dim,).
            top_k: Number of results to return.
            
        Returns:
            List of RetrievedChunk objects, sorted by similarity.
        """
        if self.num_chunks == 0:
            return []
        
        # Normalize query
        query = query_embedding.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(query)
        
        # Search
        k = min(top_k, self.num_chunks)
        similarities, indices = self.index.search(query, k)
        
        # Build results
        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            if idx < 0:  # FAISS returns -1 for empty results
                continue
            
            results.append(RetrievedChunk(
                text=self.chunks[idx],
                adapter_id=self.adapter_id,
                similarity=float(sim),
                metadata=self.metadata[idx] if idx < len(self.metadata) else None,
            ))
        
        return results
    
    def save(self, path: Path) -> None:
        """Save index to disk.
        
        Args:
            path: Directory to save to.
        """
        path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_path = path / f"{self.adapter_id}.faiss"
        
        # Convert GPU index to CPU for saving
        if self.use_gpu:
            cpu_index = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(cpu_index, str(index_path))
        else:
            faiss.write_index(self.index, str(index_path))
        
        # Save chunks and metadata
        data_path = path / f"{self.adapter_id}.pkl"
        with open(data_path, "wb") as f:
            pickle.dump({
                "adapter_id": self.adapter_id,
                "embedding_dim": self.embedding_dim,
                "chunks": self.chunks,
                "metadata": self.metadata,
            }, f)
        
        logger.info(f"Saved index for {self.adapter_id} ({self.num_chunks} chunks)")
    
    @classmethod
    def load(cls, path: Path, adapter_id: str, use_gpu: bool = False) -> AdapterIndex:
        """Load index from disk.
        
        Args:
            path: Directory containing saved index.
            adapter_id: Adapter ID to load.
            use_gpu: Whether to use GPU-accelerated FAISS.
            
        Returns:
            Loaded AdapterIndex.
        """
        index_path = path / f"{adapter_id}.faiss"
        data_path = path / f"{adapter_id}.pkl"
        
        if not index_path.exists() or not data_path.exists():
            raise FileNotFoundError(f"Index files not found for {adapter_id}")
        
        # Load metadata
        with open(data_path, "rb") as f:
            data = pickle.load(f)
        
        # Create instance
        instance = cls(
            adapter_id=data["adapter_id"],
            embedding_dim=data["embedding_dim"],
            use_gpu=use_gpu,
        )
        
        # Load FAISS index
        instance.index = faiss.read_index(str(index_path))

        # GPU acceleration if available (requires faiss-gpu). Mirror the
        # __init__ contract — `instance.use_gpu` reflects what actually
        # succeeded so a later `.save()` doesn't try `faiss.index_gpu_to_cpu`
        # under faiss-cpu.
        instance.use_gpu = False
        if use_gpu and HAS_FAISS_GPU:
            try:
                res = faiss.StandardGpuResources()
                instance.index = faiss.index_cpu_to_gpu(res, 0, instance.index)
                instance.use_gpu = True
            except Exception:
                pass
        
        # Restore chunks and metadata
        instance.chunks = data["chunks"]
        instance.metadata = data["metadata"]
        
        logger.info(f"Loaded index for {adapter_id} ({instance.num_chunks} chunks)")
        
        return instance


class SourceReplayStore:
    """Manager for multiple adapter indices.
    
    Provides a unified interface for:
    - Indexing training data from multiple adapters
    - Scoped retrieval from specific adapters
    - Cross-adapter search for conflict detection
    
    Example:
        ```python
        store = SourceReplayStore(
            embedding_fn=embedding_model.encode,
            embedding_dim=768,
        )
        
        # Index adapter training data
        store.index_adapter(
            adapter_id="patch_geo_germany",
            training_data_path="data/germany_training.jsonl",
        )
        
        # Retrieve from a specific adapter (Source-Replay)
        chunks = store.retrieve(
            query_embedding=query_vec,
            adapter_id="patch_geo_germany",
            top_k=3,
        )
        
        # Build context string for prompt injection
        context = store.build_context(chunks)
        ```
    """
    
    def __init__(
        self,
        embedding_fn: Callable[[str], np.ndarray] | None = None,
        embedding_batch_fn: Callable[[list[str]], np.ndarray] | None = None,
        embedding_dim: int = 768,
        use_gpu: bool = False,
        store_dir: str | Path | None = None,
    ) -> None:
        """Initialize the store.
        
        Args:
            embedding_fn: Function to embed a single text string.
            embedding_batch_fn: Function to embed a batch of texts (much faster).
            embedding_dim: Dimension of embedding vectors.
            use_gpu: Whether to use GPU-accelerated FAISS.
            store_dir: Directory for persisting indices.
        """
        self.embedding_fn = embedding_fn
        self.embedding_batch_fn = embedding_batch_fn
        self.embedding_dim = embedding_dim
        self.use_gpu = use_gpu
        self.store_dir = Path(store_dir) if store_dir else None
        
        self._indices: dict[str, AdapterIndex] = {}
        
        logger.info(f"Initialized SourceReplayStore (dim={embedding_dim})")
    
    @property
    def adapters(self) -> list[str]:
        """Get list of indexed adapter IDs."""
        return list(self._indices.keys())
    
    def __contains__(self, adapter_id: str) -> bool:
        """Check if adapter is indexed."""
        return adapter_id in self._indices
    
    # -------------------------------------------------------------------------
    # Indexing
    # -------------------------------------------------------------------------
    
    def index_adapter(
        self,
        adapter_id: str,
        training_data_path: str | Path,
        text_field: str = "edited_question",
        answer_field: str = "answer",
        batch_size: int = 32,
        max_chunks: int | None = None,
    ) -> int:
        """Index training data for an adapter.
        
        Reads training data JSONL file, embeds the text, and adds to index.
        
        Args:
            adapter_id: Unique adapter identifier.
            training_data_path: Path to training data JSONL.
            text_field: Field to extract text from.
            answer_field: Field to extract answer from.
            batch_size: Batch size for embedding.
            max_chunks: Maximum number of chunks to index (for testing).
            
        Returns:
            Number of chunks indexed.
        """
        if self.embedding_fn is None:
            raise ValueError("embedding_fn must be set to index adapters")
        
        training_data_path = Path(training_data_path)
        
        if not training_data_path.exists():
            raise FileNotFoundError(f"Training data not found: {training_data_path}")
        
        logger.info(f"Indexing training data for {adapter_id} from {training_data_path}")
        
        # Create index
        index = AdapterIndex(
            adapter_id=adapter_id,
            embedding_dim=self.embedding_dim,
            use_gpu=self.use_gpu,
        )
        
        # Read and process training data
        chunks: list[str] = []
        metadata_list: list[dict] = []
        
        with open(training_data_path, "r") as f:
            for line_num, line in enumerate(f):
                if max_chunks and len(chunks) >= max_chunks:
                    break
                
                try:
                    data = json.loads(line)
                    
                    # Extract text (question + answer for context)
                    question = data.get(text_field, "")
                    answers = data.get(answer_field, [])
                    if isinstance(answers, list) and answers:
                        answer = answers[0]
                    elif isinstance(answers, str):
                        answer = answers
                    else:
                        answer = ""
                    
                    # Format as Q&A chunk
                    chunk_text = f"Q: {question}\nA: {answer}"
                    
                    chunks.append(chunk_text)
                    metadata_list.append({
                        "question": question,
                        "answer": answer,
                        "date": data.get("date"),
                        "location": data.get("location"),
                        "line_num": line_num,
                    })
                    
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse line {line_num}")
                    continue
        
        if not chunks:
            logger.warning(f"No chunks extracted from {training_data_path}")
            return 0
        
        # Embed using batch function (MUCH faster) or fallback to single-text function
        if self.embedding_batch_fn is not None:
            embeddings = self.embedding_batch_fn(chunks)
        elif self.embedding_fn is not None:
            all_embeddings = []
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                batch_embeddings = np.vstack([
                    self.embedding_fn(text) for text in batch
                ])
                all_embeddings.append(batch_embeddings)
            embeddings = np.vstack(all_embeddings)
        else:
            raise ValueError("No embedding function provided")
        
        # Add to index
        index.add(embeddings, chunks, metadata_list)
        
        # Store
        self._indices[adapter_id] = index
        
        # Persist if store_dir is set
        if self.store_dir:
            index.save(self.store_dir)
        
        logger.info(f"Indexed {index.num_chunks} chunks for {adapter_id}")
        
        return index.num_chunks
    
    def index_samples(
        self,
        adapter_id: str,
        samples: list[dict[str, Any]],
        text_field: str = "edited_question",
        answer_field: str = "answer",
        batch_size: int = 32,
    ) -> int:
        """Index training samples directly from memory.
        
        This is an alternative to index_adapter() that accepts samples directly
        instead of reading from a file. Useful when samples are loaded dynamically.
        
        Args:
            adapter_id: Unique adapter identifier.
            samples: List of sample dictionaries with text and answer fields.
            text_field: Field to extract question text from.
            answer_field: Field to extract answer from.
            batch_size: Batch size for embedding.
            
        Returns:
            Number of chunks indexed.
        """
        if self.embedding_fn is None:
            raise ValueError("embedding_fn must be set to index adapters")
        
        if not samples:
            logger.warning(f"No samples provided for {adapter_id}")
            return 0
        
        logger.info(f"Indexing {len(samples)} samples for {adapter_id}")
        
        # Create index
        index = AdapterIndex(
            adapter_id=adapter_id,
            embedding_dim=self.embedding_dim,
            use_gpu=self.use_gpu,
        )
        
        # Process samples
        chunks: list[str] = []
        metadata_list: list[dict] = []
        
        for i, data in enumerate(samples):
            # Extract text (question + answer for context)
            question = data.get(text_field, "")
            answers = data.get(answer_field, [])
            if isinstance(answers, list) and answers:
                answer = answers[0]
            elif isinstance(answers, str):
                answer = answers
            else:
                answer = ""
            
            if not question:
                continue
            
            # Format as Q&A chunk
            chunk_text = f"Q: {question}\nA: {answer}"
            
            chunks.append(chunk_text)
            metadata_list.append({
                "question": question,
                "answer": answer,
                "date": data.get("date"),
                "location": data.get("location"),
                "sample_idx": i,
            })
        
        if not chunks:
            logger.warning(f"No valid chunks extracted from samples for {adapter_id}")
            return 0
        
        # Embed using batch function (MUCH faster) or fallback to single-text function
        if self.embedding_batch_fn is not None:
            # Use batch embedding function - 10-50x faster
            embeddings = self.embedding_batch_fn(chunks)
        elif self.embedding_fn is not None:
            # Fallback: embed one at a time (slow)
            all_embeddings = []
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                batch_embeddings = np.vstack([
                    self.embedding_fn(text) for text in batch
                ])
                all_embeddings.append(batch_embeddings)
            embeddings = np.vstack(all_embeddings)
        else:
            raise ValueError("No embedding function provided")
        
        # Add to index
        index.add(embeddings, chunks, metadata_list)
        
        # Store
        self._indices[adapter_id] = index
        
        # Persist if store_dir is set
        if self.store_dir:
            index.save(self.store_dir)
        
        logger.info(f"Indexed {index.num_chunks} chunks for {adapter_id}")
        
        return index.num_chunks
    
    def add_index(self, index: AdapterIndex) -> None:
        """Add a pre-built index.
        
        Args:
            index: AdapterIndex to add.
        """
        self._indices[index.adapter_id] = index
        logger.info(f"Added index for {index.adapter_id}")
    
    def load_index(self, adapter_id: str) -> bool:
        """Load an index from disk.
        
        Args:
            adapter_id: Adapter to load.
            
        Returns:
            True if loaded, False if not found.
        """
        if not self.store_dir:
            return False
        
        try:
            index = AdapterIndex.load(self.store_dir, adapter_id, self.use_gpu)
            self._indices[adapter_id] = index
            return True
        except FileNotFoundError:
            return False
    
    # -------------------------------------------------------------------------
    # Retrieval
    # -------------------------------------------------------------------------
    
    def retrieve(
        self,
        query_embedding: np.ndarray,
        adapter_id: str,
        top_k: int = 3,
    ) -> list[RetrievedChunk]:
        """Retrieve chunks from a specific adapter's index.
        
        This is the core "Source-Replay" operation for T_old adapters.
        
        Args:
            query_embedding: Query vector.
            adapter_id: Adapter to search.
            top_k: Number of results.
            
        Returns:
            List of retrieved chunks.
        """
        if adapter_id not in self._indices:
            # Try lazy loading
            if not self.load_index(adapter_id):
                logger.warning(f"No index found for adapter: {adapter_id}")
                return []
        
        return self._indices[adapter_id].search(query_embedding, top_k)
    
    def retrieve_multi(
        self,
        query_embedding: np.ndarray,
        adapter_ids: list[str],
        top_k_per_adapter: int = 2,
    ) -> list[RetrievedChunk]:
        """Retrieve from multiple adapters (for multi-conflict scenarios).
        
        Args:
            query_embedding: Query vector.
            adapter_ids: List of adapters to search.
            top_k_per_adapter: Results per adapter.
            
        Returns:
            Combined list of retrieved chunks.
        """
        all_chunks = []
        for adapter_id in adapter_ids:
            chunks = self.retrieve(query_embedding, adapter_id, top_k_per_adapter)
            all_chunks.extend(chunks)
        
        # Sort by similarity
        all_chunks.sort(key=lambda c: c.similarity, reverse=True)
        
        return all_chunks
    
    # -------------------------------------------------------------------------
    # Context Building
    # -------------------------------------------------------------------------
    
    @staticmethod
    def build_context(
        chunks: list[RetrievedChunk],
        max_context_length: int = 2000,
        separator: str = "\n---\n",
    ) -> str:
        """Build context string from retrieved chunks.
        
        Formats chunks for injection into the prompt.
        
        Args:
            chunks: List of retrieved chunks.
            max_context_length: Maximum character length.
            separator: Separator between chunks.
            
        Returns:
            Formatted context string.
        """
        if not chunks:
            return ""
        
        context_parts = []
        current_length = 0
        
        for chunk in chunks:
            chunk_text = chunk.text.strip()
            
            if current_length + len(chunk_text) > max_context_length:
                break
            
            context_parts.append(chunk_text)
            current_length += len(chunk_text) + len(separator)
        
        if not context_parts:
            return ""
        
        # Format with header
        header = "### Relevant Context from Historical Knowledge:\n"
        context = separator.join(context_parts)
        
        return header + context
    
    def get_stats(self) -> dict[str, Any]:
        """Get statistics about indexed adapters.
        
        Returns:
            Dictionary with stats.
        """
        return {
            "num_adapters": len(self._indices),
            "adapters": {
                adapter_id: {
                    "num_chunks": index.num_chunks,
                    "embedding_dim": index.embedding_dim,
                }
                for adapter_id, index in self._indices.items()
            },
        }

