"""
Inference Package
=================

Provides inference utilities for the Patch-and-Route framework.

Modules:
- vanilla_rag: Standalone RAG pipeline for QM documents
- embeddings: Embedding model wrapper
- vector_store: Vector storage backends (FAISS, ChromaDB)
- merge_adapter: LoRA adapter merging
- convert_to_gguf: GGUF format conversion
"""

from src.inference.vanilla_rag import VanillaRAG, VanillaRAGConfig
from src.inference.embeddings import EmbeddingModel, EmbeddingConfig
from src.inference.vector_store import (
    BaseVectorStore,
    FAISSVectorStore,
    FAISSConfig,
    ChromaVectorStore,
    ChromaConfig,
    SearchResult,
)

__all__ = [
    # Main RAG
    "VanillaRAG",
    "VanillaRAGConfig",
    # Embeddings
    "EmbeddingModel",
    "EmbeddingConfig",
    # Vector stores
    "BaseVectorStore",
    "FAISSVectorStore",
    "FAISSConfig",
    "ChromaVectorStore",
    "ChromaConfig",
    "SearchResult",
]
