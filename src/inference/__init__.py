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
- bm25_store: BM25 sparse retrieval for hybrid search
- reranker: Cross-encoder reranking
- rag_config: Advanced RAG server configuration
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
from src.inference.bm25_store import BM25Store, BM25Config
from src.inference.reranker import Reranker, RerankerConfig
from src.inference.rag_config import RAGServerConfig

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
    # BM25
    "BM25Store",
    "BM25Config",
    # Reranker
    "Reranker",
    "RerankerConfig",
    # RAG Server Config
    "RAGServerConfig",
]
