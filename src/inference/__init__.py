"""
Inference Package
=================

Provides the PnR inference pipeline and supporting utilities.

Modules:
- pnr:         PatchAndRouteInference — main end-to-end inference class
- vanilla_rag: Standalone RAG pipeline for document Q&A
- embeddings:  Embedding model wrapper
- vector_store: Vector storage backends (FAISS, ChromaDB)
- merge_adapter: LoRA adapter merging
- convert_to_gguf: GGUF format conversion
"""

from src.inference.pnr import (
    PatchAndRouteInference,
    GenerationConfig,
    PromptBuilder,
    generate_text,
    score_target_logprob,
)
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
    # PnR inference (was src/inference.py — now src/inference/pnr.py)
    "PatchAndRouteInference",
    "GenerationConfig",
    "PromptBuilder",
    "generate_text",
    "score_target_logprob",
    # RAG
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
