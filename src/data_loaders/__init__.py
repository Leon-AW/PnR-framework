"""
Data Loading Package
====================

Provides data loading utilities for the Patch-and-Route framework.

Modules:
- local_loader: Load JSON QA datasets for fine-tuning
- chunker: Document chunking for RAG-based training
- structure_aware_chunker: Structure-aware chunking for QM documents
"""

from src.data_loaders.local_loader import LocalJSONLoader, LocalJSONConfig
from src.data_loaders.chunker import SemanticChunker, ChunkConfig
from src.data_loaders.structure_aware_chunker import (
    StructureAwareChunker,
    StructuredChunkConfig,
    StructuredChunk,
)

__all__ = [
    "LocalJSONLoader",
    "LocalJSONConfig",
    "SemanticChunker",
    "ChunkConfig",
    "StructureAwareChunker",
    "StructuredChunkConfig",
    "StructuredChunk",
]
