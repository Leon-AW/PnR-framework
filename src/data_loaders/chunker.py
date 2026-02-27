"""
Semantic Document Chunker
=========================

Chunks documents for RAG-based fine-tuning.

Features:
- Token-based chunking with semantic boundaries
- Overlap between chunks for context continuity
- Evidence matching to find relevant chunks
- Noise chunk injection for training robustness
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ChunkConfig:
    """Configuration for document chunking.
    
    Attributes:
        max_doc_tokens: If document is smaller, use whole doc
        chunk_size: Target chunk size in tokens
        chunk_overlap: Overlap between chunks
        separator: Text to split on (default: double newline)
    """
    max_doc_tokens: int = 2500
    chunk_size: int = 750
    chunk_overlap: int = 75
    separator: str = "\n\n"


# =============================================================================
# Chunker
# =============================================================================

@dataclass
class Chunk:
    """A document chunk.
    
    Attributes:
        content: The chunk text
        source_file: Path to source document
        chunk_index: Index within document
        start_char: Starting character position
        end_char: Ending character position
    """
    content: str
    source_file: str
    chunk_index: int
    start_char: int
    end_char: int


class SemanticChunker:
    """Chunks documents semantically for RAG training.
    
    Example:
        ```python
        config = ChunkConfig(chunk_size=750)
        chunker = SemanticChunker(config)
        
        chunks = chunker.chunk_document("path/to/doc.md")
        relevant = chunker.find_relevant_chunk(chunks, "evidence text")
        noise = chunker.get_noise_chunks(all_chunks, exclude=[relevant], n=2)
        context = chunker.build_context(relevant, noise)
        ```
    """
    
    def __init__(self, config: Optional[ChunkConfig] = None):
        """Initialize the chunker.
        
        Args:
            config: Chunking configuration (uses defaults if None)
        """
        self.config = config or ChunkConfig()
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation: ~4 chars per token)."""
        return len(text) // 4
    
    def chunk_document(self, file_path: str | Path) -> list[Chunk]:
        """Chunk a document file.
        
        Args:
            file_path: Path to document
            
        Returns:
            List of document chunks
        """
        path = Path(file_path)
        
        if not path.exists():
            logger.warning(f"File not found: {path}")
            return []
        
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # If document is small enough, return as single chunk
        if self._estimate_tokens(content) <= self.config.max_doc_tokens:
            return [Chunk(
                content=content,
                source_file=str(path),
                chunk_index=0,
                start_char=0,
                end_char=len(content),
            )]
        
        # Split into paragraphs first
        paragraphs = content.split(self.config.separator)
        
        chunks = []
        current_chunk = ""
        current_start = 0
        char_pos = 0
        
        for para in paragraphs:
            para_with_sep = para + self.config.separator
            
            # Check if adding this paragraph exceeds chunk size
            if self._estimate_tokens(current_chunk + para_with_sep) > self.config.chunk_size:
                if current_chunk:
                    chunks.append(Chunk(
                        content=current_chunk.strip(),
                        source_file=str(path),
                        chunk_index=len(chunks),
                        start_char=current_start,
                        end_char=char_pos,
                    ))
                    
                    # Start new chunk with overlap
                    overlap_text = current_chunk[-self.config.chunk_overlap * 4:] if len(current_chunk) > self.config.chunk_overlap * 4 else ""
                    current_chunk = overlap_text + para_with_sep
                    current_start = char_pos - len(overlap_text)
                else:
                    current_chunk = para_with_sep
            else:
                current_chunk += para_with_sep
            
            char_pos += len(para_with_sep)
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(Chunk(
                content=current_chunk.strip(),
                source_file=str(path),
                chunk_index=len(chunks),
                start_char=current_start,
                end_char=char_pos,
            ))
        
        logger.debug(f"Created {len(chunks)} chunks from {path}")
        return chunks
    
    def find_relevant_chunk(
        self,
        chunks: list[Chunk],
        evidence: str,
    ) -> Optional[Chunk]:
        """Find the chunk that best matches evidence text.
        
        Uses simple substring matching. For production, consider
        using embeddings or fuzzy matching.
        
        Args:
            chunks: List of document chunks
            evidence: Evidence text to find
            
        Returns:
            Best matching chunk or None
        """
        if not chunks or not evidence:
            return None
        
        # Normalize evidence
        evidence_lower = evidence.lower().strip()
        
        # First try exact substring match
        for chunk in chunks:
            if evidence_lower in chunk.content.lower():
                return chunk
        
        # Fall back to overlap scoring
        best_chunk = None
        best_score = 0
        
        evidence_words = set(evidence_lower.split())
        
        for chunk in chunks:
            chunk_words = set(chunk.content.lower().split())
            overlap = len(evidence_words & chunk_words)
            score = overlap / len(evidence_words) if evidence_words else 0
            
            if score > best_score:
                best_score = score
                best_chunk = chunk
        
        return best_chunk if best_score > 0.3 else chunks[0]
    
    def get_noise_chunks(
        self,
        all_chunks: list[Chunk],
        exclude: list[Chunk],
        n: int,
    ) -> list[Chunk]:
        """Get random noise chunks excluding specified ones.
        
        Args:
            all_chunks: Pool of all available chunks
            exclude: Chunks to exclude (e.g., relevant chunk)
            n: Number of noise chunks to return
            
        Returns:
            List of noise chunks
        """
        exclude_ids = {(c.source_file, c.chunk_index) for c in exclude}
        
        candidates = [
            c for c in all_chunks 
            if (c.source_file, c.chunk_index) not in exclude_ids
        ]
        
        if len(candidates) <= n:
            return candidates
        
        return random.sample(candidates, n)
    
    def build_context(
        self,
        relevant: Chunk,
        noise: list[Chunk],
        shuffle: bool = True,
    ) -> str:
        """Build context string from chunks.
        
        Args:
            relevant: The relevant document chunk
            noise: List of noise chunks
            shuffle: Whether to shuffle chunk order
            
        Returns:
            Formatted context string
        """
        all_chunks = [relevant] + noise
        
        if shuffle:
            random.shuffle(all_chunks)
        
        context_parts = ["[Documents:]"]
        
        for i, chunk in enumerate(all_chunks, 1):
            context_parts.append(f"--- Document {i} ---")
            context_parts.append(chunk.content)
            context_parts.append("")
        
        return "\n".join(context_parts)
