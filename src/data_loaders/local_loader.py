"""
Local JSON Data Loader
======================

Loads QA datasets from local JSON files for fine-tuning.

Supports two formats:
- Simple: Direct question-answer pairs (for monolithic baseline)
- RAG: Question-answer with document context (for RAG baseline)
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

from datasets import Dataset, DatasetDict

logger = logging.getLogger(__name__)


# =============================================================================
# Default Prompts (no system prompt; all instructions in user message)
# =============================================================================

# Include all instructions in the user message rather than a system prompt
DEFAULT_SIMPLE_USER_PREFIX = (
    "Answer the following question accurately and concisely based on your knowledge.\n\n"
)

DEFAULT_RAG_USER_PREFIX = (
    "Answer the question based ONLY on the provided documents. "
    "If the answer is not in the documents, say so clearly.\n\n"
)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class LocalJSONConfig:
    """Configuration for local JSON data loading.
    
    Attributes:
        data_paths: List of paths to JSON files
        format_type: "simple" or "rag"
        include_negatives: Whether to include negative (unanswerable) samples
        validation_split: Fraction for validation set (0 to disable)
        language_filter: Optional language code to filter by
        user_prefix: Custom user prompt prefix (uses default if None)
        docs_base_path: Base path for documents (required for RAG format)
        noise_chunks: Tuple of (min, max) noise chunks for RAG
        chunk_config: Configuration for document chunking
        seed: Random seed for reproducibility
        use_chain_of_thought: Whether to include <think> blocks from analysis field
    """
    data_paths: list[str]
    format_type: str = "simple"  # "simple" or "rag"
    include_negatives: bool = True
    validation_split: float = 0.1
    language_filter: Optional[str] = None
    user_prefix: Optional[str] = None  # Renamed from system_prompt 
    docs_base_path: Optional[str] = None
    noise_chunks: tuple[int, int] = (1, 2)
    chunk_config: Optional[Any] = None  # ChunkConfig if using RAG
    seed: int = 42
    use_chain_of_thought: bool = False  # Use analysis field as <think> block


# =============================================================================
# Data Loader
# =============================================================================

class LocalJSONLoader:
    """Loads QA datasets from local JSON files.
    
    Example:
        ```python
        config = LocalJSONConfig(
            data_paths=["data/qa.json"],
            format_type="simple",
        )
        loader = LocalJSONLoader(config)
        dataset = loader.load()
        ```
    """
    
    def __init__(self, config: LocalJSONConfig):
        """Initialize the loader.
        
        Args:
            config: Loader configuration
        """
        self.config = config
        self._raw_data: list[dict] = []
        self._statistics: dict[str, Any] = {}
        
        random.seed(config.seed)
    
    def _load_json_files(self) -> list[dict]:
        """Load and combine all JSON files."""
        all_data = []
        
        for path_str in self.config.data_paths:
            path = Path(path_str)
            if not path.exists():
                logger.warning(f"File not found: {path}")
                continue
            
            logger.info(f"Loading: {path}")
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            if isinstance(data, list):
                all_data.extend(data)
            else:
                logger.warning(f"Expected list in {path}, got {type(data)}")
        
        logger.info(f"Loaded {len(all_data)} samples total")
        return all_data
    
    def _filter_data(self, data: list[dict]) -> list[dict]:
        """Apply filters to the data."""
        filtered = data
        
        # Filter by language
        if self.config.language_filter:
            filtered = [
                d for d in filtered 
                if d.get("language", "").lower() == self.config.language_filter.lower()
            ]
            logger.info(f"After language filter ({self.config.language_filter}): {len(filtered)} samples")
        
        # Filter negatives
        if not self.config.include_negatives:
            filtered = [
                d for d in filtered 
                if d.get("intention_category", "").upper() != "N"
            ]
            logger.info(f"After excluding negatives: {len(filtered)} samples")
        
        return filtered
    
    def _format_simple(self, item: dict) -> dict:
        """Format item for simple (monolithic) training.
        
        No system prompt; all instructions in user message.
        Chain-of-Thought: analysis field wrapped in <think>...</think> tags.
        """
        user_prefix = self.config.user_prefix or DEFAULT_SIMPLE_USER_PREFIX
        
        question = item.get("question", "")
        answer = item.get("answer", "")
        analysis = item.get("analysis", "")
        
        # Build user content with prefix
        user_content = f"{user_prefix}[Question:]\n{question}"
        
        # Build assistant response with optional Chain-of-Thought
        if self.config.use_chain_of_thought and analysis:
            # Format: <think>reasoning</think>\n\nfinal_answer
            assistant_content = f"<think>\n{analysis}\n</think>\n\n{answer}"
        else:
            assistant_content = answer
        
        # No system prompt
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
        
        return {
            "messages": messages,
            "question": question,
            "answer": answer,
            "analysis": analysis,
            "language": item.get("language", ""),
            "intention_category": item.get("intention_category", ""),
        }
    
    def _format_rag(self, item: dict) -> dict:
        """Format item for RAG training with document context.
        
        No system prompt; all instructions in user message.
        Chain-of-Thought: analysis field wrapped in <think>...</think> tags.
        """
        user_prefix = self.config.user_prefix or DEFAULT_RAG_USER_PREFIX
        
        question = item.get("question", "")
        answer = item.get("answer", "")
        analysis = item.get("analysis", "")
        evidence = item.get("evidence_snippet", "")
        file_path = item.get("file_path", "")
        
        # Build context from evidence (simplified - full RAG would use chunker)
        context = ""
        if evidence:
            context = f"[Documents:]\n--- Document 1 ---\n{evidence}\n\n"
        
        # All instructions in user prompt
        user_content = f"{user_prefix}{context}[Question:]\n{question}"
        
        # Build assistant response with optional Chain-of-Thought
        if self.config.use_chain_of_thought and analysis:
            # Format: <think>reasoning</think>\n\nfinal_answer
            assistant_content = f"<think>\n{analysis}\n</think>\n\n{answer}"
        else:
            assistant_content = answer
        
        # No system prompt
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
        
        return {
            "messages": messages,
            "question": question,
            "answer": answer,
            "analysis": analysis,
            "evidence_snippet": evidence,
            "file_path": file_path,
            "language": item.get("language", ""),
            "intention_category": item.get("intention_category", ""),
        }
    
    def _compute_statistics(self, data: list[dict]) -> dict[str, Any]:
        """Compute dataset statistics."""
        stats = {
            "total_samples": len(data),
            "languages": {},
            "intention_categories": {},
            "has_evidence": 0,
            "has_file_path": 0,
        }
        
        for item in data:
            # Language stats
            lang = item.get("language", "unknown")
            stats["languages"][lang] = stats["languages"].get(lang, 0) + 1
            
            # Category stats
            cat = item.get("intention_category", "unknown")
            stats["intention_categories"][cat] = stats["intention_categories"].get(cat, 0) + 1
            
            # Evidence stats
            if item.get("evidence_snippet"):
                stats["has_evidence"] += 1
            if item.get("file_path"):
                stats["has_file_path"] += 1
        
        return stats
    
    def load(self) -> Dataset | DatasetDict:
        """Load and format the dataset.
        
        Returns:
            Dataset or DatasetDict (with train/test split if validation_split > 0)
        """
        # Load raw data
        self._raw_data = self._load_json_files()
        
        # Filter
        filtered_data = self._filter_data(self._raw_data)
        
        # Compute statistics
        self._statistics = self._compute_statistics(filtered_data)
        
        # Format
        format_func = self._format_rag if self.config.format_type == "rag" else self._format_simple
        formatted_data = [format_func(item) for item in filtered_data]
        
        # Create dataset
        dataset = Dataset.from_list(formatted_data)
        
        # Split if needed
        if self.config.validation_split > 0:
            split = dataset.train_test_split(
                test_size=self.config.validation_split,
                seed=self.config.seed,
            )
            return split
        
        return dataset
    
    def get_statistics(self) -> dict[str, Any]:
        """Get dataset statistics (call after load())."""
        return self._statistics


# =============================================================================
# Factory Functions
# =============================================================================

def create_simple_loader(
    data_paths: list[str],
    include_negatives: bool = True,
    validation_split: float = 0.1,
    **kwargs,
) -> LocalJSONLoader:
    """Create a loader for simple (monolithic) training.
    
    Args:
        data_paths: Paths to JSON files
        include_negatives: Include negative samples
        validation_split: Fraction for validation
        **kwargs: Additional config options
        
    Returns:
        Configured LocalJSONLoader
    """
    config = LocalJSONConfig(
        data_paths=data_paths,
        format_type="simple",
        include_negatives=include_negatives,
        validation_split=validation_split,
        **kwargs,
    )
    return LocalJSONLoader(config)


def create_rag_loader(
    data_path: str,
    docs_path: str,
    noise_chunks: tuple[int, int] = (1, 2),
    **kwargs,
) -> LocalJSONLoader:
    """Create a loader for RAG training.
    
    Args:
        data_path: Path to JSON file
        docs_path: Base path to documents
        noise_chunks: (min, max) noise chunks to inject
        **kwargs: Additional config options
        
    Returns:
        Configured LocalJSONLoader
    """
    config = LocalJSONConfig(
        data_paths=[data_path],
        docs_base_path=docs_path,
        format_type="rag",
        noise_chunks=noise_chunks,
        **kwargs,
    )
    return LocalJSONLoader(config)
