"""
BM25 Sparse Retrieval
=====================

BM25-based sparse retrieval for hybrid search.
Supports German-aware tokenization with stop word removal.
"""

from __future__ import annotations

import logging
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

from src.inference.vector_store import SearchResult

logger = logging.getLogger(__name__)

# German stop words (common function words)
GERMAN_STOP_WORDS = frozenset({
    "der", "die", "das", "den", "dem", "des", "ein", "eine", "einer", "einem",
    "einen", "eines", "und", "oder", "aber", "doch", "sondern", "nicht", "kein",
    "keine", "keiner", "keinem", "keinen", "keines", "ist", "sind", "war",
    "waren", "wird", "werden", "wurde", "wurden", "hat", "haben", "hatte",
    "hatten", "kann", "können", "konnte", "konnten", "muss", "müssen",
    "musste", "mussten", "soll", "sollen", "sollte", "sollten", "darf",
    "dürfen", "durfte", "durften", "will", "wollen", "wollte", "wollten",
    "mag", "mögen", "mochte", "mochten", "ich", "du", "er", "sie", "es",
    "wir", "ihr", "man", "sich", "mich", "dich", "uns", "euch", "mir",
    "dir", "ihm", "ihr", "ihnen", "mein", "dein", "sein", "ihr", "unser",
    "euer", "in", "an", "auf", "aus", "bei", "mit", "nach", "seit", "von",
    "zu", "für", "um", "über", "unter", "vor", "zwischen", "durch", "gegen",
    "ohne", "bis", "als", "wie", "wenn", "dass", "ob", "weil", "da",
    "damit", "obwohl", "während", "nachdem", "bevor", "sobald", "auch",
    "noch", "schon", "nur", "sehr", "so", "dann", "hier", "dort", "wo",
    "was", "wer", "wie", "welche", "welcher", "welches", "welchem",
    "welchen", "diese", "dieser", "dieses", "diesem", "diesen", "jede",
    "jeder", "jedes", "jedem", "jeden", "alle", "alles", "allem", "allen",
    # English stop words (for bilingual support)
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "must", "and", "or",
    "but", "not", "no", "nor", "if", "then", "else", "when", "where",
    "what", "which", "who", "whom", "how", "all", "each", "every", "both",
    "few", "more", "most", "other", "some", "such", "than", "too", "very",
    "just", "because", "as", "until", "while", "of", "at", "by", "for",
    "with", "about", "against", "between", "through", "during", "before",
    "after", "above", "below", "to", "from", "up", "down", "in", "out",
    "on", "off", "over", "under", "again", "further", "it", "its", "this",
    "that", "these", "those", "i", "me", "my", "we", "our", "you", "your",
    "he", "him", "his", "she", "her", "they", "them", "their",
})

# Regex for tokenization: keeps umlauts, sharp-s, and alphanumeric characters
_TOKEN_PATTERN = re.compile(r"[a-zäöüß0-9]+", re.IGNORECASE)


def _tokenize(text: str) -> list[str]:
    """Tokenize text with German-aware processing.

    Lowercases, splits on non-alphanumeric (keeping umlauts/sharp-s),
    and removes stop words.
    """
    tokens = _TOKEN_PATTERN.findall(text.lower())
    return [t for t in tokens if t not in GERMAN_STOP_WORDS and len(t) > 1]


@dataclass
class BM25Config:
    """Configuration for BM25 retrieval.

    Attributes:
        language: Document language
        k1: Term frequency saturation parameter
        b: Document length normalization parameter
    """
    language: str = "de"
    k1: float = 1.5
    b: float = 0.75


class BM25Store:
    """BM25-based sparse retrieval store.

    Example:
        ```python
        store = BM25Store(BM25Config())
        store.build(
            ids=["doc1", "doc2"],
            contents=["First document text", "Second document text"],
            metadatas=[{"source": "a.md"}, {"source": "b.md"}],
        )
        results = store.search("query text", k=5)
        ```
    """

    def __init__(self, config: Optional[BM25Config] = None):
        self.config = config or BM25Config()
        self._bm25 = None
        self._ids: list[str] = []
        self._contents: list[str] = []
        self._metadatas: list[dict] = []
        self._corpus_tokens: list[list[str]] = []

    def build(
        self,
        ids: list[str],
        contents: list[str],
        metadatas: Optional[list[dict]] = None,
    ) -> None:
        """Build BM25 index from documents.

        Args:
            ids: Document IDs (should match FAISS IDs for RRF fusion)
            contents: Document text contents
            metadatas: Optional metadata dictionaries
        """
        from rank_bm25 import BM25Okapi

        if metadatas is None:
            metadatas = [{} for _ in ids]

        self._ids = list(ids)
        self._contents = list(contents)
        self._metadatas = list(metadatas)

        # Tokenize corpus
        self._corpus_tokens = [_tokenize(doc) for doc in contents]

        # Build BM25 index
        self._bm25 = BM25Okapi(
            self._corpus_tokens,
            k1=self.config.k1,
            b=self.config.b,
        )

        logger.info(f"Built BM25 index with {len(ids)} documents")

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        """Search for relevant documents.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of SearchResult objects
        """
        if self._bm25 is None or not self._ids:
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)

        # Get top-k indices
        top_indices = scores.argsort()[::-1][:k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score <= 0:
                continue
            results.append(SearchResult(
                id=self._ids[idx],
                score=score,
                content=self._contents[idx],
                metadata=self._metadatas[idx],
            ))

        return results

    def save(self, path: Union[str, Path]) -> None:
        """Save BM25 index to disk.

        Args:
            path: Path for the pickle file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "config": self.config,
            "ids": self._ids,
            "contents": self._contents,
            "metadatas": self._metadatas,
            "corpus_tokens": self._corpus_tokens,
        }

        with open(path, "wb") as f:
            pickle.dump(data, f)

        logger.info(f"Saved BM25 index to: {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "BM25Store":
        """Load BM25 index from disk.

        Args:
            path: Path to the pickle file

        Returns:
            BM25Store instance
        """
        from rank_bm25 import BM25Okapi

        path = Path(path)

        with open(path, "rb") as f:
            data = pickle.load(f)

        store = cls(data["config"])
        store._ids = data["ids"]
        store._contents = data["contents"]
        store._metadatas = data["metadatas"]
        store._corpus_tokens = data["corpus_tokens"]

        # Rebuild BM25 from tokens
        store._bm25 = BM25Okapi(
            store._corpus_tokens,
            k1=store.config.k1,
            b=store.config.b,
        )

        logger.info(f"Loaded BM25 index from: {path} ({len(store._ids)} documents)")
        return store

    @property
    def count(self) -> int:
        """Get the number of documents in the store."""
        return len(self._ids)
