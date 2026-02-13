"""
Query Pipeline — Core RAG Orchestrator
=======================================

Orchestrates the full RAG pipeline: data source routing, query analysis,
hybrid retrieval (FAISS + BM25 with RRF fusion), cross-encoder reranking,
context assembly with citations, and prompt building.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.inference.bm25_store import BM25Store
from src.inference.embeddings import EmbeddingConfig, EmbeddingModel
from src.inference.rag_config import RAGServerConfig
from src.inference.reranker import Reranker, RerankerConfig
from src.inference.vector_store import FAISSVectorStore, SearchResult

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class QueryAnalysis:
    """Analysis of a user query."""
    original_query: str
    reformulated_query: str
    language: str  # "de" or "en"
    intent: str  # "question", "instruction", "greeting", "other"
    keywords: list[str]
    needs_retrieval: bool
    data_source: str  # "lkr" or "ait"


@dataclass
class Citation:
    """A citation reference for a retrieved chunk."""
    index: int
    chunk_id: str
    source_file: str
    section: str
    score: float
    content_preview: str
    intranet_url: str = ""


@dataclass
class PipelineResult:
    """Result of the query pipeline."""
    messages: list[dict]  # OpenAI-format messages
    citations: list[Citation]
    query_analysis: QueryAnalysis
    metadata: dict = field(default_factory=dict)


# =============================================================================
# Data Source Routing
# =============================================================================

# Keywords that indicate LKR data source
_LKR_INDICATORS = re.compile(
    r"\b(lkr|ranshofen|leichtmetallkompetenzzentrum|leichtmetall)\b",
    re.IGNORECASE,
)

# Keywords that indicate AIT data source
_AIT_INDICATORS = re.compile(
    r"\b(ait|austrian\s+institute|giefinggasse|seibersdorf|klagenfurt)\b",
    re.IGNORECASE,
)

# Patterns for greetings / non-retrieval queries
_GREETING_PATTERN = re.compile(
    r"^(hallo|hi|hey|guten\s+(tag|morgen|abend)|servus|hello|good\s+(morning|evening)|danke|thank)",
    re.IGNORECASE,
)


# =============================================================================
# Data Source Manager
# =============================================================================

class DataSourceManager:
    """Manages per-source FAISS and BM25 indices."""

    def __init__(self):
        self._sources: dict[str, tuple[FAISSVectorStore, BM25Store]] = {}

    def load_source(self, name: str, faiss_path: str, bm25_path: str) -> None:
        """Load indices for a data source.

        Args:
            name: Source name (e.g., "ait", "lkr")
            faiss_path: Path to FAISS index directory
            bm25_path: Path to BM25 pickle file
        """
        faiss_p = Path(faiss_path)
        bm25_p = Path(bm25_path)

        if not faiss_p.exists():
            logger.warning(f"FAISS index not found for '{name}': {faiss_p}")
            return
        if not bm25_p.exists():
            logger.warning(f"BM25 index not found for '{name}': {bm25_p}")
            return

        logger.info(f"Loading data source '{name}': FAISS={faiss_p}, BM25={bm25_p}")
        faiss_store = FAISSVectorStore.load(faiss_path)
        bm25_store = BM25Store.load(bm25_path)
        self._sources[name] = (faiss_store, bm25_store)
        logger.info(
            f"Loaded '{name}': {faiss_store.count} FAISS docs, "
            f"{bm25_store.count} BM25 docs"
        )

    def get_stores(self, name: str) -> tuple[FAISSVectorStore, BM25Store]:
        """Get the FAISS and BM25 stores for a data source.

        Args:
            name: Source name

        Returns:
            Tuple of (FAISSVectorStore, BM25Store)

        Raises:
            KeyError: If source not loaded
        """
        if name not in self._sources:
            raise KeyError(
                f"Data source '{name}' not loaded. "
                f"Available: {list(self._sources.keys())}"
            )
        return self._sources[name]

    @property
    def loaded_sources(self) -> list[str]:
        """Get list of loaded source names."""
        return list(self._sources.keys())

    def source_stats(self) -> dict[str, dict]:
        """Get statistics for all loaded sources."""
        stats = {}
        for name, (faiss_store, bm25_store) in self._sources.items():
            stats[name] = {
                "faiss_count": faiss_store.count,
                "bm25_count": bm25_store.count,
            }
        return stats


# =============================================================================
# Query Pipeline
# =============================================================================

class QueryPipeline:
    """Core RAG orchestrator.

    Handles the full pipeline from user message to augmented prompt:
    1. Data source routing (LKR vs AIT)
    2. Query analysis (language, intent, keywords)
    3. Query contextualization (anaphora resolution)
    4. Hybrid retrieval (FAISS + BM25 → RRF fusion)
    5. Cross-encoder reranking
    6. Context assembly with citations
    7. Prompt building
    """

    def __init__(self, config: RAGServerConfig):
        self.config = config
        self._embedding_model: Optional[EmbeddingModel] = None
        self._reranker: Optional[Reranker] = None
        self._source_manager = DataSourceManager()
        self._intranet_links: dict[str, str] = {}
        self._query_count = 0
        self._total_retrieval_time = 0.0

    def load(self) -> None:
        """Load all models and indices."""
        # Load embedding model
        logger.info("Loading embedding model...")
        embed_config = EmbeddingConfig(model_name=self.config.embedding_model)
        self._embedding_model = EmbeddingModel(embed_config)
        # Trigger actual model load
        _ = self._embedding_model.dimension

        # Load reranker eagerly (must initialize CUDA on main thread)
        if self.config.enable_reranking:
            logger.info("Loading reranker model...")
            reranker_config = RerankerConfig(model_name=self.config.reranker_model)
            self._reranker = Reranker(reranker_config)
            self._reranker.warmup()

        # Load data sources
        for name, ds_config in self.config.data_sources.items():
            self._source_manager.load_source(
                name, ds_config.faiss_index_path, ds_config.bm25_index_path
            )

        # Load intranet link mapping
        links_path = Path(self.config.intranet_links_path)
        if links_path.exists():
            try:
                link_data = json.loads(links_path.read_text(encoding="utf-8"))
                self._intranet_links = link_data.get("links", {})
                logger.info(f"Loaded {len(self._intranet_links)} intranet link mappings")
            except Exception as e:
                logger.warning(f"Failed to load intranet links from {links_path}: {e}")
        else:
            logger.info(f"No intranet links file at {links_path} (citations won't have URLs)")

        logger.info(
            f"Pipeline loaded: sources={self._source_manager.loaded_sources}, "
            f"reranking={'on' if self.config.enable_reranking else 'off'}"
        )

    @property
    def is_loaded(self) -> bool:
        """Check if the pipeline is loaded and ready."""
        return self._embedding_model is not None and len(self._source_manager.loaded_sources) > 0

    @property
    def loaded_sources(self) -> list[str]:
        """Get list of loaded data sources."""
        return self._source_manager.loaded_sources

    def source_stats(self) -> dict:
        """Get pipeline statistics."""
        return {
            "sources": self._source_manager.source_stats(),
            "query_count": self._query_count,
            "avg_retrieval_ms": (
                (self._total_retrieval_time / self._query_count * 1000)
                if self._query_count > 0
                else 0
            ),
            "reranking_enabled": self.config.enable_reranking,
        }

    # -------------------------------------------------------------------------
    # Data Source Routing
    # -------------------------------------------------------------------------

    def detect_data_source(
        self, user_message: str, history: list[dict]
    ) -> str:
        """Detect which data source to use based on user message and history.

        Args:
            user_message: Current user message
            history: Conversation history (OpenAI message format)

        Returns:
            Data source name ("lkr" or "ait")
        """
        # Check current message for LKR indicators
        if _LKR_INDICATORS.search(user_message):
            return "lkr"

        # Check current message for AIT indicators
        if _AIT_INDICATORS.search(user_message):
            return "ait"

        # Check conversation history (most recent first) for source decisions
        for msg in reversed(history):
            content = msg.get("content", "")
            if not content:
                continue
            if _LKR_INDICATORS.search(content):
                return "lkr"
            if _AIT_INDICATORS.search(content):
                return "ait"

        return self.config.default_data_source

    # -------------------------------------------------------------------------
    # Query Analysis
    # -------------------------------------------------------------------------

    def analyze_query(
        self, user_message: str, history: list[dict]
    ) -> QueryAnalysis:
        """Analyze the user query for language, intent, and keywords.

        Args:
            user_message: Current user message
            history: Conversation history

        Returns:
            QueryAnalysis object
        """
        # Detect data source
        data_source = self.detect_data_source(user_message, history)

        # Detect language (simple heuristic)
        german_indicators = re.findall(
            r"\b(ist|sind|wie|was|der|die|das|und|oder|für|über|können|werden)\b",
            user_message,
            re.IGNORECASE,
        )
        language = "de" if len(german_indicators) >= 2 else "en"
        # If message has German umlauts/sharp-s, lean toward German
        if re.search(r"[äöüßÄÖÜ]", user_message):
            language = "de"

        # Detect intent
        if _GREETING_PATTERN.match(user_message.strip()):
            intent = "greeting"
            needs_retrieval = False
        elif len(user_message.strip()) < 5:
            intent = "other"
            needs_retrieval = False
        elif re.search(r"\?$", user_message.strip()):
            intent = "question"
            needs_retrieval = True
        else:
            intent = "instruction"
            needs_retrieval = True

        # Extract keywords (nouns/meaningful words)
        keywords = re.findall(r"\b[A-ZÄÖÜ][a-zäöüß]{2,}\b", user_message)
        keywords = list(dict.fromkeys(keywords))  # deduplicate preserving order

        # Contextualize query (resolve anaphora)
        reformulated = self._contextualize_query(user_message, history)

        return QueryAnalysis(
            original_query=user_message,
            reformulated_query=reformulated,
            language=language,
            intent=intent,
            keywords=keywords,
            needs_retrieval=needs_retrieval,
            data_source=data_source,
        )

    def _contextualize_query(
        self, user_message: str, history: list[dict]
    ) -> str:
        """Resolve anaphoric references using conversation history.

        If the user message contains pronouns like "das", "es", "this", "it"
        without clear referents, prepend context from the last assistant reply.

        Args:
            user_message: Current user message
            history: Conversation history

        Returns:
            Reformulated query string
        """
        # Check for anaphoric patterns
        anaphora = re.search(
            r"\b(das|dies|dieses|es|davon|dabei|dazu|dafür|darüber|"
            r"this|that|it|these|those)\b",
            user_message,
            re.IGNORECASE,
        )

        if not anaphora or not history:
            return user_message

        # Find the last user message to get topic context
        last_user_msg = ""
        for msg in reversed(history):
            if msg.get("role") == "user":
                last_user_msg = msg.get("content", "")
                break

        if not last_user_msg:
            return user_message

        # Prepend the previous topic as context for the search query
        return f"{last_user_msg} {user_message}"

    # -------------------------------------------------------------------------
    # Hybrid Retrieval
    # -------------------------------------------------------------------------

    def hybrid_retrieve(
        self, query: str, data_source: str
    ) -> list[SearchResult]:
        """Perform hybrid retrieval: FAISS dense + BM25 sparse with RRF fusion.

        Args:
            query: Search query (potentially reformulated)
            data_source: Data source name

        Returns:
            Fused and deduplicated list of SearchResult objects
        """
        faiss_store, bm25_store = self._source_manager.get_stores(data_source)

        # Dense retrieval (FAISS)
        query_embedding = self._embedding_model.encode(query)
        dense_results = faiss_store.search(
            query_embedding, k=self.config.dense_top_k
        )

        # Sparse retrieval (BM25)
        sparse_results = bm25_store.search(query, k=self.config.sparse_top_k)

        # Reciprocal Rank Fusion (RRF)
        return self._rrf_fuse(dense_results, sparse_results)

    def _rrf_fuse(
        self,
        dense_results: list[SearchResult],
        sparse_results: list[SearchResult],
    ) -> list[SearchResult]:
        """Fuse dense and sparse results using Reciprocal Rank Fusion.

        RRF score = sum(1 / (k + rank)) across all result lists.

        Args:
            dense_results: Results from FAISS
            sparse_results: Results from BM25

        Returns:
            Fused results sorted by RRF score
        """
        k = self.config.rrf_k
        scores: dict[str, float] = {}
        results_by_id: dict[str, SearchResult] = {}

        # Score dense results
        for rank, result in enumerate(dense_results):
            scores[result.id] = scores.get(result.id, 0) + 1.0 / (k + rank + 1)
            results_by_id[result.id] = result

        # Score sparse results
        for rank, result in enumerate(sparse_results):
            scores[result.id] = scores.get(result.id, 0) + 1.0 / (k + rank + 1)
            if result.id not in results_by_id:
                results_by_id[result.id] = result

        # Sort by fused score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        fused = []
        for doc_id in sorted_ids[: self.config.fusion_top_n]:
            result = results_by_id[doc_id]
            fused.append(
                SearchResult(
                    id=result.id,
                    score=scores[doc_id],
                    content=result.content,
                    metadata=result.metadata,
                )
            )

        return fused

    # -------------------------------------------------------------------------
    # Reranking
    # -------------------------------------------------------------------------

    def rerank(self, query: str, candidates: list[SearchResult]) -> list[SearchResult]:
        """Rerank candidates using cross-encoder.

        Args:
            query: Search query
            candidates: Retrieval candidates

        Returns:
            Reranked candidates
        """
        if not self.config.enable_reranking or self._reranker is None:
            return candidates[: self.config.rerank_top_k]

        return self._reranker.rerank(
            query, candidates, top_k=self.config.rerank_top_k
        )

    # -------------------------------------------------------------------------
    # Intranet URL Resolution
    # -------------------------------------------------------------------------

    def _resolve_intranet_url(self, source_file: str) -> str:
        """Resolve the intranet URL for a source file.

        Tries to match the document stem (filename without extension) against
        the intranet link mapping.

        Args:
            source_file: Source file path from chunk metadata

        Returns:
            Intranet URL string, or empty string if not found
        """
        if not self._intranet_links or not source_file:
            return ""

        # Extract document stem from source path
        # source_file can be like "QM/DE/AIT/.../P08-AIT.md" or full path
        stem = Path(source_file).stem
        return self._intranet_links.get(stem, "")

    # -------------------------------------------------------------------------
    # Context Assembly
    # -------------------------------------------------------------------------

    def assemble_context(
        self, results: list[SearchResult], language: str = "de"
    ) -> tuple[str, list[Citation]]:
        """Assemble context string with citation markers.

        Args:
            results: Reranked search results
            language: Detected language ("de" or "en")

        Returns:
            Tuple of (context_string, citations_list)
        """
        source_label = "Quelle" if language == "de" else "Source"
        context_parts = []
        citations = []
        total_tokens = 0

        for i, result in enumerate(results):
            # Estimate tokens for this chunk
            chunk_tokens = len(result.content) // 4
            if total_tokens + chunk_tokens > self.config.max_context_tokens:
                # Try to fit a truncated version
                remaining = self.config.max_context_tokens - total_tokens
                if remaining > 100:
                    truncated = result.content[: remaining * 4]
                    context_parts.append(f"[{source_label} {i + 1}]\n{truncated}...")
                break

            context_parts.append(f"[{source_label} {i + 1}]\n{result.content}")
            total_tokens += chunk_tokens

            # Build citation
            metadata = result.metadata
            source_file = metadata.get("source", metadata.get("source_file", ""))
            section = metadata.get("section", "")
            preview = result.content[:150].replace("\n", " ")

            # Look up intranet URL for this source file
            intranet_url = self._resolve_intranet_url(source_file)

            citations.append(
                Citation(
                    index=i + 1,
                    chunk_id=result.id,
                    source_file=source_file,
                    section=section,
                    score=result.score,
                    content_preview=preview,
                    intranet_url=intranet_url,
                )
            )

        context = "\n\n".join(context_parts)
        return context, citations

    # -------------------------------------------------------------------------
    # Prompt Building
    # -------------------------------------------------------------------------

    def build_messages(
        self,
        user_message: str,
        history: list[dict],
        context: str,
        analysis: QueryAnalysis,
        citations: list[Citation],
    ) -> list[dict]:
        """Build OpenAI-format messages with RAG context.

        Args:
            user_message: Original user message
            history: Conversation history
            context: Assembled RAG context
            analysis: Query analysis result
            citations: Citation list

        Returns:
            List of message dicts in OpenAI format
        """
        # System prompt
        source_label = "LKR (Leichtmetallkompetenzzentrum Ranshofen)" if analysis.data_source == "lkr" else "AIT (Austrian Institute of Technology)"

        if analysis.language == "de":
            system_prompt = (
                f"Du bist ein hilfreicher Assistent für Qualitätsmanagement-Dokumentation "
                f"({source_label}). "
                f"Beantworte Fragen basierend auf dem bereitgestellten Kontext. "
                f"Verwende die [Quelle N] Verweise in deinen Antworten, um auf die "
                f"relevanten Dokumente zu verweisen. "
                f"Wenn die Antwort nicht im Kontext enthalten ist, sage das ehrlich. "
                f"Antworte auf Deutsch, es sei denn, der Benutzer fragt auf Englisch."
            )
        else:
            system_prompt = (
                f"You are a helpful assistant for quality management documentation "
                f"({source_label}). "
                f"Answer questions based on the provided context. "
                f"Use [Source N] references in your answers to cite relevant documents. "
                f"If the answer is not in the context, say so honestly."
            )

        if context:
            system_prompt += f"\n\n--- Kontext ---\n{context}\n--- Ende Kontext ---"

        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history (last N turns)
        max_history = self.config.max_history_turns * 2  # user + assistant pairs
        recent_history = history[-max_history:] if history else []
        for msg in recent_history:
            if msg.get("role") in ("user", "assistant"):
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"],
                })

        # Add current user message
        messages.append({"role": "user", "content": user_message})

        return messages

    # -------------------------------------------------------------------------
    # Main Pipeline
    # -------------------------------------------------------------------------

    def run(
        self, user_message: str, history: list[dict]
    ) -> PipelineResult:
        """Run the full RAG pipeline.

        Args:
            user_message: Current user message
            history: Conversation history (OpenAI message format)

        Returns:
            PipelineResult with augmented messages and citations
        """
        start_time = time.time()
        self._query_count += 1

        # 1. Analyze query
        analysis = self.analyze_query(user_message, history)
        logger.info(
            f"Query analysis: lang={analysis.language}, intent={analysis.intent}, "
            f"source={analysis.data_source}, retrieval={analysis.needs_retrieval}"
        )

        # 2. If no retrieval needed, pass through with minimal system prompt
        if not analysis.needs_retrieval:
            messages = self.build_messages(
                user_message, history, "", analysis, []
            )
            return PipelineResult(
                messages=messages,
                citations=[],
                query_analysis=analysis,
                metadata={"retrieval_skipped": True},
            )

        # 3. Check if the requested data source is available
        if analysis.data_source not in self._source_manager.loaded_sources:
            available = self._source_manager.loaded_sources
            if available:
                analysis.data_source = available[0]
                logger.warning(
                    f"Requested source not available, falling back to "
                    f"'{analysis.data_source}'"
                )
            else:
                # No sources loaded — pass through without retrieval
                messages = self.build_messages(
                    user_message, history, "", analysis, []
                )
                return PipelineResult(
                    messages=messages,
                    citations=[],
                    query_analysis=analysis,
                    metadata={"error": "no_data_sources_loaded"},
                )

        # 4. Hybrid retrieval
        candidates = self.hybrid_retrieve(
            analysis.reformulated_query, analysis.data_source
        )
        logger.info(f"Hybrid retrieval: {len(candidates)} candidates")

        # 5. Rerank
        reranked = self.rerank(analysis.reformulated_query, candidates)
        logger.info(f"After reranking: {len(reranked)} results")

        # 6. Assemble context with citations
        context, citations = self.assemble_context(reranked, analysis.language)

        # 7. Build messages
        messages = self.build_messages(
            user_message, history, context, analysis, citations
        )

        elapsed = time.time() - start_time
        self._total_retrieval_time += elapsed

        logger.info(
            f"Pipeline complete: {len(citations)} citations, "
            f"{elapsed:.3f}s"
        )

        return PipelineResult(
            messages=messages,
            citations=citations,
            query_analysis=analysis,
            metadata={
                "retrieval_time_ms": elapsed * 1000,
                "data_source": analysis.data_source,
                "candidates_before_rerank": len(candidates),
                "results_after_rerank": len(reranked),
            },
        )
