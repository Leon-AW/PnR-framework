"""
RAG Server Configuration
========================

Centralized configuration for the advanced RAG server.
All settings controllable via RAG_* environment variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataSourceConfig:
    """Configuration for a single data source.

    Attributes:
        name: Source identifier (e.g., "ait", "lkr")
        documents_dir: Path to source documents
        faiss_index_path: Path to FAISS index directory
        bm25_index_path: Path to BM25 pickle file
    """
    name: str
    documents_dir: str
    faiss_index_path: str
    bm25_index_path: str


@dataclass
class RAGServerConfig:
    """Configuration for the RAG server.

    Attributes:
        host: Server host
        port: Server port
        llama_url: llama.cpp server URL
        embedding_model: Sentence-transformers model name
        reranker_model: Cross-encoder model name
        dense_top_k: Number of results from FAISS
        sparse_top_k: Number of results from BM25
        rrf_k: RRF fusion constant
        fusion_top_n: Number of candidates after fusion
        rerank_top_k: Final number of results after reranking
        max_context_tokens: Maximum tokens in assembled context
        max_history_turns: Number of conversation turns to include
        default_max_tokens: Default max generation tokens
        default_temperature: Default generation temperature
        enable_reranking: Whether to enable cross-encoder reranking
        enable_think_stripping: Whether to strip <think> tokens
        enable_citations: Whether to append citation footer
        data_sources: Mapping of source names to configurations
        default_data_source: Default data source name
        log_level: Logging level
    """
    host: str = "0.0.0.0"
    port: int = 8000
    llama_url: str = "http://localhost:8080"

    # Models
    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    # Retrieval parameters
    dense_top_k: int = 20
    sparse_top_k: int = 20
    rrf_k: int = 60
    fusion_top_n: int = 15
    rerank_top_k: int = 5
    max_context_tokens: int = 3000
    max_history_turns: int = 4

    # Generation defaults
    default_max_tokens: int = 8192
    default_temperature: float = 0.6
    default_frequency_penalty: float = 0.3
    default_presence_penalty: float = 0.3

    # Feature toggles
    enable_reranking: bool = True
    enable_think_stripping: bool = True
    enable_citations: bool = True

    # Intranet link mapping
    intranet_links_path: str = "./qm_vectorstore_advanced/intranet_links.json"

    # Data sources
    data_sources: dict[str, DataSourceConfig] = field(default_factory=dict)
    default_data_source: str = "ait"

    # Logging
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "RAGServerConfig":
        """Create configuration from environment variables.

        Environment variables use the RAG_ prefix. For example:
            RAG_HOST, RAG_PORT, RAG_LLAMA_URL, RAG_EMBEDDING_MODEL, etc.
        """
        config = cls(
            host=os.environ.get("RAG_HOST", cls.host),
            port=int(os.environ.get("RAG_PORT", cls.port)),
            llama_url=os.environ.get("RAG_LLAMA_URL", cls.llama_url),
            embedding_model=os.environ.get("RAG_EMBEDDING_MODEL", cls.embedding_model),
            reranker_model=os.environ.get("RAG_RERANKER_MODEL", cls.reranker_model),
            dense_top_k=int(os.environ.get("RAG_DENSE_TOP_K", cls.dense_top_k)),
            sparse_top_k=int(os.environ.get("RAG_SPARSE_TOP_K", cls.sparse_top_k)),
            rrf_k=int(os.environ.get("RAG_RRF_K", cls.rrf_k)),
            fusion_top_n=int(os.environ.get("RAG_FUSION_TOP_N", cls.fusion_top_n)),
            rerank_top_k=int(os.environ.get("RAG_RERANK_TOP_K", cls.rerank_top_k)),
            max_context_tokens=int(os.environ.get("RAG_MAX_CONTEXT_TOKENS", cls.max_context_tokens)),
            max_history_turns=int(os.environ.get("RAG_MAX_HISTORY_TURNS", cls.max_history_turns)),
            default_max_tokens=int(os.environ.get("RAG_DEFAULT_MAX_TOKENS", cls.default_max_tokens)),
            default_temperature=float(os.environ.get("RAG_DEFAULT_TEMPERATURE", cls.default_temperature)),
            default_frequency_penalty=float(os.environ.get("RAG_DEFAULT_FREQUENCY_PENALTY", cls.default_frequency_penalty)),
            default_presence_penalty=float(os.environ.get("RAG_DEFAULT_PRESENCE_PENALTY", cls.default_presence_penalty)),
            enable_reranking=os.environ.get("RAG_ENABLE_RERANKING", "true").lower() == "true",
            enable_think_stripping=os.environ.get("RAG_ENABLE_THINK_STRIPPING", "true").lower() == "true",
            enable_citations=os.environ.get("RAG_ENABLE_CITATIONS", "true").lower() == "true",
            intranet_links_path=os.environ.get("RAG_INTRANET_LINKS_PATH", cls.intranet_links_path),
            default_data_source=os.environ.get("RAG_DEFAULT_DATA_SOURCE", cls.default_data_source),
            log_level=os.environ.get("RAG_LOG_LEVEL", cls.log_level),
        )

        # Configure data sources from env or use defaults
        base_dir = os.environ.get("RAG_INDEX_BASE_DIR", "./qm_vectorstore_advanced")

        ait_docs = os.environ.get("RAG_AIT_DOCUMENTS_DIR", "src/data/cleaned_documents/DE/AIT")
        ait_faiss = os.environ.get("RAG_AIT_FAISS_PATH", f"{base_dir}/ait/faiss_index")
        ait_bm25 = os.environ.get("RAG_AIT_BM25_PATH", f"{base_dir}/ait/bm25_index.pkl")

        lkr_docs = os.environ.get("RAG_LKR_DOCUMENTS_DIR", "src/data/documents/DE/LKR")
        lkr_faiss = os.environ.get("RAG_LKR_FAISS_PATH", f"{base_dir}/lkr/faiss_index")
        lkr_bm25 = os.environ.get("RAG_LKR_BM25_PATH", f"{base_dir}/lkr/bm25_index.pkl")

        config.data_sources = {
            "ait": DataSourceConfig(
                name="ait",
                documents_dir=ait_docs,
                faiss_index_path=ait_faiss,
                bm25_index_path=ait_bm25,
            ),
            "lkr": DataSourceConfig(
                name="lkr",
                documents_dir=lkr_docs,
                faiss_index_path=lkr_faiss,
                bm25_index_path=lkr_bm25,
            ),
        }

        return config
