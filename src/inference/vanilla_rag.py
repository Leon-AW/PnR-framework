"""
VanillaRAG - Plain Python RAG Implementation
=============================================

Standalone RAG pipeline for QM document retrieval and generation.

Features:
- Structure-aware document chunking
- Multilingual embeddings
- FAISS/ChromaDB vector storage
- HuggingFace model inference
- Interactive REPL for testing
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Union

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class VanillaRAGConfig:
    """Configuration for VanillaRAG pipeline.

    Attributes:
        model_name: HuggingFace model name or path to merged model
        embedding_model: Sentence-transformers model for embeddings
        vector_store_type: Backend for vector storage ('faiss' or 'chroma')
        vector_store_path: Path for persistent vector store
        chunk_max_tokens: Maximum tokens per chunk
        top_k: Number of chunks to retrieve
        max_new_tokens: Maximum tokens to generate
        temperature: Generation temperature
        device: Device to use ('cuda', 'cpu', or None for auto)
        load_in_4bit: Use 4-bit quantization for model
        load_in_8bit: Use 8-bit quantization for model
    """
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.3"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    vector_store_type: Literal["faiss", "chroma"] = "faiss"
    vector_store_path: Optional[str] = None
    chunk_max_tokens: int = 750
    top_k: int = 5
    max_new_tokens: int = 512
    temperature: float = 0.6
    device: Optional[str] = None
    load_in_4bit: bool = False
    load_in_8bit: bool = False


# =============================================================================
# VanillaRAG
# =============================================================================

class VanillaRAG:
    """Standalone RAG pipeline for QM documents.

    Example:
        ```python
        config = VanillaRAGConfig(
            model_name="checkpoints/QM_rag/merged",
            vector_store_path="./qm_vectorstore",
        )
        rag = VanillaRAG(config)

        # Index documents
        num_indexed = rag.index_documents([
            "src/data/documents/DE/LKR/Prüfstelle/Metallografie/A02-LKR.md",
        ])

        # Query
        result = rag.query("Wie ist der Ablauf bei der Mikrohärteprüfung?")
        print(result["answer"])
        print("Sources:", result["sources"])
        ```
    """

    def __init__(self, config: Optional[VanillaRAGConfig] = None):
        """Initialize the RAG pipeline.

        Args:
            config: Pipeline configuration
        """
        self.config = config or VanillaRAGConfig()
        self._chunker = None
        self._embedding_model = None
        self._vector_store = None
        self._model = None
        self._tokenizer = None

    # -------------------------------------------------------------------------
    # Lazy Loading
    # -------------------------------------------------------------------------

    @property
    def chunker(self):
        """Get the document chunker (lazy-loaded)."""
        if self._chunker is None:
            from src.data_loaders.structure_aware_chunker import (
                StructureAwareChunker,
                StructuredChunkConfig,
            )
            chunk_config = StructuredChunkConfig(
                max_chunk_tokens=self.config.chunk_max_tokens,
            )
            self._chunker = StructureAwareChunker(chunk_config)
        return self._chunker

    @property
    def embedding_model(self):
        """Get the embedding model (lazy-loaded)."""
        if self._embedding_model is None:
            from src.inference.embeddings import EmbeddingModel, EmbeddingConfig
            embed_config = EmbeddingConfig(
                model_name=self.config.embedding_model,
                device=self.config.device,
            )
            self._embedding_model = EmbeddingModel(embed_config)
        return self._embedding_model

    @property
    def vector_store(self):
        """Get the vector store (lazy-loaded)."""
        if self._vector_store is None:
            if self.config.vector_store_type == "faiss":
                from src.inference.vector_store import FAISSVectorStore, FAISSConfig
                faiss_config = FAISSConfig(
                    dimension=self.embedding_model.dimension,
                )
                if self.config.vector_store_path and Path(self.config.vector_store_path).exists():
                    self._vector_store = FAISSVectorStore.load(self.config.vector_store_path)
                else:
                    self._vector_store = FAISSVectorStore(faiss_config)
            else:
                from src.inference.vector_store import ChromaVectorStore, ChromaConfig
                chroma_config = ChromaConfig(
                    persist_directory=self.config.vector_store_path,
                )
                self._vector_store = ChromaVectorStore(chroma_config)
        return self._vector_store

    def _load_model(self):
        """Load the generation model."""
        if self._model is not None:
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers and torch are required. "
                "Install with: pip install transformers torch"
            )

        logger.info(f"Loading model: {self.config.model_name}")

        # Determine device
        if self.config.device:
            device = self.config.device
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        # Build loading kwargs
        load_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto" if device == "cuda" else None,
        }

        if self.config.load_in_4bit:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif self.config.load_in_8bit:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        else:
            load_kwargs["torch_dtype"] = torch.float16

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
        )

        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **load_kwargs,
        )

        if device == "cpu" or self.config.load_in_4bit or self.config.load_in_8bit:
            pass  # Model already on correct device
        else:
            self._model.to(device)

        self._model.eval()
        logger.info(f"Model loaded on {device}")

    # -------------------------------------------------------------------------
    # Indexing
    # -------------------------------------------------------------------------

    def index_documents(
        self,
        paths: list[Union[str, Path]],
        show_progress: bool = True,
    ) -> int:
        """Index documents into the vector store.

        Args:
            paths: List of document paths to index
            show_progress: Show progress bar

        Returns:
            Number of chunks indexed
        """
        from tqdm import tqdm

        all_chunks = []
        paths_iter = tqdm(paths, desc="Chunking documents") if show_progress else paths

        for path in paths_iter:
            chunks = self.chunker.chunk_document(path)
            all_chunks.extend(chunks)

        if not all_chunks:
            logger.warning("No chunks created from documents")
            return 0

        logger.info(f"Created {len(all_chunks)} chunks, generating embeddings...")

        # Generate embeddings
        chunk_texts = [c.format_with_context() for c in all_chunks]
        embeddings = self.embedding_model.encode_documents(
            chunk_texts,
            show_progress_bar=show_progress,
        )

        # Generate IDs and metadata
        ids = []
        contents = []
        metadatas = []

        for i, chunk in enumerate(all_chunks):
            chunk_id = f"{Path(chunk.source_file).stem}_{i}"
            ids.append(chunk_id)
            contents.append(chunk_texts[i])
            metadatas.append({
                "source_file": chunk.source_file,
                "path": chunk.path,
                "section": chunk.section_breadcrumb,
                "content_type": chunk.content_type,
                "chunk_index": chunk.chunk_index,
            })

        # Add to vector store
        self.vector_store.add(ids, embeddings, contents, metadatas)

        # Save if path configured
        if self.config.vector_store_path:
            self.vector_store.save(self.config.vector_store_path)

        logger.info(f"Indexed {len(all_chunks)} chunks into vector store")
        return len(all_chunks)

    def index_directory(
        self,
        directory: Union[str, Path],
        pattern: str = "**/*.md",
        show_progress: bool = True,
    ) -> int:
        """Index all matching documents in a directory.

        Args:
            directory: Directory to search
            pattern: Glob pattern for files
            show_progress: Show progress bar

        Returns:
            Number of chunks indexed
        """
        directory = Path(directory)
        paths = list(directory.glob(pattern))
        logger.info(f"Found {len(paths)} documents in {directory}")
        return self.index_documents(paths, show_progress)

    # -------------------------------------------------------------------------
    # Retrieval
    # -------------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        filter_metadata: Optional[dict] = None,
    ) -> list[dict]:
        """Retrieve relevant chunks for a query.

        Args:
            query: Query text
            k: Number of chunks to retrieve (default: config.top_k)
            filter_metadata: Optional metadata filter

        Returns:
            List of retrieved chunks with scores
        """
        k = k or self.config.top_k

        # Embed query
        query_embedding = self.embedding_model.encode(query)

        # Search vector store
        results = self.vector_store.search(
            query_embedding,
            k=k,
            filter_metadata=filter_metadata,
        )

        return [
            {
                "id": r.id,
                "score": r.score,
                "content": r.content,
                "metadata": r.metadata,
            }
            for r in results
        ]

    # -------------------------------------------------------------------------
    # Generation
    # -------------------------------------------------------------------------

    def _build_prompt(self, question: str, context: str) -> str:
        """Build the prompt for generation.

        Args:
            question: User question
            context: Retrieved context

        Returns:
            Formatted prompt
        """
        return f"""Du bist ein hilfreicher Assistent für Qualitätsmanagement-Dokumentation.
Beantworte die Frage basierend auf dem folgenden Kontext. Wenn die Antwort nicht im Kontext enthalten ist, sage das ehrlich.

Kontext:
{context}

Frage: {question}

Antwort:"""

    def generate(
        self,
        question: str,
        context: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Generate an answer given question and context.

        Args:
            question: User question
            context: Retrieved context
            max_new_tokens: Override max tokens
            temperature: Override temperature

        Returns:
            Generated answer
        """
        self._load_model()

        import torch

        prompt = self._build_prompt(question, context)

        inputs = self._tokenizer(prompt, return_tensors="pt")
        if hasattr(self._model, "device"):
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or self.config.max_new_tokens,
                temperature=temperature or self.config.temperature,
                do_sample=True,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        # Decode only the new tokens
        input_length = inputs["input_ids"].shape[1]
        answer = self._tokenizer.decode(
            outputs[0][input_length:],
            skip_special_tokens=True,
        )

        return answer.strip()

    # -------------------------------------------------------------------------
    # Query Pipeline
    # -------------------------------------------------------------------------

    def query(
        self,
        question: str,
        k: Optional[int] = None,
        filter_metadata: Optional[dict] = None,
    ) -> dict:
        """Full RAG query: retrieve relevant chunks and generate answer.

        Args:
            question: User question
            k: Number of chunks to retrieve
            filter_metadata: Optional metadata filter

        Returns:
            Dictionary with:
            - answer: Generated answer
            - sources: List of source files
            - retrieved_chunks: Full chunk details
        """
        # Retrieve
        chunks = self.retrieve(question, k=k, filter_metadata=filter_metadata)

        if not chunks:
            return {
                "answer": "Keine relevanten Dokumente gefunden.",
                "sources": [],
                "retrieved_chunks": [],
            }

        # Build context from retrieved chunks
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(f"--- Dokument {i} ---")
            context_parts.append(chunk["content"])
            context_parts.append("")
        context = "\n".join(context_parts)

        # Generate
        answer = self.generate(question, context)

        # Extract sources
        sources = list(set(
            chunk["metadata"].get("source_file", "unknown")
            for chunk in chunks
        ))

        return {
            "answer": answer,
            "sources": sources,
            "retrieved_chunks": chunks,
        }

    # -------------------------------------------------------------------------
    # Interactive Mode
    # -------------------------------------------------------------------------

    def interactive_session(self):
        """Start an interactive REPL session for testing.

        Commands:
        - /quit, /exit: Exit the session
        - /index <path>: Index a document or directory
        - /search <query>: Search without generation
        - /clear: Clear the vector store
        """
        print("=" * 60)
        print("VanillaRAG Interactive Session")
        print("=" * 60)
        print(f"Model: {self.config.model_name}")
        print(f"Vector store: {self.config.vector_store_type}")
        print(f"Documents indexed: {self.vector_store.count}")
        print()
        print("Commands: /quit, /index <path>, /search <query>, /clear")
        print("=" * 60)
        print()

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not user_input:
                continue

            # Handle commands
            if user_input.startswith("/"):
                parts = user_input.split(maxsplit=1)
                cmd = parts[0].lower()
                arg = parts[1] if len(parts) > 1 else ""

                if cmd in ("/quit", "/exit"):
                    print("Goodbye!")
                    break

                elif cmd == "/index":
                    if not arg:
                        print("Usage: /index <path>")
                        continue
                    path = Path(arg)
                    if path.is_dir():
                        num = self.index_directory(path)
                    elif path.exists():
                        num = self.index_documents([path])
                    else:
                        print(f"Path not found: {path}")
                        continue
                    print(f"Indexed {num} chunks")

                elif cmd == "/search":
                    if not arg:
                        print("Usage: /search <query>")
                        continue
                    chunks = self.retrieve(arg)
                    print(f"\nFound {len(chunks)} chunks:")
                    for i, chunk in enumerate(chunks, 1):
                        print(f"\n--- Chunk {i} (score: {chunk['score']:.3f}) ---")
                        print(f"Source: {chunk['metadata'].get('source_file', 'unknown')}")
                        print(f"Section: {chunk['metadata'].get('section', 'N/A')}")
                        print(chunk["content"][:500] + "..." if len(chunk["content"]) > 500 else chunk["content"])

                elif cmd == "/clear":
                    self._vector_store = None
                    print("Vector store cleared")

                else:
                    print(f"Unknown command: {cmd}")

                continue

            # Regular query
            print("\nSearching and generating...")
            result = self.query(user_input)

            print("\n" + "=" * 40)
            print("Answer:")
            print(result["answer"])
            print("\nSources:")
            for source in result["sources"]:
                print(f"  - {source}")
            print("=" * 40 + "\n")

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------

    def save_vector_store(self, path: Optional[Union[str, Path]] = None):
        """Save the vector store to disk.

        Args:
            path: Path to save to (default: config.vector_store_path)
        """
        path = path or self.config.vector_store_path
        if path:
            self.vector_store.save(path)
            logger.info(f"Saved vector store to: {path}")
        else:
            logger.warning("No vector store path configured")

    def load_vector_store(self, path: Union[str, Path]):
        """Load a vector store from disk.

        Args:
            path: Path to load from
        """
        if self.config.vector_store_type == "faiss":
            from src.inference.vector_store import FAISSVectorStore
            self._vector_store = FAISSVectorStore.load(path)
        else:
            from src.inference.vector_store import ChromaVectorStore
            self._vector_store = ChromaVectorStore.load(path)
        logger.info(f"Loaded vector store from: {path}")


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """Command-line interface for VanillaRAG."""
    import argparse

    parser = argparse.ArgumentParser(description="VanillaRAG - QM Document RAG")
    parser.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.3",
                        help="Model name or path")
    parser.add_argument("--embedding-model",
                        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                        help="Embedding model name")
    parser.add_argument("--vector-store", default="faiss", choices=["faiss", "chroma"],
                        help="Vector store backend")
    parser.add_argument("--vector-store-path", help="Path for persistent vector store")
    parser.add_argument("--index", nargs="+", help="Documents or directories to index")
    parser.add_argument("--query", help="Single query to execute")
    parser.add_argument("--interactive", action="store_true", help="Start interactive session")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve")
    parser.add_argument("--4bit", dest="load_4bit", action="store_true",
                        help="Load model in 4-bit quantization")
    parser.add_argument("--8bit", dest="load_8bit", action="store_true",
                        help="Load model in 8-bit quantization")

    args = parser.parse_args()

    config = VanillaRAGConfig(
        model_name=args.model,
        embedding_model=args.embedding_model,
        vector_store_type=args.vector_store,
        vector_store_path=args.vector_store_path,
        top_k=args.top_k,
        load_in_4bit=args.load_4bit,
        load_in_8bit=args.load_8bit,
    )

    rag = VanillaRAG(config)

    # Index documents
    if args.index:
        for path in args.index:
            path = Path(path)
            if path.is_dir():
                rag.index_directory(path)
            else:
                rag.index_documents([path])

    # Execute query
    if args.query:
        result = rag.query(args.query)
        print("\nAnswer:")
        print(result["answer"])
        print("\nSources:")
        for source in result["sources"]:
            print(f"  - {source}")
    elif args.interactive:
        rag.interactive_session()
    elif not args.index:
        # Default to interactive if nothing specified
        rag.interactive_session()


if __name__ == "__main__":
    main()
