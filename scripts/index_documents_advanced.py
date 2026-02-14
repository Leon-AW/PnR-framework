#!/usr/bin/env python3
"""
Index Documents for Advanced RAG Server
========================================

Builds FAISS and BM25 indices for the advanced RAG server.
Separate from index_documents.py (which serves VanillaRAG).

Usage:
    python scripts/index_documents_advanced.py --source all
    python scripts/index_documents_advanced.py --source ait
    python scripts/index_documents_advanced.py --source lkr
    python scripts/index_documents_advanced.py --source all --output-dir ./qm_vectorstore_advanced
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_loaders import StructureAwareChunker, StructuredChunkConfig
from src.inference.bm25_store import BM25Store, BM25Config
from src.inference.embeddings import EmbeddingModel, EmbeddingConfig
from src.inference.vector_store import FAISSVectorStore, FAISSConfig

# Data source definitions
DATA_SOURCES = {
    "ait": {
        "documents_dir": "src/data/cleaned_documents/DE/AIT",
        "description": "AIT (Austrian Institute of Technology)",
    },
    "lkr": {
        "documents_dir": "src/data/documents/DE/LKR",
        "description": "LKR (Leichtmetallkompetenzzentrum Ranshofen)",
    },
}


def index_source(
    source_name: str,
    documents_dir: str,
    output_dir: str,
    embedder: EmbeddingModel,
    batch_size: int = 32,
) -> dict:
    """Index a single data source (FAISS + BM25).

    Args:
        source_name: Source identifier
        documents_dir: Path to documents
        output_dir: Base output directory
        embedder: Embedding model instance
        batch_size: Embedding batch size

    Returns:
        Statistics dict
    """
    docs_dir = Path(documents_dir)
    out_dir = Path(output_dir) / source_name

    if not docs_dir.exists():
        print(f"  Error: Documents directory not found: {docs_dir}")
        return {"error": f"Directory not found: {docs_dir}"}

    # Find all markdown files
    md_files = sorted(docs_dir.rglob("*.md"))
    print(f"  Found {len(md_files)} markdown documents in {docs_dir}")

    if not md_files:
        print("  No documents found!")
        return {"error": "No documents found"}

    # Initialize chunker
    chunk_config = StructuredChunkConfig(
        max_chunk_tokens=750,
        table_max_tokens=1500,
        list_max_tokens=500,
        overlap_tokens=50,
        include_breadcrumb=True,
        include_path=True,
    )
    chunker = StructureAwareChunker(chunk_config)

    # Process documents
    print(f"  Chunking {len(md_files)} documents...")
    all_chunks = []
    all_metadata = []
    failed_docs = 0

    for i, doc_path in enumerate(md_files):
        try:
            chunks = chunker.chunk_document(doc_path)
            for chunk in chunks:
                all_chunks.append(chunk.format_with_context())
                all_metadata.append({
                    "source": chunk.source_file or str(doc_path),
                    "section": chunk.section_breadcrumb or "",
                    "content_type": chunk.content_type or "text",
                    "path": chunk.path or "",
                    "chunk_index": chunk.chunk_index,
                })

            if (i + 1) % 100 == 0:
                print(f"    Processed {i + 1}/{len(md_files)} documents ({len(all_chunks)} chunks)")

        except Exception as e:
            failed_docs += 1
            if failed_docs <= 5:
                print(f"    Warning: Failed to process {doc_path.name}: {e}")
            elif failed_docs == 6:
                print("    ... (suppressing further warnings)")

    if failed_docs > 0:
        print(f"    Total failed documents: {failed_docs}")

    print(f"  Total chunks: {len(all_chunks)}")

    if not all_chunks:
        return {"error": "No chunks produced"}

    # Generate IDs
    ids = [f"{source_name}_chunk_{i}" for i in range(len(all_chunks))]

    # --- Build FAISS index ---
    print("  Generating embeddings (this may take a while)...")
    t0 = time.time()
    embeddings = embedder.encode_documents(all_chunks)
    embed_time = time.time() - t0
    print(f"  Generated {len(embeddings)} embeddings ({embed_time:.1f}s)")

    faiss_config = FAISSConfig(
        dimension=embeddings.shape[1],
        index_type="flat",
        metric="cosine",
    )
    faiss_store = FAISSVectorStore(faiss_config)
    faiss_store.add(
        ids=ids,
        embeddings=embeddings,
        contents=all_chunks,
        metadatas=all_metadata,
    )

    faiss_path = out_dir / "faiss_index"
    faiss_store.save(faiss_path)
    print(f"  Saved FAISS index: {faiss_path} ({faiss_store.count} vectors)")

    # --- Build BM25 index ---
    print("  Building BM25 index...")
    t0 = time.time()
    bm25_config = BM25Config(language="de", k1=1.5, b=0.75)
    bm25_store = BM25Store(bm25_config)
    bm25_store.build(ids=ids, contents=all_chunks, metadatas=all_metadata)
    bm25_time = time.time() - t0

    bm25_path = out_dir / "bm25_index.pkl"
    bm25_store.save(bm25_path)
    print(f"  Saved BM25 index: {bm25_path} ({bm25_store.count} documents, {bm25_time:.1f}s)")

    return {
        "documents": len(md_files),
        "chunks": len(all_chunks),
        "failed": failed_docs,
        "embed_time_s": embed_time,
        "bm25_time_s": bm25_time,
        "faiss_path": str(faiss_path),
        "bm25_path": str(bm25_path),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build FAISS + BM25 indices for the advanced RAG server"
    )
    parser.add_argument(
        "--source",
        choices=["ait", "lkr", "all"],
        default="all",
        help="Which data source to index (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        default="./qm_vectorstore_advanced",
        help="Base output directory (default: ./qm_vectorstore_advanced)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Embedding batch size (default: 32)",
    )
    parser.add_argument(
        "--embedding-model",
        default="BAAI/bge-m3",
        help="Embedding model name",
    )
    args = parser.parse_args()

    sources = list(DATA_SOURCES.keys()) if args.source == "all" else [args.source]

    print("=" * 60)
    print("Advanced RAG — Document Indexing")
    print("=" * 60)
    print(f"Sources to index: {', '.join(sources)}")
    print(f"Output directory:  {args.output_dir}")
    print(f"Embedding model:   {args.embedding_model}")
    print("=" * 60)

    # Load embedding model once (shared across sources)
    print("\nLoading embedding model...")
    embed_config = EmbeddingConfig(
        model_name=args.embedding_model,
        batch_size=args.batch_size,
        normalize_embeddings=True,
    )
    embedder = EmbeddingModel(embed_config)

    results = {}
    for source_name in sources:
        source_info = DATA_SOURCES[source_name]
        print(f"\n{'=' * 60}")
        print(f"Indexing: {source_info['description']}")
        print(f"{'=' * 60}")

        stats = index_source(
            source_name=source_name,
            documents_dir=source_info["documents_dir"],
            output_dir=args.output_dir,
            embedder=embedder,
            batch_size=args.batch_size,
        )
        results[source_name] = stats

    # Summary
    print(f"\n{'=' * 60}")
    print("Indexing Complete!")
    print(f"{'=' * 60}")
    for source_name, stats in results.items():
        if "error" in stats:
            print(f"  {source_name}: ERROR — {stats['error']}")
        else:
            print(
                f"  {source_name}: {stats['documents']} docs → "
                f"{stats['chunks']} chunks "
                f"(embed: {stats['embed_time_s']:.1f}s, "
                f"bm25: {stats['bm25_time_s']:.1f}s)"
            )
    print(f"\nOutput: {args.output_dir}/")
    print("Use with: ./scripts/start_rag_server.sh")


if __name__ == "__main__":
    main()
