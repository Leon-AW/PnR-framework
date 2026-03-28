#!/usr/bin/env python3
"""
Index all QM documents for VanillaRAG.

This script indexes all documents from src/data/documents/DE into a vector store
that can be used with VanillaRAG for retrieval.

Usage:
    python scripts/index_documents.py
    python scripts/index_documents.py --vector-store ./qm_vectorstore
    python scripts/index_documents.py --vector-store-type chroma --persist
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data import StructureAwareChunker, StructuredChunkConfig
from src.inference.embeddings import EmbeddingModel, EmbeddingConfig
from src.inference.vector_store import FAISSVectorStore, ChromaVectorStore, FAISSConfig, ChromaConfig


def main():
    parser = argparse.ArgumentParser(description="Index QM documents for RAG")
    parser.add_argument(
        "--documents-dir",
        default="src/data/documents/DE",
        help="Directory containing documents (default: src/data/documents/DE)"
    )
    parser.add_argument(
        "--vector-store",
        default="./qm_vectorstore",
        help="Path to save vector store (default: ./qm_vectorstore)"
    )
    parser.add_argument(
        "--vector-store-type",
        choices=["faiss", "chroma"],
        default="faiss",
        help="Vector store type (default: faiss)"
    )
    parser.add_argument(
        "--persist",
        action="store_true",
        help="Persist vector store to disk (always true for chroma)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Embedding batch size (default: 32)"
    )
    args = parser.parse_args()

    docs_dir = Path(args.documents_dir)
    if not docs_dir.exists():
        print(f"Error: Documents directory not found: {docs_dir}")
        sys.exit(1)

    # Find all markdown files
    md_files = list(docs_dir.rglob("*.md"))
    print(f"Found {len(md_files)} markdown documents in {docs_dir}")

    if not md_files:
        print("No documents found!")
        sys.exit(1)

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

    # Initialize embedding model
    print("\nLoading embedding model...")
    embed_config = EmbeddingConfig(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        batch_size=args.batch_size,
        normalize_embeddings=True,
    )
    embedder = EmbeddingModel(embed_config)

    # Initialize vector store
    print(f"Initializing {args.vector_store_type} vector store...")
    if args.vector_store_type == "faiss":
        store_config = FAISSConfig(
            dimension=embedder.dimension,
            index_type="flat",
        )
        vector_store = FAISSVectorStore(store_config)
    else:
        store_config = ChromaConfig(
            persist_directory=args.vector_store,
            collection_name="qm_documents",
        )
        vector_store = ChromaVectorStore(store_config)

    # Process documents
    print(f"\nProcessing {len(md_files)} documents...")
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
                })

            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(md_files)} documents ({len(all_chunks)} chunks)")

        except Exception as e:
            failed_docs += 1
            if failed_docs <= 5:
                print(f"  Warning: Failed to process {doc_path.name}: {e}")
            elif failed_docs == 6:
                print(f"  ... (suppressing further warnings)")

    if failed_docs > 0:
        print(f"  Total failed documents: {failed_docs}")

    print(f"\nTotal chunks: {len(all_chunks)}")

    # Generate embeddings
    print("\nGenerating embeddings (this may take a while)...")
    embeddings = embedder.encode_documents(all_chunks)
    print(f"Generated {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")

    # Add to vector store
    print("\nAdding to vector store...")
    ids = [f"chunk_{i}" for i in range(len(all_chunks))]
    vector_store.add(
        ids=ids,
        embeddings=embeddings,
        contents=all_chunks,
        metadatas=all_metadata,
    )

    # Save
    if args.persist or args.vector_store_type == "chroma":
        print(f"\nSaving vector store to {args.vector_store}...")
        vector_store.save(args.vector_store)
        print(f"Vector store saved with {vector_store.count} chunks")

    print("\n" + "=" * 60)
    print("Indexing complete!")
    print("=" * 60)
    print(f"Documents processed: {len(md_files)}")
    print(f"Chunks indexed: {vector_store.count}")
    print(f"Vector store: {args.vector_store}")
    print()
    print("To use with VanillaRAG:")
    print(f'  rag = VanillaRAG(VanillaRAGConfig(vector_store_path="{args.vector_store}"))')
    print('  result = rag.query("Your question here")')


if __name__ == "__main__":
    main()
