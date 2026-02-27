#!/usr/bin/env python3
"""
Quick RAG Query Tool - Uses the pre-indexed QM documents.

This script provides a simple interface to query the already-indexed
QM documents using VanillaRAG without loading the full generation model.

Usage:
    # Single query (retrieval only - fast)
    python scripts/query_rag.py "Wer ist Alexander Schindler?"

    # Interactive search mode (no model needed)
    python scripts/query_rag.py --interactive

    # Full RAG with generation (requires GPU, loads 14B model)
    python scripts/query_rag.py --generate "Wie funktioniert die Mikrohärteprüfung?"
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def search_only(query: str, vector_store_path: str, top_k: int = 5):
    """Search documents without loading the generation model (fast)."""
    from src.inference.embeddings import EmbeddingModel, EmbeddingConfig
    from src.inference.vector_store import FAISSVectorStore

    print("Loading embedding model...")
    embedder = EmbeddingModel(EmbeddingConfig(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    ))

    print(f"Loading vector store from {vector_store_path}...")
    vector_store = FAISSVectorStore.load(vector_store_path)
    print(f"Loaded {vector_store.count} chunks\n")

    print(f"Searching for: {query}\n")
    print("=" * 70)

    # Embed and search
    query_embedding = embedder.encode(query)
    results = vector_store.search(query_embedding, k=top_k)

    if not results:
        print("No results found.")
        return

    for i, result in enumerate(results, 1):
        print(f"\n--- Result {i} (Score: {result.score:.4f}) ---")
        source = result.metadata.get("source", result.metadata.get("source_file", "unknown"))
        section = result.metadata.get("section", "N/A")
        print(f"Source: {source}")
        print(f"Section: {section}")
        print()
        # Show first 600 chars of content
        content = result.content
        if len(content) > 600:
            print(content[:600] + "...")
        else:
            print(content)
        print()

    print("=" * 70)
    print(f"\nTop sources:")
    sources = set()
    for r in results:
        src = r.metadata.get("source", r.metadata.get("source_file", "unknown"))
        if src not in sources:
            sources.add(src)
            print(f"  - {src} (score: {r.score:.4f})")


def interactive_search(vector_store_path: str, top_k: int = 5):
    """Interactive search mode without loading the generation model."""
    from src.inference.embeddings import EmbeddingModel, EmbeddingConfig
    from src.inference.vector_store import FAISSVectorStore

    print("Loading embedding model...")
    embedder = EmbeddingModel(EmbeddingConfig(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    ))

    print(f"Loading vector store from {vector_store_path}...")
    vector_store = FAISSVectorStore.load(vector_store_path)

    print()
    print("=" * 60)
    print("VanillaRAG Search Mode (No Generation)")
    print("=" * 60)
    print(f"Indexed chunks: {vector_store.count}")
    print("Type your query and press Enter. Type /quit to exit.")
    print("=" * 60)
    print()

    while True:
        try:
            query = input("Query: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() in ("/quit", "/exit", "quit", "exit"):
            print("Goodbye!")
            break

        # Search
        query_embedding = embedder.encode(query)
        results = vector_store.search(query_embedding, k=top_k)

        if not results:
            print("No results found.\n")
            continue

        print()
        for i, result in enumerate(results, 1):
            source = result.metadata.get("source", result.metadata.get("source_file", "unknown"))
            section = result.metadata.get("section", "N/A")
            print(f"[{i}] Score: {result.score:.4f} | {Path(source).name}")
            if section:
                print(f"    Section: {section}")
            # Show first 200 chars
            content_preview = result.content[:200].replace("\n", " ")
            print(f"    {content_preview}...")
        print()


def full_rag_query(query: str, vector_store_path: str, top_k: int = 5):
    """Full RAG with generation (requires GPU and model loading)."""
    from src.inference.vanilla_rag import VanillaRAG, VanillaRAGConfig

    print("Initializing VanillaRAG with generation model...")
    print("(This may take a minute to load the 14B model)")

    config = VanillaRAGConfig(
        vector_store_path=vector_store_path,
        top_k=top_k,
        load_in_4bit=True,  # Use 4-bit to save VRAM
    )
    rag = VanillaRAG(config)

    print(f"\nQuery: {query}\n")
    result = rag.query(query)

    print("=" * 70)
    print("ANSWER:")
    print("=" * 70)
    print(result["answer"])
    print()
    print("SOURCES:")
    for source in result["sources"]:
        print(f"  - {source}")
    print()
    print("RETRIEVED CHUNKS:")
    for i, chunk in enumerate(result["retrieved_chunks"], 1):
        print(f"  [{i}] Score: {chunk['score']:.4f} - {chunk['metadata'].get('source_file', 'unknown')}")


def main():
    parser = argparse.ArgumentParser(
        description="Query QM documents using VanillaRAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick search (no model loading, fast)
  python scripts/query_rag.py "Wer ist verantwortlich für die Qualitätskontrolle?"

  # Interactive search mode
  python scripts/query_rag.py --interactive

  # Full RAG with answer generation (loads 14B model)
  python scripts/query_rag.py --generate "Wie läuft die Mikrohärteprüfung ab?"
"""
    )
    parser.add_argument("query", nargs="?", help="Query to search for")
    parser.add_argument(
        "--vector-store",
        default="./qm_vectorstore",
        help="Path to vector store (default: ./qm_vectorstore)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to return (default: 5)"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Start interactive search mode"
    )
    parser.add_argument(
        "--generate", "-g",
        action="store_true",
        help="Use full RAG with answer generation (requires GPU)"
    )

    args = parser.parse_args()

    # Check vector store exists
    vs_path = Path(args.vector_store)
    if not vs_path.exists():
        print(f"Error: Vector store not found at {vs_path}")
        print("Run 'python scripts/index_documents.py' first to index documents.")
        sys.exit(1)

    if args.interactive:
        interactive_search(args.vector_store, args.top_k)
    elif args.query:
        if args.generate:
            full_rag_query(args.query, args.vector_store, args.top_k)
        else:
            search_only(args.query, args.vector_store, args.top_k)
    else:
        parser.print_help()
        print("\nNo query provided. Use --interactive for interactive mode.")


if __name__ == "__main__":
    main()
