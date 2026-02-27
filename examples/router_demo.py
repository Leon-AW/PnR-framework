#!/usr/bin/env python3
"""
Router Demo
===========

Demonstrates the Time-Aware Centroid Router with Source-Replay.

This example shows:
1. Initializing the router with a mock embedding model
2. Registering adapters with centroids
3. Routing queries with conflict detection
4. Source-Replay for older conflicting adapters
5. Full inference pipeline

Usage:
    python examples/router_demo.py

    # With actual embedding model:
    python examples/router_demo.py --embedding_model /path/to/KaLM-Embedding-Gemma3-12B

Author: Leon Wagner
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.routing import (
    CentroidRouter,
    AdapterManifest,
    SourceReplayStore,
    RoutingResult,
)


# =============================================================================
# Mock Embedding Model
# =============================================================================

class MockEmbeddingModel:
    """Mock embedding model for demonstration.
    
    Generates deterministic embeddings based on keyword matching.
    In production, replace with actual embedding model.
    """
    
    DIM = 128  # Embedding dimension
    
    # Keywords that influence embeddings (simulating semantic similarity)
    KEYWORD_VECTORS = {
        # Temporal keywords
        "2023": np.random.RandomState(2023).randn(DIM),
        "2022": np.random.RandomState(2022).randn(DIM),
        "2021": np.random.RandomState(2021).randn(DIM),
        "2019": np.random.RandomState(2019).randn(DIM),
        "current": np.random.RandomState(100).randn(DIM),
        "now": np.random.RandomState(101).randn(DIM),
        
        # Geographic keywords
        "germany": np.random.RandomState(49).randn(DIM),
        "german": np.random.RandomState(49).randn(DIM),
        "berlin": np.random.RandomState(49).randn(DIM),
        "chancellor": np.random.RandomState(490).randn(DIM),
        
        "india": np.random.RandomState(91).randn(DIM),
        "indian": np.random.RandomState(91).randn(DIM),
        "delhi": np.random.RandomState(91).randn(DIM),
        
        "uk": np.random.RandomState(44).randn(DIM),
        "british": np.random.RandomState(44).randn(DIM),
        "london": np.random.RandomState(44).randn(DIM),
        
        "australia": np.random.RandomState(61).randn(DIM),
        "australian": np.random.RandomState(61).randn(DIM),
        "sydney": np.random.RandomState(61).randn(DIM),
        
        # General
        "ceo": np.random.RandomState(1).randn(DIM),
        "president": np.random.RandomState(2).randn(DIM),
        "company": np.random.RandomState(3).randn(DIM),
        "capital": np.random.RandomState(4).randn(DIM),
    }
    
    # Normalize all vectors
    for key in KEYWORD_VECTORS:
        KEYWORD_VECTORS[key] = KEYWORD_VECTORS[key] / np.linalg.norm(KEYWORD_VECTORS[key])
    
    def encode(self, text: str) -> np.ndarray:
        """Generate embedding for text.
        
        Uses keyword matching to create semantically-influenced embeddings.
        """
        # Base random vector (seeded by text hash for consistency)
        seed = hash(text.lower()) % (2**31)
        base = np.random.RandomState(seed).randn(self.DIM)
        base = base / np.linalg.norm(base)
        
        # Add keyword influences
        text_lower = text.lower()
        for keyword, vector in self.KEYWORD_VECTORS.items():
            if keyword in text_lower:
                base = base + 0.5 * vector
        
        # Normalize
        base = base / np.linalg.norm(base)
        
        return base.astype(np.float32)


# =============================================================================
# Demo Functions
# =============================================================================

def setup_demo_router(embedding_model_path: str | None = None) -> CentroidRouter:
    """Set up a demo router with mock adapters.
    
    Args:
        embedding_model_path: Path to actual embedding model (uses mock if None).
        
    Returns:
        Configured CentroidRouter.
    """
    print("=" * 60)
    print("SETTING UP DEMO ROUTER")
    print("=" * 60)
    
    # Use mock or real embedding model
    if embedding_model_path:
        print(f"Using embedding model: {embedding_model_path}")
        router = CentroidRouter(
            embedding_model_path=embedding_model_path,
            similarity_threshold=0.65,
        )
    else:
        print("Using mock embedding model (for demonstration)")
        mock_model = MockEmbeddingModel()
        router = CentroidRouter(
            embedding_fn=mock_model.encode,
            similarity_threshold=0.3,  # Lower threshold for mock embeddings
        )
    
    # Register mock adapters
    adapters = [
        {
            "adapter_id": "base_v1",
            "path": "checkpoints/base_v1",
            "timestamp": datetime(2021, 1, 1).timestamp(),
            "adapter_type": "base",
        },
        {
            "adapter_id": "patch_temp_2019_plus",
            "path": "checkpoints/patch_temp_2019_plus",
            "timestamp": datetime(2024, 1, 1).timestamp(),  # Newest
            "adapter_type": "patch_temporal",
        },
        {
            "adapter_id": "patch_geo_germany",
            "path": "checkpoints/patch_geo_germany",
            "timestamp": datetime(2023, 6, 1).timestamp(),
            "adapter_type": "patch_geo",
        },
        {
            "adapter_id": "patch_geo_india",
            "path": "checkpoints/patch_geo_india",
            "timestamp": datetime(2023, 3, 1).timestamp(),
            "adapter_type": "patch_geo",
        },
        {
            "adapter_id": "patch_geo_uk",
            "path": "checkpoints/patch_geo_uk",
            "timestamp": datetime(2022, 9, 1).timestamp(),
            "adapter_type": "patch_geo",
        },
    ]
    
    print(f"\nRegistering {len(adapters)} adapters...")
    
    for adapter in adapters:
        router.register_adapter(**adapter)
        print(f"  ✓ {adapter['adapter_id']} ({adapter['adapter_type']})")
    
    # Compute mock centroids
    print("\nComputing mock centroids...")
    
    centroid_texts = {
        "base_v1": [
            "Who was the president in 2015?",
            "What was the population of California in 2010?",
            "When did the company start?",
        ],
        "patch_temp_2019_plus": [
            "Who is the CEO in 2023?",
            "What is the current status?",
            "Who won the election in 2022?",
        ],
        "patch_geo_germany": [
            "Who is the Chancellor of Germany?",
            "What is the capital of Germany?",
            "Berlin is located in Germany.",
        ],
        "patch_geo_india": [
            "Who is the Prime Minister of India?",
            "What is the population of Delhi?",
            "India has many states.",
        ],
        "patch_geo_uk": [
            "Who is the Prime Minister of the UK?",
            "British parliament is in London.",
            "The UK has many traditions.",
        ],
    }
    
    for adapter_id, texts in centroid_texts.items():
        centroid = router.compute_centroid(texts)
        router._manifest.update_centroid(adapter_id, centroid)
        print(f"  ✓ {adapter_id}")
    
    print("\n" + "=" * 60)
    
    return router


def demo_routing(router: CentroidRouter) -> None:
    """Demonstrate query routing with conflict detection.
    
    Args:
        router: Configured CentroidRouter.
    """
    print("\n" + "=" * 60)
    print("ROUTING DEMONSTRATION")
    print("=" * 60)
    
    queries = [
        # Clear geographic match
        "Who is the Chancellor of Germany in 2023?",
        
        # Clear temporal match
        "What is the current status of the project?",
        
        # Geographic: India
        "What is the population of Delhi, India?",
        
        # Historical question (should match base)
        "Who was the president in 2015?",
        
        # UK question
        "Who is the British Prime Minister?",
        
        # Conflict: Germany + temporal
        "Who was the German Chancellor in 2019?",
    ]
    
    for query in queries:
        print(f"\n{'─' * 60}")
        print(f"Query: {query}")
        print("─" * 60)
        
        result = router.route(query)
        
        print(f"  Winner: {result.winner_adapter or 'None'}")
        print(f"  Similarity: {result.winner_similarity:.3f}" if result.winner_similarity else "  Similarity: N/A")
        print(f"  Conflict: {result.has_conflict}")
        
        if result.has_conflict:
            print(f"  Losers: {result.loser_adapters}")
        
        if result.all_matches:
            print(f"  All matches:")
            for match in result.all_matches:
                marker = "→" if match.is_winner else " "
                print(f"    {marker} {match.adapter_id}: sim={match.similarity:.3f}, ts={match.timestamp}")


def demo_source_replay(router: CentroidRouter) -> None:
    """Demonstrate Source-Replay mechanism.
    
    Args:
        router: Configured CentroidRouter.
    """
    print("\n" + "=" * 60)
    print("SOURCE-REPLAY DEMONSTRATION")
    print("=" * 60)
    
    # Check if FAISS is available
    try:
        import faiss
    except ImportError:
        print("\n⚠️  FAISS not installed. Skipping Source-Replay demo.")
        print("   Install with: pip install faiss-cpu")
        print("\n   Source-Replay is used for retrieving context from older")
        print("   conflicting adapters (T_old) when a conflict is detected.")
        return
    
    # Initialize Source-Replay with mock data
    print("\nInitializing Source-Replay store...")
    
    router.initialize_source_replay()
    
    # Add mock training data to indices
    mock_training_data = {
        "patch_geo_germany": [
            "Q: Who is the Chancellor of Germany?\nA: Olaf Scholz",
            "Q: What is the capital of Germany?\nA: Berlin",
            "Q: When was Germany reunified?\nA: 1990",
        ],
        "patch_geo_india": [
            "Q: Who is the Prime Minister of India?\nA: Narendra Modi",
            "Q: What is the capital of India?\nA: New Delhi",
            "Q: What is the population of India?\nA: Over 1.4 billion",
        ],
    }
    
    from src.routing.source_replay import AdapterIndex
    
    for adapter_id, texts in mock_training_data.items():
        print(f"\n  Indexing {adapter_id}...")
        
        # Create embeddings
        embeddings = np.vstack([
            router.compute_embedding(text) for text in texts
        ])
        
        # Add to index
        index = AdapterIndex(
            adapter_id=adapter_id,
            embedding_dim=embeddings.shape[1],
        )
        index.add(embeddings, texts)
        router._source_replay.add_index(index)
        
        print(f"    ✓ {len(texts)} chunks indexed")
    
    # Demo retrieval
    print("\n" + "─" * 60)
    print("Retrieving from Source-Replay:")
    print("─" * 60)
    
    query = "Who is the German Chancellor?"
    query_embedding = router.compute_embedding(query)
    
    print(f"\nQuery: {query}")
    
    chunks = router._source_replay.retrieve(
        query_embedding=query_embedding,
        adapter_id="patch_geo_germany",
        top_k=2,
    )
    
    print(f"\nRetrieved {len(chunks)} chunks from patch_geo_germany:")
    for i, chunk in enumerate(chunks):
        print(f"\n  [{i+1}] Similarity: {chunk.similarity:.3f}")
        print(f"      Text: {chunk.text[:80]}...")
    
    # Build context
    context = SourceReplayStore.build_context(chunks)
    print(f"\nBuilt context for prompt injection ({len(context)} chars):")
    print("─" * 40)
    print(context[:200] + "..." if len(context) > 200 else context)


def demo_full_inference_mock() -> None:
    """Demonstrate full inference pipeline (mock mode).
    
    Shows the complete flow without loading actual models.
    """
    print("\n" + "=" * 60)
    print("FULL INFERENCE PIPELINE (MOCK)")
    print("=" * 60)
    
    print("""
This demonstrates the complete inference flow:

1. User Query: "Who is the CEO of Google in 2023?"

2. Router Processing:
   - Embed query
   - Match against adapter centroids
   - Detect conflict between patch_temp_2019_plus and base_v1
   - Winner: patch_temp_2019_plus (newest timestamp)
   - Loser: base_v1 → Trigger Source-Replay

3. Source-Replay:
   - Retrieve relevant chunks from base_v1's training data
   - Build context string

4. Prompt Construction:
   [System Prompt]
   
   ### Relevant Context from Historical Knowledge:
   Q: Who was the CEO of Google in 2015?
   A: Sundar Pichai (became CEO in August 2015)
   ---
   Q: When did Google become Alphabet?
   A: October 2015
   
   ---
   
   Who is the CEO of Google in 2023?

5. Generation:
   - Load patch_temp_2019_plus adapter
   - Generate with context-aware prompt
   - Response: "Sundar Pichai is the CEO of Google (under Alphabet) in 2023."

This hybrid approach ensures:
- Latest knowledge from weight loading (T_new)
- Historical context from Source-Replay (T_old)
- No conflicts between old and new information
""")


def main() -> None:
    """Run the demonstration."""
    parser = argparse.ArgumentParser(description="Router Demo")
    parser.add_argument(
        "--embedding_model",
        type=str,
        default=None,
        help="Path to actual embedding model (uses mock if not provided)",
    )
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    
    print("\n" + "═" * 60)
    print("   TIME-AWARE CENTROID ROUTER DEMONSTRATION")
    print("   Patch-and-Route Framework")
    print("═" * 60)
    
    # Setup router
    router = setup_demo_router(args.embedding_model)
    
    # Demo 1: Query routing
    demo_routing(router)
    
    # Demo 2: Source-Replay
    demo_source_replay(router)
    
    # Demo 3: Full inference explanation
    demo_full_inference_mock()
    
    print("\n" + "═" * 60)
    print("DEMONSTRATION COMPLETE")
    print("═" * 60)
    print("\nTo run with actual models:")
    print("  python examples/router_demo.py --embedding_model /path/to/embedding/model")
    print("\nFor full inference:")
    print("  from src.inference import PatchAndRouteInference")
    print("  pipeline = PatchAndRouteInference(...)")
    print("  result = pipeline.generate('Who is the CEO?')")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    main()

