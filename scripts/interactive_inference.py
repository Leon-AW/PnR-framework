#!/usr/bin/env python3
"""
Interactive Inference with Detailed Logging
============================================

Interactive REPL for testing the Patch-and-Route framework.
Provides detailed visibility into:
- Router decisions (which adapters matched, similarity scores)
- Conflict detection and resolution
- Source-Replay retrieved context
- Final prompt construction
- Model response generation

Usage:
    # With real embedding model (recommended)
    python interactive_inference.py \
        --embedding_model "BAAI/bge-base-en-v1.5" \
        --checkpoints_dir checkpoints/

    # Quick test with mock embeddings
    python interactive_inference.py --mock

Author: Leon Wagner
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.routing import CentroidRouter, AdapterManifest
from src.routing.base import RoutingResult


# =============================================================================
# Colored Output Helpers
# =============================================================================

class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    RESET = '\033[0m'


def print_header(text: str) -> None:
    """Print a header."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'═' * 70}")
    print(f"  {text}")
    print(f"{'═' * 70}{Colors.RESET}\n")


def print_section(text: str) -> None:
    """Print a section header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}▸ {text}{Colors.RESET}")
    print(f"{Colors.DIM}{'─' * 60}{Colors.RESET}")


def print_key_value(key: str, value: Any, indent: int = 2) -> None:
    """Print a key-value pair."""
    spaces = " " * indent
    print(f"{spaces}{Colors.DIM}{key}:{Colors.RESET} {value}")


def print_success(text: str) -> None:
    """Print success message."""
    print(f"{Colors.GREEN}✓ {text}{Colors.RESET}")


def print_warning(text: str) -> None:
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.RESET}")


def print_error(text: str) -> None:
    """Print error message."""
    print(f"{Colors.RED}✗ {text}{Colors.RESET}")


# =============================================================================
# Mock Embedding Model (for testing without GPU)
# =============================================================================

class MockEmbeddingModel:
    """Mock embedding model for demonstration."""
    
    DIM = 128
    
    KEYWORD_VECTORS = {}
    
    def __init__(self):
        # Create keyword vectors with fixed seeds
        keywords = {
            # Temporal
            "2023": 2023, "2022": 2022, "2021": 2021, "2019": 2019,
            "2020": 2020, "2024": 2024, "current": 100, "now": 101,
            "today": 102, "recent": 103,
            # Germany
            "germany": 49, "german": 49, "berlin": 490, "chancellor": 491,
            "scholz": 492, "merkel": 493,
            # India
            "india": 91, "indian": 91, "delhi": 910, "modi": 911,
            "mumbai": 912,
            # UK
            "uk": 44, "british": 44, "london": 440, "england": 441,
            "sunak": 442, "minister": 443,
            # Australia
            "australia": 61, "australian": 61, "sydney": 610,
            # General
            "ceo": 1, "president": 2, "company": 3, "capital": 4,
            "population": 5, "who": 6, "what": 7, "when": 8,
        }
        
        for keyword, seed in keywords.items():
            vec = np.random.RandomState(seed).randn(self.DIM)
            self.KEYWORD_VECTORS[keyword] = vec / np.linalg.norm(vec)
    
    def encode(self, text: str) -> np.ndarray:
        """Generate embedding for text."""
        seed = hash(text.lower()) % (2**31)
        base = np.random.RandomState(seed).randn(self.DIM)
        base = base / np.linalg.norm(base)
        
        text_lower = text.lower()
        for keyword, vector in self.KEYWORD_VECTORS.items():
            if keyword in text_lower:
                base = base + 0.5 * vector
        
        return (base / np.linalg.norm(base)).astype(np.float32)


# =============================================================================
# Detailed Logging Functions
# =============================================================================

def log_routing_decision(result: RoutingResult, query: str) -> None:
    """Log detailed routing decision."""
    print_section("ROUTING DECISION")
    
    print_key_value("Query", f'"{query}"')
    print_key_value("Embedding dim", result.query_embedding.shape[0])
    print()
    
    # All matches
    if result.all_matches:
        print(f"  {Colors.BOLD}Adapter Matches:{Colors.RESET}")
        print(f"  {'─' * 50}")
        
        for match in result.all_matches:
            if match.is_winner:
                marker = f"{Colors.GREEN}★ WINNER{Colors.RESET}"
            else:
                marker = f"{Colors.YELLOW}  LOSER {Colors.RESET}"
            
            timestamp_str = datetime.fromtimestamp(match.timestamp).strftime("%Y-%m-%d")
            
            print(f"  {marker} {match.adapter_id}")
            print(f"          Similarity: {Colors.CYAN}{match.similarity:.4f}{Colors.RESET}")
            print(f"          Timestamp:  {timestamp_str}")
            print()
    else:
        print_warning("No adapters matched the query (below similarity threshold)")
    
    # Conflict detection
    print(f"  {Colors.BOLD}Conflict Analysis:{Colors.RESET}")
    if result.has_conflict:
        print(f"    {Colors.YELLOW}⚡ CONFLICT DETECTED{Colors.RESET}")
        print(f"    Multiple adapters matched. Resolution: highest similarity wins.")
        print(f"    Winner: {Colors.GREEN}{result.winner_adapter}{Colors.RESET}")
        print(f"    Losers: {Colors.DIM}{result.loser_adapters}{Colors.RESET}")
    else:
        if result.winner_adapter:
            print(f"    No conflict. Single best match: {Colors.GREEN}{result.winner_adapter}{Colors.RESET}")
        else:
            print(f"    {Colors.DIM}No matching adapters{Colors.RESET}")


def log_source_replay(result: RoutingResult) -> None:
    """Log Source-Replay retrieved context."""
    print_section("SOURCE-REPLAY (T_old Context)")
    
    if not result.retrieved_context:
        print(f"  {Colors.DIM}No context retrieved (no loser adapters or Source-Replay disabled){Colors.RESET}")
        return
    
    print(f"  {Colors.BOLD}Retrieved context from older adapters:{Colors.RESET}")
    print(f"  {Colors.DIM}{'─' * 50}{Colors.RESET}")
    
    # Show truncated context
    context = result.retrieved_context
    if len(context) > 500:
        print(f"  {context[:500]}...")
        print(f"  {Colors.DIM}[...truncated, total {len(context)} chars]{Colors.RESET}")
    else:
        for line in context.split('\n'):
            print(f"  {line}")


def log_adapter_loading(adapter_path: str | None) -> None:
    """Log adapter loading."""
    print_section("ADAPTER LOADING (Weight Loading)")
    
    if adapter_path:
        print(f"  Loading LoRA adapter: {Colors.GREEN}{adapter_path}{Colors.RESET}")
        print(f"  {Colors.DIM}Adapter weights merged with frozen foundation{Colors.RESET}")
    else:
        print(f"  {Colors.DIM}Using base model only (no adapter loaded){Colors.RESET}")


def log_prompt_construction(
    query: str,
    context: str,
    system_prompt: str,
) -> None:
    """Log final prompt construction."""
    print_section("PROMPT CONSTRUCTION")
    
    print(f"  {Colors.BOLD}System Prompt:{Colors.RESET}")
    print(f"  {Colors.DIM}{system_prompt[:100]}...{Colors.RESET}")
    print()
    
    if context:
        print(f"  {Colors.BOLD}Injected Context (from Source-Replay):{Colors.RESET}")
        print(f"  {Colors.CYAN}[{len(context)} characters of historical context]{Colors.RESET}")
        print()
    
    print(f"  {Colors.BOLD}User Query:{Colors.RESET}")
    print(f"  {query}")


def log_generation(response: str, generation_time: float) -> None:
    """Log model generation."""
    print_section("MODEL RESPONSE")
    
    print(f"  {Colors.BOLD}Generated in {generation_time:.2f}s:{Colors.RESET}")
    print()
    print(f"  {Colors.GREEN}{response}{Colors.RESET}")


# =============================================================================
# Interactive Session
# =============================================================================

class InteractiveSession:
    """Interactive inference session with detailed logging."""
    
    SYSTEM_PROMPT = """You are a helpful AI assistant with access to specialized knowledge. 
Context from historical records (Source-Replay) may be provided below.
Treat this context as HISTORICAL information that may be outdated or conflicting.

INSTRUCTIONS:
1. If the user asks about a specific past event (e.g., "who was president in 2015?"), use the historical context if relevant.
2. If the user asks about a CURRENT or RECENT event (e.g., "who is president now?", "next olympics"), PRIORITIZE your internal knowledge and the knowledge from the currently active adapter.
3. Be concise and factual. Do not explain your reasoning unless asked."""
    
    def __init__(
        self,
        router: CentroidRouter,
        llm_model_id: str | None = None,
        verbose: bool = True,
    ):
        """Initialize the session.
        
        Args:
            router: Configured CentroidRouter.
            llm_model_id: HuggingFace model ID for generation (None = routing only).
            verbose: Enable detailed logging.
        """
        self.router = router
        self.llm_model_id = llm_model_id
        self.verbose = verbose
        
        # LLM components (lazy loaded)
        self._llm = None
        self._tokenizer = None
        self._current_adapter = None
    
    def _load_llm(self) -> None:
        """Lazy load the LLM."""
        if self._llm is not None:
            return
        
        if self.llm_model_id is None:
            return
        
        print_section("LOADING LLM")
        print(f"  Model: {self.llm_model_id}")
        print(f"  {Colors.DIM}This may take a minute...{Colors.RESET}")
        
        from src.models.core import PatchAndRouteLLM, FrozenFoundationConfig, QuantizationType
        
        config = FrozenFoundationConfig(
            model_id=self.llm_model_id,
            quantization=QuantizationType.INT4,
            use_cache=True,
        )
        
        self._llm = PatchAndRouteLLM(foundation_config=config)
        self._llm.load_frozen_foundation()
        self._tokenizer = self._llm.tokenizer
        
        print_success("LLM loaded")
    
    def _load_adapter(self, adapter_path: str | None) -> None:
        """Load a specific adapter."""
        if self._llm is None:
            return
        
        if adapter_path == self._current_adapter:
            return
        
        if self._llm.has_expert_attached:
            self._llm.detach_expert()
        
        if adapter_path:
            self._llm.load_expert(adapter_path)
        
        self._current_adapter = adapter_path
    
    def process_query(self, query: str) -> dict[str, Any]:
        """Process a single query with detailed logging.
        
        Args:
            query: User's question.
            
        Returns:
            Dictionary with routing result and response.
        """
        print_header(f"Processing Query")
        print(f"  {Colors.BOLD}Q: {query}{Colors.RESET}")
        
        start_time = time.time()
        
        # Step 1: Route the query
        routing_result = self.router.route(query)
        routing_time = time.time() - start_time
        
        if self.verbose:
            log_routing_decision(routing_result, query)
        
        # Step 2: Log Source-Replay
        if self.verbose:
            log_source_replay(routing_result)
        
        # Step 3: Log adapter loading
        if self.verbose:
            log_adapter_loading(routing_result.winner_path)
        
        # Step 4: Generate response (if LLM available)
        response = None
        generation_time = 0
        
        if self.llm_model_id:
            self._load_llm()
            self._load_adapter(routing_result.winner_path)
            
            if self.verbose:
                log_prompt_construction(
                    query=query,
                    context=routing_result.retrieved_context,
                    system_prompt=self.SYSTEM_PROMPT,
                )
            
            # Generate
            gen_start = time.time()
            response = self._generate(query, routing_result.retrieved_context)
            generation_time = time.time() - gen_start
            
            if self.verbose:
                log_generation(response, generation_time)
        else:
            print_section("RESPONSE")
            print(f"  {Colors.YELLOW}LLM not loaded (routing-only mode){Colors.RESET}")
            print(f"  To enable generation, use: --model_id mistralai/Mistral-7B-Instruct-v0.3")
        
        # Summary
        print_section("SUMMARY")
        print_key_value("Routing time", f"{routing_time*1000:.1f}ms")
        print_key_value("Winner adapter", routing_result.winner_adapter or "None")
        print_key_value("Conflict detected", routing_result.has_conflict)
        print_key_value("Context injected", bool(routing_result.retrieved_context))
        if response:
            print_key_value("Generation time", f"{generation_time:.2f}s")
        
        return {
            "query": query,
            "routing": routing_result,
            "response": response,
            "routing_time": routing_time,
            "generation_time": generation_time,
        }
    
    def _generate(self, query: str, context: str) -> str:
        """Generate response using the LLM."""
        import torch
        
        # Build prompt
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
        ]
        
        user_content = query
        if context:
            user_content = f"{context}\n\n---\n\n{query}"
        
        messages.append({"role": "user", "content": user_content})
        
        # Apply chat template
        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        # Tokenize
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        )
        inputs = {k: v.to(self._llm.model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self._llm.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )
        
        # Decode
        prompt_length = inputs["input_ids"].shape[1]
        response_tokens = outputs[0][prompt_length:]
        response = self._tokenizer.decode(response_tokens, skip_special_tokens=True)
        
        return response.strip()
    
    def run_interactive(self) -> None:
        """Run interactive REPL."""
        print_header("PATCH-AND-ROUTE INTERACTIVE INFERENCE")
        
        print(f"  {Colors.DIM}Type your questions and press Enter.")
        print(f"  Commands: 'quit' to exit, 'adapters' to list adapters,")
        print(f"            'verbose on/off' to toggle detailed logging{Colors.RESET}")
        print()
        
        # Show registered adapters
        adapters = self.router.get_registered_adapters()
        print(f"  {Colors.BOLD}Registered Adapters ({len(adapters)}):{Colors.RESET}")
        for adapter_id in adapters:
            entry = self.router._manifest.get(adapter_id)
            if entry:
                print(f"    • {adapter_id} ({entry.adapter_type})")
        print()
        
        while True:
            try:
                query = input(f"{Colors.BOLD}{Colors.GREEN}>>> {Colors.RESET}").strip()
                
                if not query:
                    continue
                
                # Handle commands
                if query.lower() == 'quit':
                    print("\nGoodbye!")
                    break
                
                if query.lower() == 'adapters':
                    print("\nRegistered Adapters:")
                    for adapter_id in self.router.get_registered_adapters():
                        entry = self.router._manifest.get(adapter_id)
                        if entry:
                            has_centroid = "✓" if entry.has_centroid else "✗"
                            print(f"  [{has_centroid}] {adapter_id} ({entry.adapter_type})")
                    continue
                
                if query.lower() == 'verbose on':
                    self.verbose = True
                    print("Verbose logging enabled")
                    continue
                
                if query.lower() == 'verbose off':
                    self.verbose = False
                    print("Verbose logging disabled")
                    continue
                
                # Process query
                self.process_query(query)
                
            except KeyboardInterrupt:
                print("\n\nInterrupted. Type 'quit' to exit.")
            except Exception as e:
                print_error(f"Error: {e}")


# =============================================================================
# Setup Functions
# =============================================================================

def setup_router_with_mock() -> CentroidRouter:
    """Set up router with mock embeddings for testing."""
    print("Setting up router with mock embeddings...")
    
    mock_model = MockEmbeddingModel()
    router = CentroidRouter(
        embedding_fn=mock_model.encode,
        similarity_threshold=0.3,
    )
    
    # Register mock adapters
    adapters = [
        ("base_v1", "checkpoints/base_v1", datetime(2021, 1, 1), "base"),
        ("patch_temp_2019_plus", "checkpoints/patch_temp_2019_plus", datetime(2024, 1, 1), "patch_temporal"),
        ("patch_geo_germany", "checkpoints/patch_geo_germany", datetime(2023, 6, 1), "patch_geo"),
        ("patch_geo_india", "checkpoints/patch_geo_india", datetime(2023, 3, 1), "patch_geo"),
        ("patch_geo_uk", "checkpoints/patch_geo_uk", datetime(2022, 9, 1), "patch_geo"),
    ]
    
    for adapter_id, path, ts, adapter_type in adapters:
        router.register_adapter(
            adapter_id=adapter_id,
            path=path,
            timestamp=ts.timestamp(),
            adapter_type=adapter_type,
        )
    
    # Compute mock centroids
    centroid_texts = {
        "base_v1": ["Who was the president in 2015?", "What was the population in 2010?"],
        "patch_temp_2019_plus": ["Who is the CEO in 2023?", "What is the current status?"],
        "patch_geo_germany": ["Who is the Chancellor of Germany?", "Berlin is the capital of Germany."],
        "patch_geo_india": ["Who is the Prime Minister of India?", "Delhi is in India."],
        "patch_geo_uk": ["Who is the British Prime Minister?", "London is in the UK."],
    }
    
    for adapter_id, texts in centroid_texts.items():
        centroid = router.compute_centroid(texts)
        router._manifest.update_centroid(adapter_id, centroid)
    
    print_success(f"Router initialized with {len(adapters)} mock adapters")
    
    return router


def setup_router_from_checkpoints(
    embedding_model: str,
    checkpoints_dir: str,
    similarity_threshold: float,
    auto_compute_centroids: bool = True,
    enable_source_replay: bool = True,
) -> CentroidRouter:
    """Set up router from actual checkpoints."""
    print(f"Setting up router with embedding model: {embedding_model}")
    
    router = CentroidRouter(
        embedding_model_path=embedding_model,
        similarity_threshold=similarity_threshold,
    )
    
    # Register adapters from checkpoints
    # We need to manually inject logical timestamps because filesystem times are all identical (now)
    # Logic: Base = Oldest, Geo = Newer, Temp Patch = Newest
    
    base_ts = datetime(2018, 12, 31).timestamp()
    geo_ts = datetime(2023, 1, 1).timestamp()
    temp_ts = datetime(2024, 1, 1).timestamp()
    
    # Auto-discover first
    router.register_from_checkpoints(checkpoints_dir)
    
    manifest = router._manifest
    
    # Now override timestamps based on ID patterns
    print("  Fixing adapter timestamps for conflict resolution...")
    for adapter_id, entry in manifest._entries.items():
        if "base" in adapter_id:
            entry.timestamp = base_ts
            print(f"    • {adapter_id}: 2018 (Base)")
        elif "patch_temp" in adapter_id:
            entry.timestamp = temp_ts
            print(f"    • {adapter_id}: 2024 (Temporal Update)")
        else:
            entry.timestamp = geo_ts
            print(f"    • {adapter_id}: 2023 (Geo Specialist)")
            
    print_success(f"Router initialized with {len(router.get_registered_adapters())} adapters")
    
    # Check if centroids exist
    entries_with_centroids = router._manifest.entries_with_centroids
    if not entries_with_centroids and auto_compute_centroids:
        print_warning("No centroids found. Computing from SituatedQA training data...")
        compute_centroids_from_situatedqa(router, enable_source_replay=enable_source_replay)
    elif not entries_with_centroids:
        print_warning("Note: Centroids need to be computed. Run scripts/compute_centroids.py first.")
    
    return router


def compute_centroids_from_situatedqa(router: CentroidRouter, max_samples: int = 500, enable_source_replay: bool = True) -> None:
    """Compute centroids and index Source-Replay from ACTUAL SituatedQA training data.
    
    Uses the same data loader and filters that were used during training,
    ensuring centroids accurately represent each adapter's knowledge domain.
    Also indexes samples for Source-Replay (RAG from loser adapters).
    
    Args:
        router: The CentroidRouter to configure.
        max_samples: Maximum samples per adapter.
        enable_source_replay: Whether to index samples for Source-Replay.
    """
    print("\n  Computing centroids from ACTUAL SituatedQA training data...")
    print(f"  (Using up to {max_samples} samples per adapter)")
    if enable_source_replay:
        print(f"  Source-Replay: ENABLED (will index Q&A pairs for RAG)")
    
    from src.data.loader import SituatedQALoader, SituatedQAConfig
    
    # Initialize data loader
    config = SituatedQAConfig(
        streaming=False,  # Need to iterate multiple times
        temporal_cutoff_year=2019,
        seed=42,
    )
    loader = SituatedQALoader(config)
    
    # Define how to get training data for each adapter type
    def get_samples_for_adapter(adapter_id: str, max_n: int) -> list[dict]:
        """Extract full samples from the appropriate data stream."""
        samples = []
        
        try:
            if adapter_id == "base_v1":
                # Base: temporal < 2019 + US geo
                stream = loader.get_base_stream()
            elif adapter_id == "patch_temp_2019_plus":
                # Temporal: >= 2019
                stream = loader.get_temporal_patch_stream()
            elif adapter_id.startswith("patch_geo_"):
                # Geographic: specific country
                country = adapter_id.replace("patch_geo_", "").replace("_", " ").title()
                
                if country == "Uk":
                    country = "United Kingdom"
                elif country == "Others":
                    # For "others", get a mix of non-major countries
                    stream = loader.get_rest_of_world_stream(
                        exclude_countries=["India", "UK", "United Kingdom", "Germany", 
                                          "France", "Australia", "Canada", "Nigeria", 
                                          "Pakistan", "England", "California"]
                    )
                else:
                    stream = loader.get_geo_patch_stream(country)
            else:
                return []
            
            # Extract full samples from stream
            count = 0
            for example in stream:
                if count >= max_n:
                    break
                
                question = example.get("edited_question", "")
                if question and question.strip():
                    samples.append(dict(example))  # Keep full sample for Source-Replay
                    count += 1
                    
        except Exception as e:
            print(f"      Warning: Error loading data for {adapter_id}: {e}")
        
        return samples
    
    # Initialize Source-Replay if enabled
    if enable_source_replay:
        router.initialize_source_replay()
    
    computed = 0
    indexed = 0
    for adapter_id in router.get_registered_adapters():
        print(f"    Loading training data for {adapter_id}...")
        
        samples = get_samples_for_adapter(adapter_id, max_samples)
        
        if not samples:
            print(f"      ⚠ No training data found for {adapter_id}, skipping")
            continue
        
        print(f"      Found {len(samples)} training samples")
        
        try:
            # 1. Compute Centroid (from questions only)
            print(f"      Computing centroid...")
            questions = [s.get("edited_question", "") for s in samples]
            centroid = router.compute_centroid(questions)
            router._manifest.update_centroid(adapter_id, centroid)
            computed += 1
            
            # 2. Index for Source-Replay (full Q&A pairs)
            if enable_source_replay:
                print(f"      Indexing for Source-Replay...")
                num_chunks = router.index_samples_for_replay(adapter_id, samples)
                indexed += num_chunks
                print(f"    ✓ {adapter_id} (centroid + {num_chunks} chunks indexed)")
            else:
                print(f"    ✓ {adapter_id} (centroid from {len(questions)} questions)")
        except Exception as e:
            print(f"    ✗ {adapter_id}: {e}")
    
    if enable_source_replay:
        print_success(f"Computed {computed} centroids, indexed {indexed} chunks for Source-Replay")
    else:
        print_success(f"Computed {computed} centroids from real training data")


# =============================================================================
# Main
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Interactive inference with detailed logging",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock embeddings (for testing without GPU)",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default=None,
        help="Embedding model (HuggingFace ID or local path)",
    )
    parser.add_argument(
        "--checkpoints_dir",
        type=str,
        default="checkpoints",
        help="Directory with adapter checkpoints",
    )
    parser.add_argument(
        "--router_state",
        type=str,
        default=None,
        help="Load router from saved state (includes centroids)",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default=None,
        help="LLM model ID for generation (omit for routing-only mode)",
    )
    parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=0.55,
        help="Similarity threshold for routing (lower = more matches)",
    )
    parser.add_argument(
        "--auto_compute_centroids",
        action="store_true",
        default=True,
        help="Automatically compute centroids if missing",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable verbose logging",
    )
    parser.add_argument(
        "--source_replay",
        action="store_true",
        default=True,
        help="Enable Source-Replay (RAG from loser adapters)",
    )
    parser.add_argument(
        "--no_source_replay",
        action="store_true",
        help="Disable Source-Replay",
    )
    
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.WARNING,  # Suppress library logs
        format="%(message)s",
    )
    
    # Setup router
    if args.mock:
        router = setup_router_with_mock()
    elif args.router_state:
        print(f"Loading router from: {args.router_state}")
        router = CentroidRouter.load(
            path=args.router_state,
            embedding_model_path=args.embedding_model,
            similarity_threshold=args.similarity_threshold,
        )
        print_success("Router loaded from saved state")
    elif args.embedding_model:
        enable_replay = args.source_replay and not args.no_source_replay
        router = setup_router_from_checkpoints(
            embedding_model=args.embedding_model,
            checkpoints_dir=args.checkpoints_dir,
            similarity_threshold=args.similarity_threshold,
            enable_source_replay=enable_replay,
        )
    else:
        print_error("Must specify --mock, --router_state, or --embedding_model")
        print("\nExamples:")
        print("  python interactive_inference.py --mock")
        print("  python interactive_inference.py --embedding_model BAAI/bge-base-en-v1.5")
        print("  python interactive_inference.py --router_state router_state/")
        sys.exit(1)
    
    # Create session
    session = InteractiveSession(
        router=router,
        llm_model_id=args.model_id,
        verbose=not args.quiet,
    )
    
    # Run interactive loop
    session.run_interactive()


if __name__ == "__main__":
    main()

