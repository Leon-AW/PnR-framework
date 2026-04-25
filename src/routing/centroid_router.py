"""
Time-Aware Centroid Router
===========================

Implements the core routing logic for the Patch-and-Route framework.

The Centroid Router uses semantic similarity to match queries to adapters,
with a time-aware conflict resolution mechanism:

1. **Embed Query**: Transform user query into embedding space
2. **Match Centroids**: Find adapters whose centroids are similar to query
3. **Detect Conflicts**: If multiple adapters match above threshold
4. **Resolve Conflicts**: Newest adapter wins (Weight Loading), older adapters
   contribute via Source-Replay (retrieval from training data)

Key Design Decisions:
1. Uses cosine similarity for matching (normalized embeddings)
2. Timestamp-based conflict resolution (newest adapter = winner)
3. Source-Replay provides RAG-style context from older adapters
4. Supports any embedding model via callable interface

Reference: Section 4.4.1 of the Master's Thesis Exposé - "Time-Aware Centroid Router"
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import numpy as np
import torch

# Try sentence-transformers first (preferred for embedding models)
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

from transformers import AutoModel, AutoTokenizer

from .base import BaseRouter, RoutingResult, RoutingStrategy, AdapterMatch
from .manifest import AdapterManifest, AdapterEntry
from .source_replay import SourceReplayStore, RetrievedChunk

logger = logging.getLogger(__name__)


class CentroidRouter(BaseRouter):
    """Time-Aware Centroid Router with Source-Replay.
    
    Routes queries to adapters based on semantic similarity to adapter centroids.
    Resolves conflicts using timestamps (newest wins), with older adapters
    contributing via retrieval from their training data.
    
    Example:
        ```python
        # Initialize with embedding model
        router = CentroidRouter(
            embedding_model_path="/path/to/KaLM-Embedding-Gemma3-12B",
            similarity_threshold=0.7,
        )
        
        # Register adapters from checkpoints
        router.register_from_checkpoints("checkpoints/")
        
        # Compute centroids (offline step)
        router.compute_all_centroids()
        
        # Route a query (online step)
        result = router.route("Who is the Chancellor of Germany in 2023?")
        
        # Result contains:
        # - winner_adapter: "patch_geo_germany" (newest matching)
        # - retrieved_context: Context from older adapters if conflict
        ```
    """
    
    def __init__(
        self,
        embedding_model_path: str | None = None,
        embedding_fn: Callable[[str], np.ndarray] | None = None,
        similarity_threshold: float = 0.65,
        conflict_threshold: float = 0.1,
        top_k_retrieval: int = 3,
        retrieval_threshold: float = 0.45,
        max_context_length: int = 2000,
        use_gpu: bool = True,
        store_dir: str | Path | None = None,
    ) -> None:
        """Initialize the Centroid Router.
        
        Args:
            embedding_model_path: Path to local embedding model.
            embedding_fn: Custom embedding function (alternative to model path).
            similarity_threshold: Minimum similarity to consider a match.
            conflict_threshold: Similarity gap to consider as conflict.
            top_k_retrieval: Number of chunks to retrieve per loser adapter.
            max_context_length: Maximum context length for Source-Replay.
            use_gpu: Whether to use GPU for embedding and FAISS.
            store_dir: Directory for persisting indices and manifest.
        """
        super().__init__(strategy=RoutingStrategy.CENTROID)
        
        self.similarity_threshold = similarity_threshold
        self.conflict_threshold = conflict_threshold
        self.top_k_retrieval = top_k_retrieval
        self.retrieval_threshold = retrieval_threshold
        self.max_context_length = max_context_length
        self.use_gpu = use_gpu
        self.store_dir = Path(store_dir) if store_dir else None
        
        # Initialize embedding model
        self._embedding_model = None
        self._embedding_tokenizer = None
        self._sentence_transformer = None  # sentence-transformers model (preferred)
        self._embedding_dim: int | None = None
        self._custom_embedding_fn = embedding_fn
        
        if embedding_model_path:
            self._load_embedding_model(embedding_model_path)
        elif embedding_fn:
            # Infer dimension from a test embedding
            test_emb = embedding_fn("test")
            self._embedding_dim = test_emb.shape[0]
        
        # Initialize manifest and store
        self._manifest = AdapterManifest()
        self._source_replay: SourceReplayStore | None = None
        
        logger.info("=" * 60)
        logger.info("CENTROID ROUTER INITIALIZED")
        logger.info("=" * 60)
        logger.info(f"  Similarity threshold: {similarity_threshold}")
        logger.info(f"  Conflict threshold: {conflict_threshold}")
        logger.info(f"  Top-K retrieval: {top_k_retrieval}")
        logger.info("=" * 60)
    
    # -------------------------------------------------------------------------
    # Embedding Model
    # -------------------------------------------------------------------------
    
    def _load_embedding_model(self, model_path: str) -> None:
        """Load the embedding model.
        
        Supports two loading strategies:
        1. sentence-transformers (preferred) - for models like KaLM-Embedding, BGE, E5
        2. transformers AutoModel (fallback) - for generic transformer models
        
        Args:
            model_path: Path to the model. Can be:
                - Local path (e.g., /vol/models/KaLM-Embedding)
                - HuggingFace model ID (e.g., tencent/KaLM-Embedding-Gemma3-12B-2511)
        """
        logger.info(f"Loading embedding model from: {model_path}")
        
        # Detect if this is a local path or HuggingFace model ID
        is_local = Path(model_path).exists()
        
        if is_local:
            logger.info(f"  Detected local model path")
        else:
            logger.info(f"  Detected HuggingFace model ID")
        
        # Strategy 1: Use sentence-transformers if available (recommended)
        if HAS_SENTENCE_TRANSFORMERS:
            try:
                logger.info("  Using sentence-transformers backend")
                
                model_kwargs = {
                    "dtype": torch.bfloat16 if self.use_gpu else torch.float32,
                    "trust_remote_code": True,
                }
                
                # Add flash attention if available
                try:
                    import flash_attn
                    model_kwargs["attn_implementation"] = "flash_attention_2"
                    logger.info("  Flash Attention 2 enabled")
                except ImportError:
                    pass
                
                self._sentence_transformer = SentenceTransformer(
                    model_path,
                    trust_remote_code=True,
                    model_kwargs=model_kwargs,
                    device="cuda" if self.use_gpu else "cpu",
                )
                self._sentence_transformer.max_seq_length = 512
                
                # Get embedding dimension
                self._embedding_dim = self._sentence_transformer.get_sentence_embedding_dimension()
                
                logger.info(f"✓ Embedding model loaded via sentence-transformers (dim={self._embedding_dim})")
                return
                
            except Exception as e:
                logger.warning(f"sentence-transformers loading failed: {e}")
                logger.info("  Falling back to transformers AutoModel...")
        
        # Strategy 2: Fallback to transformers AutoModel
        try:
            logger.info("  Using transformers AutoModel backend")
            
            self._embedding_tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                local_files_only=is_local,
                trust_remote_code=True,
            )
            
            self._embedding_model = AutoModel.from_pretrained(
                model_path,
                dtype=torch.float16 if self.use_gpu else torch.float32,
                device_map="auto" if self.use_gpu else None,
                local_files_only=is_local,
                trust_remote_code=True,
            )
            
            if not self.use_gpu:
                self._embedding_model = self._embedding_model.to("cpu")
            
            self._embedding_model.eval()
            
            # Get embedding dimension
            with torch.no_grad():
                test_input = self._embedding_tokenizer(
                    "test", return_tensors="pt", padding=True, truncation=True
                )
                if self.use_gpu:
                    test_input = {k: v.to(self._embedding_model.device) for k, v in test_input.items()}
                output = self._embedding_model(**test_input)
                self._embedding_dim = output.last_hidden_state.shape[-1]
            
            logger.info(f"✓ Embedding model loaded via AutoModel (dim={self._embedding_dim})")
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def compute_embedding(self, text: str) -> np.ndarray:
        """Compute embedding vector for a text.
        
        Args:
            text: Input text.
            
        Returns:
            Embedding vector as numpy array (normalized).
        """
        # Priority 1: Custom embedding function
        if self._custom_embedding_fn:
            return self._custom_embedding_fn(text)
        
        # Priority 2: sentence-transformers (preferred for embedding models)
        if self._sentence_transformer is not None:
            embedding = self._sentence_transformer.encode(
                text,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return embedding.astype(np.float32)
        
        # Priority 3: transformers AutoModel
        if self._embedding_model is None:
            raise RuntimeError("Embedding model not loaded. Call _load_embedding_model() first.")
        
        with torch.no_grad():
            inputs = self._embedding_tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            
            if self.use_gpu:
                inputs = {k: v.to(self._embedding_model.device) for k, v in inputs.items()}
            
            outputs = self._embedding_model(**inputs)
            
            # Mean pooling over sequence length
            attention_mask = inputs["attention_mask"]
            hidden_states = outputs.last_hidden_state
            
            # Mask padding tokens
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            embeddings = sum_embeddings / sum_mask
            
            # Normalize
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            return embeddings.cpu().numpy().squeeze()
    
    def compute_embeddings_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Compute embeddings for multiple texts in batch (much faster).
        
        Args:
            texts: List of input texts.
            batch_size: Batch size for encoding.
            
        Returns:
            Array of shape (len(texts), embedding_dim).
        """
        if not texts:
            return np.array([])
        
        # Priority 1: sentence-transformers (has native batch support)
        if self._sentence_transformer is not None:
            embeddings = self._sentence_transformer.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=batch_size,
            )
            return embeddings.astype(np.float32)
        
        # Priority 2: Custom embedding function (fallback to loop)
        if self._custom_embedding_fn:
            return np.vstack([self._custom_embedding_fn(t) for t in texts])
        
        # Priority 3: AutoModel (manual batching)
        if self._embedding_model is None:
            raise RuntimeError("Embedding model not loaded.")
        
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            with torch.no_grad():
                inputs = self._embedding_tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                )
                
                if self.use_gpu:
                    inputs = {k: v.to(self._embedding_model.device) for k, v in inputs.items()}
                
                outputs = self._embedding_model(**inputs)
                
                attention_mask = inputs["attention_mask"]
                hidden_states = outputs.last_hidden_state
                
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                embeddings = sum_embeddings / sum_mask
                
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings).astype(np.float32)
    
    def compute_centroid(self, texts: list[str]) -> np.ndarray:
        """Compute centroid (mean vector) of multiple texts.

        Args:
            texts: List of training texts.

        Returns:
            Mean embedding vector (normalized).
        """
        if not texts:
            raise ValueError("Cannot compute centroid of empty text list")

        embeddings = np.vstack([self.compute_embedding(t) for t in texts])

        # Mean of normalized vectors
        centroid = np.mean(embeddings, axis=0)

        # Re-normalize
        centroid = centroid / np.linalg.norm(centroid)

        return centroid.astype(np.float32)

    def compute_cluster_centroids(
        self,
        texts: list[str],
        k: int,
        batch_size: int = 64,
        random_state: int = 42,
    ) -> list[np.ndarray]:
        """Run k-means over text embeddings and return k normalized cluster centroids.

        Intended for broad-domain adapters (e.g. patch_cf_main spanning 21k
        heterogeneous facts) where a single mean centroid collapses near origin
        and gets dominated by narrow specialists. Each cluster captures a
        subdomain (e.g. "languages of places", "creators of products", …); the
        router takes max over clusters.

        Args:
            texts: Training texts (≥ k).
            k: Number of clusters. Falls back to k=1 (plain mean) if
                len(texts) < k.
            batch_size: Embedding batch size.
            random_state: Seed for reproducible k-means.

        Returns:
            List of k L2-normalized centroids of shape (embedding_dim,).
        """
        from sklearn.cluster import KMeans

        if not texts:
            raise ValueError("Cannot cluster empty text list")

        if k <= 1 or len(texts) <= k:
            return [self.compute_centroid(texts)]

        embeddings = self.compute_embeddings_batch(texts, batch_size=batch_size)

        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        km.fit(embeddings)

        centroids: list[np.ndarray] = []
        for i in range(k):
            raw = km.cluster_centers_[i]
            norm = np.linalg.norm(raw)
            c = raw / norm if norm > 0 else raw
            centroids.append(c.astype(np.float32))

        return centroids
    
    # -------------------------------------------------------------------------
    # Adapter Registration
    # -------------------------------------------------------------------------
    
    def register_adapter(
        self,
        adapter_id: str,
        path: str,
        timestamp: float,
        adapter_type: str = "unknown",
        training_data_path: str | None = None,
        centroid: np.ndarray | None = None,
    ) -> None:
        """Register an adapter with the router.
        
        Args:
            adapter_id: Unique identifier.
            path: Path to adapter checkpoint.
            timestamp: Training timestamp (epoch seconds).
            adapter_type: Type classification.
            training_data_path: Path to training data for centroid computation.
            centroid: Pre-computed centroid (optional).
        """
        self._manifest.register(
            adapter_id=adapter_id,
            adapter_path=path,
            timestamp=timestamp,
            adapter_type=adapter_type,
            centroid=centroid,
            source_data_path=training_data_path,
        )
        
        logger.info(f"Registered adapter: {adapter_id}")
    
    def register_from_checkpoints(
        self,
        checkpoints_dir: str | Path,
        base_timestamp: float | None = None,
    ) -> int:
        """Auto-discover and register adapters from a checkpoints directory.
        
        Args:
            checkpoints_dir: Directory containing adapter checkpoints.
            base_timestamp: Default timestamp if not found in config.
            
        Returns:
            Number of adapters registered.
        """
        self._manifest = AdapterManifest.from_checkpoints_dir(
            checkpoints_dir=checkpoints_dir,
            base_timestamp=base_timestamp,
        )
        
        logger.info(f"Registered {self._manifest.num_adapters} adapters from {checkpoints_dir}")
        
        return self._manifest.num_adapters
    
    def get_registered_adapters(self) -> list[str]:
        """Get list of registered adapter IDs."""
        return self._manifest.adapters
    
    def unregister_adapter(self, adapter_id: str) -> bool:
        """Remove an adapter from the routing pool."""
        return self._manifest.unregister(adapter_id)
    
    # -------------------------------------------------------------------------
    # Centroid Computation
    # -------------------------------------------------------------------------
    
    def compute_adapter_centroid(
        self,
        adapter_id: str,
        training_data_path: str | Path | None = None,
        text_field: str = "edited_question",
        max_samples: int = 1000,
    ) -> np.ndarray:
        """Compute and store centroid for an adapter.
        
        Args:
            adapter_id: Adapter to compute centroid for.
            training_data_path: Path to training data (uses manifest if None).
            text_field: Field to extract text from.
            max_samples: Maximum samples to use for centroid.
            
        Returns:
            Computed centroid vector.
        """
        import json
        
        entry = self._manifest.get(adapter_id)
        if entry is None:
            raise KeyError(f"Adapter '{adapter_id}' not found in manifest")
        
        # Use provided path or from manifest
        data_path = training_data_path or entry.source_data_path
        if data_path is None:
            raise ValueError(f"No training data path for adapter '{adapter_id}'")
        
        data_path = Path(data_path)
        
        logger.info(f"Computing centroid for {adapter_id} from {data_path}")
        
        # Read training texts
        texts = []
        with open(data_path, "r") as f:
            for i, line in enumerate(f):
                if i >= max_samples:
                    break
                try:
                    data = json.loads(line)
                    text = data.get(text_field, "")
                    if text:
                        texts.append(text)
                except json.JSONDecodeError:
                    continue
        
        if not texts:
            raise ValueError(f"No valid texts found in {data_path}")
        
        logger.info(f"Computing centroid from {len(texts)} samples...")
        
        # Compute centroid
        centroid = self.compute_centroid(texts)
        
        # Update manifest
        self._manifest.update_centroid(adapter_id, centroid)
        
        logger.info(f"✓ Centroid computed for {adapter_id}")
        
        return centroid
    
    def compute_all_centroids(
        self,
        text_field: str = "edited_question",
        max_samples_per_adapter: int = 1000,
    ) -> int:
        """Compute centroids for all adapters with source data.
        
        Args:
            text_field: Field to extract text from.
            max_samples_per_adapter: Maximum samples per adapter.
            
        Returns:
            Number of centroids computed.
        """
        count = 0
        
        for entry in self._manifest:
            if entry.has_centroid:
                logger.info(f"Skipping {entry.adapter_id} (already has centroid)")
                continue
            
            if entry.source_data_path is None:
                logger.warning(f"Skipping {entry.adapter_id} (no source data path)")
                continue
            
            try:
                self.compute_adapter_centroid(
                    entry.adapter_id,
                    text_field=text_field,
                    max_samples=max_samples_per_adapter,
                )
                count += 1
            except Exception as e:
                logger.error(f"Failed to compute centroid for {entry.adapter_id}: {e}")
        
        logger.info(f"Computed {count} centroids")
        
        return count
    
    # -------------------------------------------------------------------------
    # Source-Replay Initialization
    # -------------------------------------------------------------------------
    
    def initialize_source_replay(self, store_dir: str | Path | None = None) -> None:
        """Initialize the Source-Replay store.
        
        Args:
            store_dir: Directory for persisting indices.
        """
        store_dir = store_dir or self.store_dir
        
        self._source_replay = SourceReplayStore(
            embedding_fn=self.compute_embedding,
            embedding_batch_fn=self.compute_embeddings_batch,  # Use batch encoding (10-50x faster)
            embedding_dim=self._embedding_dim or 768,
            use_gpu=self.use_gpu,
            store_dir=store_dir,
        )
        
        logger.info("Initialized Source-Replay store")
    
    def index_adapter_for_replay(
        self,
        adapter_id: str,
        training_data_path: str | Path | None = None,
        max_chunks: int = 5000,
    ) -> int:
        """Index an adapter's training data for Source-Replay.
        
        Args:
            adapter_id: Adapter to index.
            training_data_path: Path to training data.
            max_chunks: Maximum chunks to index.
            
        Returns:
            Number of chunks indexed.
        """
        if self._source_replay is None:
            self.initialize_source_replay()
        
        entry = self._manifest.get(adapter_id)
        data_path = training_data_path or (entry.source_data_path if entry else None)
        
        if data_path is None:
            raise ValueError(f"No training data path for adapter '{adapter_id}'")
        
        return self._source_replay.index_adapter(
            adapter_id=adapter_id,
            training_data_path=data_path,
            max_chunks=max_chunks,
        )
    
    def index_samples_for_replay(
        self,
        adapter_id: str,
        samples: list[dict],
        text_field: str = "edited_question",
        answer_field: str = "answer",
    ) -> int:
        """Index training samples directly for Source-Replay.
        
        Alternative to index_adapter_for_replay() that accepts samples
        directly instead of a file path.
        
        Args:
            adapter_id: Adapter to index.
            samples: List of sample dictionaries.
            text_field: Field for question text.
            answer_field: Field for answer.
            
        Returns:
            Number of chunks indexed.
        """
        if self._source_replay is None:
            self.initialize_source_replay()
        
        return self._source_replay.index_samples(
            adapter_id=adapter_id,
            samples=samples,
            text_field=text_field,
            answer_field=answer_field,
        )
    
    # -------------------------------------------------------------------------
    # Core Routing Logic
    # -------------------------------------------------------------------------
    
    def route(self, query: str, top_k: int = 3) -> RoutingResult:
        """Route a query to the appropriate adapter(s).
        
        This is the main entry point for the Time-Aware Centroid Router.
        
        Flow:
        1. Embed query
        2. Compute similarity to all adapter centroids
        3. Filter matches above threshold
        4. If conflict (multiple matches), resolve by timestamp
        5. Winner: Weight Loading, Losers: Source-Replay
        
        Args:
            query: User's input query.
            top_k: Maximum adapters to consider.
            
        Returns:
            RoutingResult with winner adapter and retrieved context.
        """
        logger.debug(f"Routing query: {query[:50]}...")
        
        # Step 1: Embed query
        query_embedding = self.compute_embedding(query)
        
        # Step 2: Get all cluster centroids (flat matrix, one row per cluster;
        # adapters with only a single mean centroid contribute one row)
        try:
            centroids, row_adapter_ids = self._manifest.get_cluster_centroids_flat()
        except ValueError:
            logger.warning("No adapters with centroids found")
            return RoutingResult(
                winner_adapter=None,
                winner_path=None,
                retrieved_context="",
                all_matches=[],
                query_embedding=query_embedding,
                has_conflict=False,
                routing_strategy=RoutingStrategy.CENTROID,
            )

        # Step 3: Compute similarities (cosine on normalized vectors = dot product).
        # Row-level sims are per-cluster; we then aggregate max-per-adapter so a
        # broad-domain adapter (e.g. patch_cf_main) wins as soon as ANY of its
        # subdomain cluster centroids matches the query.
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        row_similarities = np.dot(centroids, query_norm)

        best_sim_per_adapter: dict[str, float] = {}
        for adapter_id, sim in zip(row_adapter_ids, row_similarities):
            s = float(sim)
            if s > best_sim_per_adapter.get(adapter_id, -np.inf):
                best_sim_per_adapter[adapter_id] = s

        # Step 4: Filter matches above threshold
        matches = []
        for adapter_id, sim in best_sim_per_adapter.items():
            if sim >= self.similarity_threshold:
                entry = self._manifest[adapter_id]
                matches.append(AdapterMatch(
                    adapter_id=adapter_id,
                    similarity=sim,
                    timestamp=entry.timestamp,
                    is_winner=False,
                ))
        
        # Sort by similarity descending
        matches.sort(key=lambda m: m.similarity, reverse=True)
        matches = matches[:top_k]
        
        # No matches
        if not matches:
            logger.info("No matching adapters found")
            return RoutingResult(
                winner_adapter=None,
                winner_path=None,
                retrieved_context="",
                all_matches=[],
                query_embedding=query_embedding,
                has_conflict=False,
                routing_strategy=RoutingStrategy.CENTROID,
            )
        
        # Step 5: Detect conflicts
        has_conflict = len(matches) > 1
        
        if has_conflict:
            # Check if similarities are within conflict threshold
            sim_range = matches[0].similarity - matches[-1].similarity
            has_conflict = sim_range <= self.conflict_threshold
        
        # Step 6: Resolve - Winner is the best semantic match
        # Changed strategy: Prioritize Similarity > Timestamp
        if has_conflict:
            # Sort by similarity (descending) THEN timestamp (descending)
            # This ensures the most specialized adapter wins, even if older
            matches.sort(key=lambda m: (m.similarity, m.timestamp), reverse=True)
        
        # Mark winner (first in list after conflict resolution)
        matches[0].is_winner = True
        
        winner_id = matches[0].adapter_id
        winner_entry = self._manifest[winner_id]
        
        logger.info(f"Winner adapter: {winner_id} (sim={matches[0].similarity:.3f})")
        
        # Step 7: Source-Replay for losers
        retrieved_context = ""
        loser_ids = [m.adapter_id for m in matches[1:] if not m.is_winner]
        
        if loser_ids and self._source_replay:
            logger.info(f"Retrieving context from losers: {loser_ids}")
            
            chunks = self._source_replay.retrieve_multi(
                query_embedding=query_embedding,
                adapter_ids=loser_ids,
                top_k_per_adapter=self.top_k_retrieval,
            )
            
            # Filter low-relevance chunks to reduce hallucination
            original_count = len(chunks)
            chunks = [c for c in chunks if c.similarity >= self.retrieval_threshold]
            filtered_count = len(chunks)
            
            if original_count > filtered_count:
                logger.info(f"Filtered {original_count - filtered_count} chunks below threshold {self.retrieval_threshold}")

            # Store retrieved text in match objects
            for match in matches:
                if not match.is_winner:
                    match.retrieved_context = [
                        c.text for c in chunks if c.adapter_id == match.adapter_id
                    ]
            
            retrieved_context = SourceReplayStore.build_context(
                chunks,
                max_context_length=self.max_context_length,
            )
        
        return RoutingResult(
            winner_adapter=winner_id,
            winner_path=winner_entry.adapter_path,
            retrieved_context=retrieved_context,
            all_matches=matches,
            query_embedding=query_embedding,
            has_conflict=has_conflict,
            routing_strategy=RoutingStrategy.CENTROID,
        )
    
    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------
    
    def save(self, path: str | Path) -> None:
        """Save router state (manifest and indices) to disk.
        
        Args:
            path: Directory to save to.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save manifest
        self._manifest.save(path / "manifest.json")
        
        logger.info(f"Router state saved to {path}")
    
    @classmethod
    def load(
        cls,
        path: str | Path,
        embedding_model_path: str | None = None,
        **kwargs,
    ) -> CentroidRouter:
        """Load router from saved state.
        
        Args:
            path: Directory with saved state.
            embedding_model_path: Path to embedding model.
            **kwargs: Additional router arguments.
            
        Returns:
            Loaded CentroidRouter.
        """
        path = Path(path)
        
        # Create router
        router = cls(
            embedding_model_path=embedding_model_path,
            store_dir=path,
            **kwargs,
        )
        
        # Load manifest
        manifest_path = path / "manifest.json"
        if manifest_path.exists():
            router._manifest = AdapterManifest.load(manifest_path)
        
        logger.info(f"Router loaded from {path}")
        
        return router
    
    def summary(self) -> str:
        """Get a formatted summary of the router state."""
        lines = [
            "=" * 60,
            "CENTROID ROUTER STATUS",
            "=" * 60,
            f"Strategy: {self.strategy.value}",
            f"Similarity threshold: {self.similarity_threshold}",
            f"Conflict threshold: {self.conflict_threshold}",
            f"Embedding dim: {self._embedding_dim}",
            "-" * 60,
            self._manifest.summary(),
        ]
        
        if self._source_replay:
            lines.append("-" * 60)
            lines.append(f"Source-Replay adapters: {self._source_replay.adapters}")
        
        return "\n".join(lines)

