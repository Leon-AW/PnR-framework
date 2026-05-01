"""LoRA + RAG baseline inference wrapper.

Combines the existing monolithic LoRA adapter with retrieval-augmented
generation (RAG) at inference time.  No additional training is required:

- **LoRA part**: loads the trained monolithic adapter via
  ``PatchAndRouteInference`` with ``force_adapter``.
- **RAG part**: builds a FAISS index over a JSON file of QA pairs
  (e.g. ``data/edit_pairs.json``) using sentence-transformers embeddings.
  At inference time the top-k most relevant pairs are retrieved and
  prepended to the user query as context.

This is Baseline 2 from the exposé ("LoRA + RAG Hybrid"): monolithic
fine-tuning augmented with retrieval over the knowledge edit repository.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

_DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_DEFAULT_TOP_K = 3


@dataclass
class LoRARAGResult:
    response: str
    adapter_loaded: str = "lora_rag"
    routing_result: Any = None
    n_retrieved: int = 0
    retrieved_questions: list[str] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.retrieved_questions is None:
            self.retrieved_questions = []


class LoRARAGInference:
    """Monolithic LoRA adapter + QA-pair retrieval at inference time.

    Parameters
    ----------
    monolithic_adapter_path:
        Path to the trained monolithic LoRA adapter directory (produced by
        ``train_monolithic_baseline.py``).
    qa_pairs_path:
        Path to a JSON file containing knowledge edits.  Each entry must
        have at least ``"question"`` and ``"answer"`` fields.  Typically
        ``data/edit_pairs.json``.
    model_id:
        HuggingFace model identifier for the base LLM.
    quantization:
        ``"int4"`` | ``"int8"`` | ``"none"``.
    top_k:
        Number of QA pairs to retrieve per query.
    embedding_model:
        Sentence-transformers model used to embed queries and QA pairs.
    max_new_tokens:
        Maximum generation tokens.
    temperature:
        Sampling temperature.
    do_sample:
        Whether to sample (vs. greedy decode).
    use_gpu:
        Whether to use CUDA.
    """

    def __init__(
        self,
        monolithic_adapter_path: str | Path,
        qa_pairs_path: str | Path,
        model_id: str = "mistralai/Mistral-7B-Instruct-v0.3",
        quantization: str = "int4",
        top_k: int = _DEFAULT_TOP_K,
        embedding_model: str = _DEFAULT_EMBEDDING_MODEL,
        max_new_tokens: int = 256,
        temperature: float = 0.1,
        do_sample: bool = False,
        use_gpu: bool = True,
    ) -> None:
        self.monolithic_adapter_path = str(monolithic_adapter_path)
        self.qa_pairs_path = Path(qa_pairs_path)
        self.model_id = model_id
        self.quantization = quantization
        self.top_k = top_k
        self.embedding_model_name = embedding_model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample
        self.use_gpu = use_gpu

        self._pipeline = None
        self._encoder = None
        self._index = None          # np.ndarray (N, D) — row-normalised embeddings
        self._qa_pairs: list[dict] = []

    # ------------------------------------------------------------------
    # Lazy initialisation
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._pipeline is None:
            self._build_pipeline()
        if self._encoder is None or self._index is None:
            self._build_index()

    def _build_pipeline(self) -> None:
        """Initialise PatchAndRouteInference with the monolithic adapter."""
        from src.inference import PatchAndRouteInference, GenerationConfig
        from src.models.core import FrozenFoundationConfig, PatchAndRouteLLM, QuantizationType
        from src.routing import CentroidRouter

        quant_map = {
            "none": QuantizationType.NONE,
            "int8": QuantizationType.INT8,
            "int4": QuantizationType.INT4,
        }
        quantization = quant_map.get(self.quantization, QuantizationType.INT4)
        use_gpu = self.use_gpu and torch.cuda.is_available()

        router = CentroidRouter(use_gpu=use_gpu)
        gen_config = GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=self.do_sample,
        )
        self._pipeline = PatchAndRouteInference(
            model_id=self.model_id,
            router=router,
            quantization=quantization,
            generation_config=gen_config,
            use_gpu=use_gpu,
        )
        logger.info("LoRA+RAG: base pipeline loaded (adapter=%s)", self.monolithic_adapter_path)

    def _build_index(self) -> None:
        """Load QA pairs and build a FAISS-free cosine-similarity index."""
        import numpy as np
        from sentence_transformers import SentenceTransformer

        if not self.qa_pairs_path.exists():
            raise FileNotFoundError(f"QA pairs file not found: {self.qa_pairs_path}")

        with open(self.qa_pairs_path) as f:
            if self.qa_pairs_path.suffix == ".jsonl":
                raw = [json.loads(line) for line in f if line.strip()]
            else:
                raw = json.load(f)

        self._qa_pairs = [
            {"question": item["question"], "answer": item["answer"]}
            for item in raw
            if "question" in item and "answer" in item
        ]
        if not self._qa_pairs:
            raise ValueError(f"No valid QA pairs found in {self.qa_pairs_path}")

        logger.info("LoRA+RAG: encoding %d QA pairs with %s ...",
                    len(self._qa_pairs), self.embedding_model_name)

        device = "cuda" if (self.use_gpu and torch.cuda.is_available()) else "cpu"
        self._encoder = SentenceTransformer(self.embedding_model_name, device=device)

        questions = [p["question"] for p in self._qa_pairs]
        embeddings = self._encoder.encode(
            questions,
            batch_size=256,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        self._index = embeddings.astype("float32")
        logger.info("LoRA+RAG: index ready (%d vectors, dim=%d)", len(self._index), self._index.shape[1])

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def _retrieve(self, query: str) -> list[dict]:
        """Return the top-k QA pairs most similar to *query*."""
        import numpy as np

        query_emb = self._encoder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")[0]  # (D,)

        # Cosine similarity via dot product (both sides are L2-normalised)
        scores = self._index @ query_emb  # (N,)
        top_indices = np.argpartition(scores, -self.top_k)[-self.top_k:]
        top_indices = top_indices[np.argsort(-scores[top_indices])]

        return [self._qa_pairs[i] for i in top_indices]

    # ------------------------------------------------------------------
    # Context formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _format_context(pairs: list[dict]) -> str:
        lines = ["Here are some relevant facts that may help answer the question:"]
        for p in pairs:
            lines.append(f"- Q: {p['question']}")
            lines.append(f"  A: {p['answer']}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def generate(self, query: str) -> LoRARAGResult:
        """Retrieve context and generate an answer with the monolithic adapter.

        Parameters
        ----------
        query:
            The evaluation question.

        Returns
        -------
        LoRARAGResult with ``response`` and retrieval metadata.
        """
        self._ensure_loaded()

        retrieved = self._retrieve(query)
        context = self._format_context(retrieved)
        augmented_query = f"{context}\n\nQuestion: {query}"

        result = self._pipeline.generate(
            query=augmented_query,
            force_adapter=self.monolithic_adapter_path,
        )

        return LoRARAGResult(
            response=result.response,
            n_retrieved=len(retrieved),
            retrieved_questions=[p["question"] for p in retrieved],
        )

    # ------------------------------------------------------------------
    # Log-probability scoring
    # ------------------------------------------------------------------

    def score_targets(self, query: str, targets: list[str]) -> dict[str, float]:
        """Compute log P(target | augmented_prompt) under the monolithic LoRA.

        Uses the same retrieval-augmented prompt as ``generate`` so the
        log-prob view reflects the *full* LoRA+RAG system, not just the
        bare adapter.
        """
        self._ensure_loaded()
        retrieved = self._retrieve(query)
        context = self._format_context(retrieved)
        augmented_query = f"{context}\n\nQuestion: {query}"
        return self._pipeline.score_targets(
            query=augmented_query,
            targets=targets,
            force_adapter=self.monolithic_adapter_path,
        )
