"""
Baselines Package
=================

Comparison baselines for the Patch-and-Route evaluation:

- xlora:           X-LoRA soft gating (Buehler & Buehler 2024)
- recipe_official: RECIPE official repo (Chen et al., EMNLP 2024)
- lora_rag:        LoRA + RAG hybrid (monolithic adapter + QA-pair retrieval)
"""

from src.baselines.xlora import XLoRAInference
from src.baselines.lora_rag import LoRARAGInference, LoRARAGResult

__all__ = [
    "XLoRAInference",
    "LoRARAGInference",
    "LoRARAGResult",
]
