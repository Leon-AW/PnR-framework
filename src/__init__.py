"""
Patch-and-Route Framework
=========================

A modular framework for Continual Learning in Enterprise LLMs.

This framework implements the "Patch-and-Route" architecture described in:
"A Modular 'Patch-and-Route' Framework for Continual Learning in Enterprise LLMs"

Core Concepts:
- Frozen Foundation: Base LLM with frozen parameters (e.g., Mistral-7B)
- Expert Pool: Collection of domain-specific LoRA adapters
- Knowledge Router: Dynamic routing mechanism for adapter selection (Time-Aware Centroid Router)
- Source-Replay: RAG-style retrieval from older conflicting adapters

Modules:
- src.models: Frozen Foundation and Expert Adapter management
- src.data: SituatedQA and CounterFact data loaders
- src.training: SFTTrainer integration
- src.routing: Time-Aware Centroid Router with Source-Replay
- src.inference: Unified inference pipeline

Author: Leon Wagner
"""

__version__ = "0.2.0"
__author__ = "Leon Wagner"

# Convenience imports
from src.routing import CentroidRouter, AdapterManifest, SourceReplayStore

__all__ = [
    "CentroidRouter",
    "AdapterManifest",
    "SourceReplayStore",
]

