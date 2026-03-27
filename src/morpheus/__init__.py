"""
MORPHEUS — Multi-timescale Orchestrated Rehearsal with Prototype-routed
         Hierarchical Expert Unification System
=========================================================================

A cognitive multi-system architecture for continual learning in LLMs,
inspired by the multi-timescale structure of biological memory systems.

Architecture Overview:
    System 1 — Stable Core ("Neocortex"):      Deep structural knowledge
    System 2 — Expert Bank ("Cortical Columns"): Domain-specific experts
    System 3 — Fast Buffer ("Hippocampus"):      Immediate data absorption
    System 4 — Consolidation ("Sleep"):          Rehearsal & distillation
    System 5 — Knowledge Store ("Episodic"):     Explicit factual memory
    System 6 — Meta-Controller ("Prefrontal"):   System orchestration

Named after Morpheus, the Greek god of dreams, reflecting the architecture's
central "consolidation as dreaming" mechanism where self-generated rehearsal
from frozen experts prevents catastrophic forgetting.

Timescales:
    Fast (seconds):   Buffer absorbs streaming data
    Medium (hours):   Experts consolidate via interleaved rehearsal
    Slow (weeks):     Core evolves via structural distillation
    Meta (episodes):  Controller optimizes the learning process

Usage:
    from src.morpheus import MorpheusInference, MorpheusConfig

    config = MorpheusConfig()
    pipeline = MorpheusInference(config=config)
    result = pipeline.generate("Who is the Chancellor of Germany?")
"""

from .config import (
    MorpheusConfig,
    StableCoreConfig,
    ExpertBankConfig,
    FastBufferConfig,
    ConsolidationConfig,
    KnowledgeStoreConfig,
    MetaControllerConfig,
    PrototypeRouterConfig,
    ExpertState,
    ConsolidationTrigger,
    ActionReversibility,
)
from .cka import linear_cka, minibatch_cka, compute_representation_shift
from .stable_core import StableCore, CoreVersion, CompatAdapter
from .expert_bank import ExpertBank, ExpertMetadata
from .router import PrototypeRouter, ExpertPrototype
from .fast_buffer import FastBuffer, BufferSample
from .knowledge_store import KnowledgeStore, KnowledgeRecord, FactualityDecision
from .rehearsal import RehearsalEngine, Coreset, FrozenReferenceManager
from .consolidation import ConsolidationEngine, ConsolidationCycleResult
from .meta_controller import MetaController, SystemState, HeuristicPolicy
from .inference import (
    MorpheusInference,
    MorpheusInferenceResult,
    MorpheusGenerationConfig,
    MorpheusPromptBuilder,
)

__all__ = [
    # Top-level
    "MorpheusConfig",
    "MorpheusInference",
    "MorpheusInferenceResult",
    "MorpheusGenerationConfig",
    "MorpheusPromptBuilder",
    # Config
    "StableCoreConfig",
    "ExpertBankConfig",
    "FastBufferConfig",
    "ConsolidationConfig",
    "KnowledgeStoreConfig",
    "MetaControllerConfig",
    "PrototypeRouterConfig",
    "ExpertState",
    "ConsolidationTrigger",
    "ActionReversibility",
    # System 1: Stable Core
    "StableCore",
    "CoreVersion",
    "CompatAdapter",
    # System 2: Expert Bank
    "ExpertBank",
    "ExpertMetadata",
    # Router
    "PrototypeRouter",
    "ExpertPrototype",
    # System 3: Fast Buffer
    "FastBuffer",
    "BufferSample",
    # System 5: Knowledge Store
    "KnowledgeStore",
    "KnowledgeRecord",
    "FactualityDecision",
    # System 4: Consolidation
    "RehearsalEngine",
    "Coreset",
    "FrozenReferenceManager",
    "ConsolidationEngine",
    "ConsolidationCycleResult",
    # System 6: Meta-Controller
    "MetaController",
    "SystemState",
    "HeuristicPolicy",
    # CKA utilities
    "linear_cka",
    "minibatch_cka",
    "compute_representation_shift",
]
