"""
MORPHEUS Configuration
======================

Configuration dataclasses for all six cognitive subsystems of the MORPHEUS
architecture (Multi-timescale Orchestrated Rehearsal with Prototype-routed
Hierarchical Expert Unification System).

Each subsystem has its own config; MorpheusConfig aggregates them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import torch


# =============================================================================
# Enums
# =============================================================================

class ExpertState(Enum):
    """Lifecycle states for experts in the Expert Bank."""
    SHADOW = "shadow"          # Newly spawned, training but not in routing
    ACTIVE = "active"          # Actively receiving data, plastic
    FROZEN = "frozen"          # Mature, protected from modification
    MERGE_CANDIDATE = "merge"  # Flagged for potential merge
    DORMANT = "dormant"        # Compressed and archived, reactivatable


class ConsolidationTrigger(Enum):
    """When to trigger a consolidation cycle."""
    BUFFER_FULL = "buffer_full"
    LOSS_SPIKE = "loss_spike"
    SCHEDULED = "scheduled"
    META_DECISION = "meta_decision"


class ActionReversibility(Enum):
    """Classification of meta-controller actions by reversibility."""
    REVERSIBLE = "reversible"
    IRREVERSIBLE = "irreversible"


# =============================================================================
# System 1: Stable Core ("Neocortex")
# =============================================================================

@dataclass
class StableCoreConfig:
    """Configuration for the Stable Core (System 1).

    The core stores deep structural knowledge and changes only through
    carefully constrained structural distillation.
    """
    model_id: str = "mistralai/Mistral-7B-Instruct-v0.3"
    quantization: str = "int4"
    device_map: str = "auto"
    torch_dtype: torch.dtype = torch.bfloat16
    use_cache: bool = False

    cka_lambda: float = 0.5
    cka_threshold: float = 0.05
    probe_set_size: int = 512
    probe_batch_size: int = 32

    compat_adapter_rank: int = 8
    compat_adapter_alpha: int = 16

    readaptation_interval: int = 5
    max_adapter_chain_length: int = 3

    checkpoint_dir: str = "morpheus_state/core_versions"


# =============================================================================
# System 2: Expert Bank ("Cortical Columns")
# =============================================================================

@dataclass
class ExpertBankConfig:
    """Configuration for the Expert Bank (System 2).

    Dynamic collection of LoRA experts with lifecycle management.
    """
    max_experts: int = 64
    default_lora_rank: int = 16
    default_lora_alpha: int = 32
    default_lora_dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
    ])

    shadow_period_steps: int = 200
    shadow_loss_convergence_threshold: float = 0.01
    shadow_centroid_stability_threshold: float = 0.02

    merge_centroid_threshold: float = 0.92
    merge_tolerance_delta: float = 0.05

    importance_weight_frequency: float = 0.3
    importance_weight_marginal: float = 0.4
    importance_weight_uniqueness: float = 0.3
    pruning_importance_threshold: float = 0.1

    dormant_quantization_bits: int = 4
    dormant_reactivation_threshold: float = 0.85

    freeze_loss_convergence_window: int = 50
    freeze_loss_convergence_threshold: float = 0.005

    domain_adapter_rank: int = 8
    domain_adapter_alpha: int = 16

    novelty_detection_threshold: float = 0.3

    checkpoint_dir: str = "morpheus_state/experts"


# =============================================================================
# System 3: Fast Buffer ("Hippocampus")
# =============================================================================

@dataclass
class FastBufferConfig:
    """Configuration for the Fast Adaptation Buffer (System 3).

    A small, highly plastic LoRA adapter that absorbs incoming data
    immediately without touching the rest of the system.
    """
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.1
    target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
    ])
    learning_rate: float = 1e-3
    max_capacity_steps: int = 500
    pattern_separation_noise: float = 0.05
    overwrite_interval: int = 1000
    checkpoint_dir: str = "morpheus_state/buffer"


# =============================================================================
# System 4: Consolidation Engine ("Sleep")
# =============================================================================

@dataclass
class ConsolidationConfig:
    """Configuration for the Consolidation Engine (System 4).

    Handles self-rehearsal, interleaved consolidation, structural
    distillation, and expert lifecycle management.
    """
    # 4a: Self-rehearsal
    rehearsal_temperature: float = 1.2
    rehearsal_top_p: float = 0.95
    rehearsal_max_tokens: int = 256
    rehearsal_batch_size: int = 16
    rehearsal_samples_per_expert: int = 100
    tail_coverage_min_p: float = 0.02

    coreset_fraction: float = 0.001
    coreset_max_samples: int = 5000
    coreset_selection: str = "facility_location"

    frozen_reference_schedule: str = "logarithmic"
    max_frozen_references: int = 10
    kl_divergence_epsilon: float = 0.1

    discriminator_hidden_dim: int = 256
    discriminator_layers: int = 3

    # 4b: Interleaved consolidation
    default_rehearsal_ratio: float = 0.5
    consolidation_learning_rate: float = 5e-5
    consolidation_steps: int = 200

    # 4c: Structural distillation
    distillation_interval_cycles: int = 10
    distillation_learning_rate: float = 1e-5
    distillation_steps: int = 500

    # 4d: Expert lifecycle
    freeze_loss_convergence_window: int = 50
    freeze_loss_convergence_threshold: float = 0.005
    spawn_loss_threshold: float = 2.0
    spawn_uncertainty_threshold: float = 0.8

    checkpoint_dir: str = "morpheus_state/consolidation"


# =============================================================================
# System 5: Knowledge Store ("Episodic/Declarative Memory")
# =============================================================================

@dataclass
class KnowledgeStoreConfig:
    """Configuration for the Explicit Knowledge Store (System 5).

    Non-parametric store for discrete facts with graduated factuality
    override mechanism.
    """
    store_type: str = "faiss"
    embedding_dim: int = 768
    index_type: str = "IVFFlat"
    n_clusters: int = 64

    factuality_threshold_high: float = 0.8
    # Raised from 0.3: CF conflict queries hit sim=1.0; TriviaQA D_control
    # queries hit sim≤0.619. Setting low>0.62 puts all D_control in
    # parametric_freedom (no CF injection → FR≈0%) while leaving CF in
    # hard_override (bypass → ESR≈98%).
    factuality_threshold_low: float = 0.65
    novelty_threshold_shift: float = 0.15
    self_consistency_samples: int = 3

    # Authoritative-override bypass: when zone=hard_override and max_sim
    # exceeds this threshold, skip LLM generation and return the stored
    # object_value verbatim. This enforces arch §236 ("factual content
    # must come from System 5") as a hard architectural hierarchy instead
    # of a soft prompt-level nudge that Mistral tends to ignore in favor
    # of parametric belief. Set to a value <= 1.0 to enable; set > 1.0
    # to disable (falls back to prompt-level injection only).
    direct_answer_threshold: float = 0.95

    conflict_training_fraction: float = 0.1

    # Path to a trained FactualityClassifier checkpoint directory.
    # When set, inference.py should use the classifier score instead of
    # max_sim as the factuality_score passed to assess_factuality.
    classifier_path: str | None = None

    store_dir: str = "morpheus_state/knowledge_store"


# =============================================================================
# System 6: Meta-Controller ("Prefrontal Cortex")
# =============================================================================

@dataclass
class MetaControllerConfig:
    """Configuration for the Meta-Learning Controller (System 6).

    Orchestrates all other subsystems using RL on summary statistics
    with a heuristic baseline + learned residual.
    """
    ensemble_size: int = 3
    heuristic_weight: float = 0.7
    residual_hidden_dim: int = 128
    residual_layers: int = 2

    state_dim: int = 16
    action_dim: int = 8
    reward_performance_weight: float = 0.5
    reward_retention_weight: float = 0.3
    reward_efficiency_weight: float = 0.2

    irreversible_majority_threshold: float = 0.67

    staged_action_validation_window: int = 100
    staged_action_degradation_threshold: float = 0.05

    anomaly_detection_window: int = 50
    anomaly_z_threshold: float = 3.0

    probe_interval_steps: int = 100
    probe_set_size: int = 50

    checkpoint_dir: str = "morpheus_state/meta_controller"


# =============================================================================
# Prototype Router
# =============================================================================

@dataclass
class PrototypeRouterConfig:
    """Configuration for the non-parametric prototype router."""
    projection_dim: int = 256
    similarity_threshold: float = 0.55
    top_k: int = 3
    ema_decay: float = 0.99

    hierarchical_routing: bool = True
    coarse_clusters: int = 8
    recluster_interval: int = 50

    hub_detection_threshold: float = 3.0
    hub_correction_factor: float = 0.5

    embedding_model_path: str | None = None
    use_gpu: bool = True


# =============================================================================
# Top-Level MORPHEUS Config
# =============================================================================

@dataclass
class MorpheusConfig:
    """Top-level configuration aggregating all MORPHEUS subsystems.

    MORPHEUS: Multi-timescale Orchestrated Rehearsal with Prototype-routed
    Hierarchical Expert Unification System.

    A cognitive multi-system architecture for continual learning that
    separates knowledge into subsystems operating at different timescales:
    - Fast (seconds):  Buffer absorbs streaming data
    - Medium (hours):  Experts consolidate via interleaved rehearsal
    - Slow (weeks):    Core evolves via structural distillation
    - Meta (episodes): Controller optimizes the learning process
    """
    stable_core: StableCoreConfig = field(default_factory=StableCoreConfig)
    expert_bank: ExpertBankConfig = field(default_factory=ExpertBankConfig)
    fast_buffer: FastBufferConfig = field(default_factory=FastBufferConfig)
    consolidation: ConsolidationConfig = field(default_factory=ConsolidationConfig)
    knowledge_store: KnowledgeStoreConfig = field(default_factory=KnowledgeStoreConfig)
    meta_controller: MetaControllerConfig = field(default_factory=MetaControllerConfig)
    router: PrototypeRouterConfig = field(default_factory=PrototypeRouterConfig)

    random_seed: int = 42
    state_dir: str = "morpheus_state"

    def validate(self) -> list[str]:
        """Validate configuration consistency across subsystems."""
        warnings = []

        if self.fast_buffer.learning_rate <= self.consolidation.consolidation_learning_rate:
            warnings.append(
                "Buffer learning rate should be >> consolidation LR "
                f"(buffer={self.fast_buffer.learning_rate}, "
                f"consolidation={self.consolidation.consolidation_learning_rate})"
            )

        if self.consolidation.default_rehearsal_ratio < 0.3:
            warnings.append(
                "Low rehearsal ratio increases forgetting risk "
                f"(ratio={self.consolidation.default_rehearsal_ratio})"
            )

        if self.meta_controller.ensemble_size < 2:
            warnings.append(
                "Ensemble size < 2 removes irreversibility safety "
                f"(size={self.meta_controller.ensemble_size})"
            )

        if self.stable_core.cka_threshold > 0.1:
            warnings.append(
                "High CKA threshold allows large core representation shifts "
                f"(threshold={self.stable_core.cka_threshold})"
            )

        return warnings
