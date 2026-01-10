"""
Training Module
===============

Training engines and utilities for the Patch-and-Route framework.
Supports streaming datasets and parameter-efficient fine-tuning.

Key Components:
- PatchAndRouteTrainer: Core training class for Expert Adapters
- train_adapter: Adapter-aware training function (primary interface)
- train_multiple_adapters: Sequential batch training of multiple adapters
"""

from src.training.trainer import (
    PatchAndRouteTrainer,
    TrainingConfig,
    train_adapter,
    train_base_expert,
    train_multiple_adapters,
)

__all__ = [
    "PatchAndRouteTrainer",
    "TrainingConfig",
    "train_adapter",
    "train_base_expert",
    "train_multiple_adapters",
]

