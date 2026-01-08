"""
Training Module
===============

Training engines and utilities for the Patch-and-Route framework.
Supports streaming datasets and parameter-efficient fine-tuning.
"""

from .trainer import PatchAndRouteTrainer, TrainingConfig

__all__ = ["PatchAndRouteTrainer", "TrainingConfig"]

