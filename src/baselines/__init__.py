"""
Baselines Package
=================

Comparison baselines for the Patch-and-Route evaluation:

- xlora:  X-LoRA soft gating (Buehler & Buehler 2024)
- rledit: RLEdit RL hypernetwork (knowledge editing baseline)
- recipe: RECIPE retrieval-augmented continuous prompts (Chen et al. 2024)
"""

from src.baselines.xlora import XLoRAInference, XLoRAConfig
from src.baselines.rledit import RLEditInference, RLEditConfig
from src.baselines.recipe import RECIPEInference, RECIPEConfig

__all__ = [
    "XLoRAInference",
    "XLoRAConfig",
    "RLEditInference",
    "RLEditConfig",
    "RECIPEInference",
    "RECIPEConfig",
]
