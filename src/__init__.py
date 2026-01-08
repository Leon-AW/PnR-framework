"""
Patch-and-Route Framework
=========================

A modular framework for Continual Learning in Enterprise LLMs.

This framework implements the "Patch-and-Route" architecture described in:
"A Modular 'Patch-and-Route' Framework for Continual Learning in Enterprise LLMs"

Core Concepts:
- Frozen Foundation: Base LLM with frozen parameters (e.g., Mistral-7B)
- Expert Pool: Collection of domain-specific LoRA adapters
- Knowledge Router: Dynamic routing mechanism for adapter selection

Author: Leon Wagner
"""

__version__ = "0.1.0"
__author__ = "Leon Wagner"

