"""
Routing Module
==============

Implements the Intelligent Dispatcher for the Patch-and-Route framework.

Components:
- BaseRouter: Abstract base class for routing strategies
- CentroidRouter: Time-Aware Centroid Router with Source-Replay
- AdapterManifest: Registry for adapter metadata and centroids
- SourceReplayStore: FAISS-based retrieval for older adapter data

Reference: Section 4.4 of the Master's Thesis Exposé
"""

from .base import BaseRouter, RoutingResult, AdapterMatch
from .centroid_router import CentroidRouter
from .manifest import AdapterManifest, AdapterEntry
from .source_replay import SourceReplayStore

__all__ = [
    "BaseRouter",
    "RoutingResult",
    "AdapterMatch",
    "CentroidRouter",
    "AdapterManifest",
    "AdapterEntry",
    "SourceReplayStore",
]

