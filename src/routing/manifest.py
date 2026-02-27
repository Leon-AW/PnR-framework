"""
Adapter Manifest
================

Registry for adapter metadata, centroids, and source data paths.

The Manifest serves as the "offline registration" component of the Time-Aware
Centroid Router. It stores:
- Adapter metadata (ID, timestamp, type)
- Centroid vectors (mean embedding of training data)
- Paths to adapter checkpoints and source data

Key Design Decisions:
1. Centroids are stored as numpy arrays for efficient similarity computation
2. Timestamps use epoch seconds for unambiguous ordering
3. Source data paths enable the Source-Replay mechanism
4. Manifest can be serialized to JSON for persistence

Reference: Section 4.4.1 of the Master's Thesis Exposé
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AdapterEntry:
    """Entry for a single adapter in the manifest.
    
    Attributes:
        adapter_id: Unique identifier (e.g., "patch_geo_germany").
        adapter_path: Filesystem path to adapter checkpoint.
        timestamp: Training timestamp (epoch seconds).
        adapter_type: Type classification ("base", "patch_temporal", "patch_geo").
        centroid: Mean embedding vector of training data (optional until computed).
        source_data_path: Path to training data for Source-Replay.
        metadata: Additional adapter-specific metadata.
    """
    adapter_id: str
    adapter_path: str
    timestamp: float
    adapter_type: str = "unknown"
    centroid: np.ndarray | None = None
    source_data_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @property
    def has_centroid(self) -> bool:
        """Check if centroid has been computed."""
        return self.centroid is not None
    
    @property
    def timestamp_dt(self) -> datetime:
        """Get timestamp as datetime object."""
        return datetime.fromtimestamp(self.timestamp)
    
    @property
    def timestamp_str(self) -> str:
        """Get human-readable timestamp string."""
        return self.timestamp_dt.strftime("%Y-%m-%d %H:%M:%S")
    
    def to_dict(self, include_centroid: bool = False) -> dict[str, Any]:
        """Convert to dictionary for serialization.
        
        Args:
            include_centroid: Whether to include the centroid vector.
            
        Returns:
            Dictionary representation.
        """
        result = {
            "adapter_id": self.adapter_id,
            "adapter_path": self.adapter_path,
            "timestamp": self.timestamp,
            "timestamp_str": self.timestamp_str,
            "adapter_type": self.adapter_type,
            "source_data_path": self.source_data_path,
            "has_centroid": self.has_centroid,
            "metadata": self.metadata,
        }
        
        if include_centroid and self.centroid is not None:
            result["centroid"] = self.centroid.tolist()
        
        return result
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AdapterEntry:
        """Create entry from dictionary.
        
        Args:
            data: Dictionary with entry data.
            
        Returns:
            New AdapterEntry instance.
        """
        centroid = None
        if "centroid" in data and data["centroid"] is not None:
            centroid = np.array(data["centroid"], dtype=np.float32)
        
        return cls(
            adapter_id=data["adapter_id"],
            adapter_path=data["adapter_path"],
            timestamp=data["timestamp"],
            adapter_type=data.get("adapter_type", "unknown"),
            centroid=centroid,
            source_data_path=data.get("source_data_path"),
            metadata=data.get("metadata", {}),
        )


class AdapterManifest:
    """Registry for adapter metadata and centroids.
    
    The Manifest is the central registry for all adapters in the Patch-and-Route
    framework. It maintains:
    - Adapter metadata and configurations
    - Centroid vectors for similarity-based routing
    - Source data paths for the Source-Replay mechanism
    
    Example:
        ```python
        manifest = AdapterManifest()
        
        # Register adapters
        manifest.register(
            adapter_id="base_v1",
            adapter_path="checkpoints/base_v1",
            timestamp=1609459200.0,  # 2021-01-01
            adapter_type="base",
            source_data_path="data/base_training.jsonl",
        )
        
        # Update centroid after computation
        centroid = compute_centroid(training_data)
        manifest.update_centroid("base_v1", centroid)
        
        # Save to disk
        manifest.save("manifests/adapters.json")
        
        # Load from disk
        loaded = AdapterManifest.load("manifests/adapters.json")
        ```
    """
    
    def __init__(self) -> None:
        """Initialize empty manifest."""
        self._entries: dict[str, AdapterEntry] = {}
        logger.info("Initialized empty AdapterManifest")
    
    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------
    
    @property
    def adapters(self) -> list[str]:
        """Get list of registered adapter IDs."""
        return list(self._entries.keys())
    
    @property
    def num_adapters(self) -> int:
        """Get number of registered adapters."""
        return len(self._entries)
    
    @property
    def entries_with_centroids(self) -> list[AdapterEntry]:
        """Get entries that have computed centroids."""
        return [e for e in self._entries.values() if e.has_centroid]
    
    # -------------------------------------------------------------------------
    # CRUD Operations
    # -------------------------------------------------------------------------
    
    def register(
        self,
        adapter_id: str,
        adapter_path: str,
        timestamp: float,
        adapter_type: str = "unknown",
        centroid: np.ndarray | None = None,
        source_data_path: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AdapterEntry:
        """Register a new adapter.
        
        Args:
            adapter_id: Unique identifier for the adapter.
            adapter_path: Filesystem path to adapter checkpoint.
            timestamp: Training timestamp (epoch seconds).
            adapter_type: Type classification.
            centroid: Pre-computed centroid vector (optional).
            source_data_path: Path to training data for Source-Replay.
            metadata: Additional adapter-specific metadata.
            
        Returns:
            The created AdapterEntry.
            
        Raises:
            ValueError: If adapter_id already exists.
        """
        if adapter_id in self._entries:
            raise ValueError(f"Adapter '{adapter_id}' already registered. Use update() instead.")
        
        entry = AdapterEntry(
            adapter_id=adapter_id,
            adapter_path=adapter_path,
            timestamp=timestamp,
            adapter_type=adapter_type,
            centroid=centroid,
            source_data_path=source_data_path,
            metadata=metadata or {},
        )
        
        self._entries[adapter_id] = entry
        logger.info(f"Registered adapter: {adapter_id} (type={adapter_type})")
        
        return entry
    
    def get(self, adapter_id: str) -> AdapterEntry | None:
        """Get an adapter entry by ID.
        
        Args:
            adapter_id: The adapter to retrieve.
            
        Returns:
            AdapterEntry or None if not found.
        """
        return self._entries.get(adapter_id)
    
    def __getitem__(self, adapter_id: str) -> AdapterEntry:
        """Get adapter entry (raises KeyError if not found)."""
        return self._entries[adapter_id]
    
    def __contains__(self, adapter_id: str) -> bool:
        """Check if adapter is registered."""
        return adapter_id in self._entries
    
    def __iter__(self) -> Iterator[AdapterEntry]:
        """Iterate over all entries."""
        return iter(self._entries.values())
    
    def unregister(self, adapter_id: str) -> bool:
        """Remove an adapter from the manifest.
        
        Args:
            adapter_id: The adapter to remove.
            
        Returns:
            True if removed, False if not found.
        """
        if adapter_id in self._entries:
            del self._entries[adapter_id]
            logger.info(f"Unregistered adapter: {adapter_id}")
            return True
        return False
    
    def update_centroid(self, adapter_id: str, centroid: np.ndarray) -> None:
        """Update the centroid vector for an adapter.
        
        Args:
            adapter_id: The adapter to update.
            centroid: The new centroid vector.
            
        Raises:
            KeyError: If adapter not found.
        """
        if adapter_id not in self._entries:
            raise KeyError(f"Adapter '{adapter_id}' not found in manifest")
        
        self._entries[adapter_id].centroid = centroid.astype(np.float32)
        logger.info(f"Updated centroid for adapter: {adapter_id} (dim={centroid.shape[0]})")
    
    # -------------------------------------------------------------------------
    # Querying
    # -------------------------------------------------------------------------
    
    def get_by_type(self, adapter_type: str) -> list[AdapterEntry]:
        """Get all adapters of a specific type.
        
        Args:
            adapter_type: Type to filter by ("base", "patch_temporal", "patch_geo").
            
        Returns:
            List of matching entries.
        """
        return [e for e in self._entries.values() if e.adapter_type == adapter_type]
    
    def get_sorted_by_timestamp(self, descending: bool = True) -> list[AdapterEntry]:
        """Get all adapters sorted by timestamp.
        
        Args:
            descending: If True, newest first. If False, oldest first.
            
        Returns:
            Sorted list of entries.
        """
        return sorted(
            self._entries.values(),
            key=lambda e: e.timestamp,
            reverse=descending,
        )
    
    def get_centroids_matrix(self) -> tuple[np.ndarray, list[str]]:
        """Get all centroids as a matrix for batch similarity computation.
        
        Returns:
            Tuple of (centroids_matrix, adapter_ids).
            - centroids_matrix: Shape (num_adapters, embedding_dim)
            - adapter_ids: List of adapter IDs in same order as rows
            
        Raises:
            ValueError: If no adapters have centroids.
        """
        entries = self.entries_with_centroids
        
        if not entries:
            raise ValueError("No adapters have computed centroids")
        
        adapter_ids = [e.adapter_id for e in entries]
        centroids = np.vstack([e.centroid for e in entries])
        
        return centroids, adapter_ids
    
    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------
    
    def save(self, path: str | Path) -> None:
        """Save manifest to JSON file.
        
        Includes centroid vectors as lists for JSON compatibility.
        
        Args:
            path: Output file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "num_adapters": self.num_adapters,
            "adapters": {
                adapter_id: entry.to_dict(include_centroid=True)
                for adapter_id, entry in self._entries.items()
            },
        }
        
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved manifest with {self.num_adapters} adapters to {path}")
    
    @classmethod
    def load(cls, path: str | Path) -> AdapterManifest:
        """Load manifest from JSON file.
        
        Args:
            path: Input file path.
            
        Returns:
            Loaded AdapterManifest.
        """
        path = Path(path)
        
        with open(path, "r") as f:
            data = json.load(f)
        
        manifest = cls()
        
        for adapter_id, entry_data in data.get("adapters", {}).items():
            entry = AdapterEntry.from_dict(entry_data)
            manifest._entries[adapter_id] = entry
        
        logger.info(f"Loaded manifest with {manifest.num_adapters} adapters from {path}")
        
        return manifest
    
    # -------------------------------------------------------------------------
    # Auto-Discovery
    # -------------------------------------------------------------------------
    
    @classmethod
    def from_checkpoints_dir(
        cls,
        checkpoints_dir: str | Path,
        base_timestamp: float | None = None,
    ) -> AdapterManifest:
        """Auto-discover adapters from a checkpoints directory.
        
        Scans the directory for adapter checkpoints (those with adapter_config.json)
        and registers them with metadata from training_config.json.
        
        Args:
            checkpoints_dir: Directory containing adapter checkpoints.
            base_timestamp: Default timestamp if not found in config.
            
        Returns:
            Populated AdapterManifest.
        """
        checkpoints_dir = Path(checkpoints_dir)
        manifest = cls()
        
        logger.info(f"Scanning for adapters in: {checkpoints_dir}")
        
        for subdir in checkpoints_dir.iterdir():
            if not subdir.is_dir():
                continue
            
            adapter_config_path = subdir / "adapter_config.json"
            training_config_path = subdir / "training_config.json"
            
            if not adapter_config_path.exists():
                continue
            
            # Load configs
            try:
                with open(adapter_config_path) as f:
                    adapter_config = json.load(f)
                
                training_config = {}
                if training_config_path.exists():
                    with open(training_config_path) as f:
                        training_config = json.load(f)
                
                # Extract metadata
                adapter_id = training_config.get("adapter_name", subdir.name)
                adapter_type = training_config.get("adapter_type", "unknown")
                
                # Use file modification time as fallback timestamp
                timestamp = base_timestamp or subdir.stat().st_mtime
                
                manifest.register(
                    adapter_id=adapter_id,
                    adapter_path=str(subdir),
                    timestamp=timestamp,
                    adapter_type=adapter_type,
                    metadata={
                        "lora_r": adapter_config.get("r"),
                        "lora_alpha": adapter_config.get("lora_alpha"),
                        "base_model": adapter_config.get("base_model_name_or_path"),
                        "training_config": training_config,
                    },
                )
                
            except Exception as e:
                logger.warning(f"Failed to load adapter from {subdir}: {e}")
                continue
        
        logger.info(f"Discovered {manifest.num_adapters} adapters")
        
        return manifest
    
    def __repr__(self) -> str:
        """String representation."""
        return f"AdapterManifest(adapters={self.adapters})"
    
    def summary(self) -> str:
        """Get a formatted summary of the manifest."""
        lines = [
            "=" * 60,
            "ADAPTER MANIFEST",
            "=" * 60,
            f"Total adapters: {self.num_adapters}",
            f"With centroids: {len(self.entries_with_centroids)}",
            "-" * 60,
        ]
        
        for entry in self.get_sorted_by_timestamp():
            centroid_status = "✓" if entry.has_centroid else "✗"
            lines.append(
                f"  [{centroid_status}] {entry.adapter_id:30s} | "
                f"{entry.adapter_type:15s} | {entry.timestamp_str}"
            )
        
        lines.append("=" * 60)
        
        return "\n".join(lines)

