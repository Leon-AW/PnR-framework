"""
Configuration Utilities
=======================

Handles loading, saving, and managing configurations for the Patch-and-Route framework.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")


def load_config(
    config_path: str | Path,
    config_class: type[T] | None = None,
) -> dict[str, Any] | T:
    """Load configuration from a JSON file.
    
    Args:
        config_path: Path to configuration file.
        config_class: Optional dataclass to instantiate with loaded values.
        
    Returns:
        Loaded configuration as dict or dataclass instance.
        
    Raises:
        FileNotFoundError: If config file doesn't exist.
        json.JSONDecodeError: If file is not valid JSON.
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = json.load(f)
    
    if config_class is not None and is_dataclass(config_class):
        return config_class(**config_dict)
    
    return config_dict


def save_config(
    config: dict[str, Any] | Any,
    config_path: str | Path,
    indent: int = 2,
) -> Path:
    """Save configuration to a JSON file.
    
    Args:
        config: Configuration dict or dataclass instance.
        config_path: Output file path.
        indent: JSON indentation level.
        
    Returns:
        Path to saved configuration file.
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert dataclass to dict if needed
    if is_dataclass(config) and not isinstance(config, dict):
        config = asdict(config)
    
    # Handle non-serializable types
    def serialize(obj: Any) -> Any:
        if hasattr(obj, "value"):  # Enum
            return obj.value
        if hasattr(obj, "__str__"):
            return str(obj)
        return obj
    
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=indent, default=serialize)
    
    return config_path

