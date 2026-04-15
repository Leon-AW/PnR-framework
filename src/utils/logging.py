"""
Logging Utilities
=================

Centralized logging configuration for the Patch-and-Route framework.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TextIO


# Default format for framework logs
DEFAULT_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logger(
    name: str = "pnr",
    level: int | str = logging.INFO,
    log_file: str | Path | None = None,
    format_string: str = DEFAULT_FORMAT,
    stream: TextIO = sys.stdout,
) -> logging.Logger:
    """Set up and configure a logger for the framework.
    
    Args:
        name: Logger name (use "pnr" for framework root).
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional file path for logging output.
        format_string: Log message format.
        stream: Output stream for console handler.
        
    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    
    # Convert string level to int if needed
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(format_string, DEFAULT_DATE_FORMAT)
    
    # Console handler
    console_handler = logging.StreamHandler(stream)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "pnr") -> logging.Logger:
    """Get an existing logger by name.
    
    Args:
        name: Logger name.
        
    Returns:
        Logger instance.
    """
    return logging.getLogger(name)


def configure_framework_logging(
    level: int | str = logging.INFO,
    log_file: str | Path | None = None,
) -> None:
    """Configure logging for the entire Patch-and-Route framework.
    
    Sets up consistent logging across all framework modules.
    
    Args:
        level: Global logging level.
        log_file: Optional file for log output.
    """
    # Configure root framework logger
    setup_logger("pnr", level=level, log_file=log_file)

    # Configure the root logger so that training scripts using
    # logging.getLogger(__name__) (where __name__ == "__main__") are captured.
    # Without this, all INFO calls in train_*.py are silently dropped.
    setup_logger("root", level=level, log_file=log_file)
    # logging.getLogger("root") is NOT the root logger — use "" for that
    root = logging.getLogger()
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    root.setLevel(level)
    if not root.handlers:
        formatter = logging.Formatter(DEFAULT_FORMAT, DEFAULT_DATE_FORMAT)
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(formatter)
        root.addHandler(ch)
        if log_file is not None:
            from pathlib import Path as _Path
            fh = logging.FileHandler(_Path(log_file))
            fh.setLevel(level)
            fh.setFormatter(formatter)
            root.addHandler(fh)

    # Configure submodule loggers
    for module in ["pnr.data", "pnr.models", "pnr.training", "pnr.utils"]:
        logger = logging.getLogger(module)
        logger.setLevel(level)

    # Also configure src.* loggers (alternative import path)
    for module in ["src", "src.data", "src.models", "src.training", "src.utils"]:
        logger = logging.getLogger(module)
        logger.setLevel(level)

