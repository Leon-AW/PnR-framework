"""
LoRA Adapter Merging
====================

Merges a trained LoRA adapter with its base model to create
a standalone model for inference.

Usage:
    python -m src.inference.merge_adapter \
        --adapter_path checkpoints/QM_rag/checkpoint-1000 \
        --output_path checkpoints/QM_rag/merged
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import torch

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class MergeConfig:
    """Configuration for adapter merging.

    Attributes:
        base_model: Base model name or path
        adapter_path: Path to LoRA adapter checkpoint
        output_path: Path to save merged model
        torch_dtype: Data type for model weights
        device_map: Device mapping for loading
        push_to_hub: Push merged model to HuggingFace Hub
        hub_model_id: Model ID for Hub upload
    """
    base_model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    adapter_path: str = "checkpoints/QM_rag/checkpoint-1000"
    output_path: str = "checkpoints/QM_rag/merged"
    torch_dtype: str = "float16"
    device_map: str = "auto"
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None


# =============================================================================
# Merge Function
# =============================================================================

def merge_adapter(config: MergeConfig) -> Path:
    """Merge LoRA adapter with base model.

    Args:
        config: Merge configuration

    Returns:
        Path to merged model directory
    """
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output_path = Path(config.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Map dtype string to torch dtype
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map.get(config.torch_dtype, torch.float16)

    # Handle CPU-only mode to avoid CUDA initialization issues
    device_map = config.device_map
    if device_map == "cpu":
        logger.info("Running in CPU-only mode (slower but avoids CUDA issues)")
        device_map = {"": "cpu"}
        # Force float32 on CPU for stability
        torch_dtype = torch.float32

    logger.info(f"Loading base model: {config.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    logger.info(f"Loading tokenizer from: {config.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model,
        trust_remote_code=True,
    )

    logger.info(f"Loading LoRA adapter: {config.adapter_path}")
    model = PeftModel.from_pretrained(
        base_model,
        config.adapter_path,
        torch_dtype=torch_dtype,
    )

    logger.info("Merging adapter with base model...")
    merged_model = model.merge_and_unload()

    logger.info(f"Saving merged model to: {output_path}")
    merged_model.save_pretrained(
        output_path,
        safe_serialization=True,
    )
    tokenizer.save_pretrained(output_path)

    # Optionally push to Hub
    if config.push_to_hub and config.hub_model_id:
        logger.info(f"Pushing to HuggingFace Hub: {config.hub_model_id}")
        merged_model.push_to_hub(config.hub_model_id)
        tokenizer.push_to_hub(config.hub_model_id)

    logger.info("Merge complete!")
    return output_path


# =============================================================================
# Auto-detect Base Model
# =============================================================================

def detect_base_model(adapter_path: Union[str, Path]) -> Optional[str]:
    """Try to detect the base model from adapter config.

    Args:
        adapter_path: Path to adapter

    Returns:
        Base model name if detected, None otherwise
    """
    import json

    adapter_path = Path(adapter_path)
    config_path = adapter_path / "adapter_config.json"

    if config_path.exists():
        with open(config_path, "r") as f:
            config = json.load(f)
            return config.get("base_model_name_or_path")

    return None


# =============================================================================
# CLI
# =============================================================================

def main():
    """Command-line interface for adapter merging."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter with base model"
    )
    parser.add_argument(
        "--adapter_path", "-a",
        required=True,
        help="Path to LoRA adapter checkpoint"
    )
    parser.add_argument(
        "--output_path", "-o",
        required=True,
        help="Path to save merged model"
    )
    parser.add_argument(
        "--base_model", "-b",
        help="Base model name (auto-detected if not specified)"
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "float32", "bfloat16"],
        help="Model data type"
    )
    parser.add_argument(
        "--device_map",
        default="auto",
        help="Device map for loading"
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push merged model to HuggingFace Hub"
    )
    parser.add_argument(
        "--hub_model_id",
        help="Model ID for Hub upload"
    )

    args = parser.parse_args()

    # Auto-detect base model if not specified
    base_model = args.base_model
    if not base_model:
        base_model = detect_base_model(args.adapter_path)
        if base_model:
            logger.info(f"Auto-detected base model: {base_model}")
        else:
            base_model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
            logger.warning(f"Could not detect base model, using default: {base_model}")

    config = MergeConfig(
        base_model=base_model,
        adapter_path=args.adapter_path,
        output_path=args.output_path,
        torch_dtype=args.dtype,
        device_map=args.device_map,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
    )

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    merge_adapter(config)


if __name__ == "__main__":
    main()
