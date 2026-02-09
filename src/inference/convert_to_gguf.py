"""
GGUF Conversion
===============

Converts merged HuggingFace models to GGUF format for use with
llama.cpp and compatible inference engines.

Usage:
    python -m src.inference.convert_to_gguf \
        --model_path checkpoints/QM_rag/merged \
        --output_path checkpoints/QM_rag/gguf \
        --quantize q4_k_m

Requirements:
    - llama.cpp installed (with convert_hf_to_gguf.py and llama-quantize)
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class GGUFConfig:
    """Configuration for GGUF conversion.

    Attributes:
        model_path: Path to merged HuggingFace model
        output_path: Output directory for GGUF files
        llama_cpp_path: Path to llama.cpp directory
        quantize: Quantization type (None for F16 only)
        outtype: Output type for conversion ('f16', 'f32', 'bf16')
        vocab_type: Vocabulary type override
    """
    model_path: str = "checkpoints/QM_rag/merged"
    output_path: str = "checkpoints/QM_rag/gguf"
    llama_cpp_path: Optional[str] = None
    quantize: Optional[str] = "q4_k_m"
    outtype: str = "f16"
    vocab_type: Optional[str] = None


# =============================================================================
# Quantization Options
# =============================================================================

QUANTIZATION_INFO = {
    "q8_0": {"vram_14b": "~16GB", "quality": "Highest", "description": "8-bit quantization"},
    "q6_k": {"vram_14b": "~13GB", "quality": "Very High", "description": "6-bit K-quant"},
    "q5_k_m": {"vram_14b": "~11GB", "quality": "High", "description": "5-bit K-quant medium"},
    "q5_k_s": {"vram_14b": "~10GB", "quality": "High", "description": "5-bit K-quant small"},
    "q4_k_m": {"vram_14b": "~10GB", "quality": "Recommended", "description": "4-bit K-quant medium"},
    "q4_k_s": {"vram_14b": "~9GB", "quality": "Good", "description": "4-bit K-quant small"},
    "q4_0": {"vram_14b": "~8GB", "quality": "Acceptable", "description": "4-bit quantization"},
    "q3_k_m": {"vram_14b": "~7GB", "quality": "Moderate", "description": "3-bit K-quant medium"},
    "q2_k": {"vram_14b": "~5GB", "quality": "Low", "description": "2-bit K-quant"},
}


# =============================================================================
# Conversion Functions
# =============================================================================

def find_llama_cpp() -> Optional[Path]:
    """Try to find llama.cpp installation.

    Returns:
        Path to llama.cpp directory or None
    """
    # Check common locations
    common_paths = [
        Path.home() / "llama.cpp",
        Path("/opt/llama.cpp"),
        Path("./llama.cpp"),
        Path("../llama.cpp"),
    ]

    for path in common_paths:
        if path.exists() and (path / "convert_hf_to_gguf.py").exists():
            return path

    # Check if convert script is in PATH
    convert_script = shutil.which("convert_hf_to_gguf.py")
    if convert_script:
        return Path(convert_script).parent

    return None


def find_quantize_binary(llama_cpp_path: Optional[Path] = None) -> Optional[Path]:
    """Find the llama-quantize binary.

    Args:
        llama_cpp_path: Path to llama.cpp directory

    Returns:
        Path to quantize binary or None
    """
    # Check PATH first
    for name in ["llama-quantize", "quantize"]:
        binary = shutil.which(name)
        if binary:
            return Path(binary)

    # Check llama.cpp directory
    if llama_cpp_path:
        for subdir in ["build/bin", "build", ""]:
            for name in ["llama-quantize", "quantize"]:
                path = llama_cpp_path / subdir / name
                if path.exists():
                    return path

    return None


def convert_to_gguf(config: GGUFConfig) -> Path:
    """Convert HuggingFace model to GGUF format.

    Args:
        config: Conversion configuration

    Returns:
        Path to output GGUF file
    """
    model_path = Path(config.model_path)
    output_path = Path(config.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find llama.cpp
    llama_cpp_path = Path(config.llama_cpp_path) if config.llama_cpp_path else find_llama_cpp()
    if not llama_cpp_path:
        raise RuntimeError(
            "llama.cpp not found. Please install it or specify --llama_cpp_path.\n"
            "Installation: git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp && make"
        )

    convert_script = llama_cpp_path / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        raise RuntimeError(f"convert_hf_to_gguf.py not found at: {convert_script}")

    # Step 1: Convert to F16 GGUF
    model_name = model_path.name
    f16_output = output_path / f"{model_name}-f16.gguf"

    logger.info(f"Converting to F16 GGUF: {model_path} -> {f16_output}")

    convert_cmd = [
        "python", str(convert_script),
        str(model_path),
        "--outfile", str(f16_output),
        "--outtype", config.outtype,
    ]

    if config.vocab_type:
        convert_cmd.extend(["--vocab-type", config.vocab_type])

    # Build environment with conda lib path for llama.cpp binaries
    env = os.environ.copy()
    conda_prefix = env.get("CONDA_PREFIX", "")
    if conda_prefix:
        conda_lib = str(Path(conda_prefix) / "lib")
        env["LD_LIBRARY_PATH"] = conda_lib + ":" + env.get("LD_LIBRARY_PATH", "")

    logger.info(f"Running: {' '.join(convert_cmd)}")
    result = subprocess.run(convert_cmd, capture_output=True, text=True, env=env)

    if result.returncode != 0:
        logger.error(f"Conversion failed:\n{result.stderr}")
        raise RuntimeError(f"GGUF conversion failed: {result.stderr}")

    logger.info(f"F16 conversion complete: {f16_output}")

    # Step 2: Quantize if requested
    if not config.quantize:
        return f16_output

    quantize_binary = find_quantize_binary(llama_cpp_path)
    if not quantize_binary:
        logger.warning(
            "llama-quantize not found, skipping quantization. "
            "Build llama.cpp to enable quantization."
        )
        return f16_output

    quant_output = output_path / f"{model_name}-{config.quantize}.gguf"

    logger.info(f"Quantizing to {config.quantize}: {f16_output} -> {quant_output}")

    if config.quantize in QUANTIZATION_INFO:
        info = QUANTIZATION_INFO[config.quantize]
        logger.info(f"Quantization: {info['description']} - Quality: {info['quality']}, VRAM (14B): {info['vram_14b']}")

    quantize_cmd = [
        str(quantize_binary),
        str(f16_output),
        str(quant_output),
        config.quantize,
    ]

    logger.info(f"Running: {' '.join(quantize_cmd)}")
    result = subprocess.run(quantize_cmd, capture_output=True, text=True, env=env)

    if result.returncode != 0:
        logger.error(f"Quantization failed:\n{result.stderr}")
        raise RuntimeError(f"Quantization failed: {result.stderr}")

    logger.info(f"Quantization complete: {quant_output}")

    # Optionally remove F16 file to save space
    # f16_output.unlink()

    return quant_output


def list_quantization_options():
    """Print available quantization options."""
    print("\nAvailable quantization types:")
    print("-" * 70)
    print(f"{'Type':<10} {'VRAM (14B)':<12} {'Quality':<15} {'Description'}")
    print("-" * 70)
    for quant_type, info in QUANTIZATION_INFO.items():
        print(f"{quant_type:<10} {info['vram_14b']:<12} {info['quality']:<15} {info['description']}")
    print("-" * 70)
    print("\nRecommended: q4_k_m (best balance of quality and VRAM usage)")


# =============================================================================
# CLI
# =============================================================================

def main():
    """Command-line interface for GGUF conversion."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert HuggingFace model to GGUF format"
    )
    parser.add_argument(
        "--model_path", "-m",
        help="Path to merged HuggingFace model"
    )
    parser.add_argument(
        "--output_path", "-o",
        help="Output directory for GGUF files"
    )
    parser.add_argument(
        "--llama_cpp_path",
        help="Path to llama.cpp directory"
    )
    parser.add_argument(
        "--quantize", "-q",
        default="q4_k_m",
        help="Quantization type (default: q4_k_m, use 'none' for F16 only)"
    )
    parser.add_argument(
        "--outtype",
        default="f16",
        choices=["f16", "f32", "bf16"],
        help="Output type for initial conversion"
    )
    parser.add_argument(
        "--vocab_type",
        help="Vocabulary type override"
    )
    parser.add_argument(
        "--list_quantizations",
        action="store_true",
        help="List available quantization options"
    )

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    if args.list_quantizations:
        list_quantization_options()
        return

    if not args.model_path or not args.output_path:
        parser.error("--model_path/-m and --output_path/-o are required for conversion")

    config = GGUFConfig(
        model_path=args.model_path,
        output_path=args.output_path,
        llama_cpp_path=args.llama_cpp_path,
        quantize=None if args.quantize.lower() == "none" else args.quantize,
        outtype=args.outtype,
        vocab_type=args.vocab_type,
    )

    output_file = convert_to_gguf(config)
    print(f"\nOutput: {output_file}")


if __name__ == "__main__":
    main()
