#!/usr/bin/env python3
"""
GPU Validation Script
=====================

Run this BEFORE submitting training jobs to verify GPU setup is correct.

Usage:
    python validate_gpu_setup.py                    # Check all available GPUs
    python validate_gpu_setup.py --target-devices 0 1   # Check specific devices
    python validate_gpu_setup.py --dry-run          # Full validation + memory estimate

This script validates:
1. CUDA availability and device count
2. MIG mode detection
3. Memory requirements for DeepSeek-R1-14B training
4. Distributed training compatibility
"""

import argparse
import os
import sys
from pathlib import Path

def check_cuda_available():
    """Check if CUDA is available."""
    try:
        import torch
        if not torch.cuda.is_available():
            print("[FAIL] CUDA is not available")
            print("       Check: module load cuda or nvidia-smi")
            return False
        print(f"[OK] CUDA available: {torch.version.cuda}")
        return True
    except ImportError:
        print("[FAIL] PyTorch not installed")
        return False


def check_gpu_devices(target_devices=None):
    """Check GPU devices and their properties."""
    import torch

    device_count = torch.cuda.device_count()
    print(f"\n[INFO] Found {device_count} CUDA device(s)")

    if device_count == 0:
        print("[FAIL] No CUDA devices found!")
        print("       If running on SLURM, check CUDA_VISIBLE_DEVICES and --gres settings")
        return False, []

    devices = []
    mig_detected = False

    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / 1024**3

        # Detect MIG instances (typically have "MIG" in name or specific memory sizes)
        is_mig = "MIG" in props.name or memory_gb < 30  # MIG instances are typically < 30GB
        if is_mig:
            mig_detected = True

        device_info = {
            "index": i,
            "name": props.name,
            "memory_gb": memory_gb,
            "compute_capability": f"{props.major}.{props.minor}",
            "is_mig": is_mig,
        }
        devices.append(device_info)

        status = "[MIG]" if is_mig else "[GPU]"
        print(f"  {status} Device {i}: {props.name}")
        print(f"       Memory: {memory_gb:.2f} GB")
        print(f"       Compute: {props.major}.{props.minor}")

    if mig_detected:
        print(f"\n[INFO] MIG mode detected on some/all devices")
        print("       MIG instances appear as separate CUDA devices")

    # Validate target devices if specified
    if target_devices is not None:
        for d in target_devices:
            if d >= device_count:
                print(f"\n[FAIL] Target device {d} does not exist (only {device_count} devices)")
                return False, devices
        print(f"\n[OK] Target devices {target_devices} are valid")

    return True, devices


def estimate_memory_requirements():
    """Estimate memory requirements for DeepSeek-R1-14B training."""
    print("\n" + "="*60)
    print("MEMORY REQUIREMENTS ESTIMATE")
    print("="*60)
    print("Model: DeepSeek-R1-Distill-Qwen-14B (14B parameters)")
    print()
    print("4-bit Quantization (INT4):")
    print("  - Model weights:        ~8 GB")
    print("  - LoRA adapters:        ~0.5 GB")
    print("  - Optimizer states:     ~1-2 GB (paged_adamw_8bit)")
    print("  - Activations (bs=1):   ~2-4 GB (gradient checkpointing)")
    print("  - KV cache:             ~1-2 GB (max_seq=1024)")
    print("  - PyTorch overhead:     ~1-2 GB")
    print("  " + "-"*40)
    print("  TOTAL ESTIMATED:        ~14-18 GB")
    print()
    print("  Minimum VRAM required:  20 GB (tight)")
    print("  Recommended VRAM:       24 GB")
    print()
    print("8-bit Quantization (INT8):")
    print("  - Model weights:        ~14 GB")
    print("  - Total estimated:      ~20-24 GB")
    print("  - Minimum VRAM:         24 GB (tight)")
    print("  - Recommended VRAM:     32 GB")
    print("="*60)


def check_memory_sufficiency(devices, required_gb=20):
    """Check if devices have sufficient memory."""
    print(f"\n[CHECK] Validating memory requirements ({required_gb} GB minimum)...")

    all_sufficient = True
    for dev in devices:
        if dev["memory_gb"] < required_gb:
            print(f"  [WARN] Device {dev['index']}: {dev['memory_gb']:.1f} GB < {required_gb} GB")
            all_sufficient = False
        else:
            headroom = dev["memory_gb"] - required_gb
            print(f"  [OK] Device {dev['index']}: {dev['memory_gb']:.1f} GB ({headroom:.1f} GB headroom)")

    if not all_sufficient:
        print("\n[WARN] Some devices may have insufficient memory for training")
        print("       Recommendations:")
        print("       - Use batch_size=1")
        print("       - Use gradient_accumulation=16 or higher")
        print("       - Use max_seq_length=1024 or lower")
        print("       - Use 4-bit quantization (--quantization int4)")

    return all_sufficient


def check_distributed_compatibility(devices, num_processes=None):
    """Check if configuration is suitable for distributed training."""
    print("\n" + "="*60)
    print("DISTRIBUTED TRAINING COMPATIBILITY")
    print("="*60)

    device_count = len(devices)

    if num_processes is None:
        num_processes = device_count

    print(f"Devices available: {device_count}")
    print(f"Processes requested: {num_processes}")

    if num_processes > device_count:
        print(f"\n[FAIL] More processes ({num_processes}) than devices ({device_count})")
        print("       This will cause 'invalid device ordinal' errors!")
        print("       Solution: Reduce processes or use single-device mode")
        return False

    # Check for MIG mode
    mig_devices = [d for d in devices if d.get("is_mig", False)]
    if mig_devices:
        print(f"\n[INFO] MIG instances detected ({len(mig_devices)} of {device_count})")
        print("       MIG requires special handling for multi-process training")
        print()
        print("       Recommended approach for MIG:")
        print("       1. Use SINGLE device training (--target_devices 0)")
        print("       2. Or use accelerate with proper MIG config")
        print()
        if num_processes > 1:
            print("[WARN] Multi-process training on MIG requires careful setup")
            print("       Consider using single-device mode for reliability")

    print(f"\n[OK] Configuration appears compatible")
    return True


def check_environment_variables():
    """Check relevant environment variables."""
    print("\n" + "="*60)
    print("ENVIRONMENT VARIABLES")
    print("="*60)

    important_vars = [
        "CUDA_VISIBLE_DEVICES",
        "SLURM_JOB_ID",
        "SLURM_JOB_GPUS",
        "SLURM_GPUS_ON_NODE",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "RANK",
        "PYTORCH_CUDA_ALLOC_CONF",
    ]

    for var in important_vars:
        value = os.environ.get(var, "<not set>")
        if value != "<not set>":
            print(f"  {var}={value}")
        else:
            print(f"  {var}: (not set)")

    # Warn about potential issues
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd:
        visible_count = len(cvd.split(","))
        print(f"\n[INFO] CUDA_VISIBLE_DEVICES restricts to {visible_count} device(s)")


def test_model_loading(device_id=0):
    """Test if model can be loaded on the specified device."""
    print(f"\n" + "="*60)
    print(f"MODEL LOADING TEST (Device {device_id})")
    print("="*60)

    try:
        import torch
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig

        print("Testing 4-bit model load (just config, not full load)...")

        # Just test if we can create the config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        print(f"  [OK] BitsAndBytes config created successfully")
        print(f"  [OK] BF16 supported: {torch.cuda.is_bf16_supported()}")

        # Check available memory
        free_mem = torch.cuda.mem_get_info(device_id)[0] / 1024**3
        total_mem = torch.cuda.mem_get_info(device_id)[1] / 1024**3
        print(f"  [OK] Device {device_id} memory: {free_mem:.1f}/{total_mem:.1f} GB free")

        return True

    except Exception as e:
        print(f"  [FAIL] {type(e).__name__}: {e}")
        return False


def generate_recommendations(devices):
    """Generate training recommendations based on hardware."""
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)

    mig_devices = [d for d in devices if d.get("is_mig", False)]
    min_memory = min(d["memory_gb"] for d in devices) if devices else 0

    if mig_devices:
        print("\nFor MIG-enabled GPUs:")
        print("-" * 40)
        print("1. Use SINGLE-DEVICE training mode (recommended):")
        print()
        print("   python train_rag_baseline.py \\")
        print("       --data_path data/archive.json \\")
        print("       --docs_path data/documents/ \\")
        print("       --target_devices 0 \\")
        print("       --batch_size 1 \\")
        print("       --gradient_accumulation 16 \\")
        print("       --max_seq_length 1024")
        print()
        print("2. SLURM configuration for single MIG instance:")
        print()
        print("   #SBATCH --gres=gpu:1")
        print("   #SBATCH --mem=32G")
        print()
        print("3. If you need multi-GPU, use data-parallel with accelerate:")
        print("   (See accelerate_config_mig.yaml in project)")

    else:
        print("\nFor standard GPUs:")
        print("-" * 40)
        if min_memory >= 40:
            print("1. Can use larger batch sizes:")
            print("   --batch_size 2 --gradient_accumulation 8")
        else:
            print("1. Memory-optimized settings:")
            print("   --batch_size 1 --gradient_accumulation 16")

    print()
    print("Memory optimization flags to add to SLURM script:")
    print("   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
    print()


def main():
    parser = argparse.ArgumentParser(description="Validate GPU setup for training")
    parser.add_argument("--target-devices", type=int, nargs="+",
                        help="Specific device IDs to validate")
    parser.add_argument("--dry-run", action="store_true",
                        help="Full validation with memory estimates")
    parser.add_argument("--test-load", action="store_true",
                        help="Test model loading (takes ~1 min)")
    args = parser.parse_args()

    print("="*60)
    print("GPU VALIDATION FOR PnR-FRAMEWORK TRAINING")
    print("="*60)

    # Run all checks
    all_passed = True

    if not check_cuda_available():
        print("\n[ABORT] CUDA not available. Cannot continue.")
        sys.exit(1)

    ok, devices = check_gpu_devices(args.target_devices)
    if not ok:
        all_passed = False

    check_environment_variables()

    if args.dry_run:
        estimate_memory_requirements()

    check_memory_sufficiency(devices, required_gb=20)

    check_distributed_compatibility(devices)

    if args.test_load and devices:
        device_id = args.target_devices[0] if args.target_devices else 0
        if not test_model_loading(device_id):
            all_passed = False

    generate_recommendations(devices)

    print("\n" + "="*60)
    if all_passed:
        print("[SUCCESS] Validation passed - ready for training")
    else:
        print("[WARNING] Some checks failed - review recommendations above")
    print("="*60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
