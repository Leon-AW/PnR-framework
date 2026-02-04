#!/bin/bash
# Setup script for NVIDIA Blackwell GPUs (sm_120)
# ================================================
# 
# The RTX PRO 6000 Blackwell uses sm_120 compute capability which requires
# PyTorch with CUDA 12.8+ support. Standard conda PyTorch doesn't support this yet.
#
# This script installs PyTorch nightly which includes Blackwell support.

set -e

echo "==================================================="
echo "Setting up PyTorch for NVIDIA Blackwell GPUs (sm_120)"
echo "==================================================="

# Check if conda environment is activated
if [[ -z "$CONDA_DEFAULT_ENV" ]]; then
    echo "ERROR: No conda environment activated."
    echo "Run: conda activate pnr"
    exit 1
fi

echo "Using conda environment: $CONDA_DEFAULT_ENV"

# Uninstall current PyTorch
echo ""
echo "[1/3] Removing existing PyTorch..."
pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
conda remove -y pytorch torchvision torchaudio pytorch-cuda 2>/dev/null || true

# Install PyTorch nightly with CUDA 12.8 support
echo ""
echo "[2/3] Installing PyTorch nightly with CUDA 12.8 (Blackwell support)..."
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# Verify installation
echo ""
echo "[3/3] Verifying installation..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'GPU {i}: {props.name} (sm_{props.major}{props.minor})')
"

echo ""
echo "==================================================="
echo "Setup complete!"
echo ""
echo "If you see 'sm_120' in the GPU list above, Blackwell is supported."
echo ""
echo "Test the model loading with:"
echo "  python -c \"from src.models.core import PatchAndRouteLLM; llm = PatchAndRouteLLM(); llm.load_frozen_foundation()\""
echo "==================================================="
