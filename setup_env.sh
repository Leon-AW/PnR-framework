#!/bin/bash
# ==============================================================================
# Patch-and-Route Framework - Environment Setup Script
# ==============================================================================
#
# This script sets up a miniconda environment for the PnR framework.
#
# Usage:
#   chmod +x setup_env.sh
#   ./setup_env.sh
#
# Requirements:
#   - Miniconda or Anaconda installed
#   - NVIDIA GPU with CUDA support (for training)
# ==============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=============================================${NC}"
echo -e "${GREEN}  Patch-and-Route Framework Setup${NC}"
echo -e "${GREEN}=============================================${NC}"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo -e "${RED}Error: conda not found!${NC}"
    echo ""
    echo "Please install Miniconda first:"
    echo "  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    echo "  bash Miniconda3-latest-Linux-x86_64.sh"
    echo "  source ~/.bashrc"
    exit 1
fi

ENV_NAME="pnr"

# Check if environment already exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo -e "${YELLOW}Environment '${ENV_NAME}' already exists.${NC}"
    read -p "Do you want to update it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${GREEN}Updating environment...${NC}"
        conda env update -f environment.yml --prune
    else
        echo "Skipping environment update."
    fi
else
    echo -e "${GREEN}Creating new conda environment '${ENV_NAME}'...${NC}"
    conda env create -f environment.yml
fi

echo ""
echo -e "${GREEN}=============================================${NC}"
echo -e "${GREEN}  Setup Complete!${NC}"
echo -e "${GREEN}=============================================${NC}"
echo ""
echo "To activate the environment, run:"
echo -e "  ${YELLOW}conda activate ${ENV_NAME}${NC}"
echo ""
echo "To start training:"
echo -e "  ${YELLOW}python train_base_adapter.py --help${NC}"
echo ""
echo "To verify GPU availability:"
echo -e "  ${YELLOW}python -c \"import torch; print(f'CUDA available: {torch.cuda.is_available()}')\"${NC}"
echo ""

