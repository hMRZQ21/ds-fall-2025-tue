#!/bin/bash
# Installation script for Data Science Development Environment
# This script properly installs CUDA-enabled packages

set -e  # Exit on error

echo "========================================="
echo "Data Science Environment Setup"
echo "========================================="
echo ""

# Step 1: Install PyTorch with CUDA support
echo "Step 1: Installing PyTorch with CUDA 12.4 support (torch>=2.6)..."
pip install "torch>=2.6,<2.7" "torchvision>=0.21,<0.22" "torchaudio>=2.6,<2.7" --index-url https://download.pytorch.org/whl/cu124
pip install diffusers["torch"]

echo ""
echo "Step 2: Installing TensorFlow with GPU support..."
# TensorFlow will be installed via requirements.txt

echo ""
echo "Step 3: Installing all other packages from requirements.txt..."
pip install -r requirements.txt

echo ""
echo "========================================="
echo "Installation complete!"
echo "========================================="
echo ""
echo "To verify GPU support:"
echo "  python -c \"import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')\""
echo "  python -c \"import tensorflow as tf; print(f'TensorFlow: {tf.__version__}'); print(f'GPU devices: {tf.config.list_physical_devices(\"GPU\")}')\""

