#!/bin/bash
"""
Convenience script to train DINOv3 ViT-Base with optimized settings
"""

set -e

echo "🚀 Training DINOv3 ViT-Base Alpha Matting Model"
echo "=============================================="

# Check if config exists
CONFIG_FILE="configs/dinov3_alpha_config.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Config file not found: $CONFIG_FILE"
    exit 1
fi

# Set environment variables for better performance
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
export CUDA_LAUNCH_BLOCKING=0

# Detect number of GPUs
if command -v nvidia-smi &> /dev/null; then
    NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
    echo "📊 Detected $NUM_GPUS GPU(s)"
else
    NUM_GPUS=1
    echo "⚠️  nvidia-smi not found, assuming 1 GPU"
fi

# Run training with optimizations
if [ $NUM_GPUS -gt 1 ]; then
    echo "🔄 Starting multi-GPU training with $NUM_GPUS GPUs"
    python train.py \
        --config "$CONFIG_FILE" \
        --world_size $NUM_GPUS \
        "$@"
else
    echo "🏃 Starting single-GPU training"
    python train.py \
        --config "$CONFIG_FILE" \
        "$@"
fi

echo "✅ Training script completed"
