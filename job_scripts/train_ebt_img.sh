#!/bin/bash
# Job script for training EBT_IMG
#
# This script trains Energy-Based Model for image denoising on MNIST.
# Demonstrates bidirectional energy-based processing.
#
# Usage:
#   bash job_scripts/train_ebt_img.sh

echo "========================================="
echo "Training EBT for Image Denoising"
echo "========================================="

# Navigate to EBT_IMG directory
cd EBT_IMG

# Train with different noise levels
echo ""
echo "Training with noise_std=0.2..."
python train_ebt_img.py \
    --num_epochs 30 \
    --batch_size 64 \
    --lr 0.001 \
    --noise_std 0.2 \
    --num_optim_steps 5 \
    --image_size 28 \
    --save_dir ./outputs/noise_0.2

echo ""
echo "Training with noise_std=0.3..."
python train_ebt_img.py \
    --num_epochs 30 \
    --batch_size 64 \
    --lr 0.001 \
    --noise_std 0.3 \
    --num_optim_steps 7 \
    --image_size 28 \
    --save_dir ./outputs/noise_0.3

echo ""
echo "Training complete! Check ./EBT_IMG/outputs for results."

