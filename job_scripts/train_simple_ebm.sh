#!/bin/bash
# Job script for training Simple EBM (Classification)
# 
# This script demonstrates how to train the basic Energy-Based Model
# for binary classification on synthetic data.
#
# Usage:
#   bash job_scripts/train_simple_ebm.sh

echo "========================================="
echo "Training Simple EBM for Classification"
echo "========================================="

# Navigate to Simple_EBM directory
cd Simple_EBM

# Run training
python simple_ebm.py \
    --num_epochs 100 \
    --batch_size 32 \
    --lr 0.01 \
    --num_samples 1000 \
    --input_dim 2 \
    --save_dir ./plots

echo ""
echo "Training complete! Check ./Simple_EBM/plots for visualizations."

