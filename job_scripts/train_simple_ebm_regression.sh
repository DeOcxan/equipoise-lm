#!/bin/bash
# Job script for training Simple EBM (Regression)
#
# This script demonstrates iterative optimization for regression tasks.
#
# Usage:
#   bash job_scripts/train_simple_ebm_regression.sh

echo "========================================="
echo "Training Simple EBM for Regression"
echo "========================================="

# Navigate to Simple_EBM directory
cd Simple_EBM

# Train on different function types
for FUNCTION in quadratic sin cubic; do
    echo ""
    echo "Training on $FUNCTION function..."
    
    python simple_ebm_regression.py \
        --num_epochs 200 \
        --batch_size 32 \
        --lr 0.001 \
        --num_samples 1000 \
        --function_type $FUNCTION \
        --num_optim_steps 5 \
        --save_dir ./plots
done

echo ""
echo "All trainings complete! Check ./Simple_EBM/plots for visualizations."

