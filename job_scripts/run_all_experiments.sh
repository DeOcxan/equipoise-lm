#!/bin/bash
# Master script to run all experiments
#
# This script runs all training experiments in sequence:
# 1. Simple EBM (classification)
# 2. Simple EBM (regression)
# 3. EBT_NLP (text generation)
# 4. EBT_IMG (image denoising)
#
# Usage:
#   bash job_scripts/run_all_experiments.sh
#
# Warning: This will take a long time! Consider running scripts individually.

echo "========================================="
echo "Running All EBT-Mastery Experiments"
echo "========================================="
echo ""
echo "This will run all experiments sequentially."
echo "Total estimated time: ~2-3 hours (depending on hardware)"
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."

# Track start time
START_TIME=$(date +%s)

# 1. Simple EBM Classification
echo ""
echo "========== Experiment 1/4: Simple EBM Classification =========="
bash job_scripts/train_simple_ebm.sh
if [ $? -ne 0 ]; then
    echo "Error in Experiment 1. Stopping."
    exit 1
fi

# 2. Simple EBM Regression
echo ""
echo "========== Experiment 2/4: Simple EBM Regression =========="
bash job_scripts/train_simple_ebm_regression.sh
if [ $? -ne 0 ]; then
    echo "Error in Experiment 2. Stopping."
    exit 1
fi

# 3. EBT_NLP
echo ""
echo "========== Experiment 3/4: EBT_NLP =========="
bash job_scripts/train_ebt_nlp.sh
if [ $? -ne 0 ]; then
    echo "Error in Experiment 3. Stopping."
    exit 1
fi

# 4. EBT_IMG
echo ""
echo "========== Experiment 4/4: EBT_IMG =========="
bash job_scripts/train_ebt_img.sh
if [ $? -ne 0 ]; then
    echo "Error in Experiment 4. Stopping."
    exit 1
fi

# Calculate elapsed time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))

echo ""
echo "========================================="
echo "All Experiments Complete!"
echo "========================================="
echo "Total time: ${HOURS}h ${MINUTES}m"
echo ""
echo "Results locations:"
echo "  - Simple_EBM/plots/"
echo "  - EBT_NLP/outputs/"
echo "  - EBT_IMG/outputs/"
echo ""
echo "Thank you for exploring Energy-Based Transformers!"

