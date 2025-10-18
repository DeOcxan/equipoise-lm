#!/bin/bash
# Job script for training EBT_NLP
#
# This script trains Energy-Based Transformer for text generation.
# Includes both basic and advanced versions.
#
# Usage:
#   bash job_scripts/train_ebt_nlp.sh

echo "========================================="
echo "Training EBT for NLP (Text Generation)"
echo "========================================="

# Navigate to EBT_NLP directory
cd EBT_NLP

# Train basic EBT_NLP
echo ""
echo "1. Training Basic EBT_NLP..."
python train_ebt_nlp.py \
    --model_type basic \
    --num_epochs 50 \
    --batch_size 32 \
    --lr 0.0003 \
    --num_samples 1000 \
    --seq_length 32 \
    --pattern_type repetition \
    --num_optim_steps 3 \
    --save_dir ./outputs

# Train basic EBT_NLP with research features
echo ""
echo "2. Training Basic EBT_NLP with Research Features..."
echo "   (MCMC corruption + vocab_to_embed MLP + RMSNorm)"
python train_ebt_nlp.py \
    --model_type basic \
    --num_epochs 50 \
    --batch_size 32 \
    --lr 0.0003 \
    --num_samples 1000 \
    --seq_length 32 \
    --pattern_type arithmetic \
    --num_optim_steps 3 \
    --corruption_type gaussian_noise_around_embed \
    --use_vocab_to_embed_mlp \
    --vocab_to_embed_num_layers 2 \
    --use_rms_norm \
    --save_dir ./outputs

# Train advanced EBT_NLP with all features
echo ""
echo "3. Training Advanced EBT_NLP (Full Research Configuration)..."
echo "   (All features: learnable alpha + Langevin + MCMC + RMSNorm)"
python train_ebt_nlp.py \
    --model_type advanced \
    --num_epochs 50 \
    --batch_size 32 \
    --lr 0.0003 \
    --num_samples 1000 \
    --seq_length 32 \
    --pattern_type arithmetic \
    --num_optim_steps 5 \
    --corruption_type random_noise \
    --use_vocab_to_embed_mlp \
    --vocab_to_embed_num_layers 3 \
    --use_rms_norm \
    --save_dir ./outputs

echo ""
echo "Training complete! Check ./EBT_NLP/outputs for results."

