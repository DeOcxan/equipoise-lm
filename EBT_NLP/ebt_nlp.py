"""
Energy-Based Transformer for Natural Language Processing

This module implements an autoregressive Energy-Based Transformer for text generation.
Unlike standard transformers that output probability distributions, EBT outputs
energy values and finds the best token through iterative optimization.

Key differences from standard Transformer:
1. Energy head instead of softmax classifier
2. Iterative optimization for each token prediction
3. Can "think" longer on harder tokens

This demonstrates how EBMs can be scaled to sequence models like transformers.

Author: DeOcxan
Date: October 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.base_energy_model import BaseTrainableEBM
from utils.model_utils import count_parameters
from .transformer_components import (
    TransformerBlock,
    PositionalEncoding,
    CausalMask
)


class EBT_NLP(BaseTrainableEBM):
    """
    Energy-Based Transformer for NLP (Text Generation).
    
    Architecture:
        Tokens -> Embedding -> Positional Encoding -> 
        Transformer Blocks -> Energy Head -> Energy (scalar per token)
    
    Unlike standard Transformers:
    - Output is energy, not logits/probabilities
    - Generation uses gradient descent on energy landscape
    - Can adaptively allocate compute per prediction
    
    Following principles:
    - Inherits from BaseTrainableEBM
    - Modular design with separate components
    - Clear separation between training and inference
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_blocks: int = 4,
        num_heads: int = 8,
        d_ff: int = 1024,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        num_optim_steps: int = 3,
        optim_step_size: float = 0.1,
        corruption_type: str = 'random_noise',
        corruption_noise_scale: float = 1.0,
        use_vocab_to_embed_mlp: bool = True,
        vocab_to_embed_num_layers: int = 1,
        use_rms_norm: bool = False,
        device: str = 'cpu'
    ):
        """
        Initialize EBT for NLP.
        
        Args:
            vocab_size: Size of vocabulary
            d_model: Model embedding dimension
            num_blocks: Number of transformer blocks
            num_heads: Number of attention heads
            d_ff: Feed-forward dimension
            max_seq_len: Maximum sequence length
            dropout: Dropout rate
            num_optim_steps: Steps for iterative token optimization
            optim_step_size: Step size for gradient descent
            corruption_type: Type of initial corruption ('random_noise', 'gaussian_noise_around_embed', 'uniform')
            corruption_noise_scale: Scale of corruption noise
            use_vocab_to_embed_mlp: Whether to use MLP for vocab->embed conversion
            vocab_to_embed_num_layers: Number of layers in vocab_to_embed MLP
            use_rms_norm: Whether to use RMSNorm (Llama2 style) instead of LayerNorm
            device: Device to run on
        """
        config = {
            'model_name': 'EBT_NLP',
            'vocab_size': vocab_size,
            'd_model': d_model,
            'num_blocks': num_blocks,
            'num_heads': num_heads,
            'd_ff': d_ff,
            'max_seq_len': max_seq_len,
            'dropout': dropout,
            'device': torch.device(device),
            'num_optim_steps': num_optim_steps,
            'optim_step_size': optim_step_size,
            'langevin_noise_scale': 0.01,
            'l2_regularization': 0.0001,
            'corruption_type': corruption_type,
            'corruption_noise_scale': corruption_noise_scale,
            'use_vocab_to_embed_mlp': use_vocab_to_embed_mlp,
            'vocab_to_embed_num_layers': vocab_to_embed_num_layers
        }
        
        super().__init__(config)
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.corruption_type = corruption_type
        self.corruption_noise_scale = corruption_noise_scale
        self.use_vocab_to_embed_mlp = use_vocab_to_embed_mlp
        self.use_rms_norm = use_rms_norm
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer blocks with optional RMSNorm
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout, use_rms_norm=use_rms_norm)
            for _ in range(num_blocks)
        ])
        
        # vocab_to_embed: Convert probability distribution to embedding
        # Key EBT component - instead of simple prob_dist @ embeddings.weight
        if use_vocab_to_embed_mlp and vocab_to_embed_num_layers > 1:
            layers = []
            layers.append(nn.Linear(vocab_size, d_model))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(d_model))
            
            for _ in range(vocab_to_embed_num_layers - 2):
                layers.append(nn.Linear(d_model, d_model))
                layers.append(nn.ReLU())
                layers.append(nn.LayerNorm(d_model))
            
            layers.append(nn.Linear(d_model, d_model))
            self.vocab_to_embed = nn.Sequential(*layers)
        elif use_vocab_to_embed_mlp:
            # Single layer MLP
            self.vocab_to_embed = nn.Linear(vocab_size, d_model, bias=False)
        else:
            # Direct multiplication with embedding matrix
            self.vocab_to_embed = None
        
        # Energy head: maps hidden state to energy scalar
        # Instead of logits over vocab, outputs single energy value
        self.energy_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Move to device
        self.to(self.device)
        
        print(f"EBT_NLP initialized with {count_parameters(self)} parameters")
        print(f"Vocab size: {vocab_size}, d_model: {d_model}, blocks: {num_blocks}")
        print(f"Corruption type: {corruption_type}, Use vocab_to_embed MLP: {use_vocab_to_embed_mlp}")
        print(f"Normalization: {'RMSNorm (Llama2)' if use_rms_norm else 'LayerNorm'}")
        
    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def corrupt_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Corrupt embeddings with noise - key EBT concept from original paper.
        
        This provides the initial condition for MCMC optimization.
        Different corruption strategies lead to different optimization trajectories.
        
        Args:
            embeddings: Clean embeddings (batch_size, seq_len, d_model)
            
        Returns:
            Corrupted embeddings (batch_size, seq_len, d_model)
        """
        if self.corruption_type == 'random_noise':
            # Pure Gaussian noise - most aggressive corruption
            # Start from complete randomness
            return torch.randn_like(embeddings) * self.corruption_noise_scale
        
        elif self.corruption_type == 'gaussian_noise_around_embed':
            # Add noise around the real embedding - moderate corruption
            # Helps optimization start from a reasonable neighborhood
            noise = torch.randn_like(embeddings) * self.corruption_noise_scale
            return embeddings + noise
        
        elif self.corruption_type == 'uniform':
            # Uniform noise in [-scale, scale]
            return (torch.rand_like(embeddings) * 2 - 1) * self.corruption_noise_scale
        
        else:
            raise ValueError(f"Unknown corruption type: {self.corruption_type}")
    
    def convert_prob_dist_to_embedding(self, prob_dist: torch.Tensor) -> torch.Tensor:
        """
        Convert probability distribution over vocabulary to embedding.
        
        Key EBT component:
        - Option 1 (use_vocab_to_embed_mlp=True): Use MLP for non-linear transformation
        - Option 2 (use_vocab_to_embed_mlp=False): Direct multiplication with embedding matrix
        
        Args:
            prob_dist: Probability distribution (batch_size, seq_len, vocab_size)
            
        Returns:
            Embedding (batch_size, seq_len, d_model)
        """
        if self.vocab_to_embed is not None:
            # Use MLP transformation
            return self.vocab_to_embed(prob_dist)
        else:
            # Direct multiplication: prob_dist @ embeddings.weight
            # Shape: (batch, seq, vocab) @ (vocab, d_model) -> (batch, seq, d_model)
            return torch.matmul(prob_dist, self.token_embedding.weight)
    
    def encode_sequence(
        self,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode token sequence through transformer.
        
        Args:
            tokens: Token indices (batch_size, seq_len)
            mask: Optional attention mask
            
        Returns:
            Hidden states (batch_size, seq_len, d_model)
        """
        # Token embedding
        x = self.token_embedding(tokens) * math.sqrt(self.d_model)
        
        # Positional encoding
        x = self.pos_encoding(x)
        
        # Create causal mask if not provided
        if mask is None:
            seq_len = tokens.size(1)
            mask = CausalMask.create_causal_mask(seq_len, str(self.device))
        
        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x, mask)
        
        return x
    
    def compute_energy(
        self,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute energy for token sequence.
        
        Lower energy = better sequence
        
        Args:
            tokens: Token indices (batch_size, seq_len)
            mask: Optional attention mask
            
        Returns:
            Energy per position (batch_size, seq_len, 1)
        """
        # Encode sequence
        hidden_states = self.encode_sequence(tokens, mask)
        
        # Compute energy for each position
        energy = self.energy_head(hidden_states)
        
        return energy
    
    def compute_token_energy(
        self,
        context_tokens: torch.Tensor,
        candidate_token_emb: torch.Tensor,
        position: int
    ) -> torch.Tensor:
        """
        Compute energy for a candidate token at a specific position.
        
        Used during generation to evaluate candidate tokens.
        
        Args:
            context_tokens: Previous tokens (batch_size, position)
            candidate_token_emb: Embedding of candidate token (batch_size, d_model)
            position: Position to insert candidate
            
        Returns:
            Energy value (batch_size, 1)
        """
        # Get context embeddings
        context_emb = self.token_embedding(context_tokens) * math.sqrt(self.d_model)
        
        # Add candidate at position
        # Create full sequence: context + candidate
        seq_emb = torch.cat([context_emb, candidate_token_emb.unsqueeze(1)], dim=1)
        
        # Positional encoding
        seq_emb = self.pos_encoding(seq_emb)
        
        # Create causal mask
        seq_len = seq_emb.size(1)
        mask = CausalMask.create_causal_mask(seq_len, str(self.device))
        
        # Pass through transformer
        for block in self.transformer_blocks:
            seq_emb = block(seq_emb, mask)
        
        # Get energy for the candidate position
        candidate_hidden = seq_emb[:, -1, :]  # Last position
        energy = self.energy_head(candidate_hidden)
        
        return energy
    
    def generate_next_token(
        self,
        context_tokens: torch.Tensor,
        num_steps: int = None,
        temperature: float = 1.0,
        top_k: int = 50
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate next token using energy-based optimization.
        
        Process:
        1. Start with random candidate in embedding space
        2. Iteratively refine via gradient descent on energy
        3. Project back to nearest vocab token
        
        Args:
            context_tokens: Previous tokens (batch_size, seq_len)
            num_steps: Optimization steps (default from config)
            temperature: Sampling temperature
            top_k: Consider only top-k tokens
            
        Returns:
            Tuple of (next_token, energy_trajectory)
        """
        if num_steps is None:
            num_steps = self._default_num_steps
        
        batch_size = context_tokens.size(0)
        position = context_tokens.size(1)
        
        # Initialize candidate embedding from noise
        candidate_emb = torch.randn(
            batch_size,
            self.d_model,
            device=self.device,
            requires_grad=True
        )
        
        energy_trajectory = []
        
        # Iteratively optimize candidate embedding
        for step in range(num_steps):
            # Compute energy for current candidate
            energy = self.compute_token_energy(context_tokens, candidate_emb, position)
            energy_trajectory.append(energy.mean().item())
            
            # Compute gradient
            energy_sum = energy.sum()
            grads = torch.autograd.grad(
                outputs=energy_sum,
                inputs=candidate_emb,
                create_graph=False,
                retain_graph=True
            )[0]
            
            # Gradient descent step
            with torch.no_grad():
                candidate_emb = candidate_emb - self._default_step_size * grads
            
            candidate_emb = candidate_emb.detach().requires_grad_(True)
        
        # Project candidate to vocabulary
        with torch.no_grad():
            # Compute similarity to all token embeddings
            all_embeddings = self.token_embedding.weight  # (vocab_size, d_model)
            similarities = torch.matmul(
                candidate_emb,
                all_embeddings.t()
            )  # (batch_size, vocab_size)
            
            # Apply temperature
            logits = similarities / temperature
            
            # Top-k filtering
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
                logits_filtered = torch.full_like(logits, float('-inf'))
                logits_filtered.scatter_(1, top_k_indices, top_k_logits)
            else:
                logits_filtered = logits
            
            # Sample from distribution
            probs = F.softmax(logits_filtered, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        
        return next_token, torch.tensor(energy_trajectory)
    
    def generate(
        self,
        prompt_tokens: torch.Tensor,
        max_new_tokens: int = 50,
        num_steps: int = None,
        temperature: float = 1.0,
        top_k: int = 50
    ) -> torch.Tensor:
        """
        Generate a sequence autoregressively.
        
        Args:
            prompt_tokens: Initial prompt (batch_size, prompt_len)
            max_new_tokens: Number of tokens to generate
            num_steps: Optimization steps per token
            temperature: Sampling temperature
            top_k: Top-k sampling
            
        Returns:
            Generated sequence (batch_size, prompt_len + max_new_tokens)
        """
        self.eval()
        generated = prompt_tokens.clone()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Generate next token
                next_token, _ = self.generate_next_token(
                    generated,
                    num_steps=num_steps,
                    temperature=temperature,
                    top_k=top_k
                )
                
                # Append to sequence
                generated = torch.cat([generated, next_token], dim=1)
                
                # Stop if max length reached
                if generated.size(1) >= self._config['max_seq_len']:
                    break
        
        return generated
    
    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Training step for language modeling.
        
        Objective: Minimize energy for correct next tokens
        
        Args:
            batch: Tuple of (input_tokens, target_tokens)
            batch_idx: Batch index
            
        Returns:
            Dictionary with loss and metrics
        """
        input_tokens, target_tokens = batch
        input_tokens = input_tokens.to(self.device)
        target_tokens = target_tokens.to(self.device)
        
        # Compute energy for sequence
        energy = self.compute_energy(input_tokens)  # (batch, seq_len, 1)
        
        # Average energy over sequence
        avg_energy = energy.mean()
        
        # Loss is simply the energy (we want to minimize it)
        loss = avg_energy
        
        # Add L2 regularization
        if self._l2_reg > 0:
            l2_penalty = sum(torch.sum(p ** 2) for p in self.parameters())
            loss = loss + self._l2_reg * l2_penalty
        
        # Compute perplexity approximation
        # For EBM, we approximate perplexity from energy
        perplexity = torch.exp(avg_energy.detach())
        
        return {
            'loss': loss,
            'energy_mean': avg_energy,
            'energy_std': energy.std(),
            'perplexity': perplexity
        }
    
    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Validation step."""
        with torch.no_grad():
            return self.training_step(batch, batch_idx)


import math

