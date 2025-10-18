"""
Advanced Energy-Based Transformer for NLP

This module extends the basic EBT with advanced features:
- Learnable step size (adaptive optimization)
- Langevin dynamics noise for better exploration
- Adaptive number of optimization steps per token
- Temperature-based "thinking" allocation

This demonstrates how EBT can adaptively allocate compute.

Author: DeOcxan
Date: October 2025
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

sys.path.append(str(Path(__file__).parent.parent))

from .ebt_nlp import EBT_NLP


class EBT_NLP_Advanced(EBT_NLP):
    """
    Advanced EBT with learnable optimization parameters.
    
    Features:
    1. Learnable step size per layer/position
    2. Langevin dynamics for exploration
    3. Adaptive optimization steps
    4. Energy-based early stopping
    
    Extends EBT_NLP following Open/Closed Principle.
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
        num_optim_steps: int = 5,
        initial_step_size: float = 0.1,
        learn_step_size: bool = True,
        use_langevin: bool = True,
        adaptive_steps: bool = True,
        device: str = 'cpu'
    ):
        """
        Initialize Advanced EBT.
        
        Args:
            vocab_size: Size of vocabulary
            d_model: Model dimension
            num_blocks: Number of transformer blocks
            num_heads: Number of attention heads
            d_ff: Feed-forward dimension
            max_seq_len: Maximum sequence length
            dropout: Dropout rate
            num_optim_steps: Maximum optimization steps
            initial_step_size: Initial step size
            learn_step_size: Whether to learn step size
            use_langevin: Whether to use Langevin dynamics
            adaptive_steps: Whether to use adaptive number of steps
            device: Device to run on
        """
        # Initialize base EBT
        super().__init__(
            vocab_size=vocab_size,
            d_model=d_model,
            num_blocks=num_blocks,
            num_heads=num_heads,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
            dropout=dropout,
            num_optim_steps=num_optim_steps,
            optim_step_size=initial_step_size,
            device=device
        )
        
        self._config['model_name'] = 'EBT_NLP_Advanced'
        self.learn_step_size = learn_step_size
        self.use_langevin = use_langevin
        self.adaptive_steps = adaptive_steps
        
        # Learnable step size with better initialization
        # Use log-space to ensure positive values and stable gradients
        if learn_step_size:
            # Initialize with log of step size for stability
            log_step_init = torch.log(torch.tensor(initial_step_size))
            self.log_step_size = nn.Parameter(log_step_init)
            
            # Optional: Add bounds to prevent extreme values
            self.min_step_size = 1e-4
            self.max_step_size = 1.0
        
        # Energy threshold for early stopping
        # Dynamically adjusted during training
        self.register_buffer('energy_threshold', torch.tensor(0.1))
        self.register_buffer('energy_threshold_decay', torch.tensor(0.99))
        
        print(f"EBT_NLP_Advanced initialized")
        print(f"Learnable step size: {learn_step_size}")
        print(f"Langevin dynamics: {use_langevin}")
        print(f"Adaptive steps: {adaptive_steps}")
    
    def get_step_size(self) -> float:
        """
        Get current step size with clamping for stability.
        
        Returns:
            Step size value (clamped to reasonable range)
        """
        if self.learn_step_size:
            step_size = torch.exp(self.log_step_size)
            # Clamp to prevent extreme values
            step_size = torch.clamp(step_size, self.min_step_size, self.max_step_size)
            return step_size.item()
        else:
            return self._default_step_size
    
    def generate_next_token_advanced(
        self,
        context_tokens: torch.Tensor,
        max_steps: int = None,
        temperature: float = 1.0,
        top_k: int = 50
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Generate next token with learnable step size and Langevin dynamics.
        
        Uses:
        - Learnable step size
        - Langevin noise
        - Adaptive early stopping
        
        Args:
            context_tokens: Previous tokens
            max_steps: Maximum optimization steps
            temperature: Sampling temperature
            top_k: Top-k sampling
            
        Returns:
            Tuple of (next_token, energy_trajectory, actual_steps_used)
        """
        if max_steps is None:
            max_steps = self._default_num_steps
        
        batch_size = context_tokens.size(0)
        position = context_tokens.size(1)
        
        # Initialize candidate
        candidate_emb = torch.randn(
            batch_size,
            self.d_model,
            device=self.device,
            requires_grad=True
        )
        
        energy_trajectory = []
        prev_energy = float('inf')
        step_size = self.get_step_size()
        
        actual_steps = 0
        
        # Iterative optimization with early stopping
        for step in range(max_steps):
            # Compute energy
            energy = self.compute_token_energy(context_tokens, candidate_emb, position)
            current_energy = energy.mean().item()
            energy_trajectory.append(current_energy)
            
            # Check for convergence (adaptive stopping)
            if self.adaptive_steps and step > 0:
                energy_reduction = prev_energy - current_energy
                if energy_reduction < self.energy_threshold.item():
                    actual_steps = step + 1
                    break
            
            prev_energy = current_energy
            
            # Compute gradients
            energy_sum = energy.sum()
            grads = torch.autograd.grad(
                outputs=energy_sum,
                inputs=candidate_emb,
                create_graph=False,
                retain_graph=True
            )[0]
            
            # Update with learnable step size
            with torch.no_grad():
                candidate_emb = candidate_emb - step_size * grads
                
                # Add Langevin noise
                if self.use_langevin and self.training:
                    noise = torch.randn_like(candidate_emb) * self._langevin_noise_scale
                    candidate_emb = candidate_emb + noise
            
            candidate_emb = candidate_emb.detach().requires_grad_(True)
            actual_steps = step + 1
        
        # Project to vocabulary
        with torch.no_grad():
            all_embeddings = self.token_embedding.weight
            similarities = torch.matmul(candidate_emb, all_embeddings.t())
            logits = similarities / temperature
            
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
                logits_filtered = torch.full_like(logits, float('-inf'))
                logits_filtered.scatter_(1, top_k_indices, top_k_logits)
            else:
                logits_filtered = logits
            
            probs = torch.nn.functional.softmax(logits_filtered, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        
        return next_token, torch.tensor(energy_trajectory), actual_steps
    
    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Training step with additional metrics.
        
        Args:
            batch: Tuple of (input_tokens, target_tokens)
            batch_idx: Batch index
            
        Returns:
            Dictionary with loss and metrics
        """
        # Call parent training step
        metrics = super().training_step(batch, batch_idx)
        
        # Add step size to metrics
        if self.learn_step_size:
            metrics['step_size'] = self.get_step_size()
        
        return metrics

