"""
Model Utilities for Energy-Based Models

This module provides utility functions for:
- Seeding and reproducibility
- Parameter counting
- Checkpoint management
- Logging utilities
- Model initialization helpers

Following SOLID principles with clear separation of concerns.

Author: DeOcxan
Date: October 2025
"""

import torch
import torch.nn as nn
import numpy as np
import random
import os
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    
    Sets seeds for:
    - Python's random module
    - NumPy
    - PyTorch (CPU and CUDA)
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Make PyTorch deterministic (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    Count number of parameters in a model.
    
    Args:
        model: PyTorch model
        trainable_only: If True, only count trainable parameters
        
    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def get_parameter_breakdown(model: nn.Module) -> Dict[str, int]:
    """
    Get detailed breakdown of parameters by layer.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary mapping layer names to parameter counts
    """
    breakdown = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            breakdown[name] = param.numel()
    return breakdown


def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    loss: float,
    save_path: str,
    additional_info: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save model checkpoint with training state.
    
    Args:
        model: Model to save
        optimizer: Optimizer state (optional)
        epoch: Current epoch number
        loss: Current loss value
        save_path: Path to save checkpoint
        additional_info: Additional information to save (config, metrics, etc.)
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'loss': loss,
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if additional_info is not None:
        checkpoint.update(additional_info)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    Load model checkpoint and restore training state.
    
    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint file
        optimizer: Optimizer to load state into (optional)
        device: Device to load model on
        
    Returns:
        Dictionary with checkpoint information (epoch, loss, etc.)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    info = {
        'epoch': checkpoint.get('epoch', 0),
        'loss': checkpoint.get('loss', float('inf'))
    }
    
    # Extract any additional information
    for key, value in checkpoint.items():
        if key not in ['model_state_dict', 'optimizer_state_dict']:
            info[key] = value
    
    print(f"Checkpoint loaded from {checkpoint_path} (epoch {info['epoch']})")
    return info


def initialize_weights(module: nn.Module) -> None:
    """
    Initialize module weights using best practices.
    
    Args:
        module: PyTorch module to initialize
    """
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        # Kaiming initialization for layers with ReLU
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d)):
        # Standard initialization for normalization layers
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Embedding):
        # Normal initialization for embeddings
        nn.init.normal_(module.weight, mean=0, std=0.02)


def compute_perplexity(loss: float) -> float:
    """
    Compute perplexity from cross-entropy loss.
    
    Perplexity = exp(loss)
    
    Args:
        loss: Cross-entropy loss value
        
    Returns:
        Perplexity value
    """
    return np.exp(loss)


def get_device(prefer_cuda: bool = True) -> torch.device:
    """
    Get appropriate device for computation.
    
    Args:
        prefer_cuda: Whether to prefer CUDA if available
        
    Returns:
        torch.device object
    """
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU device")
    
    return device


def format_number(num: int) -> str:
    """
    Format large numbers with K/M/B suffixes.
    
    Args:
        num: Number to format
        
    Returns:
        Formatted string
    """
    if num >= 1e9:
        return f"{num/1e9:.2f}B"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    else:
        return str(num)


class GradientClipper:
    """
    Utility class for gradient clipping.
    
    Supports both norm-based and value-based clipping.
    """
    
    def __init__(
        self,
        clip_type: str = 'norm',
        clip_value: float = 1.0
    ):
        """
        Initialize gradient clipper.
        
        Args:
            clip_type: Type of clipping ('norm' or 'value')
            clip_value: Maximum norm or value for clipping
        """
        self.clip_type = clip_type
        self.clip_value = clip_value
        
    def clip(self, parameters) -> float:
        """
        Clip gradients of parameters.
        
        Args:
            parameters: Model parameters
            
        Returns:
            Total gradient norm before clipping
        """
        if self.clip_type == 'norm':
            return torch.nn.utils.clip_grad_norm_(
                parameters,
                self.clip_value
            ).item()
        elif self.clip_type == 'value':
            torch.nn.utils.clip_grad_value_(
                parameters,
                self.clip_value
            )
            # Calculate total norm for logging
            total_norm = 0.0
            for p in parameters:
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            return total_norm ** 0.5
        else:
            raise ValueError(f"Unknown clip_type: {self.clip_type}")

