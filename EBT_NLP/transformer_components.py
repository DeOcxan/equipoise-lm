"""
Transformer Components for Energy-Based Transformers

This module implements transformer building blocks from scratch:
- Multi-head Self-Attention
- Position-wise Feed-Forward Networks
- Positional Embeddings
- Transformer Blocks

All components follow OOP and SOLID principles with clear separation of concerns.

Author: DeOcxan
Date: October 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    Used in Llama2 and modern transformers. More efficient than LayerNorm
    as it doesn't compute mean and doesn't have bias.
    
    RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight
    
    Paper: https://arxiv.org/abs/1910.07467
    """
    
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize RMSNorm.
        
        Args:
            dim: Dimension of input features
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMS normalization.
        
        Args:
            x: Input tensor
            
        Returns:
            Normalized tensor
        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (*, dim)
            
        Returns:
            Normalized and scaled tensor (*, dim)
        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for transformer models.
    
    Adds position information to token embeddings using sine and cosine
    functions of different frequencies.
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    Following Single Responsibility Principle: Only handles position encoding.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Initialize positional encoding.
        
        Args:
            d_model: Embedding dimension
            max_len: Maximum sequence length
            dropout: Dropout rate
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add batch dimension
        
        # Register as buffer (not a parameter, but part of state)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism.
    
    Allows model to attend to different positions and representation
    subspaces simultaneously.
    
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
    
    Following Single Responsibility Principle: Only handles attention computation.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1
    ):
        """
        Initialize multi-head attention.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # Output projection
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Split last dimension into (num_heads, d_k).
        
        Args:
            x: Input of shape (batch, seq_len, d_model)
            
        Returns:
            Tensor of shape (batch, num_heads, seq_len, d_k)
        """
        batch_size, seq_len, d_model = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)
    
    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Merge heads back to original shape.
        
        Args:
            x: Input of shape (batch, num_heads, seq_len, d_k)
            
        Returns:
            Tensor of shape (batch, seq_len, d_model)
        """
        batch_size, num_heads, seq_len, d_k = x.size()
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, self.d_model)
    
    def _scaled_dot_product_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute scaled dot-product attention.
        
        Args:
            Q: Query tensor (batch, num_heads, seq_len, d_k)
            K: Key tensor (batch, num_heads, seq_len, d_k)
            V: Value tensor (batch, num_heads, seq_len, d_k)
            mask: Optional mask tensor
            
        Returns:
            Tuple of (attention_output, attention_weights)
        """
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided (for autoregressive generation)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax and dropout
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of multi-head attention.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            mask: Optional attention mask
            
        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        batch_size = x.size(0)
        
        # Linear projections
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # Split into multiple heads
        Q = self._split_heads(Q)
        K = self._split_heads(K)
        V = self._split_heads(V)
        
        # Apply attention
        attn_output, _ = self._scaled_dot_product_attention(Q, K, V, mask)
        
        # Merge heads
        attn_output = self._merge_heads(attn_output)
        
        # Final linear projection
        output = self.W_o(attn_output)
        
        return output


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.
    
    FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
    
    Two linear transformations with ReLU activation in between.
    
    Following Single Responsibility Principle: Only handles feed-forward computation.
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        """
        Initialize feed-forward network.
        
        Args:
            d_model: Model dimension
            d_ff: Feed-forward dimension (typically 4 * d_model)
            dropout: Dropout rate
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            
        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerBlock(nn.Module):
    """
    Single transformer block with self-attention and feed-forward layers.
    
    Implements the standard transformer architecture with:
    - Multi-head self-attention
    - Feed-forward network
    - Layer normalization
    - Residual connections
    
    Following Open/Closed Principle: Can be extended for different variants.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        use_rms_norm: bool = False
    ):
        """
        Initialize transformer block.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout rate
            use_rms_norm: Whether to use RMSNorm (Llama2 style) instead of LayerNorm
        """
        super().__init__()
        
        # Self-attention
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        
        # Feed-forward
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        # Normalization layers - choice between LayerNorm and RMSNorm
        if use_rms_norm:
            self.norm1 = RMSNorm(d_model)
            self.norm2 = RMSNorm(d_model)
        else:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through transformer block.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            mask: Optional attention mask
            
        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        # Self-attention with residual connection and layer norm
        attn_output = self.self_attn(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class CausalMask:
    """
    Utility class for creating causal (autoregressive) masks.
    
    Ensures that each position can only attend to previous positions,
    necessary for autoregressive generation.
    
    Following Single Responsibility Principle: Only handles mask creation.
    """
    
    @staticmethod
    def create_causal_mask(seq_len: int, device: str = 'cpu') -> torch.Tensor:
        """
        Create causal mask for autoregressive attention.
        
        Mask has shape (seq_len, seq_len) where:
        - 1 means attend (position is visible)
        - 0 means don't attend (position is masked)
        
        Args:
            seq_len: Sequence length
            device: Device for tensor
            
        Returns:
            Causal mask tensor
        """
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions
    
    @staticmethod
    def create_padding_mask(
        lengths: torch.Tensor,
        max_len: int,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """
        Create padding mask for variable-length sequences.
        
        Args:
            lengths: Actual lengths of each sequence (batch_size,)
            max_len: Maximum sequence length
            device: Device for tensor
            
        Returns:
            Padding mask tensor (batch_size, 1, 1, max_len)
        """
        batch_size = lengths.size(0)
        mask = torch.arange(max_len, device=device).expand(batch_size, max_len)
        mask = (mask < lengths.unsqueeze(1)).float()
        return mask.unsqueeze(1).unsqueeze(1)


def create_sinusoidal_embeddings(
    num_positions: int,
    embedding_dim: int
) -> torch.Tensor:
    """
    Create sinusoidal positional embeddings.
    
    Alternative to the PositionalEncoding module, returns raw embeddings.
    
    Args:
        num_positions: Number of positions
        embedding_dim: Embedding dimension
        
    Returns:
        Positional embeddings (num_positions, embedding_dim)
    """
    position = torch.arange(0, num_positions, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, embedding_dim, 2).float() * 
        (-math.log(10000.0) / embedding_dim)
    )
    
    embeddings = torch.zeros(num_positions, embedding_dim)
    embeddings[:, 0::2] = torch.sin(position * div_term)
    embeddings[:, 1::2] = torch.cos(position * div_term)
    
    return embeddings

