"""
Energy-Based Transformer for NLP Package

This package contains EBT implementations for text generation:
- EBT with basic autoregressive generation
- EBT with advanced features (adaptive step size, Langevin dynamics)

These models demonstrate how Energy-Based Transformers can be used
for language modeling with iterative optimization.
"""

from .ebt_nlp import EBT_NLP
from .ebt_nlp_advanced import EBT_NLP_Advanced

__all__ = ['EBT_NLP', 'EBT_NLP_Advanced']

