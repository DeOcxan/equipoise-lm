"""
Energy-Based Transformer for Image Processing Package

This package contains EBT implementations for image tasks:
- EBT for image denoising (bidirectional)
- Demonstrates energy-based iterative refinement

These models show how EBTs can be applied beyond NLP to vision tasks.
"""

from .ebt_img_denoise import EBT_IMG_Denoise

__all__ = ['EBT_IMG_Denoise']

