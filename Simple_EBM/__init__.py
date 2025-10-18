"""
Simple Energy-Based Models Package

This package contains basic EBM implementations for educational purposes:
- Simple EBM for binary classification
- Simple EBM for regression

These models demonstrate the core concepts of Energy-Based Models without
the complexity of transformers.
"""

from .simple_ebm import SimpleEBM
from .simple_ebm_regression import SimpleEBMRegression

__all__ = ['SimpleEBM', 'SimpleEBMRegression']

