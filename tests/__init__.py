"""
Unit tests for EBT-Mastery project.

Tests cover edge cases and high-dimensional scenarios:
- High-dimensional energy computations (10K+ dims)
- Numerical stability with extreme values
- Gradient flow verification
- Memory efficiency with large batches
- Device compatibility (CPU/CUDA/MPS)
- Real dataset loading (CIFAR-10/100, GLUE)

Author: DeOcxan
Date: October 2025
"""

# Test configuration
TEST_DATA_DIR = "./test_data"
DEFAULT_DEVICE = "cpu"
SKIP_SLOW_TESTS = False

# Import test utilities
import torch
import pytest


def set_test_seed(seed: int = 42):
    """Set random seed for reproducible tests."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Run seed setup on import
set_test_seed()

