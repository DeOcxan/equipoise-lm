"""
Device Management with CPU Fallback and Error Handling

This module handles device selection, CPU fallback, and version compatibility.
Addresses the weakness of assuming GPU availability and PyTorch 2.0+ without checks.

Author: DeOcxan
Date: October 2025
"""

import torch
import warnings
from typing import Optional, Tuple
import sys


class DeviceManager:
    """
    Manages device selection with intelligent fallback.
    
    Handles:
    - GPU availability detection
    - CPU fallback when GPU unavailable
    - MPS (Apple Silicon) support
    - Version compatibility checks
    """
    
    def __init__(self, preferred_device: Optional[str] = None):
        """
        Initialize device manager.
        
        Args:
            preferred_device: Preferred device ('cuda', 'mps', 'cpu', or None for auto)
        """
        self._preferred_device = preferred_device
        self._device = None
        self._device_info = {}
        
        self._initialize_device()
        self._check_pytorch_version()
    
    def _initialize_device(self):
        """Initialize device with fallback logic."""
        if self._preferred_device == 'cpu':
            self._device = torch.device('cpu')
            self._device_info = {'type': 'cpu', 'name': 'CPU'}
            print("🖥️  Using CPU (explicitly requested)")
            return
        
        # Try CUDA
        if torch.cuda.is_available():
            if self._preferred_device is None or self._preferred_device == 'cuda':
                self._device = torch.device('cuda')
                self._device_info = {
                    'type': 'cuda',
                    'name': torch.cuda.get_device_name(0),
                    'count': torch.cuda.device_count(),
                    'memory': torch.cuda.get_device_properties(0).total_memory / 1e9
                }
                print(f"🚀 Using CUDA: {self._device_info['name']} "
                      f"({self._device_info['memory']:.1f} GB)")
                return
        
        # Try MPS (Apple Silicon) with enhanced compatibility check
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            if self._preferred_device is None or self._preferred_device == 'mps':
                try:
                    # Test MPS functionality with a simple operation
                    test_tensor = torch.zeros(1, device='mps')
                    del test_tensor
                    
                    self._device = torch.device('mps')
                    self._device_info = {'type': 'mps', 'name': 'Apple Silicon (MPS)'}
                    print("🍎 Using Apple Silicon (MPS)")
                    
                    # MPS-specific warnings
                    pytorch_version = torch.__version__
                    major, minor = map(int, pytorch_version.split('.')[:2])
                    if major == 2 and minor < 1:
                        warnings.warn(
                            "⚠️  MPS support is experimental in PyTorch 2.0. "
                            "Consider upgrading to PyTorch 2.1+ for better stability.",
                            UserWarning
                        )
                    return
                except Exception as e:
                    warnings.warn(
                        f"⚠️  MPS available but failed initialization test: {e}\n"
                        f"   Falling back to CPU.",
                        UserWarning
                    )
        
        # Fallback to CPU
        self._device = torch.device('cpu')
        self._device_info = {'type': 'cpu', 'name': 'CPU'}
        
        if self._preferred_device and self._preferred_device != 'cpu':
            warnings.warn(
                f"⚠️  Preferred device '{self._preferred_device}' not available. "
                f"Falling back to CPU. Training will be slower.",
                UserWarning
            )
            print("💡 Tip: Install CUDA-enabled PyTorch for GPU support:")
            print("   pip install torch torchvision --index-url "
                  "https://download.pytorch.org/whl/cu118")
        else:
            print("🖥️  Using CPU")
    
    def _check_pytorch_version(self):
        """Check PyTorch version and warn if incompatible."""
        torch_version = torch.__version__
        major, minor = map(int, torch_version.split('.')[:2])
        
        # Check minimum version (2.0+)
        if major < 2:
            warnings.warn(
                f"⚠️  PyTorch {torch_version} detected. "
                f"This project requires PyTorch 2.0+.\n"
                f"   Some features may not work correctly.\n"
                f"   Update with: pip install --upgrade torch torchvision",
                UserWarning
            )
            print(f"❌ PyTorch {torch_version} (INCOMPATIBLE - requires 2.0+)")
        else:
            print(f"✅ PyTorch {torch_version}")
        
        # Check for known issues
        if major == 2 and minor == 0 and self._device.type == 'mps':
            warnings.warn(
                "⚠️  PyTorch 2.0 has known issues with MPS. "
                "Consider upgrading to 2.1+ or using CPU/CUDA.",
                UserWarning
            )
    
    def get_device(self) -> torch.device:
        """Get the selected device."""
        return self._device
    
    def get_device_info(self) -> dict:
        """Get device information."""
        return self._device_info.copy()
    
    def move_to_device(self, tensor_or_model):
        """Move tensor or model to device with error handling."""
        try:
            return tensor_or_model.to(self._device)
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print("❌ GPU Out of Memory! Falling back to CPU...")
                self._device = torch.device('cpu')
                self._device_info = {'type': 'cpu', 'name': 'CPU (OOM fallback)'}
                return tensor_or_model.to(self._device)
            else:
                raise
    
    def empty_cache(self):
        """Empty GPU cache if using CUDA."""
        if self._device.type == 'cuda':
            torch.cuda.empty_cache()
    
    def memory_summary(self) -> str:
        """Get memory usage summary."""
        if self._device.type == 'cuda':
            allocated = torch.cuda.memory_allocated(0) / 1e9
            reserved = torch.cuda.memory_reserved(0) / 1e9
            return (f"GPU Memory: {allocated:.2f} GB allocated, "
                   f"{reserved:.2f} GB reserved")
        else:
            return "CPU (no memory tracking)"
    
    @staticmethod
    def check_dependencies():
        """Check if all required dependencies are installed."""
        issues = []
        
        # Check PyTorch
        try:
            import torch
            if torch.__version__ < '2.0.0':
                issues.append(f"PyTorch {torch.__version__} < 2.0.0")
        except ImportError:
            issues.append("PyTorch not installed")
        
        # Check torchvision
        try:
            import torchvision
            if torchvision.__version__ < '0.15.0':
                issues.append(f"torchvision {torchvision.__version__} < 0.15.0")
        except ImportError:
            issues.append("torchvision not installed")
        
        # Check numpy
        try:
            import numpy
        except ImportError:
            issues.append("numpy not installed")
        
        # Check matplotlib
        try:
            import matplotlib
        except ImportError:
            issues.append("matplotlib not installed")
        
        if issues:
            print("❌ Dependency issues found:")
            for issue in issues:
                print(f"   - {issue}")
            print("\n💡 Install dependencies with:")
            print("   pip install -r requirements.txt")
            return False
        else:
            print("✅ All dependencies satisfied")
            return True
    
    @staticmethod
    def get_optimal_num_workers() -> int:
        """Get optimal number of DataLoader workers for current system."""
        import os
        import multiprocessing as mp
        
        if sys.platform == 'win32':
            # Windows has issues with multiprocessing in DataLoader
            return 0
        else:
            # Use 4 workers or half of CPU cores, whichever is smaller
            num_cpus = mp.cpu_count()
            return min(4, max(1, num_cpus // 2))


def auto_select_device(preferred: Optional[str] = None) -> Tuple[torch.device, dict]:
    """
    Convenience function to automatically select best available device.
    
    Args:
        preferred: Preferred device ('cuda', 'mps', 'cpu', or None)
        
    Returns:
        Tuple of (device, device_info)
    """
    manager = DeviceManager(preferred)
    return manager.get_device(), manager.get_device_info()


def setup_training_environment(
    seed: int = 42,
    preferred_device: Optional[str] = None,
    check_deps: bool = True
) -> Tuple[torch.device, dict]:
    """
    Setup complete training environment.
    
    - Sets random seeds
    - Checks dependencies
    - Selects device with fallback
    - Configures backends
    
    Args:
        seed: Random seed for reproducibility
        preferred_device: Preferred device
        check_deps: Whether to check dependencies
        
    Returns:
        Tuple of (device, device_info)
    """
    print("=" * 60)
    print("🔧 Setting up training environment...")
    print("=" * 60)
    
    # Check dependencies
    if check_deps:
        DeviceManager.check_dependencies()
    
    # Set seeds
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Configure backends for performance
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    
    # Select device
    manager = DeviceManager(preferred_device)
    device = manager.get_device()
    device_info = manager.get_device_info()
    
    print("=" * 60)
    print(f"✅ Environment ready | Seed: {seed} | Device: {device.type}")
    print("=" * 60)
    print()
    
    return device, device_info

