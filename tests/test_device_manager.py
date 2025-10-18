"""
Unit tests for device manager and CPU fallback.

Tests compatibility across different hardware configurations.

Author: DeOcxan
Date: October 2025
"""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.device_manager import DeviceManager, auto_select_device, setup_training_environment


class TestDeviceManager:
    """Test device selection and fallback logic."""
    
    def test_cpu_selection(self):
        """Test explicit CPU selection."""
        manager = DeviceManager(preferred_device='cpu')
        device = manager.get_device()
        
        assert device.type == 'cpu'
        
        info = manager.get_device_info()
        assert info['type'] == 'cpu'
    
    def test_auto_selection(self):
        """Test automatic device selection."""
        manager = DeviceManager(preferred_device=None)
        device = manager.get_device()
        
        # Should select best available device
        assert device.type in ['cpu', 'cuda', 'mps']
        
        info = manager.get_device_info()
        assert 'type' in info
        assert 'name' in info
    
    def test_move_to_device(self):
        """Test moving tensors to device."""
        manager = DeviceManager(preferred_device='cpu')
        
        tensor = torch.randn(10, 10)
        moved_tensor = manager.move_to_device(tensor)
        
        assert moved_tensor.device.type == 'cpu'
    
    def test_move_model_to_device(self):
        """Test moving model to device."""
        import torch.nn as nn
        
        manager = DeviceManager(preferred_device='cpu')
        
        model = nn.Linear(10, 10)
        moved_model = manager.move_to_device(model)
        
        # Check model parameters are on correct device
        for param in moved_model.parameters():
            assert param.device.type == 'cpu'
    
    def test_device_info_format(self):
        """Test device info contains expected fields."""
        manager = DeviceManager()
        info = manager.get_device_info()
        
        assert isinstance(info, dict)
        assert 'type' in info
        assert 'name' in info
        
        if info['type'] == 'cuda':
            assert 'count' in info
            assert 'memory' in info
    
    def test_auto_select_convenience_function(self):
        """Test convenience function for device selection."""
        device, info = auto_select_device()
        
        assert isinstance(device, torch.device)
        assert isinstance(info, dict)
        assert device.type in ['cpu', 'cuda', 'mps']
    
    def test_setup_training_environment(self):
        """Test complete environment setup."""
        device, info = setup_training_environment(
            seed=42,
            preferred_device='cpu',
            check_deps=False  # Skip dependency check in tests
        )
        
        assert isinstance(device, torch.device)
        assert device.type == 'cpu'
        
        # Check that seed was set
        rand1 = torch.rand(5)
        torch.manual_seed(42)
        rand2 = torch.rand(5)
        
        assert torch.allclose(rand1, rand2)
    
    def test_memory_summary_cpu(self):
        """Test memory summary on CPU."""
        manager = DeviceManager(preferred_device='cpu')
        summary = manager.memory_summary()
        
        assert isinstance(summary, str)
        assert 'CPU' in summary
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_selection(self):
        """Test CUDA selection when available."""
        manager = DeviceManager(preferred_device='cuda')
        device = manager.get_device()
        
        assert device.type == 'cuda'
        
        info = manager.get_device_info()
        assert info['type'] == 'cuda'
        assert 'memory' in info
        assert info['memory'] > 0
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_memory_summary_cuda(self):
        """Test memory summary on CUDA."""
        manager = DeviceManager(preferred_device='cuda')
        summary = manager.memory_summary()
        
        assert isinstance(summary, str)
        assert 'GPU Memory' in summary
        assert 'GB' in summary
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_empty_cache(self):
        """Test cache emptying on CUDA."""
        manager = DeviceManager(preferred_device='cuda')
        
        # Create some tensors
        tensors = [torch.randn(1000, 1000).cuda() for _ in range(10)]
        del tensors
        
        # Empty cache should not raise error
        manager.empty_cache()
    
    def test_optimal_num_workers(self):
        """Test optimal worker count detection."""
        num_workers = DeviceManager.get_optimal_num_workers()
        
        assert isinstance(num_workers, int)
        assert num_workers >= 0
        
        # On Windows should be 0
        import sys
        if sys.platform == 'win32':
            assert num_workers == 0


class TestDeviceCompatibility:
    """Test compatibility across different device types."""
    
    def test_tensor_operations_cpu(self):
        """Test basic tensor operations on CPU."""
        manager = DeviceManager(preferred_device='cpu')
        device = manager.get_device()
        
        a = torch.randn(10, 10).to(device)
        b = torch.randn(10, 10).to(device)
        
        c = a @ b
        
        assert c.device == device
        assert not torch.isnan(c).any()
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_tensor_operations_cuda(self):
        """Test basic tensor operations on CUDA."""
        manager = DeviceManager(preferred_device='cuda')
        device = manager.get_device()
        
        a = torch.randn(10, 10).to(device)
        b = torch.randn(10, 10).to(device)
        
        c = a @ b
        
        assert c.device.type == 'cuda'
        assert not torch.isnan(c).any()
    
    def test_mixed_device_error_handling(self):
        """Test that mixing devices raises appropriate errors."""
        a = torch.randn(10, 10).to('cpu')
        
        if torch.cuda.is_available():
            b = torch.randn(10, 10).to('cuda')
            
            with pytest.raises(RuntimeError):
                # Should raise error about different devices
                c = a @ b


class TestDependencyChecking:
    """Test dependency verification."""
    
    def test_check_dependencies(self):
        """Test dependency checking functionality."""
        # This should pass in our environment
        result = DeviceManager.check_dependencies()
        
        # Should return boolean
        assert isinstance(result, bool)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

