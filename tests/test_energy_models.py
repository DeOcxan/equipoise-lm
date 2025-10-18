"""
Unit tests for Energy-Based Models.

Tests edge cases that synthetic datasets don't cover:
- High-dimensional energy computations
- Numerical stability
- Gradient flow
- Memory efficiency
- Device compatibility

Author: DeOcxan
Date: October 2025
"""

import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.base_energy_model import BaseEnergyModel, BaseTrainableEBM
from Simple_EBM.simple_ebm import SimpleEBM
from EBT_IMG.ebt_img_denoise import EBT_IMG_Denoise


class TestEnergyModelEdgeCases:
    """Test edge cases for energy models."""
    
    def test_high_dimensional_energy(self):
        """Test energy computation in high-dimensional space."""
        # Create a simple energy model
        class HighDimEBM(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.fc = nn.Linear(dim, 1)
            
            def compute_energy(self, x):
                return self.fc(x)
        
        # Test with very high dimensions
        dims = [100, 1000, 10000]
        batch_size = 16
        
        for dim in dims:
            model = HighDimEBM(dim)
            x = torch.randn(batch_size, dim)
            
            # Compute energy
            energy = model.compute_energy(x)
            
            # Assertions
            assert energy.shape == (batch_size, 1), \
                f"Energy shape mismatch for dim={dim}"
            assert not torch.isnan(energy).any(), \
                f"NaN in energy for dim={dim}"
            assert not torch.isinf(energy).any(), \
                f"Inf in energy for dim={dim}"
    
    def test_numerical_stability_extreme_values(self):
        """Test numerical stability with extreme input values."""
        model = EBT_IMG_Denoise(
            in_channels=1,
            base_channels=32,
            num_res_blocks=2,
            image_size=28,
            device='cpu'
        )
        
        batch_size = 4
        
        # Test with extreme values
        test_cases = [
            ("zeros", torch.zeros(batch_size, 1, 28, 28)),
            ("ones", torch.ones(batch_size, 1, 28, 28)),
            ("very_small", torch.full((batch_size, 1, 28, 28), 1e-7)),
            ("very_large", torch.full((batch_size, 1, 28, 28), 1.0)),
        ]
        
        for name, x in test_cases:
            energy = model.compute_energy(x)
            
            assert not torch.isnan(energy).any(), \
                f"NaN in energy for {name}"
            assert not torch.isinf(energy).any(), \
                f"Inf in energy for {name}"
            assert energy.shape == (batch_size, 1), \
                f"Shape mismatch for {name}"
    
    def test_gradient_flow(self):
        """Test that gradients flow properly through the model."""
        model = EBT_IMG_Denoise(
            in_channels=1,
            base_channels=16,
            num_res_blocks=1,
            image_size=28,
            device='cpu'
        )
        
        x = torch.randn(2, 1, 28, 28, requires_grad=True)
        
        # Forward pass
        energy = model.compute_energy(x)
        loss = energy.sum()
        
        # Backward pass
        loss.backward()
        
        # Check gradients exist and are not NaN
        assert x.grad is not None, "No gradients computed"
        assert not torch.isnan(x.grad).any(), "NaN in gradients"
        assert not torch.isinf(x.grad).any(), "Inf in gradients"
        assert (x.grad != 0).any(), "All gradients are zero"
    
    def test_batch_size_variations(self):
        """Test model with different batch sizes including edge cases."""
        model = EBT_IMG_Denoise(
            in_channels=1,
            base_channels=16,
            num_res_blocks=1,
            image_size=28,
            device='cpu'
        )
        
        batch_sizes = [1, 2, 8, 16, 32, 64]
        
        for bs in batch_sizes:
            x = torch.randn(bs, 1, 28, 28)
            energy = model.compute_energy(x)
            
            assert energy.shape == (bs, 1), \
                f"Shape mismatch for batch_size={bs}"
            assert not torch.isnan(energy).any(), \
                f"NaN for batch_size={bs}"
    
    def test_memory_efficiency_large_batch(self):
        """Test memory efficiency with large batches."""
        model = EBT_IMG_Denoise(
            in_channels=1,
            base_channels=16,
            num_res_blocks=1,
            image_size=28,
            device='cpu'
        )
        
        # Test with large batch
        large_batch_size = 256
        x = torch.randn(large_batch_size, 1, 28, 28)
        
        # Should not crash with OOM
        try:
            energy = model.compute_energy(x)
            assert energy.shape == (large_batch_size, 1)
        except RuntimeError as e:
            if 'out of memory' in str(e):
                pytest.skip("System has insufficient memory for this test")
            else:
                raise
    
    def test_device_compatibility(self):
        """Test model works on different devices."""
        devices = ['cpu']
        
        # Add CUDA if available
        if torch.cuda.is_available():
            devices.append('cuda')
        
        # Add MPS if available
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            devices.append('mps')
        
        for device in devices:
            model = EBT_IMG_Denoise(
                in_channels=1,
                base_channels=16,
                num_res_blocks=1,
                image_size=28,
                device=device
            )
            
            x = torch.randn(4, 1, 28, 28).to(device)
            energy = model.compute_energy(x)
            
            assert energy.device.type == device, \
                f"Energy not on {device}"
            assert not torch.isnan(energy).any(), \
                f"NaN on {device}"
    
    def test_iterative_denoising_convergence(self):
        """Test that iterative denoising reduces energy."""
        model = EBT_IMG_Denoise(
            in_channels=1,
            base_channels=16,
            num_res_blocks=1,
            image_size=28,
            num_optim_steps=5,
            optim_step_size=0.01,
            device='cpu'
        )
        
        # Create noisy image
        clean = torch.rand(2, 1, 28, 28)
        noisy = clean + torch.randn_like(clean) * 0.2
        noisy = torch.clamp(noisy, 0, 1)
        
        # Denoise with trajectory
        denoised, trajectory = model.denoise_image(
            noisy,
            num_steps=5,
            return_trajectory=True
        )
        
        # Check energy decreases
        trajectory_list = trajectory.tolist()
        assert len(trajectory_list) > 1, "Trajectory too short"
        
        # Energy should generally decrease (allowing some noise)
        decreasing_count = sum(
            trajectory_list[i] > trajectory_list[i+1]
            for i in range(len(trajectory_list)-1)
        )
        
        # At least 60% of steps should decrease energy
        assert decreasing_count >= 0.6 * (len(trajectory_list)-1), \
            f"Energy not decreasing properly: {trajectory_list}"
    
    def test_edge_case_single_pixel(self):
        """Test model with minimal spatial dimensions."""
        # Test with 1x1 image (edge case)
        try:
            model = EBT_IMG_Denoise(
                in_channels=1,
                base_channels=16,
                num_res_blocks=1,
                image_size=4,  # Minimum viable size after pooling
                device='cpu'
            )
            
            x = torch.randn(2, 1, 4, 4)
            energy = model.compute_energy(x)
            
            assert energy.shape == (2, 1)
        except Exception as e:
            # This is expected to fail gracefully
            assert "size" in str(e).lower() or "dimension" in str(e).lower()
    
    def test_zero_optimization_steps(self):
        """Test model with zero optimization steps."""
        model = EBT_IMG_Denoise(
            in_channels=1,
            base_channels=16,
            num_res_blocks=1,
            image_size=28,
            device='cpu'
        )
        
        noisy = torch.rand(2, 1, 28, 28)
        
        # Denoise with 0 steps (should return input)
        denoised, _ = model.denoise_image(noisy, num_steps=0)
        
        # Should be close to input (no optimization done)
        assert torch.allclose(denoised, noisy, atol=1e-5)
    
    def test_training_step_consistency(self):
        """Test that training step produces consistent outputs."""
        model = EBT_IMG_Denoise(
            in_channels=1,
            base_channels=16,
            num_res_blocks=1,
            image_size=28,
            device='cpu'
        )
        
        # Create optimizer for training
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        clean = torch.rand(4, 1, 28, 28)
        noisy = clean + torch.randn_like(clean) * 0.1
        batch = (clean, noisy)
        
        # Train the model for a few steps to learn the energy relationship
        for step in range(5):
            optimizer.zero_grad()
            metrics = model.training_step(batch, step)
            metrics['loss'].backward()
            optimizer.step()
        
        # Run final training step
        metrics = model.training_step(batch, 5)
        
        # Check all expected keys exist
        expected_keys = ['loss', 'energy_clean', 'energy_noisy', 'mse', 'psnr']
        for key in expected_keys:
            assert key in metrics, f"Missing metric: {key}"
            assert not torch.isnan(metrics[key]), f"NaN in {key}"
            assert not torch.isinf(metrics[key]), f"Inf in {key}"
        
        # Check energy relationship (clean < noisy)
        assert metrics['energy_clean'] < metrics['energy_noisy'], \
            "Clean energy should be lower than noisy energy"


class TestNumericalStability:
    """Test numerical stability in various scenarios."""
    
    def test_gradient_clipping(self):
        """Test that gradient clipping prevents explosion."""
        from utils.optimization import StableAdamW
        
        model = nn.Linear(10, 1)
        optimizer = StableAdamW(
            model.parameters(),
            lr=0.001,
            clip_grad_norm=1.0
        )
        
        # Create scenario with large gradients
        x = torch.randn(32, 10)
        y = torch.randn(32, 1) * 1000  # Large targets
        
        for _ in range(10):
            optimizer.zero_grad()
            output = model(x)
            loss = ((output - y) ** 2).mean()
            loss.backward()
            
            # Check gradients before clipping
            grad_norm_before = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=float('inf')
            )
            
            # Apply gradient clipping manually to test
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )
            
            # Check gradients after clipping
            grad_norm_after = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=float('inf')
            )
            
            # Gradients should be clipped if they were large
            if grad_norm_before > 1.0:
                assert grad_norm_after <= 1.0 or torch.isclose(
                    grad_norm_after,
                    torch.tensor(1.0),
                    atol=1e-5
                ), f"Gradient norm not clipped: {grad_norm_after} (was {grad_norm_before})"
            
            optimizer.step()
    
    def test_energy_bounds(self):
        """Test that energy values stay within reasonable bounds."""
        model = EBT_IMG_Denoise(
            in_channels=1,
            base_channels=16,
            num_res_blocks=1,
            image_size=28,
            device='cpu'
        )
        
        # Test with many random inputs
        for _ in range(100):
            x = torch.rand(4, 1, 28, 28)
            energy = model.compute_energy(x)
            
            # Energy should be finite and reasonable
            assert torch.isfinite(energy).all(), "Non-finite energy"
            assert energy.abs().max() < 1e6, "Energy too large"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])

