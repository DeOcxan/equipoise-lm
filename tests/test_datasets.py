"""
Unit tests for dataset loaders.

Tests real dataset integration and edge cases.

Author: DeOcxan
Date: October 2025
"""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.dataset_loaders import (
    CIFAR10Loader,
    CIFAR100Loader,
    NoisyImageWrapper,
    create_dataloader
)


class TestDatasetLoaders:
    """Test dataset loading functionality."""
    
    def test_cifar10_loader_basic(self):
        """Test CIFAR-10 basic loading."""
        loader = CIFAR10Loader(root="./test_data", download=True)
        
        train_dataset, test_dataset = loader.load_dataset()
        
        # Check sizes
        assert len(train_dataset) == 50000
        assert len(test_dataset) == 10000
        
        # Check sample
        img, label = train_dataset[0]
        assert img.shape == (3, 32, 32)
        assert isinstance(label, int)
        assert 0 <= label < 10
        
        # Check metadata
        metadata = loader.get_metadata()
        assert metadata['dataset_name'] == 'CIFAR-10'
        assert metadata['num_classes'] == 10
    
    def test_cifar10_with_noise(self):
        """Test CIFAR-10 with noise for denoising."""
        loader = CIFAR10Loader(
            root="./test_data",
            download=True,
            noise_std=0.2
        )
        
        train_dataset, test_dataset = loader.load_dataset()
        
        # Check sample returns clean and noisy pair
        clean, noisy = train_dataset[0]
        
        assert clean.shape == noisy.shape == (3, 32, 32)
        assert not torch.allclose(clean, noisy), "Clean and noisy should differ"
        
        # Check noise is within bounds
        assert (noisy >= 0).all() and (noisy <= 1).all()
    
    def test_noisy_wrapper_noise_level(self):
        """Test that noise wrapper adds appropriate noise."""
        # Create simple dataset
        clean_data = torch.ones(10, 3, 32, 32)
        
        class SimpleDataset:
            def __len__(self):
                return 10
            def __getitem__(self, idx):
                return clean_data[idx]
        
        base_dataset = SimpleDataset()
        
        # Test different noise levels
        noise_levels = [0.1, 0.2, 0.5]
        
        for noise_std in noise_levels:
            wrapper = NoisyImageWrapper(base_dataset, noise_std)
            
            clean, noisy = wrapper[0]
            
            # Calculate actual noise
            noise = (noisy - clean).abs().mean()
            
            # Noise should be roughly noise_std (with tolerance)
            assert 0 < noise < noise_std * 3, \
                f"Noise level {noise} not appropriate for std {noise_std}"
    
    def test_dataloader_factory(self):
        """Test dataloader factory function."""
        train_loader, val_loader, metadata = create_dataloader(
            dataset_name='cifar10',
            batch_size=32,
            num_workers=0,  # Use 0 for testing
            root='./test_data',
            download=True
        )
        
        # Check dataloaders work
        batch = next(iter(train_loader))
        images, labels = batch
        
        assert images.shape[0] <= 32
        assert images.shape[1:] == (3, 32, 32)
        assert len(labels) == images.shape[0]
        
        # Check metadata
        assert metadata['dataset_name'] == 'CIFAR-10'
        assert metadata['num_classes'] == 10
    
    def test_invalid_dataset_name(self):
        """Test that invalid dataset name raises error."""
        with pytest.raises(ValueError, match="Unknown dataset"):
            create_dataloader(
                dataset_name='invalid_dataset',
                batch_size=32
            )
    
    def test_batch_consistency(self):
        """Test that batches are consistent across iterations."""
        train_loader, _, _ = create_dataloader(
            dataset_name='cifar10',
            batch_size=32,
            num_workers=0,
            root='./test_data',
            download=True
        )
        
        # Get first batch twice (with same seed)
        torch.manual_seed(42)
        batch1 = next(iter(train_loader))
        
        # Should be able to iterate multiple times
        for _ in range(3):
            batch = next(iter(train_loader))
            assert batch[0].shape == batch1[0].shape


class TestCIFAR100Loader:
    """Test CIFAR-100 dataset loader."""
    
    def test_cifar100_basic(self):
        """Test CIFAR-100 basic loading."""
        loader = CIFAR100Loader(root="./test_data", download=True)
        
        train_dataset, test_dataset = loader.load_dataset()
        
        # Check sizes
        assert len(train_dataset) == 50000
        assert len(test_dataset) == 10000
        
        # Check sample
        img, label = train_dataset[0]
        assert img.shape == (3, 32, 32)
        assert isinstance(label, int)
        assert 0 <= label < 100
        
        # Check metadata
        metadata = loader.get_metadata()
        assert metadata['dataset_name'] == 'CIFAR-100'
        assert metadata['num_classes'] == 100


class TestDatasetEdgeCases:
    """Test edge cases in dataset handling."""
    
    def test_empty_batch_handling(self):
        """Test handling of edge case batch sizes."""
        # Test with batch_size = 1
        train_loader, _, _ = create_dataloader(
            dataset_name='cifar10',
            batch_size=1,
            num_workers=0,
            root='./test_data',
            download=True
        )
        
        batch = next(iter(train_loader))
        assert batch[0].shape[0] == 1
    
    def test_large_batch_size(self):
        """Test with very large batch size."""
        # Test with batch_size larger than dataset split
        train_loader, _, _ = create_dataloader(
            dataset_name='cifar10',
            batch_size=60000,  # Larger than dataset
            num_workers=0,
            root='./test_data',
            download=True
        )
        
        batch = next(iter(train_loader))
        # Should still work, just return full dataset
        assert batch[0].shape[0] <= 60000
    
    def test_extreme_noise_levels(self):
        """Test denoising with extreme noise levels."""
        loader = CIFAR10Loader(
            root="./test_data",
            download=True,
            noise_std=1.0  # Very high noise
        )
        
        train_dataset, _ = loader.load_dataset()
        clean, noisy = train_dataset[0]
        
        # Even with extreme noise, values should be clipped
        assert (noisy >= 0).all() and (noisy <= 1).all()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

