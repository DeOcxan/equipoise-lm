"""
Real Dataset Loaders for EBT-Mastery

This module provides loaders for real-world datasets:
- GLUE benchmark (NLP tasks)
- CIFAR-10/100 (Image classification)
- ImageNet subset (Image classification)

Replaces synthetic datasets with production-grade data for credibility.

Author: DeOcxan
Date: October 2025
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, List
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
import urllib.request
import json
import os


class IDatasetLoader(ABC):
    """
    Interface for dataset loaders.
    
    Follows Interface Segregation Principle (ISP).
    """
    
    @abstractmethod
    def load_dataset(self) -> Tuple[Dataset, Dataset]:
        """Load train and validation datasets."""
        pass
    
    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Get dataset metadata (vocab size, num classes, etc.)."""
        pass


class CIFAR10Loader(IDatasetLoader):
    """
    CIFAR-10 dataset loader with preprocessing.
    
    Replaces synthetic MNIST with real diverse images.
    """
    
    def __init__(
        self,
        root: str = "./data",
        download: bool = True,
        augment_train: bool = True,
        noise_std: Optional[float] = None
    ):
        """
        Initialize CIFAR-10 loader.
        
        Args:
            root: Root directory for data
            download: Whether to download if not present
            augment_train: Whether to use data augmentation
            noise_std: If provided, add Gaussian noise for denoising task
        """
        self._root = Path(root)
        self._download = download
        self._augment_train = augment_train
        self._noise_std = noise_std
        
        self._root.mkdir(parents=True, exist_ok=True)
    
    def load_dataset(self) -> Tuple[Dataset, Dataset]:
        """Load CIFAR-10 train and test sets."""
        # Normalization for CIFAR-10
        normalize = transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616]
        )
        
        # Training transforms with augmentation
        train_transforms = [
            transforms.ToTensor(),
        ]
        
        if self._augment_train:
            train_transforms = [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        
        if self._noise_std is None:
            train_transforms.append(normalize)
        
        train_transform = transforms.Compose(train_transforms)
        
        # Test transforms (no augmentation)
        test_transforms = [transforms.ToTensor()]
        if self._noise_std is None:
            test_transforms.append(normalize)
        
        test_transform = transforms.Compose(test_transforms)
        
        # Load datasets
        train_dataset = torchvision.datasets.CIFAR10(
            root=str(self._root),
            train=True,
            download=self._download,
            transform=train_transform
        )
        
        test_dataset = torchvision.datasets.CIFAR10(
            root=str(self._root),
            train=False,
            download=self._download,
            transform=test_transform
        )
        
        # Wrap with noise if denoising task
        if self._noise_std is not None:
            train_dataset = NoisyImageWrapper(train_dataset, self._noise_std)
            test_dataset = NoisyImageWrapper(test_dataset, self._noise_std)
        
        return train_dataset, test_dataset
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get dataset metadata."""
        return {
            'dataset_name': 'CIFAR-10',
            'num_classes': 10,
            'image_size': 32,
            'channels': 3,
            'train_samples': 50000,
            'test_samples': 10000,
            'class_names': [
                'airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck'
            ]
        }


class CIFAR100Loader(IDatasetLoader):
    """
    CIFAR-100 dataset loader.
    
    More challenging than CIFAR-10 with 100 fine-grained classes.
    """
    
    def __init__(
        self,
        root: str = "./data",
        download: bool = True,
        augment_train: bool = True
    ):
        """Initialize CIFAR-100 loader."""
        self._root = Path(root)
        self._download = download
        self._augment_train = augment_train
        
        self._root.mkdir(parents=True, exist_ok=True)
    
    def load_dataset(self) -> Tuple[Dataset, Dataset]:
        """Load CIFAR-100 train and test sets."""
        normalize = transforms.Normalize(
            mean=[0.5071, 0.4865, 0.4409],
            std=[0.2673, 0.2564, 0.2762]
        )
        
        train_transforms = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            normalize
        ] if self._augment_train else [
            transforms.ToTensor(),
            normalize
        ]
        
        train_transform = transforms.Compose(train_transforms)
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
        
        train_dataset = torchvision.datasets.CIFAR100(
            root=str(self._root),
            train=True,
            download=self._download,
            transform=train_transform
        )
        
        test_dataset = torchvision.datasets.CIFAR100(
            root=str(self._root),
            train=False,
            download=self._download,
            transform=test_transform
        )
        
        return train_dataset, test_dataset
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get dataset metadata."""
        return {
            'dataset_name': 'CIFAR-100',
            'num_classes': 100,
            'image_size': 32,
            'channels': 3,
            'train_samples': 50000,
            'test_samples': 10000
        }


class NoisyImageWrapper(Dataset):
    """
    Wraps a clean image dataset to produce clean/noisy pairs.
    
    Used for denoising tasks on real datasets.
    """
    
    def __init__(
        self,
        clean_dataset: Dataset,
        noise_std: float = 0.1
    ):
        """
        Initialize wrapper.
        
        Args:
            clean_dataset: Clean image dataset
            noise_std: Standard deviation of Gaussian noise
        """
        self._clean_dataset = clean_dataset
        self._noise_std = noise_std
    
    def __len__(self) -> int:
        return len(self._clean_dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get clean and noisy image pair.
        
        Returns:
            Tuple of (clean_image, noisy_image)
        """
        if isinstance(self._clean_dataset[idx], tuple):
            clean_image, label = self._clean_dataset[idx]
        else:
            clean_image = self._clean_dataset[idx]
        
        # Add Gaussian noise
        noise = torch.randn_like(clean_image) * self._noise_std
        noisy_image = clean_image + noise
        noisy_image = torch.clamp(noisy_image, 0.0, 1.0)
        
        return clean_image, noisy_image


class GLUEDatasetLoader(IDatasetLoader):
    """
    GLUE (General Language Understanding Evaluation) benchmark loader.
    
    Provides real NLP tasks: sentiment analysis, natural language inference, etc.
    More credible than synthetic repetition patterns.
    """
    
    TASK_CONFIGS = {
        'cola': {'num_labels': 2, 'metric': 'matthews_corrcoef'},
        'sst2': {'num_labels': 2, 'metric': 'accuracy'},
        'mrpc': {'num_labels': 2, 'metric': 'accuracy'},
        'qqp': {'num_labels': 2, 'metric': 'accuracy'},
        'mnli': {'num_labels': 3, 'metric': 'accuracy'},
        'qnli': {'num_labels': 2, 'metric': 'accuracy'},
        'rte': {'num_labels': 2, 'metric': 'accuracy'},
    }
    
    def __init__(
        self,
        task_name: str = 'sst2',
        root: str = "./data/glue",
        max_seq_length: int = 128,
        tokenizer_type: str = 'char'
    ):
        """
        Initialize GLUE loader.
        
        Args:
            task_name: GLUE task name ('sst2', 'cola', 'mrpc')
            root: Root directory for data
            max_seq_length: Maximum sequence length
            tokenizer_type: Tokenizer type ('char' or 'word')
        """
        self._task_name = task_name.lower()
        self._root = Path(root)
        self._max_seq_length = max_seq_length
        self._tokenizer_type = tokenizer_type
        
        if self._task_name not in self.TASK_CONFIGS:
            raise ValueError(
                f"Unknown GLUE task: {task_name}. "
                f"Available: {list(self.TASK_CONFIGS.keys())}"
            )
        
        self._root.mkdir(parents=True, exist_ok=True)
        self._vocab = None
    
    def load_dataset(self) -> Tuple[Dataset, Dataset]:
        """
        Load GLUE task dataset.
        
        Note: This is a simplified implementation using local files.
        For full GLUE, use Hugging Face datasets library.
        """
        # Try to load from Hugging Face datasets if available
        try:
            from datasets import load_dataset
            
            dataset = load_dataset('glue', self._task_name, cache_dir=str(self._root))
            train_data = dataset['train']
            val_data = dataset['validation']
            
            # Build vocabulary
            self._build_vocabulary(train_data)
            
            # Convert to PyTorch datasets
            train_dataset = GLUEDataset(
                train_data,
                self._task_name,
                self._vocab,
                self._max_seq_length
            )
            val_dataset = GLUEDataset(
                val_data,
                self._task_name,
                self._vocab,
                self._max_seq_length
            )
            
            return train_dataset, val_dataset
            
        except ImportError:
            print("⚠️  Hugging Face 'datasets' not installed. "
                  "Install with: pip install datasets")
            print("Falling back to synthetic data for demonstration.")
            
            # Fallback to synthetic data
            return self._create_synthetic_fallback()
    
    def _build_vocabulary(self, data):
        """Build vocabulary from data."""
        vocab_set = set()
        
        # Get text fields based on task
        text_fields = self._get_text_fields()
        
        for example in data:
            for field in text_fields:
                if field in example and example[field] is not None:
                    text = example[field]
                    if self._tokenizer_type == 'char':
                        vocab_set.update(text)
                    else:  # word
                        vocab_set.update(text.lower().split())
        
        self._vocab = {token: idx for idx, token in enumerate(sorted(vocab_set))}
        self._vocab['<PAD>'] = len(self._vocab)
        self._vocab['<UNK>'] = len(self._vocab)
    
    def _get_text_fields(self) -> List[str]:
        """Get text field names for each task."""
        field_map = {
            'cola': ['sentence'],
            'sst2': ['sentence'],
            'mrpc': ['sentence1', 'sentence2'],
            'qqp': ['question1', 'question2'],
            'mnli': ['premise', 'hypothesis'],
            'qnli': ['question', 'sentence'],
            'rte': ['sentence1', 'sentence2'],
        }
        return field_map.get(self._task_name, ['sentence'])
    
    def _create_synthetic_fallback(self) -> Tuple[Dataset, Dataset]:
        """Create synthetic dataset as fallback."""
        print("Creating synthetic NLP data for demonstration...")
        
        # Import here to avoid circular dependency
        from .synthetic_datasets import create_synthetic_text_dataset
        
        return create_synthetic_text_dataset(
            num_samples=1000,
            seq_length=self._max_seq_length
        )
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get dataset metadata."""
        config = self.TASK_CONFIGS[self._task_name]
        
        return {
            'dataset_name': f'GLUE-{self._task_name.upper()}',
            'task_name': self._task_name,
            'num_labels': config['num_labels'],
            'metric': config['metric'],
            'vocab_size': len(self._vocab) if self._vocab else 0,
            'max_seq_length': self._max_seq_length
        }


class GLUEDataset(Dataset):
    """PyTorch Dataset wrapper for GLUE tasks."""
    
    def __init__(
        self,
        hf_dataset,
        task_name: str,
        vocab: Dict[str, int],
        max_seq_length: int
    ):
        """Initialize GLUE dataset."""
        self._data = hf_dataset
        self._task_name = task_name
        self._vocab = vocab
        self._max_seq_length = max_seq_length
    
    def __len__(self) -> int:
        return len(self._data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get tokenized example."""
        example = self._data[idx]
        
        # Get text fields based on task
        if self._task_name in ['cola', 'sst2']:
            text = example['sentence']
        elif self._task_name in ['mrpc', 'rte']:
            text = example['sentence1'] + ' ' + example['sentence2']
        elif self._task_name == 'qqp':
            text = example['question1'] + ' ' + example['question2']
        elif self._task_name == 'mnli':
            text = example['premise'] + ' ' + example['hypothesis']
        elif self._task_name == 'qnli':
            text = example['question'] + ' ' + example['sentence']
        else:
            text = str(example)
        
        # Tokenize (character-level for simplicity)
        tokens = [self._vocab.get(c, self._vocab['<UNK>']) for c in text]
        
        # Truncate or pad
        if len(tokens) > self._max_seq_length:
            tokens = tokens[:self._max_seq_length]
        else:
            tokens += [self._vocab['<PAD>']] * (self._max_seq_length - len(tokens))
        
        input_ids = torch.tensor(tokens, dtype=torch.long)
        label = torch.tensor(example['label'], dtype=torch.long)
        
        return input_ids, label


def create_dataloader(
    dataset_name: str,
    batch_size: int = 32,
    num_workers: int = 4,
    **kwargs
) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
    """
    Factory function to create dataloaders for any supported dataset.
    
    Args:
        dataset_name: Name of dataset ('cifar10', 'cifar100', 'glue-sst2', etc.)
        batch_size: Batch size
        num_workers: Number of worker processes
        **kwargs: Additional arguments for dataset loader
        
    Returns:
        Tuple of (train_loader, val_loader, metadata)
    """
    dataset_name = dataset_name.lower()
    
    # Choose appropriate loader
    if dataset_name == 'cifar10':
        loader = CIFAR10Loader(**kwargs)
    elif dataset_name == 'cifar100':
        loader = CIFAR100Loader(**kwargs)
    elif dataset_name.startswith('glue-'):
        task_name = dataset_name.split('-')[1]
        loader = GLUEDatasetLoader(task_name=task_name, **kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Load datasets
    train_dataset, val_dataset = loader.load_dataset()
    metadata = loader.get_metadata()
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader, metadata

