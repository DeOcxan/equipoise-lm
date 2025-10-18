"""
Energy-Based Model for Image Denoising

This module implements a bidirectional Energy-Based Model for image denoising.
Unlike autoregressive models, this can process all pixels simultaneously and
iteratively refine the entire image through energy minimization.

Key features:
- Bidirectional processing (no autoregressive constraints)
- Iterative denoising through gradient descent on energy
- Adaptive compute allocation per image
- Can denoise images of varying noise levels

Author: DeOcxan  
Date: October 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

sys.path.append(str(Path(__file__).parent.parent))

from utils.base_energy_model import BaseTrainableEBM
from utils.model_utils import count_parameters, format_number


class ConvBlock(nn.Module):
    """
    Convolutional block with normalization and activation.
    
    Follows Single Responsibility Principle: handles one conv operation.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        use_bn: bool = True
    ):
        """
        Initialize conv block.
        
        Args:
            in_channels: Input channels
            out_channels: Output channels
            kernel_size: Kernel size
            stride: Stride
            padding: Padding
            use_bn: Whether to use batch normalization
        """
        super().__init__()
        
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)]
        
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        
        layers.append(nn.ReLU(inplace=True))
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    """
    Residual block for deep feature extraction.
    
    Implements skip connections for better gradient flow.
    """
    
    def __init__(self, channels: int):
        """
        Initialize residual block.
        
        Args:
            channels: Number of channels
        """
        super().__init__()
        
        self.conv1 = ConvBlock(channels, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.bn = nn.BatchNorm2d(channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn(self.conv2(out))
        out = out + residual
        return F.relu(out)


class EBT_IMG_Denoise(BaseTrainableEBM):
    """
    Energy-Based Model for Image Denoising.
    
    Architecture:
        Noisy Image -> CNN Encoder -> Feature Processing -> 
        Energy Head -> Energy (scalar per image)
    
    Denoising Process:
        1. Start with noisy image
        2. Compute energy for current image
        3. Compute gradient of energy w.r.t. pixels
        4. Update pixels via gradient descent
        5. Repeat until energy minimized
    
    This demonstrates bidirectional energy-based processing for images.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        num_res_blocks: int = 4,
        image_size: int = 28,
        num_optim_steps: int = 5,
        optim_step_size: float = 0.01,
        device: str = 'cpu'
    ):
        """
        Initialize EBT for image denoising.
        
        Args:
            in_channels: Number of input channels (1 for grayscale, 3 for RGB)
            base_channels: Base number of channels for conv layers
            num_res_blocks: Number of residual blocks
            image_size: Input image size
            num_optim_steps: Steps for iterative denoising
            optim_step_size: Step size for gradient descent
            device: Device to run on
        """
        config = {
            'model_name': 'EBT_IMG_Denoise',
            'in_channels': in_channels,
            'base_channels': base_channels,
            'num_res_blocks': num_res_blocks,
            'image_size': image_size,
            'device': torch.device(device),
            'num_optim_steps': num_optim_steps,
            'optim_step_size': optim_step_size,
            'langevin_noise_scale': 0.005,
            'l2_regularization': 0.0001
        }
        
        super().__init__(config)
        
        self.in_channels = in_channels
        self.base_channels = base_channels
        
        # Encoder: extract features from image
        self.encoder = nn.Sequential(
            ConvBlock(in_channels, base_channels),
            ConvBlock(base_channels, base_channels * 2),
            nn.MaxPool2d(2),
            ConvBlock(base_channels * 2, base_channels * 4),
            nn.MaxPool2d(2)
        )
        
        # Residual blocks for deep feature processing
        self.res_blocks = nn.Sequential(*[
            ResidualBlock(base_channels * 4) for _ in range(num_res_blocks)
        ])
        
        # Global pooling and energy head
        # Calculate feature map size after pooling
        feature_size = image_size // 4  # Two maxpool layers
        
        self.energy_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global average pooling
            nn.Flatten(),
            nn.Linear(base_channels * 4, base_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(base_channels, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Move to device
        self.to(self.device)
        
        print(f"EBT_IMG_Denoise initialized with {format_number(count_parameters(self))} parameters")
        print(f"Image size: {image_size}x{image_size}, Channels: {in_channels}")
        
    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
    
    def compute_energy(self, image: torch.Tensor) -> torch.Tensor:
        """
        Compute energy for an image.
        
        Lower energy = cleaner/better image
        
        Args:
            image: Input image (batch_size, channels, height, width)
            
        Returns:
            Energy values (batch_size, 1)
        """
        # Extract features
        features = self.encoder(image)
        
        # Process with residual blocks
        features = self.res_blocks(features)
        
        # Compute energy
        energy = self.energy_head(features)
        
        return energy
    
    def denoise_image(
        self,
        noisy_image: torch.Tensor,
        num_steps: int = None,
        step_size: float = None,
        add_noise: bool = False,
        return_trajectory: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Denoise image through iterative energy minimization.
        
        Process:
        1. Start with noisy image
        2. Compute energy and gradients
        3. Update pixels to reduce energy
        4. Repeat until converged
        
        Args:
            noisy_image: Noisy input image (batch, channels, H, W)
            num_steps: Number of optimization steps
            step_size: Step size for updates
            add_noise: Whether to add Langevin noise
            return_trajectory: Whether to return energy trajectory
            
        Returns:
            Tuple of (denoised_image, energy_trajectory)
        """
        if num_steps is None:
            num_steps = self._default_num_steps
        if step_size is None:
            step_size = self._default_step_size
        
        # Start with noisy image
        current_image = noisy_image.detach().clone().requires_grad_(True)
        
        energy_trajectory = []
        
        for step in range(num_steps):
            # Compute energy
            energy = self.compute_energy(current_image)
            energy_trajectory.append(energy.mean().item())
            
            # Compute gradients w.r.t. image pixels
            energy_sum = energy.sum()
            grads = torch.autograd.grad(
                outputs=energy_sum,
                inputs=current_image,
                create_graph=False,
                retain_graph=False
            )[0]
            
            # Update image via gradient descent
            with torch.no_grad():
                current_image = current_image - step_size * grads
                
                # Clip to valid range [0, 1]
                current_image = torch.clamp(current_image, 0.0, 1.0)
                
                # Add Langevin noise
                if add_noise:
                    noise = torch.randn_like(current_image) * self._langevin_noise_scale
                    current_image = current_image + noise
                    current_image = torch.clamp(current_image, 0.0, 1.0)
            
            current_image = current_image.detach().requires_grad_(True)
        
        # Final energy
        final_energy = self.compute_energy(current_image)
        energy_trajectory.append(final_energy.mean().item())
        
        denoised = current_image.detach()
        
        if return_trajectory:
            return denoised, torch.tensor(energy_trajectory)
        else:
            return denoised, final_energy
    
    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Training step for denoising.
        
        Objective:
        - Clean images should have low energy
        - Noisy images should have high energy
        
        Args:
            batch: Tuple of (clean_images, noisy_images)
            batch_idx: Batch index
            
        Returns:
            Dictionary with loss and metrics
        """
        clean_images, noisy_images = batch
        clean_images = clean_images.to(self.device)
        noisy_images = noisy_images.to(self.device)
        
        # Compute energy for clean and noisy images
        energy_clean = self.compute_energy(clean_images)
        energy_noisy = self.compute_energy(noisy_images)
        
        # Loss: minimize energy for clean, maximize for noisy
        # This teaches the model that clean = low energy
        loss = energy_clean.mean() - energy_noisy.mean()
        
        # Add L2 regularization
        if self._l2_reg > 0:
            l2_penalty = sum(torch.sum(p ** 2) for p in self.parameters())
            loss = loss + self._l2_reg * l2_penalty
        
        # Compute denoising MSE for monitoring
        denoised, _ = self.denoise_image(noisy_images, num_steps=3)
        with torch.no_grad():
            mse = F.mse_loss(denoised, clean_images)
            psnr = 10 * torch.log10(1.0 / mse)
        
        return {
            'loss': loss,
            'energy_clean': energy_clean.mean(),
            'energy_noisy': energy_noisy.mean(),
            'mse': mse,
            'psnr': psnr
        }
    
    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Validation step."""
        with torch.no_grad():
            return self.training_step(batch, batch_idx)


class NoisyImageDataset(Dataset):
    """
    Dataset that adds noise to clean images.
    
    Useful for training denoising models.
    """
    
    def __init__(
        self,
        base_dataset: Dataset,
        noise_std: float = 0.1
    ):
        """
        Initialize noisy dataset.
        
        Args:
            base_dataset: Clean image dataset
            noise_std: Standard deviation of Gaussian noise
        """
        self.base_dataset = base_dataset
        self.noise_std = noise_std
    
    def __len__(self) -> int:
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get clean and noisy image pair.
        
        Returns:
            Tuple of (clean_image, noisy_image)
        """
        if isinstance(self.base_dataset[idx], tuple):
            clean_image, _ = self.base_dataset[idx]
        else:
            clean_image = self.base_dataset[idx]
        
        # Add Gaussian noise
        noise = torch.randn_like(clean_image) * self.noise_std
        noisy_image = clean_image + noise
        noisy_image = torch.clamp(noisy_image, 0.0, 1.0)
        
        return clean_image, noisy_image

