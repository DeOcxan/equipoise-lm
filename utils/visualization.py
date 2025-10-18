"""
Visualization Utilities for Energy-Based Models

This module provides plotting and visualization functions for:
- Energy landscapes
- Training curves
- Generated samples
- Optimization trajectories

Author: DeOcxan
Date: October 2025
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import List, Optional, Tuple, Union
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend


def plot_energy_landscape(
    energy_trajectory: Union[List[float], torch.Tensor, np.ndarray],
    save_path: str = 'energy_landscape.png',
    title: str = 'Energy Landscape During Optimization',
    show_reduction: bool = True
) -> None:
    """
    Plot energy values over optimization steps.
    
    Args:
        energy_trajectory: Energy values at each step
        save_path: Path to save plot
        title: Plot title
        show_reduction: Whether to annotate energy reduction
    """
    # Convert to numpy if needed
    if isinstance(energy_trajectory, torch.Tensor):
        energy_trajectory = energy_trajectory.cpu().numpy()
    elif isinstance(energy_trajectory, list):
        energy_trajectory = np.array(energy_trajectory)
    
    plt.figure(figsize=(10, 6))
    steps = np.arange(len(energy_trajectory))
    
    plt.plot(steps, energy_trajectory, 'b-', linewidth=2, marker='o', markersize=4)
    plt.xlabel('Optimization Step', fontsize=12)
    plt.ylabel('Energy', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Annotate start and end points
    plt.plot(0, energy_trajectory[0], 'go', markersize=10, label='Start')
    plt.plot(len(energy_trajectory)-1, energy_trajectory[-1], 'ro', markersize=10, label='End')
    
    if show_reduction and len(energy_trajectory) > 1:
        reduction = energy_trajectory[0] - energy_trajectory[-1]
        reduction_pct = (reduction / abs(energy_trajectory[0])) * 100
        plt.text(
            0.05, 0.95,
            f'Energy Reduction: {reduction:.4f} ({reduction_pct:.1f}%)',
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
    
    plt.legend()
    plt.tight_layout()
    
    # Create directory if needed
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Energy landscape saved to {save_path}")


def plot_loss_curve(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    save_path: str = 'loss_curve.png',
    title: str = 'Training and Validation Loss',
    log_scale: bool = False
) -> None:
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses (optional)
        save_path: Path to save plot
        title: Plot title
        log_scale: Whether to use log scale for y-axis
    """
    plt.figure(figsize=(10, 6))
    epochs = np.arange(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', linewidth=2, label='Training Loss', marker='o', markersize=3)
    
    if val_losses is not None:
        val_epochs = np.arange(1, len(val_losses) + 1)
        plt.plot(val_epochs, val_losses, 'r-', linewidth=2, label='Validation Loss', marker='s', markersize=3)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss' + (' (log scale)' if log_scale else ''), fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if log_scale:
        plt.yscale('log')
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Loss curve saved to {save_path}")


def plot_multiple_metrics(
    metrics_dict: dict,
    save_path: str = 'metrics.png',
    title: str = 'Training Metrics',
    subplot_layout: Optional[Tuple[int, int]] = None
) -> None:
    """
    Plot multiple metrics in subplots.
    
    Args:
        metrics_dict: Dictionary mapping metric names to value lists
        save_path: Path to save plot
        title: Main plot title
        subplot_layout: (rows, cols) for subplot layout (auto if None)
    """
    num_metrics = len(metrics_dict)
    
    if subplot_layout is None:
        # Auto-determine layout
        cols = min(2, num_metrics)
        rows = (num_metrics + cols - 1) // cols
    else:
        rows, cols = subplot_layout
    
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 4*rows))
    if num_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, (metric_name, values) in enumerate(metrics_dict.items()):
        ax = axes[idx]
        steps = np.arange(1, len(values) + 1)
        ax.plot(steps, values, 'b-', linewidth=2, marker='o', markersize=3)
        ax.set_xlabel('Step', fontsize=10)
        ax.set_ylabel(metric_name, fontsize=10)
        ax.set_title(metric_name, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(num_metrics, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Metrics plot saved to {save_path}")


def plot_generated_samples(
    original: Optional[torch.Tensor] = None,
    generated: Optional[torch.Tensor] = None,
    noisy: Optional[torch.Tensor] = None,
    save_path: str = 'generated_samples.png',
    title: str = 'Generated Samples Comparison',
    num_samples: int = 8
) -> None:
    """
    Plot comparison of original, noisy, and generated samples.
    
    Useful for image denoising and generation tasks.
    
    Args:
        original: Original images (B, C, H, W)
        generated: Generated/denoised images (B, C, H, W)
        noisy: Noisy images (B, C, H, W)
        save_path: Path to save plot
        title: Plot title
        num_samples: Number of samples to display
    """
    # Determine how many rows we need
    rows_to_plot = []
    row_labels = []
    
    if original is not None:
        rows_to_plot.append(original)
        row_labels.append('Original')
    if noisy is not None:
        rows_to_plot.append(noisy)
        row_labels.append('Noisy')
    if generated is not None:
        rows_to_plot.append(generated)
        row_labels.append('Generated')
    
    if len(rows_to_plot) == 0:
        print("No data provided for plotting")
        return
    
    num_rows = len(rows_to_plot)
    num_samples = min(num_samples, rows_to_plot[0].shape[0])
    
    fig, axes = plt.subplots(num_rows, num_samples, figsize=(2*num_samples, 2*num_rows))
    
    if num_rows == 1:
        axes = axes.reshape(1, -1)
    if num_samples == 1:
        axes = axes.reshape(-1, 1)
    
    for row_idx, images in enumerate(rows_to_plot):
        for col_idx in range(num_samples):
            ax = axes[row_idx, col_idx]
            
            # Get image and convert to numpy
            img = images[col_idx].cpu().detach()
            
            # Handle different image formats
            if img.dim() == 3:  # (C, H, W)
                if img.shape[0] == 1:  # Grayscale
                    img = img.squeeze(0)
                    ax.imshow(img, cmap='gray')
                else:  # RGB
                    img = img.permute(1, 2, 0)  # (H, W, C)
                    # Normalize to [0, 1] if needed
                    if img.min() < 0 or img.max() > 1:
                        img = (img - img.min()) / (img.max() - img.min())
                    ax.imshow(img)
            elif img.dim() == 2:  # (H, W)
                ax.imshow(img, cmap='gray')
            
            ax.axis('off')
            
            # Add row label to first column
            if col_idx == 0:
                ax.set_ylabel(row_labels[row_idx], fontsize=12, rotation=0, 
                             ha='right', va='center')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Sample comparison saved to {save_path}")


def save_image_grid(
    images: torch.Tensor,
    save_path: str,
    nrow: int = 8,
    normalize: bool = True,
    title: Optional[str] = None
) -> None:
    """
    Save a grid of images.
    
    Args:
        images: Tensor of images (B, C, H, W)
        save_path: Path to save image grid
        nrow: Number of images per row
        normalize: Whether to normalize images to [0, 1]
        title: Optional title for the grid
    """
    from torchvision.utils import make_grid
    
    # Create grid
    grid = make_grid(images, nrow=nrow, normalize=normalize, pad_value=1)
    
    # Convert to numpy and transpose
    grid_np = grid.cpu().numpy().transpose(1, 2, 0)
    
    # Plot
    plt.figure(figsize=(12, 12))
    if grid_np.shape[2] == 1:
        plt.imshow(grid_np.squeeze(), cmap='gray')
    else:
        plt.imshow(grid_np)
    plt.axis('off')
    
    if title:
        plt.title(title, fontsize=14, fontweight='bold')
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Image grid saved to {save_path}")


def plot_gradient_flow(
    named_parameters,
    save_path: str = 'gradient_flow.png',
    title: str = 'Gradient Flow'
) -> None:
    """
    Plot gradient flow through model layers.
    
    Useful for debugging gradient vanishing/exploding issues.
    
    Args:
        named_parameters: Model named parameters (from model.named_parameters())
        save_path: Path to save plot
        title: Plot title
    """
    ave_grads = []
    max_grads = []
    layers = []
    
    for n, p in named_parameters:
        if p.requires_grad and p.grad is not None:
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().item())
            max_grads.append(p.grad.abs().max().item())
    
    if len(layers) == 0:
        print("No gradients to plot")
        return
    
    plt.figure(figsize=(12, 6))
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.5, lw=1, color='c', label='Max Gradient')
    plt.bar(np.arange(len(ave_grads)), ave_grads, alpha=0.5, lw=1, color='b', label='Mean Gradient')
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color='k')
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation='vertical', fontsize=6)
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=max(max_grads) * 1.1)
    plt.xlabel('Layers', fontsize=12)
    plt.ylabel('Gradient Magnitude', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Gradient flow plot saved to {save_path}")


def plot_2d_energy_surface(
    model,
    x_range: Tuple[float, float] = (-5, 5),
    y_range: Tuple[float, float] = (-5, 5),
    resolution: int = 100,
    save_path: str = 'energy_surface.png',
    title: str = '2D Energy Surface'
) -> None:
    """
    Plot 2D energy surface for visualization.
    
    Only works for models that take 2D input.
    
    Args:
        model: Energy model
        x_range: Range for x-axis
        y_range: Range for y-axis
        resolution: Grid resolution
        save_path: Path to save plot
        title: Plot title
    """
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x, y)
    
    # Compute energy for each point
    Z = np.zeros_like(X)
    device = next(model.parameters()).device
    
    for i in range(resolution):
        for j in range(resolution):
            point = torch.tensor([[X[i, j], Y[i, j]]], dtype=torch.float32).to(device)
            with torch.no_grad():
                energy = model.compute_energy(point)
            Z[i, j] = energy.item()
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 5))
    
    # 3D surface
    ax1 = fig.add_subplot(121, projection='3d')
    surf = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Energy')
    ax1.set_title('3D Energy Surface')
    fig.colorbar(surf, ax=ax1, shrink=0.5)
    
    # Contour plot
    ax2 = fig.add_subplot(122)
    contour = ax2.contourf(X, Y, Z, levels=20, cmap='viridis')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('Energy Contours')
    fig.colorbar(contour, ax=ax2)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Energy surface saved to {save_path}")

