"""
Simple Energy-Based Model for Binary Classification

This module demonstrates the most basic EBM:
- Energy computed as a quadratic function of inputs
- Training by minimizing energy for positive samples
- Simple forward and backward passes

This is the starting point for understanding Energy-Based Models.
All concepts are implemented from scratch with detailed comments.

Author: DeOcxan
Date: October 2025
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.base_energy_model import BaseTrainableEBM
from utils.model_utils import set_seed, count_parameters
from utils.logging_utils import create_logger
from utils.visualization import plot_loss_curve, plot_2d_energy_surface
from typing import Dict, Any, Tuple


class SimpleEBM(BaseTrainableEBM):
    """
    Simple Energy-Based Model for binary classification.
    
    This model learns to assign low energy to positive class samples
    and high energy to negative class samples.
    
    Architecture:
        Input -> Linear Layer -> Energy (scalar)
    
    Energy Function:
        E(x) = W^T x + b
    
    Where W and b are learnable parameters.
    
    Following OOP principles:
    - Inherits from BaseTrainableEBM for common functionality
    - Encapsulates energy computation logic
    - Clear separation of concerns
    """
    
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 16,
        use_hidden_layer: bool = True,
        device: str = 'cpu'
    ):
        """
        Initialize Simple EBM.
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden layer (if used)
            use_hidden_layer: Whether to use a hidden layer
            device: Device to run model on
        """
        config = {
            'model_name': 'SimpleEBM',
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,
            'use_hidden_layer': use_hidden_layer,
            'device': torch.device(device),
            'num_optim_steps': 0,  # No iterative optimization for classification
            'l2_regularization': 0.0001
        }
        
        super().__init__(config)
        
        # Build energy network
        if use_hidden_layer:
            # With hidden layer: Input -> Hidden -> Energy
            self.energy_net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
        else:
            # Direct mapping: Input -> Energy
            self.energy_net = nn.Sequential(
                nn.Linear(input_dim, 1)
            )
        
        # Move to device
        self.to(self.device)
        
        print(f"SimpleEBM initialized with {count_parameters(self)} parameters")
        
    def compute_energy(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute energy for input samples.
        
        Forward Pass:
            1. Pass input through network
            2. Get energy scalar for each sample
            3. Return energies
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Energy values of shape (batch_size, 1)
        """
        # Energy computation (forward pass)
        energy = self.energy_net(x)
        return energy
    
    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Single training step.
        
        For classification:
        - Minimize energy for positive class (label=1)
        - Maximize energy for negative class (label=0)
        
        Loss = E(x_positive) - E(x_negative)
        
        Args:
            batch: Tuple of (inputs, labels)
            batch_idx: Batch index
            
        Returns:
            Dictionary with loss and metrics
        """
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        
        # Compute energy for all samples
        energy = self.compute_energy(x)
        
        # Separate positive and negative samples
        pos_mask = (y == 1).squeeze()
        neg_mask = (y == 0).squeeze()
        
        if pos_mask.any() and neg_mask.any():
            pos_energy = energy[pos_mask].mean()
            neg_energy = energy[neg_mask].mean()
            
            # Loss: minimize positive energy, maximize negative energy
            # This is equivalent to pushing positive samples to low energy
            # and negative samples to high energy
            loss = pos_energy - neg_energy
        else:
            # If batch has only one class, just minimize/maximize accordingly
            if pos_mask.any():
                loss = energy[pos_mask].mean()
            else:
                loss = -energy[neg_mask].mean()
        
        # Add L2 regularization
        if self._l2_reg > 0:
            l2_penalty = sum(torch.sum(p ** 2) for p in self.parameters())
            loss = loss + self._l2_reg * l2_penalty
        
        # Compute accuracy (samples with energy < 0 are classified as positive)
        predictions = (energy < 0).float()
        accuracy = (predictions.squeeze() == y.squeeze()).float().mean()
        
        return {
            'loss': loss,
            'energy_mean': energy.mean(),
            'energy_std': energy.std(),
            'accuracy': accuracy
        }
    
    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Single validation step.
        
        Args:
            batch: Tuple of (inputs, labels)
            batch_idx: Batch index
            
        Returns:
            Dictionary with metrics
        """
        with torch.no_grad():
            return self.training_step(batch, batch_idx)


def create_synthetic_dataset(
    num_samples: int = 1000,
    input_dim: int = 2,
    noise: float = 0.1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create synthetic binary classification dataset.
    
    Creates two classes:
    - Class 0: Points around [-1, -1, ...]
    - Class 1: Points around [1, 1, ...]
    
    Args:
        num_samples: Total number of samples
        input_dim: Dimension of input features
        noise: Noise level for data generation
        
    Returns:
        Tuple of (inputs, labels)
    """
    half = num_samples // 2
    
    # Class 0: negative samples
    class_0 = torch.randn(half, input_dim) * noise - 1.0
    labels_0 = torch.zeros(half, 1)
    
    # Class 1: positive samples
    class_1 = torch.randn(half, input_dim) * noise + 1.0
    labels_1 = torch.ones(half, 1)
    
    # Combine
    inputs = torch.cat([class_0, class_1], dim=0)
    labels = torch.cat([labels_0, labels_1], dim=0)
    
    # Shuffle
    indices = torch.randperm(num_samples)
    inputs = inputs[indices]
    labels = labels[indices]
    
    return inputs, labels


def train_simple_ebm(
    num_epochs: int = 100,
    batch_size: int = 32,
    lr: float = 0.01,
    num_samples: int = 1000,
    input_dim: int = 2,
    use_wandb: bool = False,
    use_tensorboard: bool = False,
    save_dir: str = './Simple_EBM/plots'
):
    """
    Train Simple EBM on synthetic dataset.
    
    Args:
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        num_samples: Number of samples in dataset
        input_dim: Input dimension
        use_wandb: Whether to use Weights & Biases logging
        save_dir: Directory to save plots
    """
    # Set seed for reproducibility
    set_seed(42)
    
    # Create save directory
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize logger (supports WandB + TensorBoard)
    logger = create_logger(
        experiment_name='simple_ebm_classification',
        use_wandb=use_wandb,
        use_tensorboard=use_tensorboard or (not use_wandb),
        log_dir=save_dir,
        wandb_project='EBT-Mastery',
        config={
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'lr': lr,
            'num_samples': num_samples,
            'input_dim': input_dim
        }
    )
    
    # Create dataset
    print("Creating synthetic dataset...")
    inputs, labels = create_synthetic_dataset(num_samples, input_dim)
    
    # Split into train and validation
    train_size = int(0.8 * num_samples)
    train_inputs, train_labels = inputs[:train_size], labels[:train_size]
    val_inputs, val_labels = inputs[train_size:], labels[train_size:]
    
    train_dataset = TensorDataset(train_inputs, train_labels)
    val_dataset = TensorDataset(val_inputs, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SimpleEBM(input_dim=input_dim, hidden_dim=16, use_hidden_layer=True, device=device)
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    print(f"\nTraining Simple EBM for {num_epochs} epochs...")
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        epoch_train_loss = 0
        epoch_train_acc = 0
        
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Forward pass
            metrics = model.training_step(batch, batch_idx)
            loss = metrics['loss']
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
            epoch_train_acc += metrics['accuracy'].item()
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_train_acc = epoch_train_acc / len(train_loader)
        train_losses.append(avg_train_loss)
        train_accuracies.append(avg_train_acc)
        
        # Validation
        model.eval()
        epoch_val_loss = 0
        epoch_val_acc = 0
        
        for batch_idx, batch in enumerate(val_loader):
            metrics = model.validation_step(batch, batch_idx)
            epoch_val_loss += metrics['loss'].item()
            epoch_val_acc += metrics['accuracy'].item()
        
        avg_val_loss = epoch_val_loss / len(val_loader)
        avg_val_acc = epoch_val_acc / len(val_loader)
        val_losses.append(avg_val_loss)
        val_accuracies.append(avg_val_acc)
        
        # Log metrics
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.log_metrics({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'train_accuracy': avg_train_acc,
                'val_accuracy': avg_val_acc
            }, step=epoch)
            
            print(f"Epoch [{epoch+1}/{num_epochs}] "
                  f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}")
    
    # Save visualizations
    print("\nGenerating visualizations...")
    
    # Loss curve
    plot_loss_curve(
        train_losses,
        val_losses,
        save_path=f"{save_dir}/loss_curve.png",
        title='Simple EBM Training Loss'
    )
    
    # Accuracy curve
    plot_loss_curve(
        train_accuracies,
        val_accuracies,
        save_path=f"{save_dir}/accuracy_curve.png",
        title='Simple EBM Classification Accuracy'
    )
    
    # Energy surface (only for 2D input)
    if input_dim == 2:
        plot_2d_energy_surface(
            model,
            x_range=(-2, 2),
            y_range=(-2, 2),
            resolution=100,
            save_path=f"{save_dir}/energy_surface.png",
            title='Simple EBM Energy Surface'
        )
    
    logger.finish()
    print(f"\nTraining complete! Visualizations saved to {save_dir}")
    print(f"Final validation accuracy: {val_accuracies[-1]:.4f}")


if __name__ == "__main__":
    """
    Main execution: Train Simple EBM on synthetic classification task.
    
    This demonstrates:
    1. Energy-based learning for classification
    2. Forward pass: compute energy
    3. Backward pass: gradients for learning
    4. Visualization of energy landscape
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Simple EBM for classification')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples')
    parser.add_argument('--input_dim', type=int, default=2, help='Input dimension')
    parser.add_argument('--use_wandb', action='store_true', help='Use WandB logging')
    parser.add_argument('--use_tensorboard', action='store_true', 
                       help='Use TensorBoard logging (default if no WandB)')
    parser.add_argument('--save_dir', type=str, default='./Simple_EBM/plots', help='Save directory')
    
    args = parser.parse_args()
    
    train_simple_ebm(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_samples=args.num_samples,
        input_dim=args.input_dim,
        use_wandb=args.use_wandb,
        use_tensorboard=args.use_tensorboard,
        save_dir=args.save_dir
    )

