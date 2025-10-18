"""
Simple Energy-Based Model for Regression

This module demonstrates EBM for regression with iterative optimization:
- Energy computed for continuous outputs
- Iterative gradient descent to find lowest energy prediction
- Demonstrates "thinking" through multiple optimization steps

This shows how EBMs can optimize over the output space, unlike
traditional regression which directly predicts a single output.

Author: DeOcxan
Date: October 2025
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.base_energy_model import BaseTrainableEBM
from utils.model_utils import set_seed, count_parameters
from utils.logging_utils import create_logger
from utils.visualization import plot_loss_curve
from typing import Dict, Any, Tuple, List


class SimpleEBMRegression(BaseTrainableEBM):
    """
    Simple Energy-Based Model for regression tasks.
    
    Unlike standard regression that directly outputs predictions,
    this EBM learns an energy function and finds predictions by
    minimizing energy through iterative optimization.
    
    Architecture:
        [Input, Candidate] -> Hidden Layers -> Energy (scalar)
    
    Energy Function:
        E(x, y) = f(x, y | θ)
    
    Where:
        x = input features
        y = candidate output
        θ = model parameters
    
    Prediction:
        y* = argmin_y E(x, y)  via gradient descent
    
    This demonstrates the core EBM concept: optimization over output space.
    """
    
    def __init__(
        self,
        input_dim: int = 1,
        output_dim: int = 1,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_optim_steps: int = 5,
        optim_step_size: float = 0.1,
        device: str = 'cpu'
    ):
        """
        Initialize Simple EBM for regression.
        
        Args:
            input_dim: Dimension of input features
            output_dim: Dimension of output
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
            num_optim_steps: Steps for iterative optimization during inference
            optim_step_size: Step size for gradient descent
            device: Device to run model on
        """
        config = {
            'model_name': 'SimpleEBMRegression',
            'input_dim': input_dim,
            'output_dim': output_dim,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'device': torch.device(device),
            'num_optim_steps': num_optim_steps,
            'optim_step_size': optim_step_size,
            'langevin_noise_scale': 0.005,
            'l2_regularization': 0.0001
        }
        
        super().__init__(config)
        
        self._input_dim = input_dim
        self._output_dim = output_dim
        
        # Build energy network
        # Takes concatenated [input, candidate_output] and outputs energy
        combined_dim = input_dim + output_dim
        
        layers = []
        prev_dim = combined_dim
        
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim
        
        # Final layer outputs energy (scalar)
        layers.append(nn.Linear(prev_dim, 1))
        
        self.energy_net = nn.Sequential(*layers)
        
        # Move to device
        self.to(self.device)
        
        print(f"SimpleEBMRegression initialized with {count_parameters(self)} parameters")
        print(f"Optimization steps: {num_optim_steps}, Step size: {optim_step_size}")
        
    def compute_energy(self, x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        """
        Compute energy for input-output pair.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            y: Output/candidate tensor of shape (batch_size, output_dim)
               If None, uses x directly (assumes x already contains both)
            
        Returns:
            Energy values of shape (batch_size, 1)
        """
        if y is not None:
            # Concatenate input and candidate output
            combined = torch.cat([x, y], dim=-1)
        else:
            # Assume x already contains both input and output
            combined = x
            
        energy = self.energy_net(combined)
        return energy
    
    def predict(
        self,
        x: torch.Tensor,
        initial_guess: torch.Tensor = None,
        num_steps: int = None,
        return_trajectory: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict output by minimizing energy.
        
        This is the key difference from standard regression:
        - Standard: y = f(x)  (direct prediction)
        - EBM: y* = argmin_y E(x, y)  (optimization)
        
        Args:
            x: Input tensor (batch_size, input_dim)
            initial_guess: Initial candidate (if None, starts from noise)
            num_steps: Number of optimization steps (default from config)
            return_trajectory: Whether to return energy trajectory
            
        Returns:
            Tuple of (predicted_output, energy_trajectory)
        """
        batch_size = x.shape[0]
        
        # Initialize candidate
        if initial_guess is None:
            # Start from random noise
            candidate = torch.randn(
                batch_size,
                self._output_dim,
                device=self.device,
                requires_grad=True
            )
        else:
            candidate = initial_guess.detach().clone().requires_grad_(True)
        
        # Optimize candidate by minimizing energy
        energy_trajectory = []
        
        if num_steps is None:
            num_steps = self._default_num_steps
        
        for step in range(num_steps):
            # Compute energy for current candidate
            energy = self.compute_energy(x, candidate)
            energy_trajectory.append(energy.mean().item())
            
            # Compute gradient w.r.t. candidate
            energy_sum = energy.sum()  # Sum for batch processing
            grads = torch.autograd.grad(
                outputs=energy_sum,
                inputs=candidate,
                create_graph=False,
                retain_graph=True
            )[0]
            
            # Gradient descent step
            with torch.no_grad():
                candidate = candidate - self._default_step_size * grads
                
            # Re-enable gradients for next iteration
            candidate = candidate.detach().requires_grad_(True)
        
        # Final energy
        final_energy = self.compute_energy(x, candidate)
        energy_trajectory.append(final_energy.mean().item())
        
        if return_trajectory:
            return candidate.detach(), torch.tensor(energy_trajectory)
        else:
            return candidate.detach(), final_energy.detach()
    
    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Single training step for regression.
        
        Training objective:
        - Minimize energy for true (input, output) pairs
        - This teaches the model that correct outputs have low energy
        
        Loss = E(x, y_true)
        
        Args:
            batch: Tuple of (inputs, targets)
            batch_idx: Batch index
            
        Returns:
            Dictionary with loss and metrics
        """
        x, y_true = batch
        x = x.to(self.device)
        y_true = y_true.to(self.device)
        
        # Compute energy for true pairs
        energy_true = self.compute_energy(x, y_true)
        
        # Loss: minimize energy for correct outputs
        loss = energy_true.mean()
        
        # Add L2 regularization
        if self._l2_reg > 0:
            l2_penalty = sum(torch.sum(p ** 2) for p in self.parameters())
            loss = loss + self._l2_reg * l2_penalty
        
        # Compute prediction MSE for monitoring
        with torch.no_grad():
            y_pred, _ = self.predict(x, num_steps=3)
            mse = ((y_pred - y_true) ** 2).mean()
        
        return {
            'loss': loss,
            'energy_mean': energy_true.mean(),
            'energy_std': energy_true.std(),
            'mse': mse
        }
    
    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Single validation step.
        
        Args:
            batch: Tuple of (inputs, targets)
            batch_idx: Batch index
            
        Returns:
            Dictionary with metrics
        """
        x, y_true = batch
        x = x.to(self.device)
        y_true = y_true.to(self.device)
        
        with torch.no_grad():
            # Compute energy for true pairs
            energy_true = self.compute_energy(x, y_true)
            
            # Predict using optimization
            y_pred, _ = self.predict(x, num_steps=self._default_num_steps)
            
            # Compute MSE
            mse = ((y_pred - y_true) ** 2).mean()
            
            # Compute MAE
            mae = torch.abs(y_pred - y_true).mean()
        
        return {
            'loss': energy_true.mean(),
            'energy_mean': energy_true.mean(),
            'mse': mse,
            'mae': mae
        }


def create_synthetic_regression_dataset(
    num_samples: int = 1000,
    function_type: str = 'quadratic',
    noise: float = 0.1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create synthetic regression dataset.
    
    Supports various functions:
    - 'quadratic': y = x^2
    - 'sin': y = sin(2*pi*x)
    - 'cubic': y = x^3 - x
    
    Args:
        num_samples: Number of samples
        function_type: Type of function to learn
        noise: Noise level
        
    Returns:
        Tuple of (inputs, targets)
    """
    # Generate inputs
    x = torch.linspace(-2, 2, num_samples).unsqueeze(1)
    
    # Generate targets based on function type
    if function_type == 'quadratic':
        y = x ** 2
    elif function_type == 'sin':
        y = torch.sin(2 * np.pi * x)
    elif function_type == 'cubic':
        y = x ** 3 - x
    else:
        raise ValueError(f"Unknown function type: {function_type}")
    
    # Add noise
    y = y + torch.randn_like(y) * noise
    
    return x, y


def train_simple_ebm_regression(
    num_epochs: int = 200,
    batch_size: int = 32,
    lr: float = 0.001,
    num_samples: int = 1000,
    function_type: str = 'quadratic',
    num_optim_steps: int = 5,
    use_wandb: bool = False,
    use_tensorboard: bool = False,
    save_dir: str = './Simple_EBM/plots'
):
    """
    Train Simple EBM for regression.
    
    Args:
        num_epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        num_samples: Number of samples
        function_type: Function to learn ('quadratic', 'sin', 'cubic')
        num_optim_steps: Optimization steps for prediction
        use_wandb: Whether to use WandB
        save_dir: Directory to save plots
    """
    # Set seed
    set_seed(42)
    
    # Create save directory
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize logger (supports WandB + TensorBoard)
    logger = create_logger(
        experiment_name=f'simple_ebm_regression_{function_type}',
        use_wandb=use_wandb,
        use_tensorboard=use_tensorboard or (not use_wandb),
        log_dir=save_dir,
        wandb_project='EBT-Mastery',
        config={
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'lr': lr,
            'function_type': function_type,
            'num_optim_steps': num_optim_steps
        }
    )
    
    # Create dataset
    print(f"Creating synthetic regression dataset ({function_type})...")
    inputs, targets = create_synthetic_regression_dataset(
        num_samples,
        function_type=function_type
    )
    
    # Split into train and validation
    train_size = int(0.8 * num_samples)
    train_inputs, train_targets = inputs[:train_size], targets[:train_size]
    val_inputs, val_targets = inputs[train_size:], targets[train_size:]
    
    train_dataset = TensorDataset(train_inputs, train_targets)
    val_dataset = TensorDataset(val_inputs, val_targets)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SimpleEBMRegression(
        input_dim=1,
        output_dim=1,
        hidden_dim=64,
        num_layers=2,
        num_optim_steps=num_optim_steps,
        optim_step_size=0.1,
        device=device
    )
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    print(f"\nTraining Simple EBM Regression for {num_epochs} epochs...")
    train_losses = []
    val_losses = []
    train_mses = []
    val_mses = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        epoch_train_loss = 0
        epoch_train_mse = 0
        
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            metrics = model.training_step(batch, batch_idx)
            loss = metrics['loss']
            
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
            epoch_train_mse += metrics['mse'].item()
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_train_mse = epoch_train_mse / len(train_loader)
        train_losses.append(avg_train_loss)
        train_mses.append(avg_train_mse)
        
        # Validation
        model.eval()
        epoch_val_loss = 0
        epoch_val_mse = 0
        
        for batch_idx, batch in enumerate(val_loader):
            metrics = model.validation_step(batch, batch_idx)
            epoch_val_loss += metrics['loss'].item()
            epoch_val_mse += metrics['mse'].item()
        
        avg_val_loss = epoch_val_loss / len(val_loader)
        avg_val_mse = epoch_val_mse / len(val_loader)
        val_losses.append(avg_val_loss)
        val_mses.append(avg_val_mse)
        
        # Log metrics
        if (epoch + 1) % 20 == 0 or epoch == 0:
            logger.log_metrics({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'train_mse': avg_train_mse,
                'val_mse': avg_val_mse
            }, step=epoch)
            
            print(f"Epoch [{epoch+1}/{num_epochs}] "
                  f"Train Loss: {avg_train_loss:.4f}, MSE: {avg_train_mse:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f}, MSE: {avg_val_mse:.4f}")
    
    # Visualize predictions
    print("\nGenerating visualizations...")
    model.eval()
    
    # Generate predictions for entire range
    test_x = torch.linspace(-2.5, 2.5, 200).unsqueeze(1).to(device)
    with torch.no_grad():
        pred_y, energy_traj = model.predict(test_x, num_steps=num_optim_steps, return_trajectory=True)
    
    # Plot predictions vs true function
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 5))
    
    # Subplot 1: Predictions
    plt.subplot(1, 2, 1)
    if function_type == 'quadratic':
        true_y = test_x.cpu() ** 2
    elif function_type == 'sin':
        true_y = torch.sin(2 * np.pi * test_x.cpu())
    elif function_type == 'cubic':
        true_y = test_x.cpu() ** 3 - test_x.cpu()
    
    plt.plot(test_x.cpu(), true_y, 'b-', label='True Function', linewidth=2)
    plt.plot(test_x.cpu(), pred_y.cpu(), 'r--', label='EBM Prediction', linewidth=2)
    plt.scatter(train_inputs.cpu(), train_targets.cpu(), alpha=0.3, s=10, label='Training Data')
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.title(f'EBM Regression: {function_type}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Energy trajectory for one sample
    plt.subplot(1, 2, 2)
    plt.plot(energy_traj.numpy())
    plt.xlabel('Optimization Step')
    plt.ylabel('Energy')
    plt.title('Energy Descent During Prediction')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/regression_predictions_{function_type}.png", dpi=150)
    plt.close()
    
    # Loss curves
    plot_loss_curve(
        train_losses,
        val_losses,
        save_path=f"{save_dir}/regression_loss_{function_type}.png",
        title=f'Simple EBM Regression Loss ({function_type})'
    )
    
    # MSE curves
    plot_loss_curve(
        train_mses,
        val_mses,
        save_path=f"{save_dir}/regression_mse_{function_type}.png",
        title=f'Simple EBM Regression MSE ({function_type})'
    )
    
    logger.finish()
    print(f"\nTraining complete! Visualizations saved to {save_dir}")
    print(f"Final validation MSE: {val_mses[-1]:.4f}")


if __name__ == "__main__":
    """
    Main execution: Train Simple EBM on regression task.
    
    This demonstrates:
    1. Energy-based prediction via optimization
    2. Iterative "thinking" to find best output
    3. Gradient descent on energy landscape
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Simple EBM for regression')
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples')
    parser.add_argument('--function_type', type=str, default='quadratic', 
                       choices=['quadratic', 'sin', 'cubic'], help='Function to learn')
    parser.add_argument('--num_optim_steps', type=int, default=5, help='Optimization steps for prediction')
    parser.add_argument('--use_wandb', action='store_true', help='Use WandB logging')
    parser.add_argument('--use_tensorboard', action='store_true',
                       help='Use TensorBoard logging (default if no WandB)')
    parser.add_argument('--save_dir', type=str, default='./Simple_EBM/plots', help='Save directory')
    
    args = parser.parse_args()
    
    train_simple_ebm_regression(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_samples=args.num_samples,
        function_type=args.function_type,
        num_optim_steps=args.num_optim_steps,
        use_wandb=args.use_wandb,
        use_tensorboard=args.use_tensorboard,
        save_dir=args.save_dir
    )

