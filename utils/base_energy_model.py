"""
Base Abstract Classes for Energy-Based Models

This module defines the core abstractions following SOLID principles:
- Single Responsibility: Each class has one clear purpose
- Open/Closed: Extensible through inheritance
- Liskov Substitution: Subtypes are fully substitutable
- Interface Segregation: Specific interfaces for different concerns
- Dependency Inversion: Depend on abstractions, not concrete implementations

Author: DeOcxan
Date: October 2025
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn


class IEnergyFunction(ABC):
    """
    Interface for energy computation.
    
    Defines the contract that all energy functions must follow.
    This follows the Interface Segregation Principle - classes only implement
    what they need.
    """
    
    @abstractmethod
    def compute_energy(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute energy for input tensor.
        
        Args:
            x: Input tensor of shape (batch_size, ...)
            
        Returns:
            Energy scalar(s) of shape (batch_size, 1) or (batch_size,)
        """
        pass


class IOptimizable(ABC):
    """
    Interface for optimization capabilities.
    
    Separates optimization logic from energy computation,
    following Interface Segregation Principle.
    """
    
    @abstractmethod
    def optimize_candidate(
        self,
        candidate: torch.Tensor,
        num_steps: int,
        step_size: float,
        add_noise: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform iterative optimization on candidate.
        
        Args:
            candidate: Initial candidate tensor
            num_steps: Number of gradient descent steps
            step_size: Learning rate for optimization
            add_noise: Whether to add Langevin dynamics noise
            
        Returns:
            Tuple of (optimized_candidate, energy_trajectory)
        """
        pass


class ITrainable(ABC):
    """
    Interface for trainable models.
    
    Defines training-related operations.
    """
    
    @abstractmethod
    def training_step(
        self,
        batch: Any,
        batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Single training step.
        
        Args:
            batch: Batch of training data
            batch_idx: Index of current batch
            
        Returns:
            Dictionary with 'loss' and other metrics
        """
        pass
    
    @abstractmethod
    def validation_step(
        self,
        batch: Any,
        batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Single validation step.
        
        Args:
            batch: Batch of validation data
            batch_idx: Index of current batch
            
        Returns:
            Dictionary with metrics
        """
        pass


class BaseEnergyModel(nn.Module, IEnergyFunction, ABC):
    """
    Base class for all Energy-Based Models.
    
    Follows Single Responsibility Principle - manages model parameters
    and basic energy computation structure.
    
    This abstract base class provides:
    - Parameter initialization
    - Forward pass structure
    - Energy computation interface
    - Gradient computation utilities
    
    Subclasses must implement:
    - compute_energy(): Define specific energy function
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize base energy model.
        
        Args:
            config: Configuration dictionary with model hyperparameters
        """
        super().__init__()
        self._config = config
        self._validate_config(config)
        
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate configuration parameters.
        
        Args:
            config: Configuration to validate
            
        Raises:
            ValueError: If required config parameters are missing
        """
        required_keys = ['model_name', 'device']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")
    
    @property
    def config(self) -> Dict[str, Any]:
        """Get model configuration (read-only)."""
        return self._config.copy()
    
    @property
    def device(self) -> torch.device:
        """Get device model is on."""
        return self._config.get('device', torch.device('cpu'))
    
    @abstractmethod
    def compute_energy(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute energy for input.
        
        This is the core energy function E(x) that must be minimized.
        Lower energy = better/more likely sample.
        
        Args:
            x: Input tensor
            
        Returns:
            Energy values (batch_size, 1) or (batch_size,)
        """
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computes energy.
        
        Args:
            x: Input tensor
            
        Returns:
            Energy values
        """
        return self.compute_energy(x)
    
    def compute_gradients(
        self,
        x: torch.Tensor,
        create_graph: bool = False
    ) -> torch.Tensor:
        """
        Compute gradients of energy w.r.t. input.
        
        Args:
            x: Input tensor (must have requires_grad=True)
            create_graph: Whether to create computation graph for 2nd order derivatives
            
        Returns:
            Gradients dE/dx with same shape as x
        """
        if not x.requires_grad:
            x.requires_grad_(True)
            
        energy = self.compute_energy(x)
        
        # Sum energies for batch processing
        if energy.dim() > 1:
            energy = energy.sum()
        else:
            energy = energy.sum()
            
        gradients = torch.autograd.grad(
            outputs=energy,
            inputs=x,
            create_graph=create_graph,
            retain_graph=True
        )[0]
        
        return gradients
    
    def get_num_parameters(self) -> int:
        """
        Count total number of trainable parameters.
        
        Returns:
            Number of parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BaseIterativeEBM(BaseEnergyModel, IOptimizable, ABC):
    """
    Base class for Energy-Based Models with iterative optimization.
    
    Extends BaseEnergyModel with iterative gradient-based optimization
    capabilities for generation/inference.
    
    This class demonstrates:
    - Open/Closed Principle: Extends BaseEnergyModel without modifying it
    - Single Responsibility: Handles iterative optimization logic
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize iterative EBM.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self._default_num_steps = config.get('num_optim_steps', 3)
        self._default_step_size = config.get('optim_step_size', 0.1)
        self._langevin_noise_scale = config.get('langevin_noise_scale', 0.01)
        
    def optimize_candidate(
        self,
        candidate: torch.Tensor,
        num_steps: int = None,
        step_size: float = None,
        add_noise: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Iteratively optimize candidate by minimizing energy.
        
        Performs gradient descent on the energy landscape:
        x_new = x - step_size * dE/dx (+ optional noise)
        
        Args:
            candidate: Initial candidate tensor (batch_size, ...)
            num_steps: Number of optimization steps (default from config)
            step_size: Step size for gradient descent (default from config)
            add_noise: Whether to add Langevin dynamics noise
            
        Returns:
            Tuple of:
                - optimized_candidate: Final optimized tensor
                - energy_trajectory: Energy values at each step (num_steps,)
        """
        if num_steps is None:
            num_steps = self._default_num_steps
        if step_size is None:
            step_size = self._default_step_size
            
        # Ensure candidate requires gradients
        if not candidate.requires_grad:
            candidate = candidate.detach().clone().requires_grad_(True)
        
        energy_trajectory = []
        
        for step in range(num_steps):
            # Compute current energy
            energy = self.compute_energy(candidate)
            energy_trajectory.append(energy.mean().item())
            
            # Compute gradients
            gradients = self.compute_gradients(candidate, create_graph=False)
            
            # Gradient descent step
            with torch.no_grad():
                candidate = candidate - step_size * gradients
                
                # Add Langevin dynamics noise for better exploration
                if add_noise:
                    noise = torch.randn_like(candidate) * self._langevin_noise_scale
                    candidate = candidate + noise
                    
            # Re-enable gradients for next iteration
            candidate = candidate.detach().requires_grad_(True)
        
        # Final energy computation
        final_energy = self.compute_energy(candidate)
        energy_trajectory.append(final_energy.mean().item())
        
        return candidate.detach(), torch.tensor(energy_trajectory)


class BaseTrainableEBM(BaseIterativeEBM, ITrainable, ABC):
    """
    Base class for trainable Energy-Based Models.
    
    Combines energy computation, optimization, and training logic.
    
    Demonstrates:
    - Liskov Substitution: Can be used wherever BaseIterativeEBM is expected
    - Dependency Inversion: Depends on abstractions (ITrainable interface)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize trainable EBM.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self._training_mode = True
        self._l2_reg = config.get('l2_regularization', 0.0)
        
    def compute_loss(
        self,
        positive_energy: torch.Tensor,
        negative_energy: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute training loss.
        
        For contrastive training:
            L = E(x_pos) - E(x_neg) + regularization
            
        For single-sample training:
            L = E(x) + regularization
        
        Args:
            positive_energy: Energy of real/positive samples
            negative_energy: Energy of generated/negative samples (optional)
            
        Returns:
            Loss value
        """
        if negative_energy is not None:
            # Contrastive loss: minimize positive, maximize negative
            loss = positive_energy.mean() - negative_energy.mean()
        else:
            # Simple energy minimization
            loss = positive_energy.mean()
        
        # Add L2 regularization
        if self._l2_reg > 0:
            l2_penalty = sum(
                torch.sum(p ** 2) for p in self.parameters()
            )
            loss = loss + self._l2_reg * l2_penalty
            
        return loss
    
    @abstractmethod
    def training_step(
        self,
        batch: Any,
        batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Implement specific training logic.
        
        Must be implemented by subclasses.
        """
        pass
    
    @abstractmethod
    def validation_step(
        self,
        batch: Any,
        batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Implement specific validation logic.
        
        Must be implemented by subclasses.
        """
        pass

