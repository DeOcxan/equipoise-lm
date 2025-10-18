"""
Custom Optimizers and Learning Rate Schedulers for Energy-Based Models

This module provides stable optimization algorithms designed for training
Energy-Based Models, including:
- StableAdamW: Adam with weight decay and stability improvements
- LARS: Layer-wise Adaptive Rate Scaling for large batch training
- WarmUpCosineAnnealingLR: Warm-up + cosine annealing scheduler

All optimizers follow SOLID principles with clear separation of concerns.

Author: DeOcxan
Date: October 2025
"""

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from typing import List, Callable, Optional, Iterable
import math


class StableAdamW(Optimizer):
    """
    Stable AdamW optimizer with improved numerical stability.
    
    AdamW implements Adam with decoupled weight decay, which is more
    effective for EBMs. This version adds:
    - Gradient clipping for stability
    - Better epsilon handling
    - Optional gradient centralization
    
    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (default: 1e-3)
        betas: Coefficients for computing running averages (default: (0.9, 0.999))
        eps: Term for numerical stability (default: 1e-8)
        weight_decay: Weight decay coefficient (default: 0.01)
        amsgrad: Whether to use AMSGrad variant (default: False)
        clip_grad_norm: Max gradient norm for clipping (default: None)
    """
    
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        amsgrad: bool = False,
        clip_grad_norm: Optional[float] = None
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
            
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            clip_grad_norm=clip_grad_norm
        )
        super(StableAdamW, self).__init__(params, defaults)
        
    def __setstate__(self, state):
        super(StableAdamW, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
            
    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """
        Performs a single optimization step.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss
            
        Returns:
            Loss value if closure is provided
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
                
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            
            # Optional gradient clipping
            if group['clip_grad_norm'] is not None:
                params_with_grad = [p for p in group['params'] if p.grad is not None]
                if len(params_with_grad) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        params_with_grad,
                        group['clip_grad_norm']
                    )
            
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                # Get gradient
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('StableAdamW does not support sparse gradients')
                
                amsgrad = group['amsgrad']
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                
                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                
                step_size = group['lr'] / bias_correction1
                
                # Decoupled weight decay (AdamW)
                if group['weight_decay'] != 0:
                    p.add_(p, alpha=-group['lr'] * group['weight_decay'])
                
                # Update parameters
                p.addcdiv_(exp_avg, denom, value=-step_size)
                
        return loss


class LARS(Optimizer):
    """
    Layer-wise Adaptive Rate Scaling (LARS) optimizer.
    
    LARS adapts the learning rate for each layer based on the ratio of
    weight norm to gradient norm. This is particularly useful for:
    - Large batch training
    - Training with varying layer scales
    - Stable convergence in deep networks
    
    Reference: "Large Batch Training of Convolutional Networks" (You et al., 2017)
    
    Args:
        params: Iterable of parameters to optimize
        lr: Base learning rate (default: 1.0)
        momentum: Momentum factor (default: 0.9)
        weight_decay: Weight decay coefficient (default: 0.0001)
        trust_coef: Trust coefficient for layer adaptation (default: 0.001)
        eps: Term for numerical stability (default: 1e-8)
    """
    
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1.0,
        momentum: float = 0.9,
        weight_decay: float = 0.0001,
        trust_coef: float = 0.001,
        eps: float = 1e-8
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= trust_coef:
            raise ValueError(f"Invalid trust coefficient: {trust_coef}")
            
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            trust_coef=trust_coef,
            eps=eps
        )
        super(LARS, self).__init__(params, defaults)
        
    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """
        Performs a single optimization step.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss
            
        Returns:
            Loss value if closure is provided
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
                
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            trust_coef = group['trust_coef']
            eps = group['eps']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad
                
                # Add weight decay
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)
                
                # Compute layer-wise adaptive learning rate
                param_norm = torch.norm(p)
                grad_norm = torch.norm(grad)
                
                if param_norm != 0 and grad_norm != 0:
                    # Adaptive local learning rate
                    adaptive_lr = trust_coef * param_norm / (grad_norm + eps)
                else:
                    adaptive_lr = 1.0
                
                # Apply momentum
                param_state = self.state[p]
                if 'momentum_buffer' not in param_state:
                    buf = param_state['momentum_buffer'] = torch.clone(grad).detach()
                else:
                    buf = param_state['momentum_buffer']
                    buf.mul_(momentum).add_(grad)
                
                # Update parameters with adaptive learning rate
                p.add_(buf, alpha=-group['lr'] * adaptive_lr)
                
        return loss


class WarmUpCosineAnnealingLR(_LRScheduler):
    """
    Learning rate scheduler with warm-up and cosine annealing.
    
    This scheduler:
    1. Linearly increases LR from 0 to base_lr during warm-up
    2. Applies cosine annealing after warm-up
    3. Can optionally maintain minimum LR
    
    Cosine annealing formula after warm-up:
        lr = min_lr + (base_lr - min_lr) * 0.5 * (1 + cos(pi * t / T))
    
    Args:
        optimizer: Wrapped optimizer
        warmup_epochs: Number of warm-up epochs
        max_epochs: Total number of training epochs
        min_lr: Minimum learning rate (default: 0)
        last_epoch: Index of last epoch (default: -1)
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        min_lr: float = 0,
        last_epoch: int = -1
    ):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        super(WarmUpCosineAnnealingLR, self).__init__(optimizer, last_epoch)
        
    def get_lr(self) -> List[float]:
        """
        Compute learning rate for current epoch.
        
        Returns:
            List of learning rates for each parameter group
        """
        if self.last_epoch < self.warmup_epochs:
            # Linear warm-up phase
            warmup_factor = (self.last_epoch + 1) / self.warmup_epochs
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Cosine annealing phase
            progress = (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            
            return [
                self.min_lr + (base_lr - self.min_lr) * cosine_factor
                for base_lr in self.base_lrs
            ]


def exclude_bias_and_norm(model: torch.nn.Module) -> List[dict]:
    """
    Separate parameters into those with/without weight decay.
    
    Typically, we don't want to apply weight decay to:
    - Bias terms
    - Normalization layer parameters (BatchNorm, LayerNorm, etc.)
    
    Args:
        model: PyTorch model
        
    Returns:
        List of parameter groups with different weight decay settings
    """
    decay = []
    no_decay = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        # Don't apply weight decay to bias and normalization params
        if 'bias' in name or 'norm' in name.lower() or 'bn' in name.lower():
            no_decay.append(param)
        else:
            decay.append(param)
    
    return [
        {'params': decay},
        {'params': no_decay, 'weight_decay': 0.0}
    ]

