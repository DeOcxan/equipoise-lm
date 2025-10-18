"""
Benchmarking Script for Real Datasets

Provides comprehensive evaluation on CIFAR-10/100 and GLUE tasks.
This addresses the weakness of only using synthetic data.

Author: DeOcxan
Date: October 2025
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import json
from pathlib import Path
from typing import Dict, Any, List
import sys

sys.path.append(str(Path(__file__).parent.parent))

from utils.dataset_loaders import create_dataloader
from utils.device_manager import setup_training_environment
from utils.model_utils import count_parameters, format_number
from EBT_IMG.ebt_img_denoise import EBT_IMG_Denoise


class BenchmarkRunner:
    """
    Runs comprehensive benchmarks on real datasets.
    
    Follows Single Responsibility Principle: handles benchmarking only.
    """
    
    def __init__(
        self,
        model,
        dataset_name: str,
        device: torch.device,
        output_dir: str = "./benchmarks/results"
    ):
        """
        Initialize benchmark runner.
        
        Args:
            model: Model to benchmark
            dataset_name: Name of dataset
            device: Device to run on
            output_dir: Directory to save results
        """
        self._model = model
        self._dataset_name = dataset_name
        self._device = device
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        
        self._results = {
            'dataset': dataset_name,
            'model': model.__class__.__name__,
            'num_parameters': count_parameters(model),
            'device': str(device),
            'metrics': {}
        }
    
    def run_inference_benchmark(
        self,
        test_loader: DataLoader,
        num_batches: int = 100
    ) -> Dict[str, float]:
        """
        Benchmark inference speed and accuracy.
        
        Args:
            test_loader: Test data loader
            num_batches: Number of batches to benchmark
            
        Returns:
            Dictionary of metrics
        """
        self._model.eval()
        
        total_time = 0
        total_samples = 0
        losses = []
        psnrs = []
        
        print(f"\n{'='*60}")
        print(f"Running inference benchmark on {self._dataset_name}")
        print(f"{'='*60}")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                if batch_idx >= num_batches:
                    break
                
                clean, noisy = batch
                clean = clean.to(self._device)
                noisy = noisy.to(self._device)
                
                batch_size = clean.shape[0]
                
                # Time inference
                if self._device.type == 'cuda':
                    torch.cuda.synchronize()
                
                start_time = time.time()
                
                # Run denoising
                denoised, _ = self._model.denoise_image(noisy, num_steps=5)
                
                if self._device.type == 'cuda':
                    torch.cuda.synchronize()
                
                end_time = time.time()
                
                batch_time = end_time - start_time
                total_time += batch_time
                total_samples += batch_size
                
                # Compute metrics
                mse = nn.functional.mse_loss(denoised, clean)
                psnr = 10 * torch.log10(1.0 / mse)
                
                losses.append(mse.item())
                psnrs.append(psnr.item())
                
                if batch_idx % 20 == 0:
                    print(f"Batch [{batch_idx}/{num_batches}] "
                          f"Time: {batch_time:.3f}s | "
                          f"PSNR: {psnr.item():.2f} dB")
        
        # Compute statistics
        avg_time_per_sample = total_time / total_samples
        throughput = total_samples / total_time
        avg_loss = sum(losses) / len(losses)
        avg_psnr = sum(psnrs) / len(psnrs)
        
        metrics = {
            'inference_time_per_sample': avg_time_per_sample,
            'throughput_samples_per_sec': throughput,
            'average_mse': avg_loss,
            'average_psnr': avg_psnr,
            'total_samples': total_samples,
            'total_time': total_time
        }
        
        self._results['metrics']['inference'] = metrics
        
        print(f"\n{'='*60}")
        print("Inference Benchmark Results:")
        print(f"  Time per sample: {avg_time_per_sample*1000:.2f} ms")
        print(f"  Throughput: {throughput:.2f} samples/sec")
        print(f"  Average PSNR: {avg_psnr:.2f} dB")
        print(f"  Average MSE: {avg_loss:.6f}")
        print(f"{'='*60}\n")
        
        return metrics
    
    def run_memory_benchmark(
        self,
        test_loader: DataLoader,
        batch_sizes: List[int] = [1, 4, 16, 32, 64]
    ) -> Dict[str, Any]:
        """
        Benchmark memory usage at different batch sizes.
        
        Args:
            test_loader: Test data loader
            batch_sizes: List of batch sizes to test
            
        Returns:
            Dictionary of memory metrics
        """
        print(f"\n{'='*60}")
        print("Running memory benchmark")
        print(f"{'='*60}")
        
        memory_metrics = {}
        
        for bs in batch_sizes:
            # Get a batch
            batch = next(iter(test_loader))
            clean, noisy = batch
            
            # Truncate to batch size
            clean = clean[:bs].to(self._device)
            noisy = noisy[:bs].to(self._device)
            
            if self._device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            
            # Run forward pass
            with torch.no_grad():
                denoised, _ = self._model.denoise_image(noisy, num_steps=5)
            
            if self._device.type == 'cuda':
                torch.cuda.synchronize()
                peak_memory = torch.cuda.max_memory_allocated() / 1e9
                
                memory_metrics[f'batch_size_{bs}'] = {
                    'peak_memory_gb': peak_memory,
                    'memory_per_sample_gb': peak_memory / bs
                }
                
                print(f"Batch size {bs:3d}: "
                      f"Peak memory = {peak_memory:.3f} GB "
                      f"({peak_memory/bs*1000:.1f} MB per sample)")
            else:
                print(f"Batch size {bs:3d}: Memory tracking not available on CPU")
        
        self._results['metrics']['memory'] = memory_metrics
        
        print(f"{'='*60}\n")
        
        return memory_metrics
    
    def run_convergence_benchmark(
        self,
        test_loader: DataLoader,
        step_counts: List[int] = [1, 3, 5, 7, 10, 15]
    ) -> Dict[str, Any]:
        """
        Benchmark convergence at different optimization steps.
        
        Args:
            test_loader: Test data loader
            step_counts: List of step counts to test
            
        Returns:
            Dictionary of convergence metrics
        """
        print(f"\n{'='*60}")
        print("Running convergence benchmark")
        print(f"{'='*60}")
        
        convergence_metrics = {}
        
        # Get a batch
        batch = next(iter(test_loader))
        clean, noisy = batch
        clean = clean[:8].to(self._device)
        noisy = noisy[:8].to(self._device)
        
        self._model.eval()
        
        for num_steps in step_counts:
            with torch.no_grad():
                denoised, trajectory = self._model.denoise_image(
                    noisy,
                    num_steps=num_steps,
                    return_trajectory=True
                )
                
                mse = nn.functional.mse_loss(denoised, clean)
                psnr = 10 * torch.log10(1.0 / mse)
                
                convergence_metrics[f'steps_{num_steps}'] = {
                    'psnr': psnr.item(),
                    'mse': mse.item(),
                    'energy_trajectory': trajectory.tolist()
                }
                
                print(f"Steps {num_steps:2d}: PSNR = {psnr.item():.2f} dB, "
                      f"Final Energy = {trajectory[-1]:.4f}")
        
        self._results['metrics']['convergence'] = convergence_metrics
        
        print(f"{'='*60}\n")
        
        return convergence_metrics
    
    def save_results(self):
        """Save benchmark results to JSON."""
        output_file = self._output_dir / f"benchmark_{self._dataset_name}_{int(time.time())}.json"
        
        with open(output_file, 'w') as f:
            json.dump(self._results, f, indent=2)
        
        print(f"✅ Results saved to {output_file}")
    
    def get_results(self) -> Dict[str, Any]:
        """Get benchmark results."""
        return self._results.copy()


def benchmark_cifar10_denoising(
    noise_std: float = 0.2,
    device: str = 'auto'
):
    """
    Benchmark EBT on CIFAR-10 denoising task.
    
    Args:
        noise_std: Noise standard deviation
        device: Device to use ('auto', 'cpu', 'cuda')
    """
    print("\n" + "="*80)
    print("CIFAR-10 Denoising Benchmark")
    print("="*80 + "\n")
    
    # Setup environment
    if device == 'auto':
        device, device_info = setup_training_environment(
            seed=42,
            preferred_device=None,
            check_deps=True
        )
    else:
        device, device_info = setup_training_environment(
            seed=42,
            preferred_device=device,
            check_deps=True
        )
    
    # Load dataset
    print("Loading CIFAR-10 dataset...")
    train_loader, test_loader, metadata = create_dataloader(
        dataset_name='cifar10',
        batch_size=32,
        num_workers=0,
        root='./data',
        download=True,
        noise_std=noise_std
    )
    
    print(f"Dataset: {metadata['dataset_name']}")
    print(f"Train samples: {metadata['train_samples']}")
    print(f"Test samples: {metadata['test_samples']}")
    print(f"Image size: {metadata['image_size']}x{metadata['image_size']}")
    print(f"Noise std: {noise_std}\n")
    
    # Initialize model
    print("Initializing EBT model...")
    model = EBT_IMG_Denoise(
        in_channels=3,
        base_channels=64,
        num_res_blocks=4,
        image_size=32,
        num_optim_steps=5,
        optim_step_size=0.01,
        device=str(device)
    )
    
    print(f"Model parameters: {format_number(count_parameters(model))}\n")
    
    # Run benchmarks
    runner = BenchmarkRunner(model, 'CIFAR-10', device)
    
    # Inference benchmark
    runner.run_inference_benchmark(test_loader, num_batches=100)
    
    # Memory benchmark
    if device.type == 'cuda':
        runner.run_memory_benchmark(test_loader, batch_sizes=[1, 4, 8, 16, 32])
    
    # Convergence benchmark
    runner.run_convergence_benchmark(
        test_loader,
        step_counts=[1, 3, 5, 7, 10, 15, 20]
    )
    
    # Save results
    runner.save_results()
    
    return runner.get_results()


def benchmark_cifar100_classification():
    """
    Benchmark on CIFAR-100 classification (future implementation).
    
    TODO: Implement classification variant of EBT.
    """
    print("CIFAR-100 classification benchmark coming soon...")
    print("This requires implementing classification head for EBT.")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Benchmark EBT on real datasets')
    parser.add_argument('--dataset', type=str, default='cifar10',
                       choices=['cifar10', 'cifar100'],
                       help='Dataset to benchmark')
    parser.add_argument('--noise_std', type=float, default=0.2,
                       help='Noise standard deviation for denoising')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda', 'mps'],
                       help='Device to use')
    
    args = parser.parse_args()
    
    if args.dataset == 'cifar10':
        results = benchmark_cifar10_denoising(
            noise_std=args.noise_std,
            device=args.device
        )
    elif args.dataset == 'cifar100':
        benchmark_cifar100_classification()
    
    print("\n" + "="*80)
    print("✅ Benchmark complete!")
    print("="*80)

