"""
Training script for Energy-Based Transformer Image Denoising

This script trains EBT on MNIST for image denoising demonstration.
For production, use larger datasets like ImageNet, COCO, etc.

Author: DeOcxan
Date: October 2025
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from ebt_img_denoise import EBT_IMG_Denoise, NoisyImageDataset
from utils.model_utils import set_seed, count_parameters, format_number
from utils.logging_utils import create_logger
from utils.visualization import (
    plot_loss_curve,
    plot_energy_landscape,
    plot_generated_samples,
    save_image_grid
)
from utils.optimization import StableAdamW, WarmUpCosineAnnealingLR


def train_ebt_img_denoise(
    num_epochs: int = 30,
    batch_size: int = 64,
    lr: float = 0.001,
    noise_std: float = 0.2,
    num_optim_steps: int = 5,
    image_size: int = 28,
    use_wandb: bool = False,
    use_tensorboard: bool = False,
    save_dir: str = './EBT_IMG/outputs'
):
    """
    Train EBT for image denoising.
    
    Args:
        num_epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        noise_std: Noise standard deviation for training
        num_optim_steps: Optimization steps for denoising
        image_size: Image size (MNIST is 28x28)
        use_wandb: Whether to use WandB
        save_dir: Directory to save outputs
    """
    # Set seed
    set_seed(42)
    
    # Create save directory
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    plots_dir = Path(save_dir) / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    # Initialize logger (supports WandB + TensorBoard)
    logger = create_logger(
        experiment_name=f'ebt_img_denoise_noise{noise_std}',
        use_wandb=use_wandb,
        use_tensorboard=use_tensorboard or (not use_wandb),  # Default to TensorBoard if no WandB
        log_dir=save_dir,
        wandb_project='EBT-Mastery',
        config={
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'lr': lr,
            'noise_std': noise_std,
            'num_optim_steps': num_optim_steps
        }
    )
    
    # Load MNIST dataset
    print("Loading MNIST dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        # Already normalized to [0, 1]
    ])
    
    train_dataset_clean = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    test_dataset_clean = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    # Wrap with noise
    train_dataset = NoisyImageDataset(train_dataset_clean, noise_std=noise_std)
    test_dataset = NoisyImageDataset(test_dataset_clean, noise_std=noise_std)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Initialize model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = EBT_IMG_Denoise(
        in_channels=1,
        base_channels=64,
        num_res_blocks=4,
        image_size=image_size,
        num_optim_steps=num_optim_steps,
        optim_step_size=0.01,
        device=device
    )
    
    print(f"\nModel: {model._config['model_name']}")
    print(f"Parameters: {format_number(count_parameters(model))}")
    print(f"Device: {device}")
    
    # Initialize optimizer
    optimizer = StableAdamW(
        model.parameters(),
        lr=lr,
        weight_decay=0.01,
        clip_grad_norm=1.0
    )
    
    # Learning rate scheduler
    scheduler = WarmUpCosineAnnealingLR(
        optimizer,
        warmup_epochs=3,
        max_epochs=num_epochs,
        min_lr=lr * 0.1
    )
    
    # Training loop
    print(f"\nTraining for {num_epochs} epochs...")
    train_losses = []
    test_losses = []
    train_psnrs = []
    test_psnrs = []
    
    best_test_psnr = 0.0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        epoch_train_loss = 0
        epoch_train_psnr = 0
        
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            metrics = model.training_step(batch, batch_idx)
            loss = metrics['loss']
            
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
            epoch_train_psnr += metrics['psnr'].item()
            
            # Log batch progress
            if batch_idx % 100 == 0:
                print(f"  Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_train_psnr = epoch_train_psnr / len(train_loader)
        train_losses.append(avg_train_loss)
        train_psnrs.append(avg_train_psnr)
        
        # Testing
        model.eval()
        epoch_test_loss = 0
        epoch_test_psnr = 0
        
        for batch_idx, batch in enumerate(test_loader):
            metrics = model.validation_step(batch, batch_idx)
            epoch_test_loss += metrics['loss'].item()
            epoch_test_psnr += metrics['psnr'].item()
        
        avg_test_loss = epoch_test_loss / len(test_loader)
        avg_test_psnr = epoch_test_psnr / len(test_loader)
        test_losses.append(avg_test_loss)
        test_psnrs.append(avg_test_psnr)
        
        # Step scheduler
        scheduler.step()
        
        # Log metrics
        logger.log_metrics({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'test_loss': avg_test_loss,
            'train_psnr': avg_train_psnr,
            'test_psnr': avg_test_psnr,
            'lr': optimizer.param_groups[0]['lr']
        }, step=epoch)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {avg_train_loss:.4f}, PSNR: {avg_train_psnr:.2f}dB | "
              f"Test Loss: {avg_test_loss:.4f}, PSNR: {avg_test_psnr:.2f}dB")
        
        # Save best model
        if avg_test_psnr > best_test_psnr:
            best_test_psnr = avg_test_psnr
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_psnr': best_test_psnr
            }, f"{save_dir}/best_model.pth")
            print(f"  Saved best model (PSNR: {best_test_psnr:.2f}dB)")
        
        # Generate denoising examples
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print("  Generating denoising examples...")
            model.eval()
            
            # Get a batch of test images
            clean_batch, noisy_batch = next(iter(test_loader))
            clean_batch = clean_batch[:8].to(device)
            noisy_batch = noisy_batch[:8].to(device)
            
            with torch.no_grad():
                denoised_batch, energy_traj = model.denoise_image(
                    noisy_batch,
                    num_steps=num_optim_steps,
                    return_trajectory=True
                )
            
            # Save comparison
            plot_generated_samples(
                original=clean_batch,
                generated=denoised_batch,
                noisy=noisy_batch,
                save_path=f"{plots_dir}/denoising_epoch{epoch+1}.png",
                title=f'Image Denoising - Epoch {epoch+1}',
                num_samples=8
            )
            
            # Plot energy trajectory for first image
            if epoch == 0 or (epoch + 1) % 10 == 0:
                plot_energy_landscape(
                    energy_traj,
                    save_path=f"{plots_dir}/energy_trajectory_epoch{epoch+1}.png",
                    title='Energy Reduction During Denoising'
                )
    
    # Final visualizations
    print("\nGenerating final visualizations...")
    
    # Loss curves
    plot_loss_curve(
        train_losses,
        test_losses,
        save_path=f"{plots_dir}/loss_curve.png",
        title='EBT Image Denoising Loss'
    )
    
    # PSNR curves
    plot_loss_curve(
        train_psnrs,
        test_psnrs,
        save_path=f"{plots_dir}/psnr_curve.png",
        title='EBT Image Denoising PSNR'
    )
    
    # Generate grid of denoised images
    model.eval()
    clean_batch, noisy_batch = next(iter(test_loader))
    clean_batch = clean_batch[:32].to(device)
    noisy_batch = noisy_batch[:32].to(device)
    
    with torch.no_grad():
        denoised_batch, _ = model.denoise_image(noisy_batch, num_steps=num_optim_steps)
    
    save_image_grid(
        clean_batch,
        save_path=f"{plots_dir}/clean_grid.png",
        title='Clean Images'
    )
    
    save_image_grid(
        noisy_batch,
        save_path=f"{plots_dir}/noisy_grid.png",
        title='Noisy Images'
    )
    
    save_image_grid(
        denoised_batch,
        save_path=f"{plots_dir}/denoised_grid.png",
        title='Denoised Images'
    )
    
    logger.finish()
    print(f"\nTraining complete!")
    print(f"Best test PSNR: {best_test_psnr:.2f}dB")
    print(f"Outputs saved to {save_dir}")
    print(f"Plots saved to {plots_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train EBT for image denoising')
    parser.add_argument('--num_epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--noise_std', type=float, default=0.2, help='Noise standard deviation')
    parser.add_argument('--num_optim_steps', type=int, default=5, 
                       help='Optimization steps for denoising')
    parser.add_argument('--image_size', type=int, default=28, help='Image size')
    parser.add_argument('--use_wandb', action='store_true', help='Use WandB logging')
    parser.add_argument('--use_tensorboard', action='store_true', 
                       help='Use TensorBoard logging (default if no WandB)')
    parser.add_argument('--save_dir', type=str, default='./EBT_IMG/outputs',
                       help='Save directory')
    
    args = parser.parse_args()
    
    train_ebt_img_denoise(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        noise_std=args.noise_std,
        num_optim_steps=args.num_optim_steps,
        image_size=args.image_size,
        use_wandb=args.use_wandb,
        use_tensorboard=args.use_tensorboard,
        save_dir=args.save_dir
    )

