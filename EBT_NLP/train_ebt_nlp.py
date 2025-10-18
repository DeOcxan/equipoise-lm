"""
Training script for Energy-Based Transformer NLP models

This script trains EBT on a simple synthetic text dataset for demonstration.
For real applications, use larger datasets like WikiText, OpenWebText, etc.

Author: DeOcxan
Date: October 2025
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import sys
from pathlib import Path
from typing import List, Tuple
import random

sys.path.append(str(Path(__file__).parent.parent))

from ebt_nlp import EBT_NLP
from ebt_nlp_advanced import EBT_NLP_Advanced
from utils.model_utils import set_seed, count_parameters, format_number
from utils.logging_utils import create_logger
from utils.visualization import plot_loss_curve
from utils.optimization import StableAdamW, WarmUpCosineAnnealingLR


class SimpleTokenizer:
    """
    Simple character-level tokenizer for demonstration.
    
    In practice, use proper tokenizers like SentencePiece or BPE.
    """
    
    def __init__(self, texts: List[str]):
        """
        Initialize tokenizer from texts.
        
        Args:
            texts: List of text strings
        """
        # Build vocabulary from characters
        all_chars = set(''.join(texts))
        self.char_to_idx = {char: idx for idx, char in enumerate(sorted(all_chars))}
        self.char_to_idx['<PAD>'] = len(self.char_to_idx)
        self.char_to_idx['<UNK>'] = len(self.char_to_idx)
        
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)
        
        self.pad_token_id = self.char_to_idx['<PAD>']
        
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        return [self.char_to_idx.get(char, self.char_to_idx['<UNK>']) for char in text]
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        return ''.join([self.idx_to_char.get(idx, '<UNK>') for idx in token_ids])


class SyntheticTextDataset(Dataset):
    """
    Synthetic text dataset for demonstration.
    
    Creates simple patterns that the model can learn.
    """
    
    def __init__(
        self,
        num_samples: int = 1000,
        seq_length: int = 32,
        pattern_type: str = 'repetition'
    ):
        """
        Initialize synthetic dataset.
        
        Args:
            num_samples: Number of samples
            seq_length: Sequence length
            pattern_type: Type of pattern ('repetition', 'arithmetic', 'palindrome')
        """
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.pattern_type = pattern_type
        
        # Generate samples
        self.samples = self._generate_samples()
        
        # Create tokenizer
        self.tokenizer = SimpleTokenizer(self.samples)
        
        # Tokenize samples
        self.encoded_samples = [self.tokenizer.encode(s) for s in self.samples]
        
    def _generate_samples(self) -> List[str]:
        """Generate synthetic text samples."""
        samples = []
        
        if self.pattern_type == 'repetition':
            # Generate sequences with repeated patterns
            patterns = ['abc', '123', 'xyz', 'def', '456']
            for _ in range(self.num_samples):
                pattern = random.choice(patterns)
                repeats = self.seq_length // len(pattern)
                sample = (pattern * repeats)[:self.seq_length]
                samples.append(sample)
                
        elif self.pattern_type == 'arithmetic':
            # Generate arithmetic sequences
            for _ in range(self.num_samples):
                start = random.randint(0, 5)
                seq = [str((start + i) % 10) for i in range(self.seq_length)]
                sample = ''.join(seq)
                samples.append(sample)
                
        elif self.pattern_type == 'palindrome':
            # Generate palindromes
            chars = 'abcdefghij'
            for _ in range(self.num_samples):
                half_len = self.seq_length // 2
                half = ''.join(random.choices(chars, k=half_len))
                if self.seq_length % 2 == 0:
                    sample = half + half[::-1]
                else:
                    sample = half + random.choice(chars) + half[::-1]
                samples.append(sample)
        
        return samples
    
    def __len__(self) -> int:
        return len(self.encoded_samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get training sample.
        
        Returns input and target sequences for language modeling.
        Input: tokens[:-1]
        Target: tokens[1:]
        """
        tokens = self.encoded_samples[idx]
        
        # Pad if necessary
        if len(tokens) < self.seq_length:
            tokens = tokens + [self.tokenizer.pad_token_id] * (self.seq_length - len(tokens))
        
        tokens = torch.tensor(tokens[:self.seq_length], dtype=torch.long)
        
        # Input: all but last token
        # Target: all but first token (for next-token prediction)
        input_tokens = tokens[:-1]
        target_tokens = tokens[1:]
        
        return input_tokens, target_tokens


def train_ebt_nlp(
    model_type: str = 'basic',
    num_epochs: int = 50,
    batch_size: int = 32,
    lr: float = 0.0003,
    num_samples: int = 1000,
    seq_length: int = 32,
    pattern_type: str = 'repetition',
    num_optim_steps: int = 3,
    use_wandb: bool = False,
    use_tensorboard: bool = False,
    save_dir: str = './EBT_NLP/outputs'
):
    """
    Train EBT for NLP.
    
    Args:
        model_type: 'basic' or 'advanced'
        num_epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        num_samples: Number of training samples
        seq_length: Sequence length
        pattern_type: Type of synthetic pattern
        num_optim_steps: Optimization steps per token
        use_wandb: Whether to use WandB
        save_dir: Directory to save outputs
    """
    # Set seed
    set_seed(42)
    
    # Create save directory
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize logger (supports WandB + TensorBoard)
    logger = create_logger(
        experiment_name=f'ebt_nlp_{model_type}_{pattern_type}',
        use_wandb=use_wandb,
        use_tensorboard=use_tensorboard or (not use_wandb),  # Default to TensorBoard if no WandB
        log_dir=save_dir,
        wandb_project='EBT-Mastery',
        config={
            'model_type': model_type,
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'lr': lr,
            'pattern_type': pattern_type,
            'num_optim_steps': num_optim_steps
        }
    )
    
    # Create dataset
    print(f"Creating synthetic dataset ({pattern_type})...")
    dataset = SyntheticTextDataset(
        num_samples=num_samples,
        seq_length=seq_length,
        pattern_type=pattern_type
    )
    
    print(f"Vocabulary size: {dataset.tokenizer.vocab_size}")
    print(f"Sample text: {dataset.samples[0]}")
    
    # Split into train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if model_type == 'basic':
        model = EBT_NLP(
            vocab_size=dataset.tokenizer.vocab_size,
            d_model=128,
            num_blocks=3,
            num_heads=4,
            d_ff=512,
            max_seq_len=seq_length,
            num_optim_steps=num_optim_steps,
            device=device
        )
    else:  # advanced
        model = EBT_NLP_Advanced(
            vocab_size=dataset.tokenizer.vocab_size,
            d_model=128,
            num_blocks=3,
            num_heads=4,
            d_ff=512,
            max_seq_len=seq_length,
            num_optim_steps=num_optim_steps,
            learn_step_size=True,
            use_langevin=True,
            adaptive_steps=True,
            device=device
        )
    
    print(f"\nModel: {model._config['model_name']}")
    print(f"Parameters: {format_number(count_parameters(model))}")
    
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
        warmup_epochs=5,
        max_epochs=num_epochs,
        min_lr=lr * 0.1
    )
    
    # Training loop
    print(f"\nTraining for {num_epochs} epochs...")
    train_losses = []
    val_losses = []
    train_perplexities = []
    val_perplexities = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        epoch_train_loss = 0
        epoch_train_ppl = 0
        
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            metrics = model.training_step(batch, batch_idx)
            loss = metrics['loss']
            
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
            epoch_train_ppl += metrics['perplexity'].item()
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_train_ppl = epoch_train_ppl / len(train_loader)
        train_losses.append(avg_train_loss)
        train_perplexities.append(avg_train_ppl)
        
        # Validation
        model.eval()
        epoch_val_loss = 0
        epoch_val_ppl = 0
        
        for batch_idx, batch in enumerate(val_loader):
            metrics = model.validation_step(batch, batch_idx)
            epoch_val_loss += metrics['loss'].item()
            epoch_val_ppl += metrics['perplexity'].item()
        
        avg_val_loss = epoch_val_loss / len(val_loader)
        avg_val_ppl = epoch_val_ppl / len(val_loader)
        val_losses.append(avg_val_loss)
        val_perplexities.append(avg_val_ppl)
        
        # Step scheduler
        scheduler.step()
        
        # Log metrics
        if (epoch + 1) % 5 == 0 or epoch == 0:
            log_dict = {
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'train_ppl': avg_train_ppl,
                'val_ppl': avg_val_ppl,
                'lr': optimizer.param_groups[0]['lr']
            }
            
            if model_type == 'advanced':
                log_dict['step_size'] = model.get_step_size()
            
            logger.log_metrics(log_dict, step=epoch)
            
            print(f"Epoch [{epoch+1}/{num_epochs}] "
                  f"Train Loss: {avg_train_loss:.4f}, PPL: {avg_train_ppl:.2f} | "
                  f"Val Loss: {avg_val_loss:.4f}, PPL: {avg_val_ppl:.2f}")
    
    # Generate samples
    print("\nGenerating samples...")
    model.eval()
    
    # Use first validation sample as prompt
    sample_input, _ = val_dataset[0]
    prompt_tokens = sample_input[:8].unsqueeze(0).to(device)  # Use first 8 tokens
    
    generated = model.generate(
        prompt_tokens,
        max_new_tokens=24,
        num_steps=num_optim_steps,
        temperature=0.8,
        top_k=10
    )
    
    prompt_text = dataset.tokenizer.decode(prompt_tokens[0].cpu().tolist())
    generated_text = dataset.tokenizer.decode(generated[0].cpu().tolist())
    
    print(f"Prompt: {prompt_text}")
    print(f"Generated: {generated_text}")
    
    # Visualizations
    print("\nGenerating visualizations...")
    
    plot_loss_curve(
        train_losses,
        val_losses,
        save_path=f"{save_dir}/loss_curve_{model_type}.png",
        title=f'EBT_NLP Training Loss ({model_type})'
    )
    
    plot_loss_curve(
        train_perplexities,
        val_perplexities,
        save_path=f"{save_dir}/perplexity_{model_type}.png",
        title=f'EBT_NLP Perplexity ({model_type})'
    )
    
    logger.finish()
    print(f"\nTraining complete! Outputs saved to {save_dir}")
    print(f"Final validation perplexity: {val_perplexities[-1]:.2f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train EBT for NLP')
    parser.add_argument('--model_type', type=str, default='basic', 
                       choices=['basic', 'advanced'], help='Model type')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0003, help='Learning rate')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples')
    parser.add_argument('--seq_length', type=int, default=32, help='Sequence length')
    parser.add_argument('--pattern_type', type=str, default='repetition',
                       choices=['repetition', 'arithmetic', 'palindrome'],
                       help='Pattern type')
    parser.add_argument('--num_optim_steps', type=int, default=3, 
                       help='Optimization steps per token')
    parser.add_argument('--use_wandb', action='store_true', help='Use WandB logging')
    parser.add_argument('--use_tensorboard', action='store_true',
                       help='Use TensorBoard logging (default if no WandB)')
    parser.add_argument('--save_dir', type=str, default='./EBT_NLP/outputs',
                       help='Save directory')
    
    args = parser.parse_args()
    
    train_ebt_nlp(
        model_type=args.model_type,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_samples=args.num_samples,
        seq_length=args.seq_length,
        pattern_type=args.pattern_type,
        num_optim_steps=args.num_optim_steps,
        use_wandb=args.use_wandb,
        use_tensorboard=args.use_tensorboard,
        save_dir=args.save_dir
    )

