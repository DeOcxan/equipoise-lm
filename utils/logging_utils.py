"""
Logging Utilities with Multiple Backend Support

Supports WandB and TensorBoard for experiment tracking.
Addresses the limitation of only having WandB support.

Author: DeOcxan
Date: October 2025
"""

from typing import Dict, Any, Optional
from pathlib import Path
import warnings
import numpy as np
import matplotlib.pyplot as plt


class ExperimentLogger:
    """
    Unified interface for experiment logging.
    
    Supports multiple backends: WandB, TensorBoard, or simple file logging.
    Follows Interface Segregation Principle.
    """
    
    def __init__(
        self,
        experiment_name: str,
        log_dir: str = "./logs",
        backend: str = "auto",  # 'wandb', 'tensorboard', 'file', 'auto'
        wandb_project: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize logger.
        
        Args:
            experiment_name: Name of experiment
            log_dir: Directory for logs
            backend: Logging backend ('wandb', 'tensorboard', 'file', 'auto')
            wandb_project: WandB project name (if using WandB)
            config: Configuration dictionary to log
        """
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.backend = backend
        self.config = config or {}
        
        self._wandb = None
        self._tensorboard_writer = None
        self._file_logger = None
        
        self._initialize_backend(wandb_project)
    
    def _initialize_backend(self, wandb_project: Optional[str]):
        """Initialize logging backend."""
        if self.backend == "auto":
            # Try WandB first, then TensorBoard, then file
            try:
                import wandb
                self.backend = "wandb"
            except ImportError:
                try:
                    from torch.utils.tensorboard import SummaryWriter
                    self.backend = "tensorboard"
                except ImportError:
                    self.backend = "file"
                    warnings.warn(
                        "Neither WandB nor TensorBoard available. Using file logging.",
                        UserWarning
                    )
        
        if self.backend == "wandb":
            self._initialize_wandb(wandb_project)
        elif self.backend == "tensorboard":
            self._initialize_tensorboard()
        elif self.backend == "file":
            self._initialize_file_logger()
    
    def _initialize_wandb(self, project: Optional[str]):
        """Initialize WandB."""
        try:
            import wandb
            
            self._wandb = wandb.init(
                project=project or "EBT-Mastery",
                name=self.experiment_name,
                config=self.config
            )
            print(f"✅ WandB initialized: {self.experiment_name}")
        except ImportError:
            warnings.warn("WandB not installed. Falling back to TensorBoard.", UserWarning)
            self.backend = "tensorboard"
            self._initialize_tensorboard()
        except Exception as e:
            warnings.warn(f"WandB initialization failed: {e}. Falling back.", UserWarning)
            self.backend = "file"
            self._initialize_file_logger()
    
    def _initialize_tensorboard(self):
        """Initialize TensorBoard."""
        try:
            from torch.utils.tensorboard import SummaryWriter
            
            tb_dir = self.log_dir / "tensorboard" / self.experiment_name
            self._tensorboard_writer = SummaryWriter(str(tb_dir))
            
            # Log config as text
            if self.config:
                config_text = "\n".join(f"{k}: {v}" for k, v in self.config.items())
                self._tensorboard_writer.add_text("config", config_text, 0)
            
            print(f"✅ TensorBoard initialized: {tb_dir}")
            print(f"   Run: tensorboard --logdir {self.log_dir / 'tensorboard'}")
        except ImportError:
            warnings.warn("TensorBoard not available. Using file logging.", UserWarning)
            self.backend = "file"
            self._initialize_file_logger()
    
    def _initialize_file_logger(self):
        """Initialize simple file logger."""
        log_file = self.log_dir / f"{self.experiment_name}.log"
        self._file_logger = open(log_file, 'w')
        
        # Write config
        self._file_logger.write(f"Experiment: {self.experiment_name}\n")
        self._file_logger.write("=" * 60 + "\n")
        for k, v in self.config.items():
            self._file_logger.write(f"{k}: {v}\n")
        self._file_logger.write("=" * 60 + "\n\n")
        self._file_logger.flush()
        
        # Create directories for images and histograms
        self._image_dir = self.log_dir / "images" / self.experiment_name
        self._image_dir.mkdir(parents=True, exist_ok=True)
        self._hist_dir = self.log_dir / "histograms" / self.experiment_name
        self._hist_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"✅ File logging initialized: {log_file}")
    
    def log_metrics(self, metrics: Dict[str, Any], step: int):
        """
        Log metrics.
        
        Args:
            metrics: Dictionary of metric names to values
            step: Step/epoch number
        """
        if self.backend == "wandb" and self._wandb:
            self._wandb.log(metrics, step=step)
        
        elif self.backend == "tensorboard" and self._tensorboard_writer:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self._tensorboard_writer.add_scalar(key, value, step)
        
        elif self.backend == "file" and self._file_logger:
            self._file_logger.write(f"Step {step}:\n")
            for key, value in metrics.items():
                self._file_logger.write(f"  {key}: {value}\n")
            self._file_logger.write("\n")
            self._file_logger.flush()
    
    def log_image(self, tag: str, image, step: int):
        """Log image (TensorBoard/WandB only, with file fallback saving as PNG)."""
        if self.backend == "wandb" and self._wandb:
            import wandb
            self._wandb.log({tag: wandb.Image(image)}, step=step)
        
        elif self.backend == "tensorboard" and self._tensorboard_writer:
            # Assume image is torch tensor or numpy array
            self._tensorboard_writer.add_image(tag, image, step)
        
        elif self.backend == "file":
            # Fallback: Save image to file using matplotlib
            img_path = self._image_dir / f"{tag}_step_{step}.png"
            if isinstance(image, np.ndarray):
                plt.imsave(img_path, image)
            else:
                # Assume torch tensor, convert to numpy
                try:
                    import torch
                    if isinstance(image, torch.Tensor):
                        image = image.cpu().numpy()
                        plt.imsave(img_path, image)
                except ImportError:
                    pass
            self._file_logger.write(f"Image logged: {tag} at step {step} saved to {img_path}\n")
            self._file_logger.flush()
    
    def log_histogram(self, tag: str, values, step: int):
        """Log histogram (TensorBoard/WandB only, with file fallback plotting and saving)."""
        if self.backend == "wandb" and self._wandb:
            import wandb
            self._wandb.log({tag: wandb.Histogram(values)}, step=step)
        
        elif self.backend == "tensorboard" and self._tensorboard_writer:
            self._tensorboard_writer.add_histogram(tag, values, step)
        
        elif self.backend == "file":
            # Fallback: Plot histogram and save as PNG
            hist_path = self._hist_dir / f"{tag}_step_{step}.png"
            fig, ax = plt.subplots()
            ax.hist(values, bins=50)
            ax.set_title(f"{tag} at step {step}")
            fig.savefig(hist_path)
            plt.close(fig)
            self._file_logger.write(f"Histogram logged: {tag} at step {step} saved to {hist_path}\n")
            self._file_logger.flush()
    
    def finish(self):
        """Close logger."""
        if self.backend == "wandb" and self._wandb:
            self._wandb.finish()
        
        elif self.backend == "tensorboard" and self._tensorboard_writer:
            self._tensorboard_writer.close()
        
        elif self.backend == "file" and self._file_logger:
            self._file_logger.close()
        
        print(f"✅ Logger closed: {self.experiment_name}")


def create_logger(
    experiment_name: str,
    use_wandb: bool = False,
    use_tensorboard: bool = False,
    **kwargs
) -> ExperimentLogger:
    """
    Factory function to create logger.
    
    Args:
        experiment_name: Experiment name
        use_wandb: Force WandB
        use_tensorboard: Force TensorBoard
        **kwargs: Additional arguments
        
    Returns:
        ExperimentLogger instance
    """
    if use_wandb:
        backend = "wandb"
    elif use_tensorboard:
        backend = "tensorboard"
    else:
        backend = "auto"
    
    return ExperimentLogger(
        experiment_name=experiment_name,
        backend=backend,
        **kwargs
    )