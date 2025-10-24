"""
configuration management.
"""
from dataclasses import dataclass
from typing import Optional
import torch

@dataclass
class Config:
    """configuration class."""
    batch_size: int = 128
    num_workers: int = 4
    data_root: str = "data"
    num_classes: int = 10
    input_size: int = 32
    epochs: int = 30
    lr: float = 1e-2
    weight_decay: float = 1e-4
    momentum: float = 0.9
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_dir: str = "experiments/logs"
    ckpt_dir: str = "experiments/checkpoints"
    use_augmentation: bool = True
    
    def __post_init__(self):
        """validate configuration."""
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.lr <= 0:
            raise ValueError("learning rate must be positive")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
