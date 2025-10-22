"""
microvision: a minimal deep learning research lab for small-scale vision tasks.
"""

__version__ = "0.1.0"
__author__ = "microvision"

from .config import Config
from .datasets import get_cifar10_loaders
from .models import SimpleCNN
from .trainer import Trainer
from .utils import plot_training_curves, plot_confusion_matrix

__all__ = [
    "Config",
    "get_cifar10_loaders", 
    "SimpleCNN",
    "Trainer",
    "plot_training_curves",
    "plot_confusion_matrix"
]
