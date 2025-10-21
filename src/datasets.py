"""
Dataset loading and preprocessing for microvision.
"""
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Tuple, Optional
import os

def get_cifar10_loaders(config) -> Tuple[DataLoader, DataLoader]:
    """
    Create CIFAR-10 data loaders with appropriate transforms.
    
    Args:
        config: Configuration object containing data parameters
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Define transforms
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Create datasets
    train_dataset = datasets.CIFAR10(
        root=config.data_root,
        train=True,
        download=True,
        transform=train_transforms
    )
    
    test_dataset = datasets.CIFAR10(
        root=config.data_root,
        train=False,
        download=True,
        transform=test_transforms
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader

def get_cifar10_classes() -> list:
    """Get CIFAR-10 class names."""
    return [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]
