#!/usr/bin/env python3
"""
Quick training script for testing advanced models.
"""
import torch
from src.config import Config
from src.datasets import get_cifar10_loaders
from src.models import create_model, list_available_models, get_model_info
from src.trainer import Trainer
from src.utils import set_seed

def quick_train(model_name: str, epochs: int = 5):
    """Quick training for testing models."""
    print(f"Training {model_name} for {epochs} epochs...")
    
    # Setup
    config = Config()
    config.epochs = epochs
    set_seed(config.seed)
    
    # Get model info
    info = get_model_info(model_name)
    print(f"Model: {model_name}")
    print(f"Parameters: {info['params']}")
    print(f"Description: {info['description']}")
    
    # Create data loaders
    train_loader, val_loader, test_loader = get_cifar10_loaders(config)
    print(f"Data loaded: {len(train_loader)} train batches, {len(val_loader)} val batches")
    
    # Create model
    model = create_model(model_name, config)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model created with {param_count:,} parameters")
    
    # Train
    trainer = Trainer(model, config)
    history = trainer.fit(train_loader, val_loader)
    
    # Test
    test_results = trainer.evaluate(test_loader)
    print(f"Test accuracy: {test_results['accuracy']:.2f}%")
    
    return history, test_results

if __name__ == "__main__":
    print("Available models:")
    for model_name in list_available_models():
        info = get_model_info(model_name)
        print(f"  {model_name:20} - {info['params']:8} - {info['description']}")
    
    print("\nChoose a model to train:")
    print("Examples: resnet18, efficientnet_b0, vit, mobilenet_v2")
    
    # Uncomment to run specific model
    # quick_train('resnet18', epochs=3)
