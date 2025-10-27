#!/usr/bin/env python3
"""
advanced usage example for microvision.

this script demonstrates advanced features including
model comparison, hyperparameter tuning, and experiment tracking.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.config import Config
from src.datasets import get_cifar10_loaders
from src.models import create_model
from src.trainer import Trainer
from src.utils import set_seed, calculate_model_size
from src.optimization import create_advanced_optimizer
from src.experiment_tracker import create_experiment_tracker
import torch

def compare_models():
    """compare different model architectures."""
    print("=== model comparison ===")
    
    config = Config()
    train_loader, val_loader, test_loader = get_cifar10_loaders(config)
    
    models_to_test = ['simple_cnn', 'improved_cnn', 'resnet18', 'vgg11']
    results = {}
    
    for model_name in models_to_test:
        print(f"\ntesting {model_name}...")
        model = create_model(model_name, config)
        model = model.to(config.device)
        num_params, size_mb = calculate_model_size(model)
        print(f"parameters: {num_params:,}, size: {size_mb:.2f} mb")

        trainer = Trainer(model, config)
        history = trainer.fit(train_loader, val_loader)
        test_results = trainer.evaluate(test_loader)
        print(f"test accuracy: {test_results['accuracy']:.2f}%")
        results[model_name] = {
            'test_accuracy': test_results['accuracy'],
            'test_loss': test_results['loss'],
            'best_val_accuracy': max(history['val_acc']),
            'parameters': num_params,
            'size_mb': size_mb
        }
        
        print(f"test accuracy: {test_results['accuracy']:.2f}%")
    print("\n=== model comparison results ===")
    print(f"{'model':<15} {'test_acc':<10} {'params':<12} {'size_mb':<8}")
    print("-" * 50)
    for model_name, metrics in results.items():
        print(f"{model_name:<15} {metrics['test_accuracy']:<10.2f} "
              f"{metrics['parameters']:<12,} {metrics['size_mb']:<8.2f}")
    
    return results
def hyperparameter_tuning():
    """demonstrate hyperparameter tuning."""
    print("\n=== hyperparameter tuning ===")
    learning_rates = [0.001, 0.01, 0.1]
    results = {}
    
    for lr in learning_rates:
        print(f"\ntesting learning rate: {lr}")
        config = Config()
        config.lr = lr
        config.epochs = 10
        
        train_loader, val_loader, test_loader = get_cifar10_loaders(config)
        
        model = create_model('simple_cnn', config)
        model = model.to(config.device)
        
        trainer = Trainer(model, config)
        
        history = trainer.fit(train_loader, val_loader)
        
        test_results = trainer.evaluate(test_loader)
        
        results[lr] = {
            'test_accuracy': test_results['accuracy'],
            'best_val_accuracy': max(history['val_acc'])
        }
        
        print(f"test accuracy: {test_results['accuracy']:.2f}%")
    
    print("\n=== hyperparameter tuning results ===")
    print(f"{'learning_rate':<15} {'test_acc':<10} {'best_val_acc':<12}")
    print("-" * 40)
    for lr, metrics in results.items():
        print(f"{lr:<15} {metrics['test_accuracy']:<10.2f} {metrics['best_val_accuracy']:<12.2f}")
    
    return results

def advanced_experiment():
    """demonstrate advanced experiment tracking."""
    print("\n=== advanced experiment tracking ===")
    
    config = Config()
    config.model_name = 'improved_cnn'
    config.epochs = 5

    with create_experiment_tracker(config, "advanced_experiment") as tracker:
        train_loader, val_loader, test_loader = get_cifar10_loaders(config)
        model = create_model(config.model_name, config)
        model = model.to(config.device)
        
        trainer = Trainer(model, config)
        
        hparams = {
            'batch_size': config.batch_size,
            'lr': config.lr,
            'epochs': config.epochs,
            'model': config.model_name,
            'optimizer': 'sgd'
        }
        
        print("training with advanced tracking...")
        history = trainer.fit(train_loader, val_loader)
        
        test_results = trainer.evaluate(test_loader)
        
        final_metrics = {
            'test_accuracy': test_results['accuracy'],
            'test_loss': test_results['loss'],
            'best_val_accuracy': max(history['val_acc'])
        }
        
        tracker.log_experiment_summary(final_metrics)
        
        print(f"experiment completed! check logs in: {tracker.log_dir}")
        print(f"test accuracy: {test_results['accuracy']:.2f}%")

def main():
    """main function."""
    set_seed(42)
    
    print("microvision advanced usage example")
    print("=" * 50)
    
    model_results = compare_models()
    
    hp_results = hyperparameter_tuning()
    
    advanced_experiment()
    
    print("\n=== summary ===")
    print("all experiments completed successfully!")
    print("check the experiments/ directory for detailed logs and results.")

if __name__ == "__main__":
    main()

