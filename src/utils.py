"""
utility functions.
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report
from typing import List, Optional, Tuple, Dict, Any
import seaborn as sns
import json
import os
from datetime import datetime
import torch.nn as nn

def plot_training_curves(train_losses=None, val_losses=None, train_accs=None, val_accs=None, 
                        history=None, save_path: Optional[str] = None):
    """plot training curves."""
    if history is not None:
        train_losses = history['train_loss']
        val_losses = history['val_loss']
        train_accs = history['train_acc']
        val_accs = history['val_acc']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    ax1.plot(train_losses, label='Train Loss', color='blue')
    ax1.plot(val_losses, label='Val Loss', color='red')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(train_accs, label='Train Acc', color='blue')
    ax2.plot(val_accs, label='Val Acc', color='red')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                         class_names: List[str], save_path: Optional[str] = None):
    """plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_sample_predictions(model: torch.nn.Module, test_loader, class_names: List[str],
                           num_samples: int = 16, device: str = 'cpu'):
    """plot sample predictions."""
    model.eval()
    model.to(device)
    
    # Get a batch of samples
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    images, labels = images.to(device), labels.to(device)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
    
    # Select random samples
    indices = torch.randperm(len(images))[:num_samples]
    
    # Create subplot
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.ravel()
    
    for i, idx in enumerate(indices):
        img = images[idx].cpu()
        true_label = labels[idx].item()
        pred_label = predicted[idx].item()
        
        # Denormalize image
        img = img * 0.2023 + 0.1994  # Approximate denormalization
        img = torch.clamp(img, 0, 1)
        
        axes[i].imshow(img.permute(1, 2, 0))
        axes[i].set_title(f'True: {class_names[true_label]}\nPred: {class_names[pred_label]}')
        axes[i].axis('off')
        
        # Color code: green for correct, red for incorrect
        color = 'green' if true_label == pred_label else 'red'
        axes[i].set_title(f'True: {class_names[true_label]}\nPred: {class_names[pred_label]}',
                         color=color)
    
    plt.tight_layout()
    plt.show()

def calculate_model_size(model: torch.nn.Module) -> Tuple[int, float]:
    """calculate model size."""
    num_params = sum(p.numel() for p in model.parameters())
    size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    return num_params, size_mb

def plot_class_distribution(labels: List[int], class_names: List[str], save_path: Optional[str] = None):
    """plot class distribution."""
    unique, counts = np.unique(labels, return_counts=True)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(unique)), counts, color='skyblue', edgecolor='navy', alpha=0.7)
    
    plt.xlabel('class')
    plt.ylabel('count')
    plt.title('class distribution')
    plt.xticks(range(len(unique)), [class_names[i] for i in unique], rotation=45)
    
    for i, (bar, count) in enumerate(zip(bars, counts)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def set_seed(seed: int):
    """set random seed."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_experiment_results(results: Dict[str, Any], save_path: str):
    """save experiment results."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)

def load_experiment_results(load_path: str) -> Dict[str, Any]:
    """load experiment results."""
    with open(load_path, 'r') as f:
        return json.load(f)

def analyze_model_performance(y_true: np.ndarray, y_pred: np.ndarray, 
                            class_names: List[str]) -> Dict[str, Any]:
    """analyze model performance."""
    accuracy = np.mean(y_true == y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    precision = report['macro avg']['precision']
    recall = report['macro avg']['recall']
    f1_score = report['macro avg']['f1-score']
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'classification_report': report,
        'confusion_matrix': cm.tolist()
    }

def create_experiment_summary(config, results: Dict[str, Any], 
                             model: nn.Module) -> Dict[str, Any]:
    """create experiment summary."""
    num_params, size_mb = calculate_model_size(model)
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'batch_size': config.batch_size,
            'epochs': config.epochs,
            'lr': config.lr,
            'weight_decay': config.weight_decay,
            'device': config.device
        },
        'model': {
            'num_parameters': num_params,
            'size_mb': size_mb,
            'architecture': model.__class__.__name__
        },
        'results': results
    }
    
    return summary

def load_pretrained_model(model_name: str, config, model_id: Optional[str] = None):
    """load a pre-trained model from registry."""
    from .model_registry import registry
    
    if model_id is None:
        model_id = registry.get_latest_model(model_name)
    
    model_info = registry.get_model_info(model_id)
    model_config = model_info["config"]
    
    # create model with original config
    from .models import create_model
    # convert class name to model name for create_model
    model_name_lower = model_name.lower().replace('cnn', '_cnn')
    model = create_model(model_name_lower, config)
    
    # load trained weights
    model_path = model_info["model_path"]
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    return model, model_info["history"], model_info
