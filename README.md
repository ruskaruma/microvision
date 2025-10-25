# microvision

a comprehensive deep learning research lab for computer vision tasks. built with pytorch, numpy, and managed with uv for fast reproducibility and advanced experimentation.

## features

- **12 advanced model architectures**: ResNet, VGG, EfficientNet, Vision Transformer, MobileNet, DenseNet, SqueezeNet
- **modular design**: models, datasets, trainers, model registry
- **cifar-10 experiments** with data augmentation and validation splits
- **pure pytorch training loops** with gpu acceleration
- **config-driven reproducibility** with model checkpointing
- **integrated tensorboard logging** and experiment tracking
- **professional notebook workflow** with model sharing
- **comprehensive analysis tools**: model comparison, performance benchmarking, interpretability
- **advanced experiments**: hyperparameter optimization, ensemble methods, ablation studies

## getting started

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/ruskaruma/microvision.git
cd microvision
uv sync
uv run jupyter lab
```

## quick start

```python
from src.config import Config
from src.datasets import get_cifar10_loaders
from src.models import create_model, list_available_models
from src.trainer import Trainer

# setup
config = Config()
train_loader, val_loader, test_loader = get_cifar10_loaders(config)

# see all available models
print("Available models:", list_available_models())

# train advanced model
model = create_model('resnet18', config)  # or 'efficientnet_b0', 'vit', etc.
trainer = Trainer(model, config)
history = trainer.fit(train_loader, val_loader)

# evaluate
test_results = trainer.evaluate(test_loader)
print(f"test accuracy: {test_results['accuracy']:.2f}%")
```

## notebook workflow

1. **01_data_exploration.ipynb** - explore cifar-10 dataset and class distributions
2. **02_model_training.ipynb** - train and compare 4 model architectures
3. **03_model_analysis.ipynb** - comprehensive analysis of trained models with performance comparison, interpretability, and feature visualization

## project structure

```
microvision/
├── src/                    # core modules
│   ├── config.py          # configuration management
│   ├── datasets.py        # data loading and transforms
│   ├── models.py          # neural network architectures
│   ├── trainer.py         # training and evaluation
│   ├── utils.py           # visualization and metrics
│   └── model_registry.py  # model persistence and sharing
├── notebooks/             # jupyter notebooks
├── experiments/           # logs, checkpoints, and model registry
└── pyproject.toml         # dependencies and project metadata
```

## available models

- **simple_cnn**: custom 3-layer cnn with batch normalization
- **improved_cnn**: cnn with residual connections
- **resnet18**: resnet architecture with pretrained weights
- **efficientnet_b0**: efficientnet with pretrained weights

## key components

- **config system**: centralized hyperparameter management
- **model registry**: automatic model saving and loading
- **training pipeline**: gpu-accelerated training with validation
- **visualization**: training curves, confusion matrices, sample predictions, grad-cam
- **experiment tracking**: tensorboard integration and checkpointing
- **comprehensive analysis**: model comparison, performance benchmarking, interpretability

microvision provides a comprehensive, extensible framework for computer vision experiments with advanced models and analysis capabilities.
