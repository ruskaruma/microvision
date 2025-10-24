# microvision

a minimal deep learning research lab for small-scale vision tasks. built with pytorch, numpy, and managed with uv for fast reproducibility.

## features

- modular design: models, datasets, trainers, model registry
- cifar-10 experiments with data augmentation and validation splits
- pure pytorch training loops with gpu acceleration
- config-driven reproducibility with model checkpointing
- integrated tensorboard logging and experiment tracking
- professional notebook workflow with model sharing

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
from src.models import create_model
from src.trainer import Trainer

# setup
config = Config()
train_loader, val_loader, test_loader = get_cifar10_loaders(config)

# train model
model = create_model('simple_cnn', config)
trainer = Trainer(model, config)
history = trainer.fit(train_loader, val_loader)

# evaluate
test_results = trainer.evaluate(test_loader)
print(f"test accuracy: {test_results['accuracy']:.2f}%")
```

## notebook workflow

1. **01_data_exploration.ipynb** - explore cifar-10 dataset and class distributions
2. **02_model_training.ipynb** - train simplecnn and improvedcnn models
3. **03_model_analysis.ipynb** - analyze pre-trained models from registry
4. **04_experiments.ipynb** - compare models and run new experiments

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

## key components

- **config system**: centralized hyperparameter management
- **model registry**: automatic model saving and loading
- **training pipeline**: gpu-accelerated training with validation
- **visualization**: training curves, confusion matrices, sample predictions
- **experiment tracking**: tensorboard integration and checkpointing

microvision provides a clean, extensible framework for computer vision experiments with minimal dependencies and maximum transparency.
