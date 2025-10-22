# microvision

a minimal deep learning research lab for small-scale vision tasks. built with pytorch, numpy, and managed with uv for fast reproducibility.

## features

- modular design: models, datasets, trainers
- cifar-10 experiments with data augmentation
- pure pytorch training loops, no boilerplate
- config-driven reproducibility
- integrated tensorboard logging

## getting started

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/ruskaruma/microvision.git
cd microvision
uv sync
uv run jupyter lab
```

## example

```python
from src.models import SimpleCNN
from src.trainer import Trainer
from src.datasets import get_cifar10_loaders
from src.config import Config

cfg = Config()
train_loader, test_loader = get_cifar10_loaders(cfg)
model = SimpleCNN(num_classes=10)
trainer = Trainer(model, cfg)
trainer.fit(train_loader, test_loader)
```

## project structure

```
microvision/
├── src/                    # core modules
├── notebooks/              # jupyter notebooks
├── experiments/            # logs and checkpoints
├── data/                   # datasets
└── pyproject.toml          # dependencies
```

microvision provides a clean, extensible framework for computer vision experiments with minimal dependencies and maximum transparency.
