"""
neural network models.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class SimpleCNN(nn.Module):
    """simple cnn architecture."""
    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.5):
        super(SimpleCNN, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)        
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class ImprovedCNN(nn.Module):
    """improved cnn with residual connections."""
    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.3):
        super(ImprovedCNN, self).__init__()
        self.num_classes = num_classes
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.res_block1 = self._make_residual_block(64, 64)
        self.res_block2 = self._make_residual_block(64, 128, stride=2)
        self.res_block3 = self._make_residual_block(128, 256, stride=2)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        self.fc = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(dropout_rate)
        
    def _make_residual_block(self, in_channels: int, out_channels: int, stride: int = 1):
        """create residual block."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        
        x = self.res_block1(x) + x
        x = self.res_block2(x)
        x = self.res_block3(x)
        
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x

def create_model(model_name: str, config=None, num_classes: int = 10, **kwargs) -> nn.Module:
    """factory function to create models."""
    if config is not None:
        num_classes = config.num_classes
        dropout_rate = getattr(config, 'dropout_rate', 0.5)
        kwargs['dropout_rate'] = dropout_rate
    
    if model_name == "simple_cnn":
        return SimpleCNN(num_classes=num_classes, **kwargs)
    elif model_name == "improved_cnn":
        return ImprovedCNN(num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}")
