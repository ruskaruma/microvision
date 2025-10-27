"""
neural network models.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as torchvision_models
from typing import Optional
import math

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


class ResNet18(nn.Module):
    """ResNet-18 architecture with pretrained backbone."""
    def __init__(self, num_classes: int = 10, pretrained: bool = True, dropout_rate: float = 0.5):
        super(ResNet18, self).__init__()
        self.backbone = torchvision_models.resnet18(pretrained=pretrained)
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.backbone.fc.in_features, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)

class ResNet34(nn.Module):
    """ResNet-34 architecture with pretrained backbone."""
    def __init__(self, num_classes: int = 10, pretrained: bool = True, dropout_rate: float = 0.5):
        super(ResNet34, self).__init__()
        self.backbone = torchvision_models.resnet34(pretrained=pretrained)
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.backbone.fc.in_features, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)

class ResNet50(nn.Module):
    """ResNet-50 architecture with pretrained backbone."""
    def __init__(self, num_classes: int = 10, pretrained: bool = True, dropout_rate: float = 0.5):
        super(ResNet50, self).__init__()
        self.backbone = torchvision_models.resnet50(pretrained=pretrained)
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.backbone.fc.in_features, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)

class VGG11(nn.Module):
    """VGG-11 architecture with pretrained backbone."""
    def __init__(self, num_classes: int = 10, pretrained: bool = True, dropout_rate: float = 0.5):
        super(VGG11, self).__init__()
        self.backbone = torchvision_models.vgg11(pretrained=pretrained)
        self.backbone.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(True),
            nn.Dropout(dropout_rate),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(dropout_rate),
            nn.Linear(4096, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)

class VGG16(nn.Module):
    """VGG-16 architecture with pretrained backbone."""
    def __init__(self, num_classes: int = 10, pretrained: bool = True, dropout_rate: float = 0.5):
        super(VGG16, self).__init__()
        self.backbone = torchvision_models.vgg16(pretrained=pretrained)
        self.backbone.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(True),
            nn.Dropout(dropout_rate),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(dropout_rate),
            nn.Linear(4096, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)

class EfficientNetB0(nn.Module):
    """EfficientNet-B0 architecture with pretrained backbone."""
    def __init__(self, num_classes: int = 10, pretrained: bool = True, dropout_rate: float = 0.2):
        super(EfficientNetB0, self).__init__()
        self.backbone = torchvision_models.efficientnet_b0(pretrained=pretrained)
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.backbone.classifier[1].in_features, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)

class MobileNetV2(nn.Module):
    """MobileNetV2 architecture with pretrained backbone."""
    def __init__(self, num_classes: int = 10, pretrained: bool = True, dropout_rate: float = 0.2):
        super(MobileNetV2, self).__init__()
        self.backbone = torchvision_models.mobilenet_v2(pretrained=pretrained)
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.backbone.classifier[1].in_features, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)

class VisionTransformer(nn.Module):
    """Vision Transformer (ViT) architecture."""
    def __init__(self, num_classes: int = 10, pretrained: bool = True, dropout_rate: float = 0.1):
        super(VisionTransformer, self).__init__()
        self.backbone = torchvision_models.vit_b_16(pretrained=pretrained)
        self.backbone.heads = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.backbone.heads.head.in_features, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)

class DenseNet121(nn.Module):
    """DenseNet-121 architecture with pretrained backbone."""
    def __init__(self, num_classes: int = 10, pretrained: bool = True, dropout_rate: float = 0.5):
        super(DenseNet121, self).__init__()
        self.backbone = torchvision_models.densenet121(pretrained=pretrained)
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.backbone.classifier.in_features, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)

class SqueezeNet(nn.Module):
    """SqueezeNet architecture with pretrained backbone."""
    def __init__(self, num_classes: int = 10, pretrained: bool = True, dropout_rate: float = 0.5):
        super(SqueezeNet, self).__init__()
        self.backbone = torchvision_models.squeezenet1_1(pretrained=pretrained)
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Conv2d(512, num_classes, kernel_size=1)
        )
        
    def forward(self, x):
        return self.backbone(x)

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
    
    elif model_name == "resnet18":
        return ResNet18(num_classes=num_classes, **kwargs)
    elif model_name == "resnet34":
        return ResNet34(num_classes=num_classes, **kwargs)
    elif model_name == "resnet50":
        return ResNet50(num_classes=num_classes, **kwargs)
    
    elif model_name == "vgg11":
        return VGG11(num_classes=num_classes, **kwargs)
    elif model_name == "vgg16":
        return VGG16(num_classes=num_classes, **kwargs)
    
    elif model_name == "efficientnet_b0":
        return EfficientNetB0(num_classes=num_classes, **kwargs)
    
    elif model_name == "mobilenet_v2":
        return MobileNetV2(num_classes=num_classes, **kwargs)
    
    elif model_name == "vit":
        return VisionTransformer(num_classes=num_classes, **kwargs)
    
    elif model_name == "densenet121":
        return DenseNet121(num_classes=num_classes, **kwargs)
    
    elif model_name == "squeezenet":
        return SqueezeNet(num_classes=num_classes, **kwargs)
    
    else:
        available_models = [
            "simple_cnn", "improved_cnn",
            "resnet18", "resnet34", "resnet50",
            "vgg11", "vgg16",
            "efficientnet_b0",
            "mobilenet_v2",
            "vit",
            "densenet121",
            "squeezenet"
        ]
        raise ValueError(f"Unknown model: {model_name}. Available models: {available_models}")

def get_model_info(model_name: str) -> dict:
    """Get information about a model."""
    model_info = {
        "simple_cnn": {"params": "~1M", "description": "Simple CNN with 3 conv layers"},
        "improved_cnn": {"params": "~2M", "description": "CNN with residual connections"},
        "resnet18": {"params": "~11M", "description": "ResNet-18 with pretrained backbone"},
        "resnet34": {"params": "~21M", "description": "ResNet-34 with pretrained backbone"},
        "resnet50": {"params": "~25M", "description": "ResNet-50 with pretrained backbone"},
        "vgg11": {"params": "~9M", "description": "VGG-11 with pretrained backbone"},
        "vgg16": {"params": "~138M", "description": "VGG-16 with pretrained backbone"},
        "efficientnet_b0": {"params": "~5M", "description": "EfficientNet-B0 with pretrained backbone"},
        "mobilenet_v2": {"params": "~3M", "description": "MobileNetV2 with pretrained backbone"},
        "vit": {"params": "~86M", "description": "Vision Transformer with pretrained backbone"},
        "densenet121": {"params": "~8M", "description": "DenseNet-121 with pretrained backbone"},
        "squeezenet": {"params": "~1M", "description": "SqueezeNet with pretrained backbone"}
    }
    return model_info.get(model_name, {"params": "Unknown", "description": "Unknown model"})

def list_available_models() -> list:
    """List all available models."""
    return [
        "simple_cnn", "improved_cnn",
        "resnet18", "resnet34", "resnet50",
        "vgg11", "vgg16",
        "efficientnet_b0",
        "mobilenet_v2",
        "vit",
        "densenet121",
        "squeezenet"
    ]
