"""Convolutional Neural Network models for histopathology tile classification."""

from __future__ import annotations

from typing import Optional
import torch
import torch.nn as nn
from torchvision import models


class SimpleCNN(nn.Module):
    """Lightweight 4-stage convolutional baseline for fast training or constrained environments."""

    def __init__(self, num_classes: int = 2, in_channels: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 112x112
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 56x56
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 28x28
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),  # 1x1
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        return torch.flatten(feat, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.extract_features(x)
        return self.classifier(feat)


def build_resnet18(
    num_classes: int = 2,
    pretrained: bool = True,
    dropout: float = 0.2,
) -> nn.Module:
    """Instantiate a ResNet-18 model adapted for histopathology tile classification.

    Args:
        num_classes: Number of target classes.
        pretrained: If True, initialize with ImageNet weights.
        dropout: Dropout probability before the classification linear head.
    """
    if pretrained:
        weights = models.ResNet18_Weights.DEFAULT
    else:
        weights = None

    model = models.resnet18(weights=weights)

    in_features = model.fc.in_features
    if dropout > 0.0:
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
    else:
        model.fc = nn.Linear(in_features, num_classes)

    return model


def get_model(
    model_name: str = "resnet18",
    num_classes: int = 2,
    pretrained: bool = True,
) -> nn.Module:
    """Model factory returning configured neural network architecture."""
    name = model_name.lower()
    if name == "resnet18":
        return build_resnet18(num_classes=num_classes, pretrained=pretrained)
    elif name == "simple_cnn":
        return SimpleCNN(num_classes=num_classes)
    elif name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        m = models.resnet50(weights=weights)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    else:
        raise ValueError(f"Unknown neural network model architecture: {model_name}")
