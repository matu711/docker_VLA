from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.net(x))


class SmallCNNEncoder(nn.Module):
    def __init__(self, image_feat_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ResidualBlock(32),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResidualBlock(64),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            ResidualBlock(128),
            nn.Conv2d(128, image_feat_dim, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(image_feat_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.net(image).flatten(1)


class RobotEncoder(nn.Module):
    def __init__(self, robot_state_dim: int = 3, robot_feat_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(robot_state_dim, robot_feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(robot_feat_dim, robot_feat_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, robot_state: torch.Tensor) -> torch.Tensor:
        return self.net(robot_state)


def mlp(in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0, tanh: bool = False):
    layers = [
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(inplace=True),
    ]
    if dropout > 0.0:
        layers.append(nn.Dropout(dropout))
    layers += [
        nn.Linear(hidden_dim, hidden_dim // 2),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim // 2, out_dim),
    ]
    if tanh:
        layers.append(nn.Tanh())
    return nn.Sequential(*layers)
