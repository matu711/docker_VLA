from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import RobotEncoder, SmallCNNEncoder, mlp


class CNNBC(nn.Module):
    """画像 + ロボット状態 -> 1 step action の基本BC．"""

    def __init__(
        self,
        action_dim: int = 4,
        robot_state_dim: int = 3,
        image_feat_dim: int = 256,
        robot_feat_dim: int = 64,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.image_encoder = SmallCNNEncoder(image_feat_dim=image_feat_dim)
        self.robot_encoder = RobotEncoder(robot_state_dim, robot_feat_dim)
        self.policy_head = mlp(
            image_feat_dim + robot_feat_dim,
            hidden_dim,
            action_dim,
            dropout=dropout,
            tanh=True,
        )

    def forward(self, image: torch.Tensor, robot_state: torch.Tensor) -> torch.Tensor:
        image_feat = self.image_encoder(image)
        robot_feat = self.robot_encoder(robot_state)
        return self.policy_head(torch.cat([image_feat, robot_feat], dim=1))

    def compute_loss(self, batch, loss_name: str = "smooth_l1"):
        pred = self(batch["image"], batch["robot_state"])
        target = batch["action"]
        if loss_name == "mse":
            return F.mse_loss(pred, target), {"bc_loss": F.mse_loss(pred, target).detach()}
        loss = F.smooth_l1_loss(pred, target)
        return loss, {"bc_loss": loss.detach()}
