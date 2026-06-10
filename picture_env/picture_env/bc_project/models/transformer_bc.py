from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import RobotEncoder, SmallCNNEncoder, mlp


class TransformerBC(nn.Module):
    """Transformer版BC．

    画像特徴，ロボット状態特徴，CLS token を Transformer Encoder に入れて action を出す．
    履歴を使う本格版に拡張する前の，差し替え用の最小構成．
    """

    def __init__(
        self,
        action_dim: int = 4,
        robot_state_dim: int = 3,
        image_feat_dim: int = 256,
        robot_feat_dim: int = 64,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        **kwargs,
    ):
        super().__init__()
        self.image_encoder = SmallCNNEncoder(image_feat_dim=image_feat_dim)
        self.robot_encoder = RobotEncoder(robot_state_dim, robot_feat_dim)
        self.image_proj = nn.Linear(image_feat_dim, d_model)
        self.robot_proj = nn.Linear(robot_feat_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, 3, d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.policy_head = mlp(d_model, hidden_dim, action_dim, dropout=dropout, tanh=True)

    def forward(self, image: torch.Tensor, robot_state: torch.Tensor) -> torch.Tensor:
        b = image.shape[0]
        img_tok = self.image_proj(self.image_encoder(image)).unsqueeze(1)
        state_tok = self.robot_proj(self.robot_encoder(robot_state)).unsqueeze(1)
        cls_tok = self.cls_token.expand(b, -1, -1)
        tokens = torch.cat([cls_tok, img_tok, state_tok], dim=1) + self.pos_embed
        out = self.transformer(tokens)
        return self.policy_head(out[:, 0])

    def compute_loss(self, batch, loss_name: str = "smooth_l1"):
        pred = self(batch["image"], batch["robot_state"])
        target = batch["action"]
        if loss_name == "mse":
            loss = F.mse_loss(pred, target)
        else:
            loss = F.smooth_l1_loss(pred, target)
        return loss, {"bc_loss": loss.detach()}
