from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import RobotEncoder, SmallCNNEncoder


class ACTBC(nn.Module):
    """Action Chunking Transformer の最小版．

    obs_t から action_t:t+K をまとめて出す．評価時の forward は先頭 action だけ返す．
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
        action_horizon: int = 16,
        **kwargs,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.action_horizon = action_horizon

        self.image_encoder = SmallCNNEncoder(image_feat_dim=image_feat_dim)
        self.robot_encoder = RobotEncoder(robot_state_dim, robot_feat_dim)
        self.obs_proj = nn.Linear(image_feat_dim + robot_feat_dim, d_model)
        self.query_embed = nn.Parameter(torch.zeros(1, action_horizon, d_model))

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_layers)
        self.action_head = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def predict_chunk(self, image: torch.Tensor, robot_state: torch.Tensor) -> torch.Tensor:
        b = image.shape[0]
        image_feat = self.image_encoder(image)
        robot_feat = self.robot_encoder(robot_state)
        memory = self.obs_proj(torch.cat([image_feat, robot_feat], dim=1)).unsqueeze(1)
        query = self.query_embed.expand(b, -1, -1)
        dec = self.decoder(tgt=query, memory=memory)
        return self.action_head(dec)

    def forward(self, image: torch.Tensor, robot_state: torch.Tensor) -> torch.Tensor:
        return self.predict_chunk(image, robot_state)[:, 0]

    def compute_loss(self, batch, loss_name: str = "smooth_l1"):
        pred_chunk = self.predict_chunk(batch["image"], batch["robot_state"])
        target_chunk = batch["action_chunk"][:, : self.action_horizon]
        if pred_chunk.shape[1] != target_chunk.shape[1]:
            pred_chunk = pred_chunk[:, : target_chunk.shape[1]]
        if loss_name == "mse":
            loss = F.mse_loss(pred_chunk, target_chunk)
        else:
            loss = F.smooth_l1_loss(pred_chunk, target_chunk)
        return loss, {"act_loss": loss.detach()}
