from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import RobotEncoder, SmallCNNEncoder, mlp


class CVAEBC(nn.Module):
    """CVAE版BC．

    学習時: q(z | obs, action) から z をサンプルし，action を再構成．
    評価時: z=0 を使って決定論的に action を出す．
    """

    def __init__(
        self,
        action_dim: int = 4,
        robot_state_dim: int = 3,
        image_feat_dim: int = 256,
        robot_feat_dim: int = 64,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        latent_dim: int = 32,
        kl_weight: float = 0.01,
        **kwargs,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight
        self.image_encoder = SmallCNNEncoder(image_feat_dim=image_feat_dim)
        self.robot_encoder = RobotEncoder(robot_state_dim, robot_feat_dim)
        obs_dim = image_feat_dim + robot_feat_dim

        self.posterior = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)
        self.decoder = mlp(obs_dim + latent_dim, hidden_dim, action_dim, dropout=dropout, tanh=True)

    def encode_obs(self, image, robot_state):
        return torch.cat([self.image_encoder(image), self.robot_encoder(robot_state)], dim=1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, image: torch.Tensor, robot_state: torch.Tensor) -> torch.Tensor:
        obs_feat = self.encode_obs(image, robot_state)
        z = torch.zeros(obs_feat.shape[0], self.latent_dim, device=obs_feat.device)
        return self.decoder(torch.cat([obs_feat, z], dim=1))

    def compute_loss(self, batch, loss_name: str = "smooth_l1"):
        obs_feat = self.encode_obs(batch["image"], batch["robot_state"])
        action = batch["action"]
        h = self.posterior(torch.cat([obs_feat, action], dim=1))
        mu = self.mu_head(h)
        logvar = self.logvar_head(h).clamp(-10.0, 10.0)
        z = self.reparameterize(mu, logvar)
        pred = self.decoder(torch.cat([obs_feat, z], dim=1))

        if loss_name == "mse":
            recon = F.mse_loss(pred, action)
        else:
            recon = F.smooth_l1_loss(pred, action)
        kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
        loss = recon + self.kl_weight * kl
        return loss, {"recon_loss": recon.detach(), "kl_loss": kl.detach()}
