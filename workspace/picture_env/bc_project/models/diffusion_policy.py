from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import RobotEncoder, SmallCNNEncoder


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            torch.arange(half, device=t.device, dtype=torch.float32)
            * -(math.log(10000.0) / max(half - 1, 1))
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class DiffusionPolicy(nn.Module):
    """行動をDDPM風に生成する最小Diffusion Policy．

    ここでは action chunk ではなく，まず1 step actionをdenoiseする形にしている．
    ACTと同じaction_horizon版に拡張する時は，action_dimを action_horizon*action_dim にする．
    """

    def __init__(
        self,
        action_dim: int = 4,
        robot_state_dim: int = 3,
        image_feat_dim: int = 256,
        robot_feat_dim: int = 64,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        diffusion_steps: int = 50,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        **kwargs,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.diffusion_steps = diffusion_steps
        self.image_encoder = SmallCNNEncoder(image_feat_dim=image_feat_dim)
        self.robot_encoder = RobotEncoder(robot_state_dim, robot_feat_dim)
        self.time_embed = SinusoidalTimeEmbedding(hidden_dim)

        cond_dim = image_feat_dim + robot_feat_dim
        self.noise_pred_net = nn.Sequential(
            nn.Linear(action_dim + cond_dim + hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_dim),
        )

        betas = torch.linspace(beta_start, beta_end, diffusion_steps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)

    def encode_cond(self, image, robot_state):
        return torch.cat([self.image_encoder(image), self.robot_encoder(robot_state)], dim=1)

    def predict_noise(self, noisy_action, t, cond):
        t_emb = self.time_embed(t)
        x = torch.cat([noisy_action, cond, t_emb], dim=1)
        return self.noise_pred_net(x)

    def compute_loss(self, batch, loss_name: str = "mse"):
        action = batch["action"]
        cond = self.encode_cond(batch["image"], batch["robot_state"])
        b = action.shape[0]
        t = torch.randint(0, self.diffusion_steps, (b,), device=action.device)
        noise = torch.randn_like(action)
        alpha_bar = self.alpha_bars[t].unsqueeze(1)
        noisy_action = torch.sqrt(alpha_bar) * action + torch.sqrt(1.0 - alpha_bar) * noise
        pred_noise = self.predict_noise(noisy_action, t, cond)
        loss = F.mse_loss(pred_noise, noise)
        return loss, {"diffusion_loss": loss.detach()}

    @torch.no_grad()
    def forward(self, image: torch.Tensor, robot_state: torch.Tensor) -> torch.Tensor:
        cond = self.encode_cond(image, robot_state)
        b = image.shape[0]
        x = torch.randn(b, self.action_dim, device=image.device)

        for step in reversed(range(self.diffusion_steps)):
            t = torch.full((b,), step, device=image.device, dtype=torch.long)
            pred_noise = self.predict_noise(x, t, cond)
            beta = self.betas[step]
            alpha = self.alphas[step]
            alpha_bar = self.alpha_bars[step]
            x = (1.0 / torch.sqrt(alpha)) * (
                x - (beta / torch.sqrt(1.0 - alpha_bar)) * pred_noise
            )
            if step > 0:
                x = x + torch.sqrt(beta) * torch.randn_like(x)

        return torch.tanh(x)
