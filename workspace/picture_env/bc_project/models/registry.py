from __future__ import annotations

from .act_bc import ACTBC
from .cnn_bc import CNNBC
from .cvae_bc import CVAEBC
from .diffusion_policy import DiffusionPolicy
from .transformer_bc import TransformerBC


MODEL_REGISTRY = {
    "cnn_bc": CNNBC,
    "transformer_bc": TransformerBC,
    "act_bc": ACTBC,
    "diffusion_policy": DiffusionPolicy,
    "cvae_bc": CVAEBC,
}


def _get(cfg, key: str, default):
    return cfg.model[key] if key in cfg.model else default


def build_model(cfg):
    name = cfg.model.name
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"unknown model.name={name}. available={list(MODEL_REGISTRY.keys())}"
        )

    model_cls = MODEL_REGISTRY[name]
    return model_cls(
        action_dim=int(cfg.data.action_dim),
        robot_state_dim=int(cfg.data.robot_state_dim),
        image_feat_dim=int(_get(cfg, "image_feat_dim", 256)),
        robot_feat_dim=int(_get(cfg, "robot_feat_dim", 64)),
        hidden_dim=int(_get(cfg, "hidden_dim", 256)),
        dropout=float(_get(cfg, "dropout", 0.1)),
        d_model=int(_get(cfg, "d_model", 256)),
        n_heads=int(_get(cfg, "n_heads", 8)),
        n_layers=int(_get(cfg, "n_layers", 4)),
        action_horizon=int(_get(cfg, "action_horizon", 1)),
        latent_dim=int(_get(cfg, "latent_dim", 32)),
        kl_weight=float(_get(cfg, "kl_weight", 0.01)),
        diffusion_steps=int(_get(cfg, "diffusion_steps", 50)),
        beta_start=float(_get(cfg, "beta_start", 1e-4)),
        beta_end=float(_get(cfg, "beta_end", 2e-2)),
    )
