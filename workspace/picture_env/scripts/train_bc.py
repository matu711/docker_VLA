from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from bc_project.config import load_config, resolve_device
from bc_project.dataset import BCDataset
from bc_project.models.registry import build_model
from bc_project.utils import ensure_dir, set_seed


def move_batch_to_device(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def compute_model_loss(model, batch, loss_name: str):
    if hasattr(model, "compute_loss"):
        return model.compute_loss(batch, loss_name=loss_name)

    pred_actions = model(batch["image"], batch["robot_state"])
    if loss_name == "mse":
        loss = F.mse_loss(pred_actions, batch["action"])
    else:
        loss = F.smooth_l1_loss(pred_actions, batch["action"])
    return loss, {"bc_loss": loss.detach()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.seed))
    device = resolve_device(cfg.device)
    ensure_dir(cfg.paths.model_dir)

    print("device:", device)
    print("model:", cfg.model.name)

    dataset = BCDataset(cfg)
    n_total = len(dataset)
    n_val = max(1, int(n_total * float(cfg.train.val_ratio)))
    n_train = n_total - n_val

    train_set, val_set = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(int(cfg.seed)),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=int(cfg.train.batch_size),
        shuffle=True,
        num_workers=int(cfg.train.num_workers),
        pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=int(cfg.train.batch_size),
        shuffle=False,
        num_workers=int(cfg.train.num_workers),
        pin_memory=(device == "cuda"),
    )

    model = build_model(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.train.lr),
        weight_decay=float(cfg.train.weight_decay),
    )

    best_val_loss = float("inf")

    for epoch in range(int(cfg.train.epochs)):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            batch = move_batch_to_device(batch, device)
            loss, _ = compute_model_loss(model, batch, cfg.train.loss)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=float(cfg.train.grad_clip_norm),
            )
            optimizer.step()

            train_loss += loss.item() * batch["image"].size(0)

        train_loss /= len(train_set)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = move_batch_to_device(batch, device)
                loss, _ = compute_model_loss(model, batch, cfg.train.loss)
                val_loss += loss.item() * batch["image"].size(0)

        val_loss /= len(val_set)

        print(
            f"epoch={epoch + 1:03d} "
            f"train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = cfg.paths.model_path
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "image_size": int(cfg.data.image_size),
                    "robot_state_dim": int(cfg.data.robot_state_dim),
                    "action_dim": int(cfg.data.action_dim),
                    "action_horizon": int(cfg.data.action_horizon),
                    "robot_mean": dataset.robot_mean,
                    "robot_std": dataset.robot_std,
                    "model_name": cfg.model.name,
                    "model_config": dict(cfg.model),
                    "loss_type": cfg.train.loss,
                },
                save_path,
            )
            print("saved:", save_path)

    print("done")


if __name__ == "__main__":
    main()
