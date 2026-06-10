from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gymnasium as gym
import gymnasium_robotics  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import torch

from bc_project.config import load_config, resolve_device
from bc_project.envs.fetch_pick_place import (
    CameraRenderer,
    get_gripper_pos,
    get_object_pos,
    make_object_pos,
    set_object_and_goal,
)
from bc_project.models.registry import build_model
from bc_project.preprocess import preprocess_image_np
from bc_project.utils import as_np_float_array, set_seed


def load_model(cfg, device):
    checkpoint = torch.load(cfg.paths.model_path, map_location=device)

    # checkpoint側の情報を優先して，config変更による読み込み事故を減らす．
    if "image_size" in checkpoint:
        cfg.data.image_size = int(checkpoint["image_size"])
    if "robot_state_dim" in checkpoint:
        cfg.data.robot_state_dim = int(checkpoint["robot_state_dim"])
    if "action_dim" in checkpoint:
        cfg.data.action_dim = int(checkpoint["action_dim"])
    if "model_name" in checkpoint:
        cfg.model.name = checkpoint["model_name"]
    if "model_config" in checkpoint:
        for key, value in checkpoint["model_config"].items():
            cfg.model[key] = value

    model = build_model(cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    robot_mean = checkpoint["robot_mean"].astype(np.float32)
    robot_std = checkpoint["robot_std"].astype(np.float32)

    print("loaded:", cfg.paths.model_path)
    print("checkpoint epoch:", checkpoint.get("epoch"))
    print("checkpoint val_loss:", checkpoint.get("val_loss"))
    print("model:", cfg.model.name)
    print("image_size:", cfg.data.image_size)
    print("robot_mean:", robot_mean)
    print("robot_std:", robot_std)

    return model, robot_mean, robot_std


def run_one_episode(cfg, env, renderer, model, robot_mean, robot_std, object_pos, goal_pos, label, device):
    obs, info = env.reset()
    set_object_and_goal(env, object_pos, goal_pos)

    print("\n-------------------------")
    print(label)
    print("object:", object_pos)
    print("goal  :", goal_pos)

    plt.ion()
    fig, ax = plt.subplots(figsize=(7, 7))
    img = renderer.render()
    im = ax.imshow(img)
    ax.axis("off")

    for step in range(int(cfg.env.max_steps)):
        img = renderer.render()
        image = preprocess_image_np(img, int(cfg.data.image_size)).unsqueeze(0).to(device)

        robot_state = get_gripper_pos(env).astype(np.float32)
        robot_state = (robot_state - robot_mean) / robot_std
        robot_state = torch.from_numpy(robot_state).float().unsqueeze(0).to(device)

        with torch.no_grad():
            action = model(image, robot_state).cpu().numpy()[0]

        obs, reward, terminated, truncated, info = env.step(action)

        im.set_data(img)
        ax.set_title(
            f"{label} | step={step} | reward={reward:.3f}\n"
            f"action=[{action[0]:+.2f}, {action[1]:+.2f}, {action[2]:+.2f}, {action[3]:+.2f}]"
        )
        fig.canvas.draw()
        fig.canvas.flush_events()

        if step % 20 == 0:
            print(
                f"step={step:03d}",
                f"reward={reward:.3f}",
                f"action={action}",
                f"object={get_object_pos(env)}",
                f"gripper={get_gripper_pos(env)}",
            )

        time.sleep(float(cfg.test.sleep_sec))

        if terminated or truncated:
            break

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.seed))
    device = resolve_device(cfg.device)

    print("device:", device)
    model, robot_mean, robot_std = load_model(cfg, device)

    env = gym.make(
        cfg.env.id,
        render_mode=None,
        max_episode_steps=int(cfg.env.max_episode_steps),
    )

    print("available cameras:")
    for i in range(env.unwrapped.model.ncam):
        print(i, env.unwrapped.model.camera(i).name)

    renderer = CameraRenderer(
        env=env,
        camera_name=cfg.env.camera_name,
        width=int(cfg.env.render_width),
        height=int(cfg.env.render_height),
    )

    base_object_pos = as_np_float_array(cfg.collect.base_object_pos)
    goal_pos = as_np_float_array(cfg.collect.goal_pos)

    seen_x = np.linspace(
        float(cfg.collect.x_min),
        float(cfg.collect.x_max),
        int(cfg.test.n_seen_pos),
    )
    unseen_x = (seen_x[:-1] + seen_x[1:]) / 2.0

    print("seen x:", seen_x)
    print("unseen x:", unseen_x)

    try:
        if bool(cfg.test.run_seen):
            print("\n====================")
            print("SEEN POSITIONS")
            print("====================")
            for x in seen_x:
                object_pos = make_object_pos(base_object_pos, x)
                run_one_episode(
                    cfg, env, renderer, model, robot_mean, robot_std,
                    object_pos, goal_pos, f"SEEN x={x:.3f}", device,
                )
                time.sleep(0.5)

        if bool(cfg.test.run_unseen):
            print("\n====================")
            print("UNSEEN POSITIONS")
            print("====================")
            for x in unseen_x:
                object_pos = make_object_pos(base_object_pos, x)
                run_one_episode(
                    cfg, env, renderer, model, robot_mean, robot_std,
                    object_pos, goal_pos, f"UNSEEN x={x:.3f}", device,
                )
                time.sleep(0.5)
    finally:
        renderer.close()
        env.close()
        plt.close("all")


if __name__ == "__main__":
    main()
