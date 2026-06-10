from __future__ import annotations


import sys

sys.path.append("/home/shunsuke-m/docker_VLA/workspace/gymnasium_robotics/envs/fetch/__init__.py")

import pick_and_place

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gymnasium as gym

import matplotlib.pyplot as plt
import numpy as np

from bc_project.config import load_config
from bc_project.envs.fetch_pick_place import (
    CameraRenderer,
    get_gripper_pos,
    get_object_pos,
    set_object_and_goal,
)
from bc_project.envs.scripted_policy import ScriptedPickPlacePolicy
from bc_project.utils import as_np_float_array, ensure_dir, set_seed


def make_empty_dataset_dict():
    return {
        "images": [],
        "actions": [],
        "gripper_pos": [],
        "object_pos": [],
        "goal_pos": [],
        "demo_id": [],
        "step_id": [],
        "phase": [],
        "reward": [],
    }


def sample_object_pos(cfg, base_object_pos: np.ndarray) -> np.ndarray:
    object_pos = base_object_pos.copy()
    object_pos[0] = np.random.uniform(
        float(cfg.collect.x_min),
        float(cfg.collect.x_max),
    )

    # y方向も使いたくなったら config で有効化できる．
    if bool(cfg.collect.sample_y):
        object_pos[1] = np.random.uniform(
            float(cfg.collect.y_min),
            float(cfg.collect.y_max),
        )

    return object_pos


def collect_one_demo(cfg, env, renderer, policy, object_pos, goal_pos, demo_id: int):
    obs, info = env.reset()
    set_object_and_goal(env, object_pos, goal_pos)

    data = make_empty_dataset_dict()

    show = bool(cfg.collect.show)
    if show:
        plt.ion()
        fig, ax = plt.subplots(figsize=(7, 7))
        im = ax.imshow(renderer.render())
        ax.axis("off")
    else:
        fig = ax = im = None

    phase = 0
    for step in range(int(cfg.env.max_steps)):
        img = renderer.render()
        action, phase = policy(env, phase)

        noise_std = float(cfg.collect.action_noise_std)
        if noise_std > 0.0:
            action[:3] += np.random.normal(0.0, noise_std, size=3)
            action[:3] = np.clip(action[:3], -1.0, 1.0)

        data["images"].append(img)
        data["actions"].append(action.copy())
        data["gripper_pos"].append(get_gripper_pos(env))
        data["object_pos"].append(get_object_pos(env))
        data["goal_pos"].append(env.unwrapped.goal.copy())
        data["demo_id"].append(demo_id)
        data["step_id"].append(step)
        data["phase"].append(phase)

        if show:
            im.set_data(img)
            ax.set_title(
                f"demo={demo_id:04d} | step={step} | phase={phase}\n"
                f"object=[{object_pos[0]:.3f}, {object_pos[1]:.3f}, {object_pos[2]:.3f}] "
                f"goal=[{goal_pos[0]:.3f}, {goal_pos[1]:.3f}, {goal_pos[2]:.3f}]"
            )
            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(float(cfg.collect.show_sleep_sec))

        obs, reward, terminated, truncated, info = env.step(action)
        data["reward"].append(reward)

        if phase == 6 and step > 20:
            break

    if show:
        plt.close(fig)

    return data


def append_demo_data(all_data, demo_data):
    for key in all_data:
        all_data[key].extend(demo_data[key])


def save_dataset(save_path, all_data):
    np.savez_compressed(
        save_path,
        images=np.array(all_data["images"], dtype=np.uint8),
        actions=np.array(all_data["actions"], dtype=np.float32),
        gripper_pos=np.array(all_data["gripper_pos"], dtype=np.float32),
        object_pos=np.array(all_data["object_pos"], dtype=np.float32),
        goal_pos=np.array(all_data["goal_pos"], dtype=np.float32),
        demo_id=np.array(all_data["demo_id"], dtype=np.int32),
        step_id=np.array(all_data["step_id"], dtype=np.int32),
        phase=np.array(all_data["phase"], dtype=np.int32),
        reward=np.array(all_data["reward"], dtype=np.float32),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.seed))
    ensure_dir(cfg.paths.log_dir)

    env = gym.make(cfg.env.id, render_mode=None)

    print("available cameras:")
    for i in range(env.unwrapped.model.ncam):
        print(i, env.unwrapped.model.camera(i).name)

    renderer = CameraRenderer(
        env=env,
        camera_name=cfg.env.camera_name,
        width=int(cfg.env.render_width),
        height=int(cfg.env.render_height),
    )
    policy = ScriptedPickPlacePolicy(cfg)

    all_data = make_empty_dataset_dict()
    base_object_pos = as_np_float_array(cfg.collect.base_object_pos)
    goal_pos = as_np_float_array(cfg.collect.goal_pos)

    n_demo = int(cfg.collect.n_demo)

    try:
        for demo_id in range(n_demo):
            object_pos = sample_object_pos(cfg, base_object_pos)

            demo_data = collect_one_demo(
                cfg=cfg,
                env=env,
                renderer=renderer,
                policy=policy,
                object_pos=object_pos,
                goal_pos=goal_pos,
                demo_id=demo_id,
            )
            append_demo_data(all_data, demo_data)

            print(
                f"demo={demo_id:04d}",
                f"x={object_pos[0]:.3f}",
                f"y={object_pos[1]:.3f}",
                f"steps={len(demo_data['actions'])}",
            )
    finally:
        renderer.close()
        env.close()

    save_dataset(cfg.paths.dataset_path, all_data)

    print("saved:", cfg.paths.dataset_path)
    print("total demos:", n_demo)
    print("total steps:", len(all_data["actions"]))


if __name__ == "__main__":
    main()
