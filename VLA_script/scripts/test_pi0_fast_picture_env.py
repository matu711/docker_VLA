from __future__ import annotations

import pathlib
import sys
import time

import gymnasium as gym
import gymnasium_robotics  # noqa: F401
import mujoco
import numpy as np
import matplotlib.pyplot as plt

from openpi.training import config as _config
from openpi.policies import policy_config


ENV_ID = "FetchPickAndPlaceDense-v4"
CAMERA_NAME = "external_camera_0"

CHECKPOINT_DIR = pathlib.Path(
    "./checkpoints/pi0_fast_picture_env_lora/test_picture_env/999"
)

PROMPT = "pick up the object and place it at the goal"

BASE_OBJECT_POS = np.array([1.30, 0.50, 0.42], dtype=np.float32)
GOAL_POS = np.array([1.30, 1.00, 0.42], dtype=np.float32)

MAX_STEPS = 220


class CameraRenderer:
    def __init__(self, env, camera_name, width=480, height=480):
        self.env = env
        self.camera_id = mujoco.mj_name2id(
            env.unwrapped.model,
            mujoco.mjtObj.mjOBJ_CAMERA,
            camera_name,
        )
        if self.camera_id < 0:
            raise ValueError(f"camera not found: {camera_name}")

        self.renderer = mujoco.Renderer(
            env.unwrapped.model,
            height=height,
            width=width,
        )

    def render(self):
        mujoco.mj_forward(
            self.env.unwrapped.model,
            self.env.unwrapped.data,
        )
        self.renderer.update_scene(
            self.env.unwrapped.data,
            camera=self.camera_id,
        )
        return self.renderer.render()

    def close(self):
        self.renderer.close()


def set_object_and_goal(env, object_pos, goal_pos):
    env.unwrapped.data.joint("object0:joint").qpos[:] = [
        object_pos[0],
        object_pos[1],
        object_pos[2],
        1.0,
        0.0,
        0.0,
        0.0,
    ]

    env.unwrapped.goal = goal_pos.copy()

    if hasattr(env.unwrapped, "_render_callback"):
        env.unwrapped._render_callback()

    mujoco.mj_forward(
        env.unwrapped.model,
        env.unwrapped.data,
    )


def get_gripper_pos(env):
    return env.unwrapped.data.body("robot0:gripper_link").xpos.copy()


def get_object_pos(env):
    return env.unwrapped.data.body("object0").xpos.copy()


def make_obs(img, gripper_pos):
    state = np.zeros(7, dtype=np.float32)
    state[:3] = gripper_pos.astype(np.float32)

    return {
        "image": {
            "base_0_rgb": img,
            "base_1_rgb": img,
            "wrist_0_rgb": img,
        },
        "image_mask": {
            "base_0_rgb": np.array(True),
            "base_1_rgb": np.array(True),
            "wrist_0_rgb": np.array(True),
        },
        "state": state,
        "prompt": PROMPT,
    }


def main():
    cfg = next(
        x for x in _config._CONFIGS
        if x.name == "pi0_fast_picture_env_lora"
    )

    policy = policy_config.create_trained_policy(
        cfg,
        CHECKPOINT_DIR,
        default_prompt=PROMPT,
        norm_stats={},
    )

    env = gym.make(
        ENV_ID,
        render_mode=None,
        max_episode_steps=500,
    )

    renderer = CameraRenderer(
        env=env,
        camera_name=CAMERA_NAME,
        width=480,
        height=480,
    )

    obs, info = env.reset()
    set_object_and_goal(
        env,
        object_pos=BASE_OBJECT_POS,
        goal_pos=GOAL_POS,
    )

    plt.ion()
    fig, ax = plt.subplots(figsize=(7, 7))
    img = renderer.render()
    im = ax.imshow(img)
    ax.axis("off")

    try:
        for step in range(MAX_STEPS):
            img = renderer.render()
            gripper_pos = get_gripper_pos(env)

            pi_obs = make_obs(img, gripper_pos)

            out = policy.infer(pi_obs)

            if isinstance(out, dict):
                actions = out.get("actions", None)
                if actions is None:
                    print("policy output keys:", out.keys())
                    raise KeyError("actions not found in policy output")
            else:
                actions = out

            actions = np.asarray(actions)

            if actions.ndim == 2:
                action = actions[0]
            else:
                action = actions

            action = action.astype(np.float32)
            action = np.clip(action, -1.0, 1.0)

            obs, reward, terminated, truncated, info = env.step(action)

            object_pos = get_object_pos(env)
            goal_dist = np.linalg.norm(object_pos - GOAL_POS)

            im.set_data(img)
            ax.set_title(
                f"pi0-fast LoRA | step={step} | reward={reward:.3f} | goal_dist={goal_dist:.3f}\n"
                f"action={np.array2string(action, precision=3)}"
            )
            fig.canvas.draw()
            fig.canvas.flush_events()

            if step % 10 == 0:
                print(
                    f"step={step:03d}",
                    f"reward={reward:.3f}",
                    f"goal_dist={goal_dist:.4f}",
                    f"action={action}",
                    f"object={object_pos}",
                    f"gripper={gripper_pos}",
                )

            if goal_dist < 0.05:
                print("success")
                break

            if terminated or truncated:
                print("env done")
                break

            time.sleep(0.02)

    finally:
        renderer.close()
        env.close()
        plt.close("all")


if __name__ == "__main__":
    main()