from __future__ import annotations

import time
import sys
from pathlib import Path
import imageio.v2 as imageio

import gymnasium as gym
import gymnasium_robotics  # noqa: F401
import numpy as np
import matplotlib.pyplot as plt
from openpi_client import websocket_client_policy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bc_project.envs.fetch_pick_place import (
    CameraRenderer,
    get_gripper_pos,
    get_object_pos,
    set_object_and_goal,
)


HOST = "localhost"
PORT = 8000
PROMPT = "pick up the object and place it at the goal"
VIDEO_DIR = Path("./log/success_videos")
VIDEO_DIR.mkdir(parents=True, exist_ok=True)

ENV_ID = "FetchPickAndPlaceDense-v4"
CAMERA_NAME = "external_camera_0"

IMAGE_SIZE = 224
MAX_STEPS = 300

X_MIN = 1.10
X_MAX = 1.50
N_TRIALS = 100
OBJECT_POS_BASE = np.array([1.30, 0.50, 0.42], dtype=np.float32)
GOAL_POS = np.array([1.30, 1.00, 0.42], dtype=np.float32)


def resize_image(img: np.ndarray, size: int = 224) -> np.ndarray:
    """
    renderer.render() の画像を 224x224 に変換する。
    matplotlibではなくOpenCVがあればcv2の方がよいが，
    ここではPILなしでも動きやすいように簡単に書く。
    """
    try:
        import cv2

        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        return img.astype(np.uint8)
    except ImportError:
        from PIL import Image

        img = Image.fromarray(img)
        img = img.resize((size, size))
        return np.array(img, dtype=np.uint8)


def make_openpi_obs(env, renderer):
    """
    FetchPickAndPlace環境からOpenPIへ送るobsを作る。
    """

    img = renderer.render()
    img = resize_image(img, IMAGE_SIZE)

    # ここでは状態として gripper_pos + object_pos + goal_pos を入れる例
    # ただし，学習時のstate次元と合わせる必要がある。
    gripper_pos = get_gripper_pos(env).astype(np.float32)
    object_pos = get_object_pos(env).astype(np.float32)
    goal_pos = env.unwrapped.goal.copy().astype(np.float32)

    # 例1: stateを4次元にしたい場合
    # state = np.zeros(4, dtype=np.float32)

    # 例2: 位置情報を入れる場合
    state = np.concatenate(
        [
            gripper_pos,  # 3次元
            object_pos,   # 3次元
            goal_pos,     # 3次元
        ],
        axis=0,
    ).astype(np.float32)

    return {
        "image": {
            "base_0_rgb": img,
            "left_wrist_0_rgb": img,
            "right_wrist_0_rgb": img,
        },
        "image_mask": {
            "base_0_rgb": np.array(True),
            "left_wrist_0_rgb": np.array(True),
            "right_wrist_0_rgb": np.array(True),
        },
        "state": state,
        "prompt": PROMPT,
    }


def to_fetch_action(openpi_action: np.ndarray) -> np.ndarray:
    """
    OpenPIから返ってきたactionをFetchPickAndPlace用の4次元actionに変換する。

    FetchPickAndPlaceのactionは基本的に
    [dx, dy, dz, gripper]
    の4次元。
    """

    action = np.asarray(openpi_action, dtype=np.float32)

    # OpenPIのaction次元が4より大きい場合は先頭4次元だけ使う
    if action.shape[0] > 4:
        action = action[:4]

    # 4次元より少ない場合はエラー
    if action.shape[0] < 4:
        raise ValueError(f"action dim must be >= 4, but got {action.shape}")

    # Fetch環境のaction範囲に合わせる
    action = np.clip(action, -1.0, 1.0)

    return action.astype(np.float32)

def sample_object_pos():
    object_pos = OBJECT_POS_BASE.copy()
    object_pos[0] = np.random.uniform(X_MIN, X_MAX)
    return object_pos.astype(np.float32)


def main():
    policy = websocket_client_policy.WebsocketClientPolicy(
        host=HOST,
        port=PORT,
    )

    env = gym.make(
        ENV_ID,
        render_mode=None,
        max_episode_steps=150,
    )

    print("available cameras:")
    for i in range(env.unwrapped.model.ncam):
        print(i, env.unwrapped.model.camera(i).name)

    renderer = CameraRenderer(
        env=env,
        camera_name=CAMERA_NAME,
        width=480,
        height=480,
    )

    success_count = 0
    trial_results = []

    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(np.zeros((480, 480, 3), dtype=np.uint8))
    ax.axis("off")

    try:
        for trial in range(N_TRIALS):
            object_pos = sample_object_pos()
            goal_pos = GOAL_POS.copy()

            obs, info = env.reset()
            set_object_and_goal(env, object_pos, goal_pos)

            frames = []
            episode_success = False
            final_dist = None

            print(
                f"\ntrial={trial:03d}",
                f"object_pos={object_pos}",
                f"goal_pos={goal_pos}",
            )

            for step in range(MAX_STEPS):
                openpi_obs = make_openpi_obs(env, renderer)

                out = policy.infer(openpi_obs)
                actions = out["actions"]
                first_action = actions[0]

                env_action = to_fetch_action(first_action)

                obs, reward, terminated, truncated, info = env.step(env_action)

                success_by_info = bool(info.get("is_success", False))

                img = renderer.render()
                frames.append(img.copy())

                im.set_data(img)
                ax.set_title(
                    f"trial={trial} step={step} reward={reward:.3f}\n"
                    f"success={success_by_info} is_success={info.get('is_success', None)}\n"
                    f"action={env_action}"
                )
                fig.canvas.draw()
                fig.canvas.flush_events()

                if success_by_info:
                    episode_success = True
                    print("success!")

                    video_path = VIDEO_DIR / f"success_trial_{trial:03d}_x_{object_pos[0]:.3f}.mp4"
                    imageio.mimsave(video_path, frames, fps=30)
                    print("saved video:", video_path)

                    break

                if terminated or truncated:
                    print("episode finished")
                    break


            if episode_success:
                success_count += 1

            trial_results.append(
                {
                    "trial": trial,
                    "object_x": float(object_pos[0]),
                    "success": episode_success,
                }
            )

        success_rate = success_count / N_TRIALS

        print("\n========== result ==========")
        print("trials:", N_TRIALS)
        print("success:", success_count)
        print("success rate:", success_rate)

    finally:
        renderer.close()
        env.close()
        plt.close(fig)

if __name__ == "__main__":
    main()