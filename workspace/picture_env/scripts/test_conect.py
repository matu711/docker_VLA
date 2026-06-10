import numpy as np
from openpi_client import websocket_client_policy
import gymnasium as gym

HOST = "localhost"
PORT = 8000
PROMPT = "pick up the object and place it at the goal"


def make_dummy_obs():
    img = np.zeros((224, 224, 3), dtype=np.uint8)

    state = np.zeros(4, dtype=np.float32)

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


def main():
    policy = websocket_client_policy.WebsocketClientPolicy(
        host=HOST,
        port=PORT,
    )

    env = gym.make(ENV_ID, render_mode=None)

    print("available cameras:")
    for i in range(env.unwrapped.model.ncam):
        print(i, env.unwrapped.model.camera(i).name)

    renderer = CameraRenderer(
        env=env,
        camera_name=CAMERA_NAME,
        width=480,
        height=480,
    )

    obs, info = env.reset()
    set_object_and_goal(env, OBJECT_POS, GOAL_POS)

    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(renderer.render())
    ax.axis("off")

    try:
        for step in range(MAX_STEPS):
            openpi_obs = make_openpi_obs(env, renderer)

            out = policy.infer(openpi_obs)

            actions = out["actions"]

            # OpenPIは複数ステップ分のactionを返すことがある
            # まずは一番先頭だけ使う
            first_action = actions[0]

            env_action = to_fetch_action(first_action)

            obs, reward, terminated, truncated, info = env.step(env_action)

            img = renderer.render()
            im.set_data(img)
            ax.set_title(
                f"step={step} reward={reward:.3f}\n"
                f"action={env_action}"
            )
            fig.canvas.draw()
            fig.canvas.flush_events()

            print(
                f"step={step:03d}",
                "action=", env_action,
                "reward=", reward,
            )

            time.sleep(0.02)

            if terminated or truncated:
                print("episode finished")
                break

    finally:
        renderer.close()
        env.close()
        plt.close(fig)




if __name__ == "__main__":
    main()