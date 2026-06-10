from __future__ import annotations

import numpy as np
import mujoco


class CameraRenderer:
    def __init__(self, env, camera_name: str, width: int, height: int):
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
        mujoco.mj_forward(self.env.unwrapped.model, self.env.unwrapped.data)
        self.renderer.update_scene(self.env.unwrapped.data, camera=self.camera_id)
        return self.renderer.render()

    def close(self):
        self.renderer.close()


def set_object_and_goal(env, object_pos, goal_pos):
    env.unwrapped.data.joint("object0:joint").qpos[:] = [
        object_pos[0], object_pos[1], object_pos[2],
        1.0, 0.0, 0.0, 0.0,
    ]
    env.unwrapped.goal = goal_pos.copy()

    if hasattr(env.unwrapped, "_render_callback"):
        env.unwrapped._render_callback()

    mujoco.mj_forward(env.unwrapped.model, env.unwrapped.data)


def get_gripper_pos(env):
    return env.unwrapped.data.body("robot0:gripper_link").xpos.copy()


def get_object_pos(env):
    return env.unwrapped.data.body("object0").xpos.copy()


def make_object_pos(base_object_pos, x):
    object_pos = np.array(base_object_pos, dtype=np.float32).copy()
    object_pos[0] = x
    return object_pos
