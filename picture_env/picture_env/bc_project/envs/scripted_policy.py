from __future__ import annotations

import numpy as np

from .fetch_pick_place import get_gripper_pos, get_object_pos


class ScriptedPickPlacePolicy:
    """Fetch Pick&Place 用の教示方策．

    速度，掴み保持，設置保持を config から調整できるようにしている．
    """

    def __init__(self, cfg):
        self.above_z = float(cfg.scripted_policy.above_z)
        self.grasp_z_offset = float(cfg.scripted_policy.grasp_z_offset)
        self.lift_z = float(cfg.scripted_policy.lift_z)
        self.place_z_offset = float(cfg.scripted_policy.place_z_offset)

        self.kp = float(cfg.scripted_policy.kp)
        self.max_action_xyz = float(cfg.scripted_policy.max_action_xyz)
        self.pos_eps = float(cfg.scripted_policy.pos_eps)

        self.grasp_hold_steps = int(cfg.scripted_policy.grasp_hold_steps)
        self.place_hold_steps = int(cfg.scripted_policy.place_hold_steps)
        self._hold_count = 0

    def reached(self, current_pos, target_pos) -> bool:
        return np.linalg.norm(current_pos - target_pos) < self.pos_eps

    def make_move_action(self, current_pos, target_pos, gripper_cmd):
        delta = target_pos - current_pos
        action_xyz = np.clip(
            self.kp * delta,
            -self.max_action_xyz,
            self.max_action_xyz,
        )
        action = np.zeros(4, dtype=np.float32)
        action[:3] = action_xyz
        action[3] = gripper_cmd
        return action

    def _hold_then_next(self, next_phase: int, stay_phase: int, hold_steps: int) -> int:
        self._hold_count += 1
        if self._hold_count >= hold_steps:
            self._hold_count = 0
            return next_phase
        return stay_phase

    def __call__(self, env, phase: int):
        grip_pos = get_gripper_pos(env)
        obj_pos = get_object_pos(env)
        goal_pos = env.unwrapped.goal.copy()

        above_obj = obj_pos.copy()
        above_obj[2] = self.above_z

        grasp_pos = obj_pos.copy()
        grasp_pos[2] = obj_pos[2] + self.grasp_z_offset

        lift_pos = obj_pos.copy()
        lift_pos[2] = self.lift_z

        above_goal = goal_pos.copy()
        above_goal[2] = self.lift_z

        place_pos = goal_pos.copy()
        place_pos[2] = goal_pos[2] + self.place_z_offset

        if phase == 0:
            target, gripper = above_obj, 1.0
            next_phase = 1 if self.reached(grip_pos, target) else 0

        elif phase == 1:
            target, gripper = grasp_pos, 1.0
            next_phase = 2 if self.reached(grip_pos, target) else 1

        elif phase == 2:
            # ここで数step閉じ続ける．掴みが甘いまま持ち上げるのを避ける．
            target, gripper = grasp_pos, -1.0
            next_phase = self._hold_then_next(
                next_phase=3,
                stay_phase=2,
                hold_steps=self.grasp_hold_steps,
            )

        elif phase == 3:
            target, gripper = lift_pos, -1.0
            next_phase = 4 if self.reached(grip_pos, target) else 3

        elif phase == 4:
            target, gripper = above_goal, -1.0
            next_phase = 5 if self.reached(grip_pos, target) else 4

        elif phase == 5:
            # 置く位置に到達してからも少し保持して，その後に開く．
            target, gripper = place_pos, -1.0
            if self.reached(grip_pos, target):
                next_phase = self._hold_then_next(
                    next_phase=6,
                    stay_phase=5,
                    hold_steps=self.place_hold_steps,
                )
            else:
                next_phase = 5

        else:
            target, gripper = place_pos, 1.0
            next_phase = 6

        return self.make_move_action(grip_pos, target, gripper), next_phase
