from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .preprocess import preprocess_image_np


class BCDataset(Dataset):
    """BC用Dataset．

    通常BC用の action に加えて，ACT用の action_chunk も返す．
    demo_id を使い，同じdemoの範囲内で未来actionを取り出す．足りない部分は最後のactionで埋める．
    """

    def __init__(self, cfg):
        data = np.load(cfg.paths.dataset_path)

        self.image_size = int(cfg.data.image_size)
        self.images = data[cfg.data.image_key]
        self.robot_states = data[cfg.data.robot_state_key].astype(np.float32)
        self.actions = data[cfg.data.action_key].astype(np.float32)
        self.demo_id = data["demo_id"].astype(np.int32) if "demo_id" in data else np.zeros(len(self.actions), dtype=np.int32)
        self.action_horizon = int(cfg.data.action_horizon)

        self.robot_mean = self.robot_states.mean(axis=0).astype(np.float32)
        self.robot_std = (self.robot_states.std(axis=0) + 1e-6).astype(np.float32)

        print("images:", self.images.shape, self.images.dtype)
        print("robot_states:", self.robot_states.shape, self.robot_states.dtype)
        print("actions:", self.actions.shape, self.actions.dtype)
        print("action_horizon:", self.action_horizon)
        print("robot_mean:", self.robot_mean)
        print("robot_std:", self.robot_std)

    def __len__(self):
        return len(self.actions)

    def _get_action_chunk(self, idx: int) -> np.ndarray:
        end = idx + self.action_horizon
        chunk = []
        current_demo = self.demo_id[idx]
        last_action = self.actions[idx]

        for j in range(idx, end):
            if j < len(self.actions) and self.demo_id[j] == current_demo:
                last_action = self.actions[j]
            chunk.append(last_action)

        return np.stack(chunk, axis=0).astype(np.float32)

    def __getitem__(self, idx):
        image = preprocess_image_np(self.images[idx], self.image_size)

        robot_state = self.robot_states[idx]
        robot_state = (robot_state - self.robot_mean) / self.robot_std

        action = self.actions[idx]
        action_chunk = self._get_action_chunk(idx)

        return {
            "image": image,
            "robot_state": torch.from_numpy(robot_state).float(),
            "action": torch.from_numpy(action).float(),
            "action_chunk": torch.from_numpy(action_chunk).float(),
        }
