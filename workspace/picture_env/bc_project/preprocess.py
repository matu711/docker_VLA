from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def preprocess_image_np(img: np.ndarray, image_size: int) -> torch.Tensor:
    img = np.ascontiguousarray(img.copy())
    x = torch.from_numpy(img).float() / 255.0
    x = x.permute(2, 0, 1)
    x = F.interpolate(
        x.unsqueeze(0),
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    return x
