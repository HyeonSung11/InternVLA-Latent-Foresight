from __future__ import annotations

from typing import Any

import numpy as np
import torch

from lerobot.transforms.core import ResizeImagesWithPadFn
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE

from .input_semantics import prepare_example_images, validate_example_state


def to_chw_float01(image: np.ndarray) -> torch.Tensor:
    if image.ndim != 3:
        raise ValueError(f"Expected HWC image, got shape={image.shape}")
    # Some env buffers are read-only; copy to avoid PyTorch non-writable-array warnings.
    tensor = torch.from_numpy(np.array(image, copy=True))
    if tensor.dtype == torch.uint8:
        tensor = tensor.float() / 255.0
    else:
        tensor = tensor.float()
        if torch.max(tensor) > 1.0:
            tensor = tensor / 255.0
    return tensor.permute(2, 0, 1).contiguous()


def build_base_sample(
    example: dict[str, Any],
    *,
    robot_type: str | None,
    expected_state_dim: int,
    resize_transform: ResizeImagesWithPadFn,
    mask_as_tensor: bool = False,
) -> dict[str, Any]:
    task = str(example.get("lang") or example.get("task") or "")
    state = validate_example_state(example, expected_dim=expected_state_dim)
    images, image_masks, _ = prepare_example_images(example, robot_type=robot_type)

    sample: dict[str, Any] = {
        OBS_STATE: torch.from_numpy(state),
        "task": task,
    }

    for idx in range(3):
        sample[f"{OBS_IMAGES}.image{idx}"] = to_chw_float01(np.asarray(images[idx]))

    sample = resize_transform(sample)
    for idx in range(3):
        key = f"{OBS_IMAGES}.image{idx}_mask"
        if mask_as_tensor:
            sample[key] = torch.tensor(bool(image_masks[idx]), dtype=torch.bool)
        else:
            sample[key] = bool(image_masks[idx])
    return sample
