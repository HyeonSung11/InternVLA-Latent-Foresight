from __future__ import annotations

from typing import Any

import numpy as np

from lerobot.dataset_schemas import get_schema
from lerobot.utils.constants import OBS_IMAGES


def _error(code: str, message: str) -> ValueError:
    return ValueError(f"{code}: {message}")


_DEFAULT_IMAGE_MAPPING: dict[str, str] = {
    "observation.image": f"{OBS_IMAGES}.image0",
}


def _get_image_mapping(robot_type: str | None) -> dict[str, str]:
    if not robot_type:
        return dict(_DEFAULT_IMAGE_MAPPING)
    try:
        schema = get_schema(robot_type)
        if schema.image_mapping:
            return dict(schema.image_mapping)
        return dict(_DEFAULT_IMAGE_MAPPING)
    except ValueError:
        return dict(_DEFAULT_IMAGE_MAPPING)


def _parse_image_slot(mapped_key: str) -> int:
    prefix = f"{OBS_IMAGES}.image"
    if not mapped_key.startswith(prefix):
        raise _error(
            "INVALID_IMAGE_MAPPING",
            f"Unsupported mapped image key '{mapped_key}', expected prefix '{prefix}'.",
        )
    suffix = mapped_key[len(prefix) :]
    if not suffix.isdigit():
        raise _error(
            "INVALID_IMAGE_MAPPING",
            f"Mapped image key '{mapped_key}' must end with numeric slot index.",
        )
    slot = int(suffix)
    if slot < 0 or slot > 2:
        raise _error(
            "INVALID_IMAGE_MAPPING",
            f"Mapped image key '{mapped_key}' points to slot={slot}, but supported range is [0, 2].",
        )
    return slot


def image_slots_for_robot(robot_type: str | None) -> list[int]:
    mapping = _get_image_mapping(robot_type)
    slots = sorted({_parse_image_slot(v) for v in mapping.values()})
    if not slots:
        raise _error("INVALID_IMAGE_MAPPING", f"Image mapping for robot_type='{robot_type}' is empty.")
    if len(slots) > 3:
        raise _error(
            "INVALID_IMAGE_MAPPING",
            f"Image mapping for robot_type='{robot_type}' has {len(slots)} slots, max supported is 3.",
        )
    return slots


def required_image_keys_for_robot(robot_type: str | None) -> list[str]:
    mapping = _get_image_mapping(robot_type)
    by_slot: dict[int, str] = {}
    for raw_key, mapped_key in mapping.items():
        slot = _parse_image_slot(mapped_key)
        if slot in by_slot:
            raise _error(
                "INVALID_IMAGE_MAPPING",
                (
                    f"Image mapping for robot_type='{robot_type}' maps multiple raw keys "
                    f"to slot={slot}: '{by_slot[slot]}' and '{raw_key}'."
                ),
            )
        by_slot[slot] = raw_key

    if not by_slot:
        raise _error("INVALID_IMAGE_MAPPING", f"Image mapping for robot_type='{robot_type}' is empty.")
    return [by_slot[s] for s in sorted(by_slot.keys())]


def expected_num_input_images(robot_type: str | None) -> int:
    return len(image_slots_for_robot(robot_type))


def _to_hwc_uint8(image: Any, image_idx: int) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 3:
        raise _error(
            "INVALID_IMAGE_SHAPE",
            f"Image[{image_idx}] must be 3D HWC/CHW, got shape={arr.shape}.",
        )

    # CHW -> HWC
    if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))

    if arr.shape[-1] not in (1, 3, 4):
        raise _error(
            "INVALID_IMAGE_SHAPE",
            f"Image[{image_idx}] channel dim must be 1/3/4, got shape={arr.shape}.",
        )

    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            max_val = float(np.max(arr)) if arr.size > 0 else 0.0
            if max_val <= 1.0:
                arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
            else:
                arr = arr.clip(0, 255).astype(np.uint8)
        else:
            arr = arr.clip(0, 255).astype(np.uint8)

    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    elif arr.shape[-1] == 4:
        arr = arr[..., :3]

    return np.ascontiguousarray(arr)


def prepare_example_images(
    example: dict[str, Any],
    robot_type: str | None,
) -> tuple[list[np.ndarray], list[bool], int]:
    if "image" in example:
        raw_images = example["image"]
    elif "images" in example:
        raw_images = example["images"]
    else:
        raise _error(
            "MISSING_IMAGE_FIELD",
            "Example must provide 'image' (or legacy alias 'images').",
        )

    if not isinstance(raw_images, list):
        raise _error(
            "INVALID_IMAGE_FIELD",
            f"Example image field must be a list, got type={type(raw_images)}.",
        )

    slots = image_slots_for_robot(robot_type)
    expected = len(slots)
    if len(raw_images) != expected:
        raise _error(
            "INVALID_IMAGE_COUNT",
            (
                f"robot_type='{robot_type}' expects {expected} image(s) for slots={slots}, "
                f"but got {len(raw_images)}."
            ),
        )

    converted = [_to_hwc_uint8(img, idx) for idx, img in enumerate(raw_images)]
    if not converted:
        raise _error("INVALID_IMAGE_COUNT", "Image list is empty after parsing.")

    ref = converted[0]
    canvas_h, canvas_w = int(ref.shape[0]), int(ref.shape[1])
    # Match training RemapImageKeyTransformFn semantics:
    # missing camera slots are filled with ones-like(image0).
    white = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

    packed: list[np.ndarray | None] = [None, None, None]
    masks = [False, False, False]
    for img, slot in zip(converted, slots):
        packed[slot] = img
        masks[slot] = True

    final_images = [img if img is not None else white.copy() for img in packed]
    return final_images, masks, expected


def validate_example_state(example: dict[str, Any], expected_dim: int) -> np.ndarray:
    if "state" not in example:
        raise _error("MISSING_STATE", "Example must provide 'state'.")

    state = np.asarray(example["state"], dtype=np.float32).reshape(-1)
    if state.shape[0] != expected_dim:
        raise _error(
            "INVALID_STATE_DIM",
            f"Expected state dim={expected_dim}, got dim={state.shape[0]}.",
        )
    return state.astype(np.float32)
