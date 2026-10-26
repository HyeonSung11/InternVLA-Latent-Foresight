from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.internvla_a1_5 import InternVLAA15Config
from lerobot.policies.pi05 import PI05Config

from .base_backend import PROTOCOL_VERSION
from .policy_backend_internvla_a1_5 import InternVLAA15Backend
from .policy_backend_pi05 import PI05Backend


@dataclass
class RandomMockBackend:
    action_dim: int = 7
    chunk_size: int = 16

    def infer(self, payload: dict[str, Any]) -> dict[str, Any]:
        examples = payload.get("examples")
        batch = len(examples) if isinstance(examples, list) and examples else 1
        actions = np.random.uniform(
            low=-1.0,
            high=1.0,
            size=(batch, self.chunk_size, self.action_dim),
        ).astype(np.float32)
        low = np.full((self.action_dim,), -1.0, dtype=np.float32)
        high = np.full((self.action_dim,), 1.0, dtype=np.float32)
        return {
            "actions": actions,
            "action_space": {"low": low, "high": high, "dim": self.action_dim},
            "action_dim": self.action_dim,
            "chunk_size": self.chunk_size,
            "postprocess": {
                "clip": True,
                "gripper_binarize": False,
            },
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "policy_type": "mock_random",
            "ckpt_path": "",
            "stats_key": "",
            "robot_type": "",
            "chunk_size": self.chunk_size,
            "action_dim": self.action_dim,
            "protocol_version": PROTOCOL_VERSION,
            "returns": "actions",
            "expected_num_input_images": 0,
            "image_mask_policy": "training_semantics_strict",
            "strict_input_validation": False,
            "preprocessing_owner": "server_canonical",
            "deterministic_inference_preprocess": True,
            "required_image_keys": [],
            "expected_state_dim": 0,
        }


def build_backend(
    ckpt_path: str,
    device: str = "cuda",
    stats_key: str | None = None,
    robot_type: str | None = None,
    resize_size: int = 224,
    mock_policy: str | None = None,
    mock_action_dim: int = 7,
    mock_chunk_size: int = 16,
    vlm_model_path: str | None = None,
    no_state_prompt: bool = False,
    wan_model_path: str | None = None,
    wan_vae_path: str | None = None,
    action_loss_only: bool = True,
    inference_backend: str = "standard",
):
    if mock_policy:
        if mock_policy != "random":
            raise ValueError(f"Unsupported mock_policy='{mock_policy}'. Expected: random")
        return RandomMockBackend(action_dim=mock_action_dim, chunk_size=mock_chunk_size)

    config = PreTrainedConfig.from_pretrained(Path(ckpt_path))

    if isinstance(config, PI05Config):
        return PI05Backend(
            ckpt_path=ckpt_path,
            device=device,
            stats_key=stats_key,
            robot_type=robot_type,
            resize_size=resize_size,
        )

    if isinstance(config, InternVLAA15Config):
        return InternVLAA15Backend(
            ckpt_path=ckpt_path,
            device=device,
            stats_key=stats_key,
            robot_type=robot_type,
            resize_size=resize_size,
            vlm_model_path=vlm_model_path,
            no_state_prompt=no_state_prompt,
            wan_model_path=wan_model_path,
            wan_vae_path=wan_vae_path,
            action_loss_only=action_loss_only,
            inference_backend=inference_backend,
        )

    raise TypeError(
        f"Unsupported policy config type: {type(config)}. "
        "Only PI05 and InternVLA-A1.5 are supported by evaluation/LIBERO policy server."
    )
