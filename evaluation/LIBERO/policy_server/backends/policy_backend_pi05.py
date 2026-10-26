from __future__ import annotations

from typing import Any

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi05 import PI05Config, PI05Policy
from lerobot.policies.pi05.transform_pi05 import PI05GemmaTokenizerTransformFn
from lerobot.transforms.core import NormalizeTransformFn, ResizeImagesWithPadFn
from lerobot.utils.constants import OBS_STATE

from .base_backend import BasePolicyBackend, PROTOCOL_VERSION
from .canonical_preprocess import build_base_sample
from .input_semantics import expected_num_input_images, required_image_keys_for_robot


class PI05Backend(BasePolicyBackend):
    def __init__(
        self,
        ckpt_path: str,
        device: str = "cuda",
        stats_key: str | None = None,
        robot_type: str | None = None,
        resize_size: int = 224,
    ) -> None:
        super().__init__(ckpt_path=ckpt_path, device=device, stats_key=stats_key, robot_type=robot_type)

        config = PreTrainedConfig.from_pretrained(self.ckpt_path)
        if not isinstance(config, PI05Config):
            raise TypeError(f"Expected PI05Config, got {type(config)}")

        self.policy = PI05Policy.from_pretrained(config=config, pretrained_name_or_path=self.ckpt_path)
        self.policy.to(self.device)
        self.policy.eval()

        self.state_dim = config.input_features[OBS_STATE].shape[0]
        self.chunk_size = int(config.chunk_size)
        self.resize_size = int(resize_size)

        self.stats_key, self.robot_type, self.state_stat, self.action_stat = self._load_stats()
        self.state_input_dim = int(np.asarray(self.state_stat[OBS_STATE]["mean"], dtype=np.float32).reshape(-1).shape[0])
        self.expected_num_input_images = expected_num_input_images(self.robot_type)
        self.state_normalizer = NormalizeTransformFn(selected_keys=[OBS_STATE], norm_stats=self.state_stat)
        self.resize = ResizeImagesWithPadFn(height=self.resize_size, width=self.resize_size)
        self.tokenizer = PI05GemmaTokenizerTransformFn(max_state_dim=self.state_dim)

    def _prepare_single(self, example: dict[str, Any]) -> dict[str, torch.Tensor | str]:
        sample = build_base_sample(
            example,
            robot_type=self.robot_type,
            expected_state_dim=self.state_input_dim,
            resize_transform=self.resize,
            mask_as_tensor=True,
        )
        sample = self.state_normalizer(sample)
        sample = self.tokenizer(sample)
        return sample

    def _sample_to_inputs(self, sample: dict[str, Any]) -> dict[str, Any]:
        inputs: dict[str, Any] = {}
        for key, value in sample.items():
            if key == "task":
                inputs[key] = [value]
                continue

            if not isinstance(value, torch.Tensor):
                raise TypeError(f"Unexpected value type for key='{key}': {type(value)}")
            # Keep tensor dtype unchanged to respect checkpoint mixed precision.
            inputs[key] = value.unsqueeze(0).to(self.device)
        return inputs

    def infer(self, payload: dict[str, Any]) -> dict[str, Any]:
        examples = payload.get("examples")
        if not isinstance(examples, list) or len(examples) == 0:
            raise ValueError("payload.examples must be a non-empty list")

        outputs = []
        with torch.no_grad():
            for example in examples:
                sample = self._prepare_single(example)
                inputs = self._sample_to_inputs(sample)
                chunk = self.policy.predict_action_chunk(inputs)
                outputs.append(chunk.detach().float().cpu().numpy())

        normalized_actions = np.concatenate(outputs, axis=0)
        actions = self.denormalize_actions(normalized_actions)
        actions = self.postprocess_actions(actions)
        return {
            "actions": actions,
            "action_space": self.action_space(),
            "action_dim": int(actions.shape[-1]),
            "chunk_size": self.chunk_size,
            "postprocess": {
                "clip": True,
                "gripper_binarize": False,
            },
        }

    def metadata(self) -> dict[str, Any]:
        action_space = self.action_space()
        return {
            "policy_type": "pi05",
            "ckpt_path": str(self.ckpt_path),
            "stats_key": self.stats_key,
            "robot_type": self.robot_type,
            "chunk_size": self.chunk_size,
            "action_dim": int(action_space["dim"]),
            "action_denorm_mode": self.action_denorm_mode,
            "protocol_version": PROTOCOL_VERSION,
            "returns": "actions",
            "expected_num_input_images": self.expected_num_input_images,
            "image_mask_policy": "training_semantics_strict",
            "strict_input_validation": True,
            "preprocessing_owner": "server_canonical",
            "deterministic_inference_preprocess": True,
            "required_image_keys": required_image_keys_for_robot(self.robot_type),
            "expected_state_dim": self.state_input_dim,
        }
