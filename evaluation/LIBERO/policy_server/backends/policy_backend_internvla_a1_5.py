from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.dataset_schemas import get_schema
from lerobot.policies.internvla_a1_5 import InternVLAA15Config, InternVLAA15Policy
from lerobot.policies.internvla_a1_5.transform_internvla_a1_5 import (
    InternVLAA15ChatProcessorTransformFn,
)
from lerobot.transforms.core import NormalizeTransformFn, ResizeImagesWithPadFn
from lerobot.utils.constants import OBS_STATE

from .base_backend import BasePolicyBackend, PROTOCOL_VERSION
from .canonical_preprocess import build_base_sample
from .input_semantics import expected_num_input_images, required_image_keys_for_robot


class InternVLAA15Backend(BasePolicyBackend):
    """Inference backend for InternVLA-A1.5 (aka the former qwen3_5vla_ki_wan).

    The action-prediction forward path (`predict_action_chunk` ->
    `model.sample_actions`) does not touch the WAN DiT/VAE, so eval-time
    inference works with `action_loss_only=True` and either the standard or
    optimized inference backend. Key steps:

      * Load `InternVLAA15Policy` (deserializes learnable_tokens / WAN video
        weights cleanly even when action_loss_only=True).
      * Use `InternVLAA15ChatProcessorTransformFn(mode="eval")` to mirror
        training-time prompt construction (adds "Control Mode: <action_mode>;"
        and the eval-mode "Output: <Subtask, Action>" suffix).
      * Resolve `action_mode` from the dataset schema for the eval robot type
        so the "Control Mode: <...>" prompt tag matches training.
      * Default `max_length=650` to match the A1.5 dataset config's
        `max_prompt_length`.
    """

    def __init__(
        self,
        ckpt_path: str,
        device: str = "cuda",
        stats_key: str | None = None,
        robot_type: str | None = None,
        resize_size: int = 224,
        vlm_model_path: str | None = None,
        no_state_prompt: bool = False,
        max_prompt_length: int = 650,
        wan_model_path: str | None = None,
        wan_vae_path: str | None = None,
        action_loss_only: bool = True,
        inference_backend: str = "standard",
    ) -> None:
        super().__init__(ckpt_path=ckpt_path, device=device, stats_key=stats_key, robot_type=robot_type)

        config = PreTrainedConfig.from_pretrained(self.ckpt_path)
        if not isinstance(config, InternVLAA15Config):
            raise TypeError(f"Expected InternVLAA15Config, got {type(config)}")

        # Apply CLI overrides so the policy __init__ sees valid paths.
        if vlm_model_path:
            config.vlm_model_name_or_path = vlm_model_path
        if not config.vlm_model_name_or_path:
            raise ValueError(
                "InternVLA-A1.5 checkpoint has an empty vlm_model_name_or_path. "
                "Pass --vlm_model_path <hf-id-or-local-dir> to override."
            )

        # Repoint the WAN dir + VAE if the paths baked into the checkpoint
        # config don't exist locally.
        if wan_model_path:
            config.wan_checkpoint_path = wan_model_path
            config.wan_config_path = wan_model_path
            if not wan_vae_path:
                default_vae = Path(wan_model_path) / "Wan2.2_VAE.pth"
                if default_vae.exists():
                    config.vae_path = str(default_vae)
        if wan_vae_path:
            config.vae_path = wan_vae_path

        # action_loss_only=True skips WAN weight loading during inference; the
        # public release ships checkpoints tuned to this setting.
        config.action_loss_only = bool(action_loss_only)
        config.inference_backend = inference_backend
        if config.inference_backend == "optimized" and not config.action_loss_only:
            raise ValueError("inference_backend='optimized' requires action_loss_only=True")

        if not config.action_loss_only:
            wan_config_json = Path(config.wan_config_path) / "config.json"
            if not wan_config_json.exists():
                raise FileNotFoundError(
                    f"WAN config.json not found at {wan_config_json}. "
                    "Pass --wan_model_path <local-wan-dir> to override the path "
                    "baked into the checkpoint, or run with action_loss_only=True."
                )
            if not Path(config.vae_path).exists():
                raise FileNotFoundError(
                    f"WAN VAE weights not found at {config.vae_path}. "
                    "Pass --wan_vae_path <vae.pth> to override."
                )

        self.policy = InternVLAA15Policy.from_pretrained(
            config=config, pretrained_name_or_path=self.ckpt_path
        )
        self.policy.to(self.device)
        self.policy.eval()

        if config.dtype == "bfloat16":
            self.compute_dtype = torch.bfloat16
        elif config.dtype == "float32":
            self.compute_dtype = torch.float32
        else:
            raise ValueError(f"Unsupported config.dtype={config.dtype!r}")

        self.state_dim = config.input_features[OBS_STATE].shape[0]
        self.chunk_size = int(config.chunk_size)
        self.resize_size = int(resize_size)

        self.stats_key, self.robot_type, self.state_stat, self.action_stat = self._load_stats()
        self.state_input_dim = int(
            np.asarray(self.state_stat[OBS_STATE]["mean"], dtype=np.float32).reshape(-1).shape[0]
        )
        self.expected_num_input_images = expected_num_input_images(self.robot_type)
        self.state_normalizer = NormalizeTransformFn(selected_keys=[OBS_STATE], norm_stats=self.state_stat)
        self.resize = ResizeImagesWithPadFn(height=self.resize_size, width=self.resize_size)

        # Resolve action_mode from the schema for the eval robot. Training-time
        # hydration reads it from the dataset; without a dataset at eval time we
        # look it up here so the "Control Mode: <...>" prompt tag matches.
        action_mode = "joint"
        if self.robot_type:
            try:
                schema = get_schema(self.robot_type)
                action_mode = getattr(schema, "action_mode", "joint")
            except ValueError:
                pass

        processor_path = vlm_model_path or config.vlm_model_name_or_path
        processor_tokenize_state = False if no_state_prompt else config.tokenize_state
        # Memory ω: 학습 때 켰으면 checkpoint config에 플래그가 실려 있음 → 동일하게 eval에도 적용.
        self.use_action_memory = bool(getattr(config, "use_action_memory", False))
        self.action_memory_len = int(getattr(config, "action_memory_len", 8))
        self.processor = InternVLAA15ChatProcessorTransformFn(
            pretrained_model_name_or_path=processor_path,
            max_length=int(max_prompt_length),
            tokenize_state=processor_tokenize_state,
            max_state_dim=config.max_state_dim,
            use_fast_action_tokens=False,
            mode="eval",
            action_mode=action_mode,
            use_action_memory=self.use_action_memory,
            action_memory_len=self.action_memory_len,
            max_action_dim=config.max_action_dim,
        )
        self.no_state_prompt = bool(no_state_prompt)
        self.action_mode = action_mode

    def _prepare_single(self, example: dict[str, Any]) -> dict[str, Any]:
        sample = build_base_sample(
            example,
            robot_type=self.robot_type,
            expected_state_dim=self.state_input_dim,
            resize_transform=self.resize,
            mask_as_tensor=False,
        )
        sample = self.state_normalizer(sample)
        # Memory ω: 클라이언트가 보낸 과거 실행 액션을 sample에 주입 → processor가 문자열로 인코딩.
        if self.use_action_memory and example.get("action_memory") is not None:
            sample[self.processor.action_memory_key] = torch.as_tensor(
                example["action_memory"], dtype=torch.float32
            )
        sample = self.processor(sample)
        return sample

    def _sample_to_inputs(self, sample: dict[str, Any]) -> dict[str, Any]:
        inputs: dict[str, Any] = {}
        for key, value in sample.items():
            if key == "task":
                inputs[key] = [value]
                continue

            if isinstance(value, bool):
                continue

            if not isinstance(value, torch.Tensor):
                continue

            inputs[key] = value.unsqueeze(0).to(self.device)
        return inputs

    def infer(self, payload: dict[str, Any]) -> dict[str, Any]:
        examples = payload.get("examples")
        if not isinstance(examples, list) or len(examples) == 0:
            raise ValueError("payload.examples must be a non-empty list")

        autocast_device = self.device.type if self.device.type != "cpu" else "cpu"
        autocast_enabled = self.compute_dtype != torch.float32

        outputs = []
        with torch.no_grad(), torch.amp.autocast(
            device_type=autocast_device,
            dtype=self.compute_dtype,
            enabled=autocast_enabled,
        ):
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
            "policy_type": "internvla_a1_5",
            "ckpt_path": str(self.ckpt_path),
            "stats_key": self.stats_key,
            "robot_type": self.robot_type,
            "action_mode": self.action_mode,
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
            "use_action_memory": self.use_action_memory,
            "action_memory_len": self.action_memory_len,
        }
