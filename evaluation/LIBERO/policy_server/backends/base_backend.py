from __future__ import annotations

import abc
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.datasets.utils import load_json
from lerobot.dataset_schemas import get_schema
from lerobot.utils.constants import ACTION, OBS_STATE

STAT_KEYS = ("min", "max", "mean", "std")
PROTOCOL_VERSION = "2.1"


class BasePolicyBackend(abc.ABC):
    def __init__(
        self,
        ckpt_path: str | Path,
        device: str = "cuda",
        stats_key: str | None = None,
        robot_type: str | None = None,
    ) -> None:
        self.ckpt_path = Path(ckpt_path)
        self.device = torch.device(device)
        self.stats_key = stats_key
        self.robot_type = robot_type
        self.action_denorm_mode = self._resolve_action_denorm_mode()

    @staticmethod
    def _try_get_schema(robot_type):
        if not robot_type:
            return None
        try:
            return get_schema(robot_type)
        except ValueError:
            return None

    def _resolve_action_denorm_mode(self) -> str:
        train_cfg_path = self.ckpt_path / "train_config.json"
        if train_cfg_path.exists():
            try:
                train_cfg = load_json(train_cfg_path)
                transforms = (
                    train_cfg.get("dataset", {})
                    .get("data_transforms", {})
                    .get("inputs", [])
                )
                if isinstance(transforms, list):
                    for transform in transforms:
                        if not isinstance(transform, dict):
                            continue
                        if transform.get("type") != "normalize":
                            continue
                        mode = str(transform.get("mode", "")).strip().lower()
                        if mode in {"mean_std", "min_max"}:
                            logging.info(
                                "Using action denorm mode '%s' from %s",
                                mode,
                                train_cfg_path,
                            )
                            return mode
            except Exception as exc:
                logging.warning("Failed to parse %s for denorm mode: %s", train_cfg_path, exc)

        # Backward-compatible fallback for older checkpoints without train_config.json.
        return "mean_std"

    @staticmethod
    def _concat_feature_stats(stats_for_key: dict, feature_keys: list[str]) -> dict[str, np.ndarray]:
        concat_stats = {}
        for stat in STAT_KEYS:
            parts = [np.asarray(stats_for_key[k][stat], dtype=np.float32) for k in feature_keys]
            concat_stats[stat] = np.concatenate(parts, axis=-1)
        return concat_stats

    @staticmethod
    def _pick_stats_key(all_stats: dict, stats_key: str | None, robot_type: str | None) -> str:
        if stats_key:
            if stats_key not in all_stats:
                raise KeyError(f"stats_key '{stats_key}' not found in stats.json keys={list(all_stats.keys())}")
            return stats_key

        if robot_type and robot_type in all_stats:
            return robot_type

        if len(all_stats) == 1:
            return next(iter(all_stats.keys()))

        raise ValueError(
            "stats.json contains multiple keys; pass --stats_key explicitly. "
            f"Available: {list(all_stats.keys())}"
        )

    def _load_stats(self) -> tuple[str, str | None, dict[str, dict[str, np.ndarray]], dict[str, dict[str, np.ndarray]]]:
        all_stats = load_json(self.ckpt_path / "stats.json")
        selected_key = self._pick_stats_key(all_stats, self.stats_key, self.robot_type)
        selected_stats = all_stats[selected_key]

        schema = self._try_get_schema(self.robot_type) or self._try_get_schema(selected_key)
        robot_type = schema.robot_type if schema else None

        if OBS_STATE in selected_stats:
            state_stat = {OBS_STATE: {k: np.asarray(v, dtype=np.float32) for k, v in selected_stats[OBS_STATE].items()}}
        elif schema:
            state_keys = schema.get_state_keys()
            state_stat = {OBS_STATE: self._concat_feature_stats(selected_stats, state_keys)}
        else:
            raise KeyError(
                f"Cannot build '{OBS_STATE}' stats for key '{selected_key}'. "
                f"Available feature keys: {list(selected_stats.keys())[:20]}"
            )

        if ACTION in selected_stats:
            action_stat = {ACTION: {k: np.asarray(v, dtype=np.float32) for k, v in selected_stats[ACTION].items()}}
        elif schema:
            action_keys = schema.get_action_keys()
            action_stat = {ACTION: self._concat_feature_stats(selected_stats, action_keys)}
        else:
            raise KeyError(
                f"Cannot build '{ACTION}' stats for key '{selected_key}'. "
                f"Available feature keys: {list(selected_stats.keys())[:20]}"
            )

        return selected_key, robot_type, state_stat, action_stat

    def _get_action_stats(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not hasattr(self, "action_stat"):
            raise RuntimeError("Backend action_stat is not initialized")
        action_stats = self.action_stat[ACTION]
        low = np.asarray(action_stats["min"], dtype=np.float32).reshape(-1)
        high = np.asarray(action_stats["max"], dtype=np.float32).reshape(-1)
        mask = np.asarray(action_stats.get("mask", np.ones_like(low, dtype=bool)), dtype=bool).reshape(-1)
        mean = np.asarray(action_stats.get("mean", np.zeros_like(low)), dtype=np.float32).reshape(-1)
        std = np.asarray(action_stats.get("std", np.ones_like(low)), dtype=np.float32).reshape(-1)
        if mean.shape != low.shape:
            mean = np.zeros_like(low)
        if std.shape != low.shape:
            std = np.ones_like(low)
        return low, high, mask, mean, std

    def denormalize_actions(self, normalized_actions: np.ndarray) -> np.ndarray:
        arr = np.asarray(normalized_actions, dtype=np.float32)
        if arr.ndim != 3:
            raise ValueError(f"Expected normalized_actions with ndim=3, got shape={arr.shape}")

        low, high, mask, mean, std = self._get_action_stats()
        action_dim = int(low.shape[-1])
        if arr.shape[-1] < action_dim:
            raise ValueError(
                f"Model action dim {arr.shape[-1]} is smaller than stats action dim {action_dim}"
            )

        normalized = arr[..., :action_dim]
        if self.action_denorm_mode == "mean_std":
            safe_std = np.clip(std, 1e-6, None)
            denorm = np.where(mask, normalized * safe_std + mean, normalized)
        elif self.action_denorm_mode == "min_max":
            clipped = np.clip(normalized, 0.0, 1.0)
            denorm = np.where(mask, clipped * (high - low) + low, clipped)
        else:
            raise ValueError(f"Unknown action_denorm_mode='{self.action_denorm_mode}'")
        return denorm.astype(np.float32)

    def postprocess_actions(self, actions: np.ndarray) -> np.ndarray:
        arr = np.asarray(actions, dtype=np.float32)
        low, high, _, _, _ = self._get_action_stats()
        arr = np.clip(arr, low, high)
        return arr.astype(np.float32)

    def action_space(self) -> dict[str, np.ndarray | int]:
        low, high, _, _, _ = self._get_action_stats()
        return {
            "low": low.astype(np.float32),
            "high": high.astype(np.float32),
            "dim": int(low.shape[-1]),
        }

    @abc.abstractmethod
    def infer(self, payload: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    @abc.abstractmethod
    def metadata(self) -> dict[str, Any]:
        raise NotImplementedError
