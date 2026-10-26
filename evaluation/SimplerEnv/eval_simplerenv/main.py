"""SimplerEnv (Bridge / Fractal) evaluation client.

Drives the SAPIEN-based SimplerEnv (`allenzren/SimplerEnv`) against the
websocket policy server under `evaluation/LIBERO/policy_server/`.

Server contract (handled inside the backend, e.g. InternVLA-A1.5 / PI05):
  - Receives raw 8-dim state `[xyz(3), rpy(3), pad(1), gripper(1)]` for bridge
    (or 8-dim `[xyz, quat_xyzw, gripper_closedness]` for fractal). Normalizes,
    pads to `max_state_dim=32`, and discretizes via the chat processor.
  - Receives a raw uint8 HWC image; resizes to `resize_size` and tokenizes via
    the same chat processor used at training time.
  - Returns a `{actions: (B, T, action_dim), action_space, ...}` dict where the
    action chunk is already denormalized and clipped to the training stats.

Client responsibilities (this script):
  1. Build the SAPIEN env from `env_options/{dataset_name}.json`.
  2. Extract `obs["agent"]["eef_pos"]` (8-dim) every `replan_steps` and call
     `preprocess_proprio_bridge` / `preprocess_proprio_fractal`.
  3. For bridge, insert one zero before gripper so the state is the 8-dim
     `[xyz, rpy, pad, gripper]` schema the bridge training data uses.
  4. Send `{image, state, lang}` to the server; convert returned rpy → axangle
     and gripper → ±1 before stepping the env.

The conda env for this script is `simplerenv_state` (SimplerEnv from
`https://github.com/allenzren/SimplerEnv`, which exposes `eef_pos` directly
under `obs["agent"]`).
"""

from __future__ import annotations

import collections
import dataclasses
import hashlib
import json
import logging
import pathlib
import sys
import time
from pathlib import Path
from typing import Literal

import cv2 as cv
import imageio
import numpy as np
import pandas as pd
import torch
import tyro
from PIL import Image
from transforms3d.euler import euler2axangle, mat2euler, quat2mat
from transforms3d.quaternions import mat2quat

# Make the policy_server modules importable regardless of CWD.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.LIBERO.policy_server.tools.websocket_policy_client import WebsocketClientPolicy
from simpler_env.utils.env.env_builder import build_maniskill2_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict


@dataclasses.dataclass
class Args:
    # ---------------- Model server ----------------
    host: str = "0.0.0.0"
    port: int = 10093
    replan_steps: int = 4

    # ---------------- Dataset / sharding ----------------
    dataset_name: Literal["fractal", "bridge"] = "bridge"
    worker_id: int = 0
    num_workers: int = 1

    repeat_times: int = 1
    record_video: bool = True

    # Match the training-time InternVLA-A1.5 / PI05 `height/width=224`. The
    # server's `ResizeImagesWithPadFn` is a no-op at eval time (its `mapping`
    # is empty when the backend instantiates it), so the client must resize
    # before sending — otherwise the Qwen3-VL image processor tiles the raw
    # SimplerEnv frame into ~30x30 patches x 3 views = 900 image tokens and
    # `truncation='max_length'` silently drops some, causing the "Mismatch in
    # `image` token count" error on the server.
    image_size: int = 224

    # ---------------- IO ----------------
    experiment_root: str = "./logs/"
    # Where the SimplerEnv repo is checked out — used to resolve the
    # `rgb_overlay_path` strings stored inside env_options/*.json.
    simpler_env_root: str = ""
    # Optional override: path to env_options/*.json file. When empty, falls
    # back to `<this_dir>/env_options/{dataset_name}.json`.
    env_options_path: str = ""

    seed: int = 42


# ----------------------------------------------------------------------
# Reproducibility helper — matches the reference client's seed derivation so
# results align across pipelines.
# ----------------------------------------------------------------------
def hash_data_to_seed(data, max_bytes: int = 4) -> int:
    def custom_encoder(obj):
        if isinstance(obj, np.ndarray):
            return {
                "__type__": "numpy",
                "dtype": str(obj.dtype),
                "shape": obj.shape,
                "data": obj.tobytes().hex(),
            }
        if isinstance(obj, Image.Image):
            img_hash = hashlib.md5(obj.tobytes()).hexdigest()
            return {
                "__type__": "PIL.Image",
                "mode": obj.mode,
                "size": obj.size,
                "content_hash": img_hash,
            }
        if isinstance(obj, set):
            return sorted(list(obj))
        raise TypeError(f"Type {type(obj)} is not JSON serializable")

    json_str = json.dumps(
        data,
        default=custom_encoder,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    seed_int = int(hashlib.sha256(json_str.encode("utf-8")).hexdigest(), 16)
    if max_bytes > 0:
        seed_int = seed_int % (2 ** (8 * max_bytes))
    return seed_int


# ----------------------------------------------------------------------
# State preprocessing. Both the bridge top-down rotation calibration and the
# fractal wxyz->xyzw quaternion convention live here so the wire-format state
# matches what the model saw at training time.
# ----------------------------------------------------------------------
def preprocess_proprio_fractal(proprio: np.ndarray) -> np.ndarray:
    gripper_quat_wxyz = proprio[3:7]
    gripper_rotm = quat2mat(gripper_quat_wxyz)
    gripper_xyzw = mat2quat(gripper_rotm)[[1, 2, 3, 0]]
    gripper_width = proprio[7]  # simpler convention: 0 close, 1 open
    gripper_closedness = 1 - gripper_width
    return np.concatenate((proprio[:3], gripper_xyzw, [gripper_closedness])).astype(np.float32)


def postprocess_gripper_fractal(action: float) -> float:
    # action in [0, 1]; -1 close, 1 open after this convert
    action = (action * 2) - 1
    relative_gripper_action = -action
    return float(np.clip(relative_gripper_action, -1, 1))


def preprocess_proprio_bridge(proprio: np.ndarray) -> np.ndarray:
    """Convert SimplerEnv `eef_pos` (8-dim) to the bridge 7-dim raw state.

    Layout returned: [x, y, z, roll, pitch, yaw, gripper_openness].
    """
    default_rot = np.array([[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]])
    rm_bridge = quat2mat(proprio[3:7])
    rpy_bridge_converted = mat2euler(rm_bridge @ default_rot.T)
    gripper_openness = proprio[7]
    return np.concatenate(
        [proprio[:3], rpy_bridge_converted, [gripper_openness]],
    ).astype(np.float32)


def postprocess_gripper_bridge(action: float) -> float:
    return float(2.0 * (action > 0.5) - 1.0)


def build_state_payload(task_id: str, raw_proprio: np.ndarray) -> np.ndarray:
    """Pad the raw proprio into the 8-dim training schema expected by the server.

    The server backend handles state normalization and `max_state_dim=32` zero-
    padding internally, so we only need to deliver the 8-dim raw vector that
    matches the training data layout.

      - bridge_delta : 7-dim raw -> 8-dim by inserting `0` before gripper
        (state[6]=pad).
      - fractal_delta: 8-dim raw is passed through as-is.
    """
    if "bridge" in task_id:
        zero_state = np.zeros_like(raw_proprio)[..., -1:]
        return np.concatenate(
            [raw_proprio[..., :-1], zero_state, raw_proprio[..., -1:]],
            axis=-1,
        ).astype(np.float32)
    return raw_proprio.astype(np.float32)


def _resize_image(image: np.ndarray, image_size: int) -> np.ndarray:
    """Match training's `ResizeImagesWithPadFn(height=224, width=224)`.

    cv2.INTER_AREA down-samples without aliasing for raw SimplerEnv frames
    (typically 480x640).
    """
    if image.shape[0] == image_size and image.shape[1] == image_size:
        return np.ascontiguousarray(image)
    resized = cv.resize(image, (image_size, image_size), interpolation=cv.INTER_AREA)
    return np.ascontiguousarray(resized)


# ----------------------------------------------------------------------
# Env-options resolution — `bridge.json` ships with a relative
# `rgb_overlay_path` (`eval_simplerenv/SimplerEnv/...`). Substitute the
# user-supplied SimplerEnv root so the config resolves from any CWD.
# ----------------------------------------------------------------------
def _resolve_overlay_path(raw_path: str | None, simpler_env_root: str | None) -> str | None:
    if not raw_path:
        return raw_path
    if Path(raw_path).is_absolute():
        return raw_path
    if simpler_env_root and raw_path.startswith("eval_simplerenv/SimplerEnv/"):
        rel = raw_path[len("eval_simplerenv/SimplerEnv/"):]
        return str(Path(simpler_env_root) / rel)
    if simpler_env_root and raw_path.startswith("SimplerEnv/"):
        rel = raw_path[len("SimplerEnv/"):]
        return str(Path(simpler_env_root) / rel)
    return raw_path


def _load_env_configs(args: Args) -> list[dict]:
    if args.env_options_path:
        cfg_path = Path(args.env_options_path)
    else:
        cfg_path = Path(__file__).resolve().parent / "env_options" / f"{args.dataset_name}.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"env_options file not found: {cfg_path}")

    with cfg_path.open("r") as f:
        configs = json.load(f)

    simpler_env_root = args.simpler_env_root or None
    for cfg in configs:
        env_kwargs = cfg["env_kwargs"]
        env_kwargs["max_episode_steps"] = int(env_kwargs["max_episode_steps"])
        overlay = env_kwargs.get("rgb_overlay_path")
        env_kwargs["rgb_overlay_path"] = _resolve_overlay_path(overlay, simpler_env_root)
    return configs


# ----------------------------------------------------------------------
# Client wrapper — turn the raw (image, state, lang) into the websocket
# `examples` payload the policy server expects, and surface the returned
# (T, D) chunk back to the caller.
# ----------------------------------------------------------------------
class PolicyServerClient:
    def __init__(self, host: str, port: int) -> None:
        self.client = WebsocketClientPolicy(host=host, port=port)
        meta = self.client.get_server_metadata()
        self.expected_state_dim = int(meta.get("expected_state_dim", 0))
        self.action_dim = int(meta.get("action_dim", 7))
        self.chunk_size = int(meta.get("chunk_size", 0))
        logging.info(
            "Connected to policy server: state_dim=%s action_dim=%s chunk_size=%s",
            self.expected_state_dim,
            self.action_dim,
            self.chunk_size,
        )

    def predict_chunk(
        self,
        image: np.ndarray,
        state: np.ndarray,
        instruction: str,
        seed: int | None = None,
    ) -> np.ndarray:
        if self.expected_state_dim and state.shape[-1] != self.expected_state_dim:
            raise ValueError(
                f"State dim mismatch: server expects {self.expected_state_dim}, got {state.shape[-1]}"
            )

        example = {
            "image": [np.ascontiguousarray(image)],
            "state": state.astype(np.float32),
            "lang": instruction,
        }
        payload: dict = {"examples": [example]}
        if seed is not None:
            payload["seed"] = int(seed)

        response = self.client.predict_action(payload)
        if not response.get("ok", True):
            err = response.get("error", {})
            raise RuntimeError(f"Policy server returned error: {err}")
        actions = np.asarray(response["data"]["actions"][0], dtype=np.float32)  # (T, D)
        return actions


# ----------------------------------------------------------------------
# Main eval loop.
#
# Concurrency note (SAPIEN 2.2.2): `env.close()` only sets `_scene = None` and
# does NOT release `_engine` / `_renderer`, so the underlying Vulkan device
# handles leak across episodes within a single Python process. After 1 build
# the next `gym.make()` reliably crashes with
# `vk::PhysicalDevice::createDeviceUnique: ErrorInitializationFailed`. The
# fix is to build the env exactly once per `(env_kwargs)` group and iterate
# episodes via `env.reset(options=...)`. We sort the rank's slice by
# `task_suite_name` so episodes that share env_kwargs run back-to-back, and
# rebuild only when env_kwargs changes.
# ----------------------------------------------------------------------
def _env_kwargs_key(env_kwargs: dict) -> str:
    """Hashable key identifying a unique env-construction signature."""
    return json.dumps(env_kwargs, sort_keys=True, separators=(",", ":"))


def eval_simplerenv(args: Args) -> None:
    np.random.seed(args.seed)
    print(f"Running dataset: {args.dataset_name} with {args.num_workers} workers")
    print(f"Worker ID: {args.worker_id}")

    total_indices = args.num_workers
    global_index = args.worker_id

    exp_root = pathlib.Path(args.experiment_root)
    exp_root.mkdir(parents=True, exist_ok=True)
    log_dir = exp_root / "results"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{global_index}.csv"
    end_dir = exp_root / "end"
    end_dir.mkdir(parents=True, exist_ok=True)
    end_path = end_dir / f"{global_index}.end"

    video_out_path = None
    if args.record_video:
        video_out_path = exp_root / "videos"
        video_out_path.mkdir(parents=True, exist_ok=True)

    with open(log_path, "w") as f:
        f.write(
            "exp_name,dataset_name,task_suite_name,global_index,total_indices,success_episodes,total_episodes\n"
        )

    env_configs = _load_env_configs(args)

    options_list: list[dict] = []
    for cfg in env_configs:
        options_list.extend([cfg] * args.repeat_times)

    # Sort by env_kwargs first so configs sharing an env land next to each
    # other, then carry the post-sort full-list index through.
    indexed_options = sorted(
        list(enumerate(options_list)),
        key=lambda item: (_env_kwargs_key(item[1]["env_kwargs"]), item[0]),
    )

    # Contiguous-chunk sharding: with 4 task suites x 144 episodes = 576
    # configs and 8 workers, each worker owns 72 contiguous episodes — all
    # inside ONE task suite. That keeps SAPIEN 2.2.2's `vk::createDeviceUnique`
    # failure mode boxed in: every worker only builds the env once
    # (`env.reset(options=...)` for each subsequent episode is leak-safe).
    n = len(indexed_options)
    chunk = (n + total_indices - 1) // total_indices
    rank_slice = indexed_options[global_index * chunk : (global_index + 1) * chunk]

    rank_keys = []
    seen = set()
    for _, cfg in rank_slice:
        k = _env_kwargs_key(cfg["env_kwargs"])
        if k not in seen:
            rank_keys.append(k)
            seen.add(k)
    print(
        f"[Worker {global_index}] processes {len(rank_slice)} episodes "
        f"across {len(rank_keys)} env_kwargs group(s). "
        f"More than 1 group means rebuilds at boundaries (SAPIEN 2.2.2 "
        f"may fail there)."
    )

    client = PolicyServerClient(args.host, args.port)

    total_episodes_dict: dict[str, int] = collections.defaultdict(int)
    total_successes_dict: dict[str, int] = collections.defaultdict(int)
    start_time = time.time()

    env = None
    current_env_key: str | None = None

    try:
        for processed_count, (i, config) in enumerate(rank_slice):
            env_kwargs = config["env_kwargs"]
            env_reset_options = config["env_reset_options"]
            env_key = _env_kwargs_key(env_kwargs)

            if env is None or env_key != current_env_key:
                # Tear down the previous env BEFORE building the next one so
                # SAPIEN's renderer destructor runs while no live env holds
                # the device.
                if env is not None:
                    try:
                        env.close()
                    except Exception:
                        pass
                    env = None
                    import gc
                    gc.collect()
                env = build_maniskill2_env(**env_kwargs)
                current_env_key = env_key

            success = run_single_episode(
                env,
                i,
                config["task_suite_name"],
                args,
                env_reset_options,
                client,
                video_out_path,
            )
            total_successes_dict[config["task_suite_name"]] += int(success)
            total_episodes_dict[config["task_suite_name"]] += 1

            time_elapsed = time.time() - start_time
            time_remaining = (
                (time_elapsed / (processed_count + 1)) * (len(rank_slice) - processed_count - 1)
            )
            print(
                f"[Worker {args.worker_id} / {args.num_workers}] "
                f"{config['task_suite_name']} - {i}/{len(options_list)}, "
                f"success: {success}, "
                f"time elapsed: {time_elapsed:.2f}s, "
                f"time remaining: {time_remaining:.2f}s"
            )
            with open(log_path, "a") as f:
                f.write(
                    f"lerobot_lab,{args.dataset_name},{config['task_suite_name']},"
                    f"{global_index},{total_indices},{int(success)},1\n"
                )
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass

    end_path.write_text("")

    # Wait for all peers to finish before the worker-0 aggregator runs.
    while True:
        end_files = list(end_dir.glob("*.end"))
        if len(end_files) >= total_indices:
            break
        print(
            f"[Worker {args.worker_id} / {args.num_workers}] Waiting for all end files... "
            f"({len(end_files)}/{total_indices}) files present. Sleeping 30s."
        )
        time.sleep(30)

    if global_index == 0:
        csv_files = list(log_dir.glob("*.csv"))
        dataframes = [pd.read_csv(f) for f in csv_files]
        data = pd.concat(dataframes, ignore_index=True)
        grouped = data.groupby(
            ["exp_name", "dataset_name", "task_suite_name"], as_index=False
        )[["success_episodes", "total_episodes"]].sum()
        grouped["success_rate"] = grouped["success_episodes"] / grouped["total_episodes"]
        grouped.to_csv(exp_root / "all_results.csv", index=False)
        print(grouped)


def run_single_episode(
    env,
    config_id: int,
    task_suite_name: str,
    args: Args,
    env_reset_options: dict,
    client: PolicyServerClient,
    video_out_path: pathlib.Path | None = None,
) -> bool:
    """Run one episode on a pre-built shared `env`.

    The env is owned by the caller; this function MUST NOT close it (closing
    inside the loop re-triggers the SAPIEN 2.2.2 Vulkan-device leak that
    breaks the next `gym.make()` in the same process). We only reset the env
    with the per-episode init options and roll out one episode.
    """
    obs, _ = env.reset(options=env_reset_options)
    instruction = env.get_language_instruction()

    image = get_image_from_maniskill2_obs_dict(env, obs)
    done, truncated = False, False

    writer = None
    if video_out_path:
        (video_out_path / task_suite_name).mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(
            video_out_path / task_suite_name / f"{config_id:04d}.mp4",
            fps=30,
        )
        writer.append_data(image)

    action_plan: collections.deque = collections.deque()

    try:
        while not truncated:
            if len(action_plan) <= 0:
                if "bridge" in args.dataset_name:
                    raw_proprio = preprocess_proprio_bridge(obs["agent"]["eef_pos"])
                    task_id = "bridge_delta"
                else:
                    raw_proprio = preprocess_proprio_fractal(obs["agent"]["eef_pos"])
                    task_id = "fractal_delta"

                state = build_state_payload(task_id, raw_proprio)

                # Bridge training tasks are stored raw and predominantly
                # lowercase without trailing periods; the reference client's
                # `Upper+.` rewrite turns "put carrot on plate" (present in
                # training) into "Put carrot on plate." (absent), which is the
                # opposite of what we want. Skip normalization for bridge and
                # pass SimplerEnv's raw instruction through. Fractal is out of
                # scope here, so keep its original normalization.
                if "bridge" not in args.dataset_name:
                    instruction = instruction[0].upper() + instruction[1:] + "."
                base_obs = Image.fromarray(image, mode="RGB")
                seed = hash_data_to_seed(
                    {
                        "task_id": task_id,
                        "state": state,
                        "language": instruction,
                        "base": base_obs,
                    }
                )

                action_chunk = client.predict_chunk(
                    image=_resize_image(image, args.image_size),
                    state=state,
                    instruction=instruction,
                    seed=seed,
                )

                if action_chunk.shape[0] < args.replan_steps:
                    raise RuntimeError(
                        f"replan_steps={args.replan_steps} > chunk size {action_chunk.shape[0]}"
                    )
                action_plan.extend(action_chunk[: args.replan_steps, :7])

            raw_action = action_plan.popleft()

            roll, pitch, yaw = raw_action[3:6]
            ax, angle = euler2axangle(roll, pitch, yaw)
            rot_axangle = ax * angle

            if "bridge" in args.dataset_name:
                gripper = postprocess_gripper_bridge(raw_action[-1])
            else:
                gripper = postprocess_gripper_fractal(raw_action[-1])

            action = np.concatenate([raw_action[:3], rot_axangle, [gripper]])

            obs, _, done, truncated, _ = env.step(action)
            image = get_image_from_maniskill2_obs_dict(env, obs)
            instruction = env.get_language_instruction()
            if writer is not None:
                writer.append_data(image)

            if done:
                break
    finally:
        if writer is not None:
            writer.close()

    return bool(done)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    torch.set_printoptions(3, sci_mode=False)
    tyro.cli(eval_simplerenv)
