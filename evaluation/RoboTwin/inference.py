#!/usr/bin/env python

from __future__ import annotations

import argparse
import importlib
import logging
import shutil
import sys
import traceback
from collections import deque
from pathlib import Path

import imageio
import numpy as np
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
ROBOTWIN_ROOT = REPO_ROOT / "third_party" / "RoboTwin"

if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from lerobot.configs.policies import PreTrainedConfig
from lerobot.dataset_schemas import get_schema
from lerobot.datasets.utils import load_json
from lerobot.policies.factory import get_policy_class
from lerobot.policies.internvla_a1_5.configuration_internvla_a1_5 import InternVLAA15Config
from lerobot.policies.internvla_a1_5.transform_internvla_a1_5 import (
    InternVLAA15ChatProcessorTransformFn,
)
from lerobot.transforms.core import (
    NormalizeTransformFn,
    PadStateAndActionTransformFn,
    RemapImageKeyTransformFn,
    ReorderStateActionTransform,
    ResizeImagesWithPadFn,
    UnNormalizeTransformFn,
    compose,
)
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE


TASK_NAMES = [
    "adjust_bottle",
    "beat_block_hammer",
    "blocks_ranking_rgb",
    "blocks_ranking_size",
    "click_alarmclock",
    "click_bell",
    "dump_bin_bigbin",
    "grab_roller",
    "handover_block",
    "handover_mic",
    "hanging_mug",
    "lift_pot",
    "move_can_pot",
    "move_pillbottle_pad",
    "move_playingcard_away",
    "move_stapler_pad",
    "open_laptop",
    "open_microwave",
    "pick_diverse_bottles",
    "pick_dual_bottles",
    "place_a2b_left",
    "place_a2b_right",
    "place_bread_basket",
    "place_bread_skillet",
    "place_burger_fries",
    "place_can_basket",
    "place_cans_plasticbox",
    "place_container_plate",
    "place_dual_shoes",
    "place_empty_cup",
    "place_fan",
    "place_mouse_pad",
    "place_object_basket",
    "place_object_scale",
    "place_object_stand",
    "place_phone_stand",
    "place_shoe",
    "press_stapler",
    "put_bottles_dustbin",
    "put_object_cabinet",
    "rotate_qrcode",
    "scan_object",
    "shake_bottle",
    "shake_bottle_horizontally",
    "stack_blocks_three",
    "stack_blocks_two",
    "stack_bowls_three",
    "stack_bowls_two",
    "stamp_seal",
    "turn_switch",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RoboTwin evaluation for InternVLA-A1.5 policies.")
    parser.add_argument("--ckpt-path", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, default=Path("outputs/robotwin/internvla_a1_5"))
    parser.add_argument("--task-config", default="demo_clean")
    parser.add_argument("--task-idx", type=int, default=0)
    parser.add_argument("--instruction-type", default="unseen")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stats-key", default="aloha")
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--action-mode", choices=("delta", "abs"), default="abs")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--infer-horizon", type=int, default=20)
    parser.add_argument("--inference-backend", choices=("standard", "optimized"), default="standard")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def require_robotwin():
    if not (ROBOTWIN_ROOT / "envs").exists():
        raise RuntimeError(
            "RoboTwin is not initialized. Run `git submodule update --init third_party/RoboTwin` "
            "and install RoboTwin dependencies first."
        )

    for path in [
        ROBOTWIN_ROOT,
        ROBOTWIN_ROOT / "policy",
        ROBOTWIN_ROOT / "description" / "utils",
    ]:
        if str(path) not in sys.path:
            sys.path.append(str(path))

    from envs import CONFIGS_PATH
    from envs.utils.create_actor import UnStableError
    from generate_episode_instructions import generate_episode_descriptions

    return CONFIGS_PATH, UnStableError, generate_episode_descriptions


def get_embodiment_config(robot_file: str):
    robot_config_file = Path(robot_file) / "config.yml"
    with open(robot_config_file, "r", encoding="utf-8") as f:
        return OmegaConf.load(f)


def make_env(task_name: str):
    envs_module = importlib.import_module(f"envs.{task_name}")
    env_class = getattr(envs_module, task_name)
    return env_class()


def build_task_args(task_config: str, task_name: str, configs_path: str):
    with open(ROBOTWIN_ROOT / "task_config" / f"{task_config}.yml", "r", encoding="utf-8") as f:
        task_args = OmegaConf.to_container(OmegaConf.load(f), resolve=True)

    with open(configs_path + "_embodiment_config.yml", "r", encoding="utf-8") as f:
        embodiment_types = OmegaConf.to_container(OmegaConf.load(f), resolve=True)
    with open(configs_path + "_camera_config.yml", "r", encoding="utf-8") as f:
        camera_cfg = OmegaConf.to_container(OmegaConf.load(f), resolve=True)

    def get_embodiment_file(embodiment_type):
        robot_file = embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise RuntimeError(f"No embodiment file for {embodiment_type}")
        return robot_file

    embodiment_type = task_args["embodiment"]
    head_camera_type = task_args["camera"]["head_camera_type"]
    task_args["head_camera_h"] = camera_cfg[head_camera_type]["h"]
    task_args["head_camera_w"] = camera_cfg[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        robot_file = str(ROBOTWIN_ROOT / get_embodiment_file(embodiment_type[0]))
        task_args["left_robot_file"] = robot_file
        task_args["right_robot_file"] = robot_file
        task_args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        task_args["left_robot_file"] = str(ROBOTWIN_ROOT / get_embodiment_file(embodiment_type[0]))
        task_args["right_robot_file"] = str(ROBOTWIN_ROOT / get_embodiment_file(embodiment_type[1]))
        task_args["embodiment_dis"] = embodiment_type[2]
        task_args["dual_arm_embodied"] = False
    else:
        raise RuntimeError("RoboTwin embodiment list must contain 1 or 3 items.")

    task_args["left_embodiment_config"] = get_embodiment_config(task_args["left_robot_file"])
    task_args["right_embodiment_config"] = get_embodiment_config(task_args["right_robot_file"])
    task_args["task_name"] = task_name
    task_args["task_config"] = task_config
    task_args["eval_mode"] = True
    return task_args


def load_stats(ckpt_path: Path, stats_key: str):
    stats = load_json(ckpt_path / "stats.json")
    if stats_key not in stats:
        raise KeyError(f"stats_key '{stats_key}' not found in {ckpt_path / 'stats.json'}")

    selected = stats[stats_key]
    stat_keys = ["min", "max", "mean", "std"]
    state_stat = {OBS_STATE: {k: np.asarray(selected[OBS_STATE][k]) for k in stat_keys}}
    action_stat = {ACTION: {k: np.asarray(selected[ACTION][k]) for k in stat_keys}}
    return state_stat, action_stat


def build_input_transforms(resize_size: int, state_stat: dict, stats_key: str, config: InternVLAA15Config):
    schema = get_schema(stats_key)
    return compose(
        [
            ResizeImagesWithPadFn(height=resize_size, width=resize_size, mapping=schema.image_mapping),
            RemapImageKeyTransformFn(mapping=schema.image_mapping),
            NormalizeTransformFn(selected_keys=[OBS_STATE], norm_stats=state_stat),
            InternVLAA15ChatProcessorTransformFn(
                mode="eval",
                tokenize_state=getattr(config, "tokenize_state", True),
                max_state_dim=getattr(config, "max_state_dim", 32),
            ),
            PadStateAndActionTransformFn(
                max_state_dim=getattr(config, "max_state_dim", 32),
                max_action_dim=getattr(config, "max_action_dim", 32),
            ),
            ReorderStateActionTransform(
                state_reorder=schema.state_reorder,
                action_reorder=schema.action_reorder,
            ),
        ]
    )


def load_policy(args: argparse.Namespace, dtype: torch.dtype):
    config = PreTrainedConfig.from_pretrained(args.ckpt_path)
    if not isinstance(config, InternVLAA15Config):
        raise ValueError(f"Checkpoint policy.type must be 'internvla_a1_5', got {config.type!r}.")

    config.action_loss_only = True
    config.inference_backend = args.inference_backend
    config.device = "cuda" if torch.cuda.is_available() else "cpu"

    policy_cls = get_policy_class(config.type)
    policy = policy_cls.from_pretrained(args.ckpt_path, config=config)
    device = torch.device(config.device)
    policy.to(device=device, dtype=dtype)
    policy.eval()
    return policy, device, config


def to_policy_batch(sample: dict, device: torch.device, dtype: torch.dtype) -> dict:
    batch = {}
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            value = value.unsqueeze(0)
            if value.dtype.is_floating_point:
                value = value.to(device=device, dtype=dtype)
            else:
                value = value.to(device=device)
            batch[key] = value
        else:
            batch[key] = [value]
    return batch


def build_sample(observation: dict, instruction: str, dtype: torch.dtype) -> dict:
    obs = observation["observation"]
    sample = {
        OBS_STATE: torch.from_numpy(observation["joint_action"]["vector"]).float(),
        ACTION: torch.zeros(50, 14, dtype=torch.float32),
        "task": instruction,
        f"{OBS_IMAGES}.cam_high": torch.as_tensor(obs["head_camera"]["rgb"]).contiguous().to(dtype=dtype) / 255.0,
        f"{OBS_IMAGES}.cam_left_wrist": torch.as_tensor(obs["left_camera"]["rgb"]).contiguous().to(dtype=dtype) / 255.0,
        f"{OBS_IMAGES}.cam_right_wrist": torch.as_tensor(obs["right_camera"]["rgb"]).contiguous().to(dtype=dtype) / 255.0,
    }

    for key in list(sample.keys()):
        if key.startswith(OBS_IMAGES):
            sample[key] = sample[key].permute(2, 0, 1)
    return sample


def compact_reordered_dual_arm_actions(actions: torch.Tensor) -> torch.Tensor:
    if actions.shape[-1] < 16:
        raise ValueError("InternVLA-A1.5 RoboTwin action output must have at least 16 reordered dimensions.")
    return torch.cat(
        [
            actions[..., :6],
            actions[..., 7:8],
            actions[..., 8:14],
            actions[..., 15:16],
        ],
        dim=-1,
    )


def tensor_chw_to_uint8_hwc(image_chw: torch.Tensor) -> np.ndarray:
    image = image_chw.detach().float().cpu().clamp(0, 1)
    image = (image * 255.0).to(torch.uint8)
    return image.permute(1, 2, 0).numpy()


def save_replay_video(video_path: Path, replay_images: list[np.ndarray], fps: int):
    video_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(video_path, replay_images, fps=fps)


def maybe_close_env(task_env, *, clear_cache: bool = False):
    try:
        task_env.close_env(clear_cache=clear_cache)
    except TypeError:
        task_env.close_env()
    except Exception:
        logging.debug("Failed to close RoboTwin env cleanly.", exc_info=True)


def infer_once(args: argparse.Namespace):
    if not 0 <= args.task_idx < len(TASK_NAMES):
        raise IndexError(f"task_idx must be in [0, {len(TASK_NAMES) - 1}], got {args.task_idx}")
    if args.infer_horizon <= 0:
        raise ValueError("--infer-horizon must be positive.")

    configs_path, unstable_error, generate_episode_descriptions = require_robotwin()
    task_name = TASK_NAMES[args.task_idx]
    task_args = build_task_args(args.task_config, task_name, configs_path)
    task_env = make_env(task_name)

    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    policy, device, config = load_policy(args, dtype)
    state_stat, action_stat = load_stats(args.ckpt_path, args.stats_key)
    input_transforms = build_input_transforms(args.resize_size, state_stat, args.stats_key, config)
    unnormalize_fn = UnNormalizeTransformFn(
        selected_keys=[ACTION],
        mode="mean_std",
        norm_stats=action_stat,
    )

    task_env.suc = 0
    task_env.test_num = 0
    np.random.seed(args.seed)

    seed_start = 100000 * (1 + args.seed)
    seed_cursor = seed_start
    episode_id = 0
    clear_cache_freq = task_args["clear_cache_freq"]
    successful_seed_count = 0
    seed_candidates = list(range(seed_start, seed_start * 2))

    args.video_dir.mkdir(parents=True, exist_ok=True)

    while successful_seed_count < args.num_episodes:
        render_freq = task_args["render_freq"]
        task_args["render_freq"] = 0
        seed_value = seed_candidates[seed_cursor - seed_start]

        try:
            task_env.setup_demo(now_ep_num=episode_id, seed=seed_value, is_test=True, **task_args)
            episode_info = task_env.play_once()
            maybe_close_env(task_env)
        except unstable_error as exc:
            logging.warning("Skipping unstable seed for task=%s seed=%s: %s", task_name, seed_value, exc)
            maybe_close_env(task_env)
            seed_cursor += 1
            task_args["render_freq"] = render_freq
            continue
        except Exception:
            logging.error("Expert rollout failed for task=%s seed=%s", task_name, seed_value)
            logging.error(traceback.format_exc())
            maybe_close_env(task_env)
            seed_cursor += 1
            task_args["render_freq"] = render_freq
            continue

        if task_env.plan_success and task_env.check_success():
            successful_seed_count += 1
        else:
            seed_cursor += 1
            task_args["render_freq"] = render_freq
            continue

        task_args["render_freq"] = render_freq
        task_env.setup_demo(now_ep_num=episode_id, seed=seed_value, is_test=True, **task_args)

        descriptions = generate_episode_descriptions(task_name, [episode_info["info"]], args.num_episodes)
        instruction = str(np.random.choice(descriptions[0][args.instruction_type]))
        task_env.set_instruction(instruction=instruction)

        policy.reset()
        action_plan = deque([], maxlen=args.infer_horizon)
        replay_images: list[np.ndarray] = []
        success = False

        while task_env.take_action_cnt < task_env.step_lim:
            observation = task_env.get_obs()
            sample = build_sample(observation, task_env.get_instruction(), dtype)
            sample = input_transforms(sample)

            transformed_image = sample[f"{OBS_IMAGES}.image0"]
            replay_images.append(tensor_chw_to_uint8_hwc(transformed_image))

            if not action_plan:
                batch = to_policy_batch(sample, device, dtype)
                with torch.no_grad():
                    action_pred = policy.predict_action_chunk(batch)

                if action_pred.ndim == 3:
                    action_pred = action_pred[0]
                action_pred = compact_reordered_dual_arm_actions(action_pred[: args.infer_horizon])
                action_pred = unnormalize_fn({ACTION: action_pred})[ACTION]

                if args.action_mode == "delta":
                    current_action = torch.from_numpy(observation["joint_action"]["vector"]).to(action_pred)
                    current_action[6] = 0.0
                    current_action[13] = 0.0
                    action_pred = action_pred + current_action[:14]

                action_plan.extend(action_pred.detach().float().cpu().numpy())

            action = action_plan.popleft()[:14]
            action[6] = np.clip(action[6], 0, 1)
            action[13] = np.clip(action[13], 0, 1)
            task_env.take_action(action, action_type="qpos")

            if task_env.eval_success:
                success = True
                break

        if success:
            task_env.suc += 1
            print("\033[92mSuccess!\033[0m")
        else:
            print("\033[91mFail!\033[0m")

        suffix = "success" if success else "failure"
        save_replay_video(args.video_dir / f"{suffix}_{successful_seed_count}.mp4", replay_images, args.fps)

        episode_id += 1
        maybe_close_env(task_env, clear_cache=((successful_seed_count + 1) % clear_cache_freq == 0))
        if getattr(task_env, "render_freq", 0):
            task_env.viewer.close()

        task_env.test_num += 1
        print(
            f"\033[93m{task_name}\033[0m | \033[92m{task_args['task_config']}\033[0m\n"
            f"Success rate: \033[96m{task_env.suc}/{task_env.test_num}\033[0m => "
            f"\033[95m{round(task_env.suc / task_env.test_num * 100, 1)}%\033[0m, "
            f"current seed: \033[90m{seed_cursor}\033[0m\n"
        )
        seed_cursor += 1


def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )
    logging.getLogger("curobo").setLevel(logging.WARNING)

    if args.video_dir.exists():
        shutil.rmtree(args.video_dir)
    args.video_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Starting RoboTwin inference for InternVLA-A1.5.")
    logging.info("task_idx=%s ckpt=%s output=%s", args.task_idx, args.ckpt_path, args.video_dir)
    infer_once(args)


if __name__ == "__main__":
    main()
