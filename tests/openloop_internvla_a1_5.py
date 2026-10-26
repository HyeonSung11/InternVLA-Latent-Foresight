#!/usr/bin/env python
"""Open-loop action evaluation for InternVLA-A1.5.

The workflow intentionally mirrors the local Qwen3.5 KI-Wan open-loop scripts:
load one checkpoint, load one LeRobot demo dataset, run chunked action
prediction, unnormalize the default A1 action layout, plot trajectories, and
optionally save predicted future-frame visualizations.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

ACTION = "action"
OBS_STATE = "observation.state"
DEFAULT_REPO_ID = "."
DEFAULT_DEVICE = "cuda"
DEFAULT_DTYPE = "bfloat16"
DEFAULT_NORM_MODE = "mean_std"
BLACK_THRESHOLD = 0.03

ACTION_STAT_KEYS = [
    "actions.left_joint.position",
    "actions.left_gripper.position",
    "actions.right_joint.position",
    "actions.right_gripper.position",
]
STATE_STAT_KEYS = [
    "states.left_joint.position",
    "states.left_gripper.position",
    "states.right_joint.position",
    "states.right_gripper.position",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("--ckpt-path", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/openloop/internvla_a1_5"))
    parser.add_argument("--num-episodes", type=int, default=2)
    parser.add_argument(
        "--max-samples-per-episode",
        type=int,
        default=8,
        help="0 means evaluate every open-loop step in each selected episode.",
    )
    parser.add_argument("--visualize-future", action="store_true")
    parser.add_argument("--video-denoise-steps", type=int, default=20)
    parser.add_argument("--num-future-frames", type=int, default=4)
    return parser.parse_args()


def torch_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def compact_a1_dims(tensor: torch.Tensor) -> torch.Tensor:
    """Default A1 layout: keep 6 joints + gripper for each arm, dropping gaps."""
    tensor = tensor[..., :16]
    return torch.cat(
        [
            tensor[..., :6],
            tensor[..., 7:8],
            tensor[..., 8:14],
            tensor[..., 15:16],
        ],
        dim=-1,
    )


def make_batch(sample: dict, device: torch.device, dtype: torch.dtype) -> dict:
    batch = {}
    for key, value in sample.items():
        if key in {"repo_id", "dataset_index"}:
            continue
        if key in {"task", "robot_type"}:
            batch[key] = [value]
        elif torch.is_tensor(value):
            if value.dtype in (torch.int64, torch.int32, torch.bool):
                batch[key] = value[None].to(device=device)
            else:
                batch[key] = value[None].to(device=device, dtype=dtype)
        else:
            batch[key] = [value]
    return batch


def concat_stats(dataset, keys: list[str]) -> dict | None:
    meta_stats = dataset.meta.stats
    if not all(key in meta_stats for key in keys):
        return None

    stats = {}
    for stat_name in ("min", "max", "mean", "std", "q01", "q99"):
        if all(stat_name in meta_stats[key] for key in keys):
            stats[stat_name] = np.concatenate(
                [np.asarray(meta_stats[key][stat_name]) for key in keys],
                axis=-1,
            )
    return stats or None


def make_unnorm_fn(
    dataset,
    key: str,
    stat_keys: list[str],
    mode: str,
) -> UnNormalizeTransformFn | None:
    from lerobot.transforms.core import UnNormalizeTransformFn

    stats = concat_stats(dataset, stat_keys)
    if stats is None:
        stats = dataset.meta.stats.get(key)
    if stats is None:
        logging.warning("No stats found for %s; values stay in normalized space.", key)
        return None
    return UnNormalizeTransformFn(
        selected_keys=[key],
        mode=mode,
        norm_stats={key: stats},
    )


def tensor_frames_to_uint8(frames: torch.Tensor) -> np.ndarray:
    """Convert [T, C, H, W] frames in [-1, 1] or [0, 1] to uint8 [T, H, W, C]."""
    frames = frames.detach().float().cpu()
    if float(frames.min()) < -0.05:
        frames = (frames + 1.0) / 2.0
    frames = frames.clamp(0.0, 1.0)
    frames = frames.permute(0, 2, 3, 1).numpy()
    return (frames * 255.0 + 0.5).astype(np.uint8)


def find_nonblack_bbox(frame: np.ndarray, threshold: float) -> tuple[int, int, int, int]:
    mask = frame.max(axis=2) > int(round(threshold * 255.0))
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if len(rows) == 0 or len(cols) == 0:
        h, w = frame.shape[:2]
        return 0, h, 0, w
    return int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1


def crop_frames(frames: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    top, bottom, left, right = bbox
    return frames[:, top:bottom, left:right, :]


def make_strip(
    frames: list[np.ndarray],
    labels: list[str] | None,
    gap: int = 4,
    label_height: int = 18,
) -> Image.Image:
    h, w = frames[0].shape[:2]
    top_pad = label_height if labels else 0
    canvas = Image.new(
        "RGB",
        (len(frames) * w + (len(frames) + 1) * gap, h + top_pad + 2 * gap),
        color=(245, 245, 245),
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for i, frame in enumerate(frames):
        x = gap + i * (w + gap)
        y = gap + top_pad
        if labels:
            draw.text((x, gap), labels[i], fill=(20, 20, 20), font=font)
        canvas.paste(Image.fromarray(frame), (x, y))
    return canvas


def make_pred_gt_grid(
    pred_frames: list[np.ndarray],
    gt_frames: list[np.ndarray],
    labels: list[str] | None,
    gap: int = 4,
    label_height: int = 18,
    row_label_width: int = 38,
) -> Image.Image:
    h, w = pred_frames[0].shape[:2]
    top_pad = label_height if labels else 0
    canvas = Image.new(
        "RGB",
        (
            row_label_width + len(pred_frames) * w + (len(pred_frames) + 1) * gap,
            top_pad + 2 * h + 3 * gap,
        ),
        color=(245, 245, 245),
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    if labels:
        for i, label in enumerate(labels):
            x = row_label_width + gap + i * (w + gap)
            draw.text((x, gap), label, fill=(20, 20, 20), font=font)

    for row_idx, (row_name, row_frames) in enumerate(
        [("pred", pred_frames), ("gt", gt_frames)]
    ):
        y = top_pad + gap + row_idx * (h + gap)
        draw.text((gap, y + max(0, (h - 10) // 2)), row_name, fill=(20, 20, 20), font=font)
        for i, frame in enumerate(row_frames):
            x = row_label_width + gap + i * (w + gap)
            canvas.paste(Image.fromarray(frame), (x, y))
    return canvas


def save_future_visualization(
    *,
    sample: dict,
    generated_video: torch.Tensor,
    config: InternVLAA15Config,
    args: argparse.Namespace,
    input_pred_dir: Path,
    compare_dir: Path,
    ep_id: int,
    idx: int,
) -> dict | None:
    if "observation.video_frames" not in sample:
        print(f"skip ep{ep_id} idx{idx}: no observation.video_frames")
        return None

    gt_video = sample["observation.video_frames"].detach().cpu()
    pred_video = generated_video[0].detach().cpu()

    pred_start = 1 if pred_video.shape[0] >= args.num_future_frames + 1 else 0
    future_count = min(
        args.num_future_frames,
        config.num_video_frames,
        gt_video.shape[0] - 1,
        pred_video.shape[0] - pred_start,
    )
    if future_count <= 0:
        print(
            f"skip ep{ep_id} idx{idx}: insufficient frames "
            f"(pred={pred_video.shape[0]}, gt={gt_video.shape[0]})"
        )
        return None

    gt_uint8 = tensor_frames_to_uint8(gt_video[: future_count + 1])
    pred_uint8 = tensor_frames_to_uint8(pred_video[pred_start : pred_start + future_count])

    bbox = find_nonblack_bbox(gt_uint8[0], BLACK_THRESHOLD)
    gt_uint8 = crop_frames(gt_uint8, bbox)
    pred_uint8 = crop_frames(pred_uint8, bbox)

    input_frame = gt_uint8[0]
    gt_future = [gt_uint8[i] for i in range(1, future_count + 1)]
    pred_future = [pred_uint8[i] for i in range(future_count)]

    future_labels = [f"t+{i}" for i in range(1, future_count + 1)]
    strip_labels = ["input"] + [f"pred {x}" for x in future_labels]
    compare_labels = ["obs"] + future_labels

    stem = f"ep{ep_id:03d}_idx{idx:08d}"
    input_pred_path = input_pred_dir / f"{stem}.png"
    compare_path = compare_dir / f"{stem}.png"

    make_strip([input_frame] + pred_future, labels=strip_labels).save(input_pred_path)
    make_pred_gt_grid(
        [input_frame] + pred_future,
        [input_frame] + gt_future,
        labels=compare_labels,
    ).save(compare_path)

    print(f"saved {input_pred_path} and {compare_path}")
    return {
        "episode": ep_id,
        "dataset_index": idx,
        "future_count": future_count,
        "crop_bbox_top_bottom_left_right": list(bbox),
        "input_pred": str(input_pred_path),
        "pred_gt_compare": str(compare_path),
    }


def save_episode_plot(gt: np.ndarray, pred: np.ndarray, save_path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(8, 2, figsize=(16, 12))
    axs = axs.ravel()
    x_values = np.arange(gt.shape[0])

    for dim in range(len(axs)):
        if dim >= gt.shape[1]:
            axs[dim].axis("off")
            continue
        axs[dim].plot(x_values, gt[:, dim], label="Ground Truth", color="blue", linewidth=1.5)
        axs[dim].plot(
            x_values,
            pred[:, dim],
            label="FM Decode",
            color="red",
            linestyle="--",
            linewidth=1.5,
        )
        axs[dim].set_title(f"Dimension {dim + 1}")
        axs[dim].set_xlabel("Time Step")
        axs[dim].set_ylabel("Value")
        axs[dim].legend(loc="upper right")
        axs[dim].grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.suptitle(title, fontsize=16, y=1.02)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger().handlers.clear()

    global F, torch
    import torch
    import torch.nn.functional as F

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.datasets.factory import make_dataset
    from lerobot.policies.internvla_a1_5 import InternVLAA15Config, InternVLAA15Policy

    if DEFAULT_DEVICE == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by this open-loop script.")

    device = torch.device(DEFAULT_DEVICE)
    dtype = torch_dtype(DEFAULT_DTYPE)

    config = PreTrainedConfig.from_pretrained(args.ckpt_path)
    assert isinstance(config, InternVLAA15Config), (
        f"Expected InternVLAA15Config, got {type(config)}"
    )
    config.compile_model = False
    config.compile_mode = "reduce-overhead"
    config.device = DEFAULT_DEVICE

    if args.visualize_future:
        config.inference_backend = "standard"
        config.action_loss_only = False
    else:
        config.inference_backend = "optimized"
        config.action_loss_only = True

    policy = InternVLAA15Policy.from_pretrained(args.ckpt_path, config=config)
    policy.to(device=device)
    policy.to(dtype=dtype)
    policy.eval()
    print(policy)

    cfg = TrainPipelineConfig.from_pretrained(args.ckpt_path)
    cfg.policy = config
    cfg.vqa_dataset = None
    cfg.dataset.repo_id = DEFAULT_REPO_ID
    cfg.dataset.root = str(args.dataset_root)
    cfg.dataset.mode = "eval"
    cfg.dataset.streaming = False
    cfg.dataset.dist_loading = False
    cfg.dataset.__post_init__()
    action_mode = cfg.dataset.action_mode
    print(f"Set mode=eval for dataset; action_mode={action_mode}")

    dataset, _ = make_dataset(cfg)

    act_unnorm_fn = make_unnorm_fn(dataset, ACTION, ACTION_STAT_KEYS, DEFAULT_NORM_MODE)
    state_unnorm_fn = make_unnorm_fn(dataset, OBS_STATE, STATE_STAT_KEYS, DEFAULT_NORM_MODE)

    from_ids = np.asarray(dataset.meta.episodes["dataset_from_index"]).tolist()
    to_ids = np.asarray(dataset.meta.episodes["dataset_to_index"]).tolist()
    num_episodes = min(dataset.num_episodes, args.num_episodes)
    stride = config.chunk_size

    output_dir = args.out_dir
    (output_dir / "plots").mkdir(parents=True, exist_ok=True)

    input_pred_dir = output_dir / "input_pred"
    compare_dir = output_dir / "pred_gt_compare"
    if args.visualize_future:
        input_pred_dir.mkdir(parents=True, exist_ok=True)
        compare_dir.mkdir(parents=True, exist_ok=True)

    metric_mse = []
    mse_joint = []
    mse_gripper = []
    future_saved = []
    num_steps = 0
    elapse_time = 0.0

    for ep_id in range(num_episodes):
        print(f"episode: {ep_id}")
        print(f"from_idx: {from_ids[ep_id]}, to_idx: {to_ids[ep_id]}")

        action_gt_list = []
        action_pred_list = []
        state_list = []

        ep_indices = list(range(from_ids[ep_id], to_ids[ep_id], stride))
        if args.max_samples_per_episode > 0:
            ep_indices = ep_indices[: args.max_samples_per_episode]

        for idx in ep_indices:
            print(f"compute sample {idx}")
            sample = dataset[idx]
            inputs = make_batch(sample, device=device, dtype=dtype)

            if device.type == "cuda":
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            with torch.no_grad():
                if args.visualize_future:
                    action_pred, generated_video = policy.predict_action_chunk_with_video(
                        inputs,
                        num_video_steps=args.video_denoise_steps,
                    )
                else:
                    action_pred = policy.predict_action_chunk(inputs)
                    generated_video = None
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapse_time += time.perf_counter() - start_time

            action_pred = compact_a1_dims(action_pred[0]).to(dtype=torch.float32)
            chunk_size = config.chunk_size
            action_gt = compact_a1_dims(sample[ACTION][:chunk_size].to(device=device))
            action_gt = action_gt.to(dtype=torch.float32)

            valid_steps = min(action_pred.shape[0], action_gt.shape[0])
            action_pred_list.append(action_pred[:valid_steps].clone())
            action_gt_list.append(action_gt[:valid_steps])

            if action_mode == "delta":
                state = sample[OBS_STATE].clone().repeat(valid_steps, 1).to(device=device)
                state = compact_a1_dims(state).to(dtype=torch.float32)
                state[:, 6] *= 0.0
                state[:, 13] *= 0.0
                state_list.append(state)

            if args.visualize_future and generated_video is not None:
                saved = save_future_visualization(
                    sample=sample,
                    generated_video=generated_video,
                    config=config,
                    args=args,
                    input_pred_dir=input_pred_dir,
                    compare_dir=compare_dir,
                    ep_id=ep_id,
                    idx=idx,
                )
                if saved is not None:
                    future_saved.append(saved)

            num_steps += 1

        if not action_gt_list:
            print(f"skip episode {ep_id}: no samples")
            continue

        action_gt_tensor = torch.cat(action_gt_list, dim=0)
        action_pred_tensor = torch.cat(action_pred_list, dim=0)
        if act_unnorm_fn is not None:
            action_gt_tensor = act_unnorm_fn({ACTION: action_gt_tensor})[ACTION]
            action_pred_tensor = act_unnorm_fn({ACTION: action_pred_tensor})[ACTION]

        if action_mode == "delta":
            state_tensor = torch.cat(state_list, dim=0)
            if state_unnorm_fn is not None:
                state_tensor = state_unnorm_fn({OBS_STATE: state_tensor})[OBS_STATE]
            state_tensor = state_tensor.to(device=device)
            action_gt_tensor[:, :14] += state_tensor[:, :14]
            action_pred_tensor[:, :14] += state_tensor[:, :14]

        action_gt_tensor = action_gt_tensor.to(torch.float32)
        action_pred_tensor = action_pred_tensor.to(torch.float32)

        action_gt_gripper = torch.cat(
            [action_gt_tensor[:, 6:7], action_gt_tensor[:, 13:14]],
            dim=-1,
        )
        action_gt_joint = torch.cat(
            [action_gt_tensor[:, :6], action_gt_tensor[:, 7:13]],
            dim=-1,
        )
        action_pred_gripper = torch.cat(
            [action_pred_tensor[:, 6:7], action_pred_tensor[:, 13:14]],
            dim=-1,
        )
        action_pred_joint = torch.cat(
            [action_pred_tensor[:, :6], action_pred_tensor[:, 7:13]],
            dim=-1,
        )

        metric_mse.append(float(F.mse_loss(action_gt_tensor, action_pred_tensor).cpu()))
        mse_joint.append(float(F.mse_loss(action_gt_joint, action_pred_joint).cpu()))
        mse_gripper.append(float(F.mse_loss(action_gt_gripper, action_pred_gripper).cpu()))

        action_gt_numpy = action_gt_tensor.detach().cpu().numpy()
        action_pred_numpy = action_pred_tensor.detach().cpu().numpy()
        np.savez(
            output_dir / f"actions_ep{ep_id:03d}.npz",
            ground_truth=action_gt_numpy,
            prediction=action_pred_numpy,
        )
        save_episode_plot(
            action_gt_numpy,
            action_pred_numpy,
            output_dir / "plots" / f"internvla_a1_5_open_loop_ep{ep_id}.jpg",
            f"Ground Truth vs Prediction (InternVLA-A1.5) ep{ep_id}",
        )

    log = {
        "ckpt_path": str(args.ckpt_path),
        "repo_id": DEFAULT_REPO_ID,
        "dataset_root": str(args.dataset_root),
        "num_steps": num_steps,
        "sample_stride": stride,
        "visualize_future": args.visualize_future,
        "FM_decode": {
            "MSE": metric_mse,
            "Average_MSE": float(np.mean(metric_mse)) if metric_mse else None,
            "MSE_joints": mse_joint,
            "Average_MSE_joints": float(np.mean(mse_joint)) if mse_joint else None,
            "MSE_gripper": mse_gripper,
            "Average_MSE_gripper": float(np.mean(mse_gripper)) if mse_gripper else None,
            "elapse_time": elapse_time,
            "fps": num_steps / elapse_time if elapse_time > 0 else 0,
        },
    }
    with (output_dir / "log.json").open("w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)

    if args.visualize_future:
        manifest = {
            "ckpt_path": str(args.ckpt_path),
            "repo_id": DEFAULT_REPO_ID,
            "video_denoise_steps": args.video_denoise_steps,
            "num_future_frames_requested": args.num_future_frames,
            "sample_stride": stride,
            "crop_padding": True,
            "saved": future_saved,
        }
        with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    print("\n=== Performance Metrics ===")
    print("[FM Decode]")
    print(f"  Average MSE (total): {log['FM_decode']['Average_MSE']}")
    print(f"  Average MSE (joints): {log['FM_decode']['Average_MSE_joints']}")
    print(f"  Average MSE (gripper): {log['FM_decode']['Average_MSE_gripper']}")
    print(f"  Total time: {elapse_time:.2f}s | FPS: {log['FM_decode']['fps']:.2f}")
    if args.visualize_future:
        print(f"  Future visualizations: {len(future_saved)}")
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
