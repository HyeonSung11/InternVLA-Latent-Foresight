from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import sys
from pathlib import Path

import imageio
import numpy as np
from termcolor import colored
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Reuse the stock-LIBERO env-side adapter unchanged: LIBERO-plus keeps the same
# obs keys / action convention and only bakes perturbations into bddl+init files.
from evaluation.LIBERO.model2libero_interface import LiberoModelClient

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
LOGGER = logging.getLogger(__name__)

TASK_SUITE_MAX_STEPS: dict[str, int] = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


def _get_libero_env(task, resolution: int, seed: int):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(task_bddl_file),
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(seed)
    return env, task_description


def evaluate_task(
    task,
    initial_states,
    args: argparse.Namespace,
    client: LiberoModelClient,
    max_steps: int,
    video_dir: Path,
    video_tag: str,
):
    """Run all episodes for one LIBERO-plus task; returns list[bool] of successes.

    `video_tag` is the perturbation-aware clean name from task_classification.json,
    used for video filenames because LIBERO-plus task `.language` repeats across
    perturbations of the same base task.
    """
    env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
    n_episodes = min(args.num_trials_per_task, len(initial_states))
    successes: list[bool] = []

    for episode_idx in tqdm(range(n_episodes), desc=f"task: {video_tag}", leave=False):
        client.reset(task_description)
        env.reset()
        obs = env.set_init_state(initial_states[episode_idx])

        replay_images: list[np.ndarray] = []
        done = False
        for t in range(max_steps + args.num_steps_wait):
            if t < args.num_steps_wait:
                obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                continue

            action = client.step(obs, task_description)
            replay_images.append(np.ascontiguousarray(np.asarray(obs["agentview_image"])[::-1, ::-1]))
            obs, _, done, _ = env.step(action.tolist())
            if done:
                break

        successes.append(bool(done))

        if args.save_videos and replay_images:
            suffix = "success" if done else "failure"
            out_path = video_dir / f"rollout_{video_tag}_episode{episode_idx}_{suffix}.mp4"
            imageio.mimwrite(out_path, [np.asarray(x) for x in replay_images], fps=10)

    env.close()
    return successes, task_description


def _load_id2category(task_classification_path: str, suite: str) -> tuple[dict[int, tuple[str, str]], dict[str, dict]]:
    with open(task_classification_path, encoding="utf-8") as f:
        mapping = json.load(f)
    if suite not in mapping:
        raise KeyError(
            f"Suite '{suite}' not in task_classification.json. Available: {list(mapping.keys())}"
        )
    id2category: dict[int, tuple[str, str]] = {}
    disturb_res: dict[str, dict] = {}
    for item in mapping[suite]:
        category = item["category"]
        id2category[int(item["id"])] = (category, item["name"])
        disturb_res.setdefault(category, {"total_count": 0, "success_count": 0})
    return id2category, disturb_res


def evaluate_policy(args: argparse.Namespace, client: LiberoModelClient) -> dict:
    from libero.libero import benchmark

    benchmark_dict = benchmark.get_benchmark_dict()
    if args.task_suite_name not in benchmark_dict:
        raise KeyError(
            f"Unknown task_suite_name '{args.task_suite_name}'. Available: {list(benchmark_dict.keys())}"
        )
    task_suite = benchmark_dict[args.task_suite_name]()
    n_tasks_in_suite = task_suite.n_tasks

    # Task-id shard range [start_idx, end_idx). Defaults to the full suite.
    start_idx = max(0, args.start_idx) if args.start_idx >= 0 else 0
    end_idx = args.end_idx if args.end_idx >= 0 else n_tasks_in_suite
    end_idx = min(end_idx, n_tasks_in_suite)
    if start_idx >= end_idx:
        raise ValueError(f"Empty shard: start_idx={start_idx} end_idx={end_idx} (n_tasks={n_tasks_in_suite})")

    # task_classification.json id is 1-indexed and matches suite task order, so
    # JSON id == env task_id + 1.
    id2category, disturb_res = _load_id2category(args.task_classification_path, args.task_suite_name)

    max_steps = args.max_steps_override if args.max_steps_override > 0 else TASK_SUITE_MAX_STEPS[args.task_suite_name]

    LOGGER.info(
        "Suite=%s | shard [%d, %d) of %d tasks | max_steps=%d | trials/task=%d",
        args.task_suite_name, start_idx, end_idx, n_tasks_in_suite, max_steps, args.num_trials_per_task,
    )

    eval_log_dir = Path(args.eval_log_dir)
    video_dir = eval_log_dir / "videos" / args.task_suite_name
    if args.save_videos:
        video_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = eval_log_dir / "logs" / args.task_suite_name
    logs_dir.mkdir(parents=True, exist_ok=True)

    total_episodes = 0
    total_successes = 0

    for task_id in tqdm(range(start_idx, end_idx), desc=f"{args.task_suite_name}[{start_idx}:{end_idx}]"):
        category, clean_name = id2category[task_id + 1]
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        successes, task_desc = evaluate_task(
            task, initial_states, args, client, max_steps, video_dir, clean_name,
        )
        n_succ = int(sum(successes))
        disturb_res[category]["total_count"] += len(successes)
        disturb_res[category]["success_count"] += n_succ
        total_episodes += len(successes)
        total_successes += n_succ
        LOGGER.info(
            "task_id=%d [%s] '%s' -> %d/%d (running total %d/%d = %.2f%%)",
            task_id, category, task_desc, n_succ, len(successes),
            total_successes, total_episodes, 100.0 * total_successes / max(total_episodes, 1),
        )

    shard_summary = {
        "task_suite": args.task_suite_name,
        "start_idx": start_idx,
        "end_idx": end_idx,
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "overall_success_rate": total_successes / max(total_episodes, 1),
        "per_category": disturb_res,
    }

    # Per-shard file consumed by aggregate_results.py: {category: {total_count, success_count}}.
    out_json = logs_dir / f"{start_idx}_to_{end_idx}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(disturb_res, f, indent=2)
    LOGGER.info("Saved per-shard category results to %s", out_json)

    return shard_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LIBERO-plus evaluation with websocket policy server")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5694)

    parser.add_argument(
        "--task_suite_name",
        type=str,
        default="libero_goal",
        choices=list(TASK_SUITE_MAX_STEPS.keys()),
    )
    # LIBERO-plus default is 1 trial per (already perturbed) task.
    parser.add_argument("--num_trials_per_task", type=int, default=1)
    parser.add_argument("--num_steps_wait", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--replan_steps",
        type=int,
        default=8,
        help="Re-request a new action chunk every N env steps (must be <= server chunk_size).",
    )

    # Sharding over task ids for parallel multi-GPU runs.
    parser.add_argument("--start_idx", type=int, default=-1, help="Shard start task id (inclusive). -1 = 0.")
    parser.add_argument("--end_idx", type=int, default=-1, help="Shard end task id (exclusive). -1 = n_tasks.")
    parser.add_argument(
        "--task_classification_path",
        type=str,
        required=True,
        help="Path to LIBERO-plus libero/libero/benchmark/task_classification.json",
    )
    parser.add_argument(
        "--max_steps_override",
        type=int,
        default=-1,
        help="If > 0, override the per-suite max_steps.",
    )

    parser.add_argument("--eval_log_dir", type=str, default="outputs/sim_eval/libero_plus")
    parser.add_argument("--save_videos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--no_rotate_images",
        action="store_true",
        help="Disable 180 deg image rotation (only when training data is in raw LIBERO orientation).",
    )
    parser.add_argument(
        "--no_binarize_gripper",
        action="store_true",
        help="Disable sign-binarization of action[-1] for LIBERO env (-1=open / +1=close).",
    )
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s  %(levelname)-8s | %(message)s",
        datefmt="%m/%d [%H:%M:%S]",
        force=True,
    )
    np.random.seed(args.seed)

    eval_log_dir = Path(args.eval_log_dir)
    eval_log_dir.mkdir(parents=True, exist_ok=True)

    client = LiberoModelClient(
        host=args.host,
        port=args.port,
        rotate_images=not args.no_rotate_images,
        binarize_gripper=not args.no_binarize_gripper,
        replan_steps=args.replan_steps,
    )

    summary = evaluate_policy(args, client)
    LOGGER.info(
        "LIBERO-plus eval done. Suite=%s shard=[%d,%d) SR=%.4f (%d/%d)",
        summary["task_suite"],
        summary["start_idx"],
        summary["end_idx"],
        summary["overall_success_rate"],
        summary["total_successes"],
        summary["total_episodes"],
    )
    LOGGER.info(colored(f"Saved results under {args.eval_log_dir}", "green"))


if __name__ == "__main__":
    # headless GL backends; user can override via env vars before launching.
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
