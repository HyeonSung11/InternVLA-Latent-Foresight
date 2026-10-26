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
):
    env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
    n_episodes = min(args.num_trials_per_task, len(initial_states))
    successes: list[bool] = []

    for episode_idx in tqdm(range(n_episodes), desc=f"task: {task_description}", leave=False):
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
            seg = task_description.replace(" ", "_")
            out_path = video_dir / f"rollout_{seg}_episode{episode_idx}_{suffix}.mp4"
            imageio.mimwrite(out_path, [np.asarray(x) for x in replay_images], fps=10)

    env.close()
    return successes, task_description


def evaluate_policy(args: argparse.Namespace, client: LiberoModelClient) -> dict:
    from libero.libero import benchmark

    benchmark_dict = benchmark.get_benchmark_dict()
    if args.task_suite_name not in benchmark_dict:
        raise KeyError(
            f"Unknown task_suite_name '{args.task_suite_name}'. Available: {list(benchmark_dict.keys())}"
        )
    task_suite = benchmark_dict[args.task_suite_name]()
    n_tasks_in_suite = task_suite.n_tasks
    n_tasks = n_tasks_in_suite if args.max_tasks <= 0 else min(args.max_tasks, n_tasks_in_suite)

    start_idx = max(0, args.start_idx) if args.start_idx >= 0 else 0
    end_idx = args.end_idx if args.end_idx >= 0 else n_tasks
    end_idx = min(end_idx, n_tasks)
    if start_idx >= end_idx:
        raise ValueError(
            f"Empty shard: start_idx={start_idx} end_idx={end_idx} (n_tasks={n_tasks})"
        )
    sharded = not (start_idx == 0 and end_idx == n_tasks)
    LOGGER.info(
        "Task suite: %s | evaluating tasks [%d, %d) of %d (max_tasks=%d)",
        args.task_suite_name, start_idx, end_idx, n_tasks_in_suite, args.max_tasks,
    )

    max_steps = TASK_SUITE_MAX_STEPS[args.task_suite_name]

    eval_log_dir = Path(args.eval_log_dir)
    video_dir = eval_log_dir / "videos" / args.task_suite_name
    if args.save_videos:
        video_dir.mkdir(parents=True, exist_ok=True)

    per_task: dict[str, dict] = {}
    total_episodes = 0
    total_successes = 0

    for task_id in tqdm(range(start_idx, end_idx), desc=args.task_suite_name):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        successes, task_desc = evaluate_task(
            task, initial_states, args, client, max_steps, video_dir,
        )
        sr = float(np.mean(successes)) if successes else 0.0
        per_task[task_desc] = {
            "n_episodes": len(successes),
            "successes": int(sum(successes)),
            "success_rate": sr,
        }
        total_episodes += len(successes)
        total_successes += int(sum(successes))
        LOGGER.info(
            "Task '%s' SR=%.2f%% (%d/%d)",
            task_desc, sr * 100.0, sum(successes), len(successes),
        )

    overall = total_successes / max(total_episodes, 1)
    summary = {
        "task_suite": args.task_suite_name,
        "start_idx": start_idx,
        "end_idx": end_idx,
        "n_tasks": end_idx - start_idx,
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "overall_success_rate": overall,
        "per_task": per_task,
    }

    if sharded:
        out_json = eval_log_dir / f"results_{args.task_suite_name}_{start_idx}_{end_idx}.json"
    else:
        out_json = eval_log_dir / f"results_{args.task_suite_name}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    LOGGER.info("Saved per-suite results to %s", out_json)

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LIBERO evaluation with websocket policy server")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5694)

    parser.add_argument(
        "--task_suite_name",
        type=str,
        default="libero_goal",
        choices=list(TASK_SUITE_MAX_STEPS.keys()),
    )
    parser.add_argument("--num_trials_per_task", type=int, default=50)
    parser.add_argument("--max_tasks", type=int, default=-1, help="If > 0, limit number of tasks (smoke test).")
    parser.add_argument("--start_idx", type=int, default=-1, help="Shard start task id (inclusive). -1 = 0.")
    parser.add_argument("--end_idx", type=int, default=-1, help="Shard end task id (exclusive). -1 = n_tasks.")
    parser.add_argument("--num_steps_wait", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--replan_steps",
        type=int,
        default=8,
        help="Re-request a new action chunk every N env steps (must be <= server chunk_size).",
    )

    parser.add_argument("--eval_log_dir", type=str, default="outputs/sim_eval/libero")
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
        "LIBERO eval done. Suite=%s SR=%.4f (%d/%d)",
        summary["task_suite"],
        summary["overall_success_rate"],
        summary["total_successes"],
        summary["total_episodes"],
    )
    LOGGER.info(colored(f"Saved results to {args.eval_log_dir}", "green"))


if __name__ == "__main__":
    # headless GL backends; user can override via env vars before launching.
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
