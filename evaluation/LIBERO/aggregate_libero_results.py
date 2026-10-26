from __future__ import annotations

import argparse
import glob
import json
import os

# Standard LIBERO suites in leaderboard order.
TASK_SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"]


def _rate(succ: int, total: int) -> float:
    return float(succ) / float(total) if total else 0.0


def aggregate(root_path: str) -> dict:
    """Merge all results_{suite}*.json shards under root_path into per-suite and
    overall success rates.

    Each shard file is written by eval_libero_server_client.py and holds a
    `per_task` mapping {task_desc: {n_episodes, successes, success_rate}}. Tasks
    are uniquely keyed by description, so shards of the same suite merge cleanly.
    """
    per_suite: dict[str, dict] = {}
    grand_succ = 0
    grand_eps = 0

    for suite in TASK_SUITES:
        candidates = glob.glob(
            os.path.join(root_path, "**", f"results_{suite}*.json"), recursive=True
        )
        per_task: dict[str, dict] = {}
        for file in candidates:
            base = os.path.basename(file)[len("results_"):-len(".json")]
            # base is either "{suite}" or "{suite}_{start}_{end}"; the suite token
            # must match exactly to avoid e.g. libero_10 picking up libero_100.
            if base != suite and not base.startswith(f"{suite}_"):
                continue
            with open(file, encoding="utf-8") as f:
                data = json.load(f)
            for task_desc, counts in data.get("per_task", {}).items():
                bucket = per_task.setdefault(task_desc, {"n_episodes": 0, "successes": 0})
                bucket["n_episodes"] += int(counts.get("n_episodes", 0))
                bucket["successes"] += int(counts.get("successes", 0))

        if not per_task:
            continue

        for counts in per_task.values():
            counts["success_rate"] = _rate(counts["successes"], counts["n_episodes"])
        suite_succ = sum(c["successes"] for c in per_task.values())
        suite_eps = sum(c["n_episodes"] for c in per_task.values())
        per_suite[suite] = {
            "n_tasks": len(per_task),
            "total_episodes": suite_eps,
            "total_successes": suite_succ,
            "success_rate": _rate(suite_succ, suite_eps),
            "per_task": per_task,
        }
        grand_succ += suite_succ
        grand_eps += suite_eps

    return {
        "overall": {
            "total_episodes": grand_eps,
            "total_successes": grand_succ,
            "success_rate": _rate(grand_succ, grand_eps),
        },
        "per_suite": per_suite,
    }


def _print_table(result: dict) -> None:
    print("\n| Suite           | Tasks |   Eps | Succ |    SR |")
    print("|-----------------|-------|-------|------|-------|")
    for suite, s in result["per_suite"].items():
        print(
            f"| {suite:<15} | {s['n_tasks']:>5} | {s['total_episodes']:>5} | "
            f"{s['total_successes']:>4} | {100.0 * s['success_rate']:>4.1f} |"
        )
    o = result["overall"]
    print(
        f"| {'OVERALL':<15} | {'':>5} | {o['total_episodes']:>5} | "
        f"{o['total_successes']:>4} | {100.0 * o['success_rate']:>4.1f} |"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate per-(suite,task) LIBERO eval shards")
    parser.add_argument("--root", required=True, help="eval_log_dir containing results_{suite}*.json")
    args = parser.parse_args()

    result = aggregate(args.root)
    out_path = os.path.join(args.root, "overall_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    _print_table(result)
    print(f"\nSaved aggregated results to {out_path}")


if __name__ == "__main__":
    main()
