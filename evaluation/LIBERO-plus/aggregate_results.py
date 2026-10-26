from __future__ import annotations

import argparse
import glob
import json
import os

# 4 evaluated suites and the 7 perturbation categories, in leaderboard order.
TASK_SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
CATEGORY_ORDER = [
    "Camera Viewpoints",
    "Robot Initial States",
    "Language Instructions",
    "Light Conditions",
    "Background Textures",
    "Sensor Noise",
    "Objects Layout",
]


def _merge(into: dict, item: str, counts: dict) -> None:
    bucket = into.setdefault(item, {"total_count": 0, "success_count": 0})
    bucket["total_count"] += counts.get("total_count", 0)
    bucket["success_count"] += counts.get("success_count", 0)


def _rate(counts: dict) -> float:
    total = counts.get("total_count", 0)
    return float(counts.get("success_count", 0)) / float(total) if total else 0.0


def aggregate(root_path: str) -> dict:
    # category -> counts (across all suites); also per-suite breakdown.
    overall: dict[str, dict] = {}
    per_suite: dict[str, dict] = {}
    grand = {"total_count": 0, "success_count": 0}

    for suite in TASK_SUITES:
        suite_dir = os.path.join(root_path, "logs", suite)
        json_files = glob.glob(os.path.join(suite_dir, "*.json"))
        if not json_files:
            continue
        suite_cat: dict[str, dict] = {}
        for file in json_files:
            with open(file, encoding="utf-8") as f:
                results = json.load(f)
            for category, counts in results.items():
                _merge(overall, category, counts)
                _merge(suite_cat, category, counts)
                grand["total_count"] += counts.get("total_count", 0)
                grand["success_count"] += counts.get("success_count", 0)
        for counts in suite_cat.values():
            counts["success_rate"] = _rate(counts)
        suite_total = {
            "total_count": sum(c["total_count"] for c in suite_cat.values()),
            "success_count": sum(c["success_count"] for c in suite_cat.values()),
        }
        suite_total["success_rate"] = _rate(suite_total)
        per_suite[suite] = {"per_category": suite_cat, "total": suite_total}

    for counts in overall.values():
        counts["success_rate"] = _rate(counts)
    grand["success_rate"] = _rate(grand)

    # Leaderboard-style summary row (percentages), matching the README table columns.
    summary_row = {cat: round(100.0 * _rate(overall.get(cat, {})), 1) for cat in CATEGORY_ORDER}
    summary_row["Total"] = round(100.0 * grand["success_rate"], 1)

    return {
        "overall": {**grand, "per_category": overall},
        "per_suite": per_suite,
        "leaderboard_summary_percent": summary_row,
    }


def _print_table(result: dict) -> None:
    row = result["leaderboard_summary_percent"]
    cols = ["Camera", "Robot", "Language", "Light", "Background", "Noise", "Layout", "Total"]
    keys = CATEGORY_ORDER + ["Total"]
    print("\n| " + " | ".join(cols) + " |")
    print("|" + "|".join(["-------"] * len(cols)) + "|")
    print("| " + " | ".join(f"{row[k]:>5}" for k in keys) + " |")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate LIBERO-plus per-shard results")
    parser.add_argument("--root", required=True, help="eval_log_dir containing logs/{suite}/*.json")
    args = parser.parse_args()

    result = aggregate(args.root)
    out_path = os.path.join(args.root, "overall_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    _print_table(result)
    print(f"\nSaved aggregated results to {out_path}")


if __name__ == "__main__":
    main()
