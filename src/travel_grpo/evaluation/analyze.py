"""Summarize TravelGym-only evaluation caches.

Each cache stores terminal rewards under one or more TravelGym dataset
variants.  The analyzer intentionally ignores protocol manifests and does
not aggregate legacy per-step rewards from removed environments.
"""

from __future__ import annotations

import json
from pathlib import Path

from .._paths import REPOSITORY_ROOT

# Evaluation outputs have one canonical runtime location, separate from the
# source package and from training experiment directories.
OUTPUT_DIR = REPOSITORY_ROOT / "outputs" / "evaluation"


def summarize_cache(path: Path) -> dict[str, float | int | str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    scores: list[float] = []
    valid = 0
    total = 0
    for env_name, task_entries in payload.items():
        if str(env_name).startswith("_"):
            continue
        if not str(env_name).startswith("travel"):
            raise ValueError(f"Non-TravelGym cache entry found: {env_name!r}")
        if not isinstance(task_entries, dict):
            continue
        for entry in task_entries.values():
            rewards = entry.get("reward", []) if isinstance(entry, dict) else []
            reports = entry.get("reward_report", []) if isinstance(entry, dict) else []
            if not isinstance(rewards, list):
                rewards = [rewards]
            scores.extend(float(value) for value in rewards if isinstance(value, (int, float)))
            total += len(rewards)
            if isinstance(reports, list):
                valid += sum(
                    int(bool(report.get("reward_valid", False)))
                    for report in reports
                    if isinstance(report, dict)
                )
    return {
        "micro_avg": sum(scores) / len(scores) if scores else 0.0,
        "terminal_max": max(scores) if scores else 0.0,
        "valid_rollout_rate": valid / total if total else 0.0,
        "rollout_count": total,
    }


def main() -> None:
    if not OUTPUT_DIR.exists():
        raise FileNotFoundError(f"Evaluation output directory not found: {OUTPUT_DIR}")
    summaries = {}
    for path in sorted(OUTPUT_DIR.glob("*_reward_cache.json")):
        summaries[path.stem.removesuffix("_reward_cache")] = summarize_cache(path)
    print("TravelGym terminal reward ranking:")
    for name, summary in sorted(
        summaries.items(), key=lambda item: item[1]["micro_avg"], reverse=True
    ):
        print(
            f"{name}: micro_avg={summary['micro_avg']:.4f}, "
            f"terminal_max={summary['terminal_max']:.4f}, "
            f"valid_rate={summary['valid_rollout_rate']:.3f}, "
            f"rollouts={summary['rollout_count']}"
        )


if __name__ == "__main__":
    main()
