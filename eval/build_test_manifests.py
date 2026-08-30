"""Build the fixed TravelGym evaluation task manifests.

The active evaluator has 471 test tasks spread across eight composition
variants.  This script makes two smaller, reproducible sets without changing
the training data or the TravelGym interaction protocol:

* ``smoke20.json``: 20 tasks, three from each of the four larger variants and
  two from each of the other four variants;
* ``final200.json``: 200 tasks allocated proportionally to the source split.

Tasks are ranked deterministically from ``seed + env_name + task_id`` instead
of relying on parquet row order.  The smoke set is the prefix of the same
per-variant ranking used by the final set, so every smoke task is also in the
final set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


SUPPORTED_ENVS = (
    "travel22",
    "travel33",
    "travel44",
    "travel233",
    "travel333",
    "travel334",
    "travel444",
    "travel2222",
)

# Keep this value stable.  Changing it intentionally produces a new test-set
# revision and should be reflected in the manifest name/version.
SELECTION_SEED = 20260801

# Balanced smoke coverage: every composition appears at least twice.
SMOKE_QUOTAS = {
    "travel22": 3,
    "travel33": 3,
    "travel44": 3,
    "travel233": 3,
    "travel333": 2,
    "travel334": 2,
    "travel444": 2,
    "travel2222": 2,
}

# Largest-remainder proportional allocation for the 200-task final
# Validation-Task-Pool.  The first 20 ranked rows of every composition are the
# existing smoke20 selection, so smoke20 remains a subset of this manifest.
FINAL200_QUOTAS = {
    "travel22": 43,
    "travel33": 37,
    "travel44": 28,
    "travel233": 26,
    "travel333": 23,
    "travel334": 20,
    "travel444": 16,
    "travel2222": 7,
}


def _task_rank(seed: int, env_name: str, task_id: str) -> str:
    payload = f"{seed}:{env_name}:{task_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_source_tasks(project_root: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    """Load task IDs and source metadata from each test parquet."""
    tasks: dict[str, list[dict[str, Any]]] = {}
    sources: dict[str, dict[str, Any]] = {}
    for env_name in SUPPORTED_ENVS:
        path = project_root / "data" / f"{env_name}_multiturn_onechoice" / "test.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"missing TravelGym test split: {path}")
        frame = pd.read_parquet(path)
        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for source_index, value in enumerate(frame["reward_model"]):
            if not isinstance(value, dict) or not value.get("id"):
                raise ValueError(f"{path} row {source_index} has no reward_model.id")
            task_id = str(value["id"])
            if task_id in seen:
                raise ValueError(f"duplicate task ID {task_id!r} in {path}")
            seen.add(task_id)
            rows.append({
                "env_name": env_name,
                "task_id": task_id,
                "source_index": source_index,
                "rank": _task_rank(SELECTION_SEED, env_name, task_id),
            })
        rows.sort(key=lambda item: item["rank"])
        tasks[env_name] = rows
        sources[env_name] = {
            "path": str(path.relative_to(project_root)).replace("\\", "/"),
            "task_count": len(rows),
        }
    return tasks, sources


def _select(tasks: dict[str, list[dict[str, Any]]], quotas: dict[str, int]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for env_name in SUPPORTED_ENVS:
        quota = int(quotas[env_name])
        available = tasks[env_name]
        if quota <= 0 or quota > len(available):
            raise ValueError(f"invalid quota {quota} for {env_name}; available={len(available)}")
        selected.extend(available[:quota])
    # Manifest order is stable and human-auditable: composition first, then
    # deterministic hash rank within that composition.
    selected.sort(key=lambda item: (SUPPORTED_ENVS.index(item["env_name"]), item["rank"]))
    return [
        {
            "env_name": item["env_name"],
            "task_id": item["task_id"],
            "source_index": item["source_index"],
        }
        for item in selected
    ]


def _make_manifest(
    *,
    project_root: Path,
    name: str,
    quotas: dict[str, int],
    tasks: dict[str, list[dict[str, Any]]],
    sources: dict[str, dict[str, Any]],
    source_name: str,
) -> dict[str, Any]:
    records = _select(tasks, quotas)
    return {
        "manifest_version": "travelgym-test-set-v1",
        "name": name,
        "split": "test",
        "dataset_variant": "multiturn_onechoice",
        "selection_strategy": "stratified_hash_v1",
        "selection_seed": SELECTION_SEED,
        "source_task_count": sum(len(items) for items in tasks.values()),
        "task_count": len(records),
        "source_name": source_name,
        "source_env_counts": {env: len(tasks[env]) for env in SUPPORTED_ENVS},
        "selected_env_counts": {env: int(quotas[env]) for env in SUPPORTED_ENVS},
        "source_files": sources,
        "records": records,
    }


def build(project_root: Path, output_dir: Path) -> tuple[Path, Path]:
    tasks, sources = load_source_tasks(project_root)
    if sum(SMOKE_QUOTAS.values()) != 20:
        raise AssertionError("smoke quotas must total 20")
    if sum(FINAL200_QUOTAS.values()) != 200:
        raise AssertionError("final200 quotas must total 200")
    final200 = _make_manifest(
        project_root=project_root,
        name="travelgym-final200",
        quotas=FINAL200_QUOTAS,
        tasks=tasks,
        sources=sources,
        source_name="all eight TravelGym one-choice test splits (471 tasks)",
    )
    smoke = _make_manifest(
        project_root=project_root,
        name="travelgym-smoke20",
        quotas=SMOKE_QUOTAS,
        tasks=tasks,
        sources=sources,
        source_name="travelgym-final200 (fixed subset)",
    )
    final_ids = {(item["env_name"], item["task_id"]) for item in final200["records"]}
    smoke_ids = {(item["env_name"], item["task_id"]) for item in smoke["records"]}
    if not smoke_ids.issubset(final_ids):
        raise AssertionError("smoke20 must be a subset of final200")

    output_dir.mkdir(parents=True, exist_ok=True)
    smoke_path = output_dir / "smoke20.json"
    final200_path = output_dir / "final200.json"
    for path, manifest in ((smoke_path, smoke), (final200_path, final200)):
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return smoke_path, final200_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="TravelGym project root (defaults to the repository root).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "test_manifests",
        help="Directory receiving smoke20.json and final200.json.",
    )
    args = parser.parse_args()
    smoke_path, final_path = build(args.project_root.resolve(), args.output_dir.resolve())
    print(f"wrote {smoke_path}")
    print(f"wrote {final_path}")


if __name__ == "__main__":
    main()
