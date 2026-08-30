"""Create a secret-free summary for a DeepSeek Teacher collection cache.

The cache keeps private trajectories for later canonicalisation.  This
report intentionally contains only aggregate counts, labels, and protocol
diagnostics; it never copies task IDs, preference IDs, candidate attributes,
reasoning text, or credentials.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from teacher_collection import code_revision, summarize_pass_stats  # noqa: E402
from sft.clean_travel_trajectories import clean_trajectory  # noqa: E402
from sft.travel_task_resolver import TravelTaskResolver  # noqa: E402


_PRIVATE_MARKERS = (
    "preference_id",
    "correct_by_aspect",
    "correct_ids",
    "best_id",
    "reward_report",
    "preference_coverage",
)


def _quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"p50": None, "p90": None, "p95": None, "p99": None, "max": None}
    ordered = sorted(values)
    if len(ordered) == 1:
        return {"p50": ordered[0], "p90": ordered[0], "p95": ordered[0], "p99": ordered[0], "max": ordered[0]}
    quantile_values = statistics.quantiles(ordered, n=100, method="inclusive")
    return {
        "p50": quantile_values[49],
        "p90": quantile_values[89],
        "p95": quantile_values[94],
        "p99": quantile_values[98],
        "max": ordered[-1],
    }


def _pool_task_keys(manifest: dict[str, Any], pool_name: str) -> set[str]:
    pool = manifest.get("pools", {}).get(pool_name, {})
    return {
        f"{row.get('env_name')}::{row.get('task_id')}"
        for row in pool.get("records", [])
        if isinstance(row, dict) and row.get("env_name") and row.get("task_id")
    }


def build_summary(cache_path: Path, task_pool_path: Path | None = None) -> dict[str, Any]:
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    records = payload.get("records", {})
    if not isinstance(records, dict):
        raise ValueError("Teacher cache records must be an object")
    values = [value for value in records.values() if isinstance(value, dict)]
    status_counts = Counter(str(value.get("status", "unknown")) for value in values)
    provenance = [value.get("provenance", {}) for value in values if isinstance(value.get("provenance"), dict)]
    first = provenance[0] if provenance else {}
    task_keys_seen = {
        str(item.get("task_key"))
        for item in provenance
        if item.get("task_key")
    }
    complete_task_keys = {
        str(item.get("task_key"))
        for item in provenance
        if item.get("task_key")
        and all(
            any(
                value.get("status") == "success"
                and value.get("provenance", {}).get("task_key") == item.get("task_key")
                and int(value.get("provenance", {}).get("pass_index", -1)) == pass_index
                for value in values
            )
            for pass_index in (0, 1)
        )
    }

    pass_stats = summarize_pass_stats(records)
    token_quantiles: dict[str, dict[str, float | None]] = {}
    elapsed_quantiles: dict[str, dict[str, float | None]] = {}
    for status in ("success", "invalid", "abandoned", "in_flight"):
        token_quantiles[status] = _quantiles(
            [float(value.get("token_count", 0) or 0) for value in values if value.get("status") == status]
        )
        elapsed_quantiles[status] = _quantiles(
            [float(value.get("elapsed_seconds", 0) or 0) for value in values if value.get("status") == status]
        )

    resolver = TravelTaskResolver(project_root=ROOT)
    classification = Counter()
    fatal_reasons = Counter()
    assistant_messages = 0
    missing_think = 0
    tool_call_blocks = 0
    private_markers = 0
    for value in values:
        if value.get("status") in {"in_flight", "abandoned"}:
            continue
        trajectory = value.get("trajectory", {})
        messages = trajectory.get("messages", []) if isinstance(trajectory, dict) else []
        for message in messages:
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            assistant_messages += 1
            content = str(message.get("content", "") or "")
            if not re.search(r"<think>\s*.*?\s*</think>", content, re.S):
                missing_think += 1
            if "<tool_call>" in content:
                tool_call_blocks += 1
            if any(marker in content for marker in _PRIVATE_MARKERS):
                private_markers += 1
        task_id = value.get("provenance", {}).get("task_id")
        spec = resolver.tasks.get(str(task_id)) if task_id else None
        cleaned, audit = clean_trajectory(
            trajectory,
            task=spec,
            task_id=str(task_id) if task_id else None,
            source="teacher-smoke-summary",
            require_think=True,
        )
        classification[f"{value.get('status')}::{cleaned.get('trainer_metadata', {}).get('trajectory_class', 'unknown')}"] += 1
        if audit.get("fatal_kind"):
            fatal_reasons[str(audit["fatal_kind"])] += 1

    requested_task_keys: set[str] = set()
    if task_pool_path and task_pool_path.is_file():
        pool_manifest = json.loads(task_pool_path.read_text(encoding="utf-8"))
        requested_task_keys = _pool_task_keys(pool_manifest, "sft_smoke")

    unique_values = {str(value) for value in task_keys_seen}
    recorded_code_revision = first.get("code_revision")
    resolved_code_revision = (
        recorded_code_revision
        if recorded_code_revision and str(recorded_code_revision).lower() != "unknown"
        else code_revision()
    )
    summary = {
        "schema_version": "travelgym-teacher-smoke-summary-v1",
        "cache": str(cache_path.resolve()),
        "collection_run_id": payload.get("collection_run_id"),
        "model": first.get("model"),
        "task_pool_hash": first.get("task_pool_hash"),
        "api_endpoint_label": first.get("api_endpoint_label"),
        "thinking": first.get("thinking"),
        "reasoning_effort": first.get("reasoning_effort"),
        "max_turns": first.get("max_turns"),
        "pass_k": 2,
        "code_revision": resolved_code_revision,
        "provenance_code_revision": recorded_code_revision,
        "code_revision_reconciled": resolved_code_revision != recorded_code_revision,
        "requested_task_count": len(requested_task_keys) or None,
        "seen_task_count": len(unique_values),
        "completed_task_count": len(complete_task_keys),
        "unseen_task_count": len(requested_task_keys - task_keys_seen) if requested_task_keys else None,
        "status_counts": dict(sorted(status_counts.items())),
        "cache_stats": payload.get("stats", {}),
        "pass_stats": pass_stats,
        "token_count_quantiles": token_quantiles,
        "elapsed_seconds_quantiles": elapsed_quantiles,
        "classification": dict(sorted(classification.items())),
        "fatal_reasons": dict(sorted(fatal_reasons.items())),
        "reasoning_integrity": {
            "assistant_messages_checked": assistant_messages,
            "missing_think_blocks": missing_think,
            "tool_call_blocks": tool_call_blocks,
            "private_marker_messages": private_markers,
        },
        "manual_stop": True,
        "full_collection_started": False,
        "note": "One in-flight pass was intentionally left for explicit reconciliation; do not retry it automatically.",
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--task-pool", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    summary = build_summary(Path(args.cache), Path(args.task_pool) if args.task_pool else None)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
