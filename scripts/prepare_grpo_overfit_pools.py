#!/usr/bin/env python3
"""Build deterministic GRPO overfit pools from trajectories used by SFT."""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
TRAVELGYM_SOURCE_ROOT = ROOT / "environments" / "TravelGym"
for import_root in (SOURCE_ROOT, TRAVELGYM_SOURCE_ROOT, ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from travel_grpo.collection.task_pools import assert_task_pools_disjoint
from travel_grpo.collection.travel_canonical import canonical_hash

ENV_ORDER = (
    "travel22",
    "travel33",
    "travel44",
    "travel233",
    "travel333",
    "travel334",
    "travel444",
    "travel2222",
)
# The four-task diagnostic keeps the three shortest two-tool compositions and
# the travel233 three-tool representative used by the one-task diagnostic.
FOUR_TASK_ENVS = ("travel22", "travel33", "travel44", "travel233")
ONE_TASK_ENV = "travel233"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _trajectory_chars(record: dict[str, Any]) -> int:
    return sum(
        len(str(message.get("content", "")))
        + len(str(message.get("reasoning_content", "")))
        for message in record.get("messages", [])
    )


def build_manifest(
    *,
    source_manifest_path: Path,
    sft_train_path: Path,
    sft_audit_path: Path,
) -> dict[str, Any]:
    manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    train_records = _read_jsonl(sft_train_path)
    audits = json.loads(sft_audit_path.read_text(encoding="utf-8"))["records"]

    audit_by_hash: dict[str, dict[str, Any]] = {}
    for audit in audits:
        digest = str(audit.get("canonical_hash_after", ""))
        if digest:
            audit_by_hash[digest] = audit

    matched: list[tuple[int, dict[str, Any], dict[str, Any], int]] = []
    for train_index, record in enumerate(train_records):
        digest = canonical_hash(record)
        audit = audit_by_hash.get(digest)
        if audit is None:
            raise RuntimeError(f"SFT train record {train_index} has no canonical audit match")
        matched.append((train_index, record, audit, _trajectory_chars(record)))

    # One representative per task: prefer the most complete partial trajectory,
    # then the shorter transcript when the same task has multiple SFT records.
    by_task: dict[str, tuple[int, dict[str, Any], dict[str, Any], int]] = {}
    for item in matched:
        train_index, _record, audit, char_count = item
        metrics = audit.get("metrics") or {}
        score = float(metrics.get("correct_completion", 0.0) or 0.0)
        eligible = (
            audit.get("trajectory_class") == "partial_correct"
            and float(audit.get("sample_weight", 0.0) or 0.0) == 0.5
            and bool(metrics.get("metrics_available"))
            and not bool(metrics.get("completion_success"))
            and 0.0 < score < 1.0
            and audit.get("env_name") in ENV_ORDER
            and bool(audit.get("task_key"))
        )
        if not eligible:
            continue
        task_key = str(audit["task_key"])
        previous = by_task.get(task_key)
        rank = (-score, char_count, train_index)
        if previous is None:
            by_task[task_key] = item
        else:
            p_index, _p_record, p_audit, p_chars = previous
            p_score = float((p_audit.get("metrics") or {}).get("correct_completion", 0.0) or 0.0)
            if rank < (-p_score, p_chars, p_index):
                by_task[task_key] = item

    selected: list[tuple[int, dict[str, Any], dict[str, Any], int]] = []
    for env_name in ENV_ORDER:
        candidates = [item for item in by_task.values() if item[2].get("env_name") == env_name]
        if not candidates:
            raise RuntimeError(f"no eligible SFT partial-correct task for {env_name}")
        median_chars = statistics.median(item[3] for item in candidates)
        candidates.sort(key=lambda item: (abs(item[3] - median_chars), str(item[2]["task_key"])))
        selected.append(candidates[0])

    selected_by_env = {str(item[2]["env_name"]): item for item in selected}
    one = selected_by_env[ONE_TASK_ENV]
    selected_four = [selected_by_env[env_name] for env_name in FOUR_TASK_ENVS]

    sft_pool_records = manifest.get("pools", {}).get("sft", {}).get("records", [])
    formal_sft_by_key = {str(record.get("task_key")): record for record in sft_pool_records}
    selected_keys = [str(item[2]["task_key"]) for item in selected]
    missing = [key for key in selected_keys if key not in formal_sft_by_key]
    if missing:
        raise RuntimeError(f"selected task(s) are outside the formal SFT pool: {missing}")

    one_key = str(one[2]["task_key"])
    four_keys = [str(item[2]["task_key"]) for item in selected_four]
    manifest = copy.deepcopy(manifest)
    manifest["name"] = "travelgym-grpo-overfit-pools"
    manifest["pools"]["grpo_overfit_one"] = {
        "pool": "grpo_overfit_one",
        "parent_pool": "sft",
        "split": "train",
        "purpose": "one-task GRPO overfit diagnostic using a task seen by SFT",
        "records": [copy.deepcopy(formal_sft_by_key[one_key])],
    }
    manifest["pools"]["grpo_overfit_four"] = {
        "pool": "grpo_overfit_four",
        "parent_pool": "sft",
        "split": "train",
        "purpose": (
            "four-task GRPO overfit diagnostic using the three shortest "
            "two-tool compositions plus the travel233 one-task representative"
        ),
        "records": [copy.deepcopy(formal_sft_by_key[key]) for key in four_keys],
    }
    manifest.setdefault("pool_task_counts", {})["grpo_overfit_one"] = 1
    manifest["pool_task_counts"]["grpo_overfit_four"] = 4
    manifest["grpo_overfit_selection"] = {
        "schema_version": "travelgym-grpo-overfit-selection-v2",
        "source_sft_train": str(sft_train_path.relative_to(ROOT)),
        "source_sft_audit": str(sft_audit_path.relative_to(ROOT)),
        "policy": (
            "actual SFT train split only; partial_correct with weight 0.5 and 0 < correct_completion < 1; "
            "one median-length representative per environment; four-task diagnostic keeps travel22, travel33, "
            "travel44, and travel233; one-task diagnostic uses travel233"
        ),
        "one_task_key": one_key,
        "four_task_keys": four_keys,
        "records": [
            {
                "env_name": item[2]["env_name"],
                "task_id": item[2]["task_id"],
                "task_key": item[2]["task_key"],
                "sft_train_index": item[0],
                "sft_source": item[2].get("source"),
                "sft_source_index": (item[2].get("source_meta") or {}).get("source_index"),
                "trajectory_class": item[2].get("trajectory_class"),
                "sample_weight": item[2].get("sample_weight"),
                "correct_completion": (item[2].get("metrics") or {}).get("correct_completion"),
                "completion_success": (item[2].get("metrics") or {}).get("completion_success"),
                "trajectory_char_count": item[3],
            }
            for item in selected_four
        ],
    }

    # Existing formal SFT/GRPO/Validation isolation must remain intact.
    assert_task_pools_disjoint(manifest, require_strict=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=ROOT / "data/task_pools/travel_task_pools.json",
    )
    parser.add_argument(
        "--sft-train",
        type=Path,
        default=ROOT / "data/sft/travel_sft_qwen35_split/train.jsonl",
    )
    parser.add_argument(
        "--sft-audit",
        type=Path,
        default=ROOT / "data/sft/travel_sft_qwen35_merged.audit.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/task_pools/travel_grpo_overfit_pools.json",
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    result = build_manifest(
        source_manifest_path=args.source_manifest.resolve(),
        sft_train_path=args.sft_train.resolve(),
        sft_audit_path=args.sft_audit.resolve(),
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    output = args.output.resolve()
    if args.check:
        if not output.is_file() or output.read_text(encoding="utf-8") != rendered:
            raise SystemExit(f"stale or missing generated manifest: {output}")
        print(f"OK: {output}")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(output)
    print(f"wrote {output}")
    print(f"one={result['grpo_overfit_selection']['one_task_key']}")
    for key in result["grpo_overfit_selection"]["four_task_keys"]:
        print(f"four={key}")


if __name__ == "__main__":
    main()
