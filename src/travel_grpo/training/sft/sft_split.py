"""Task-grouped SFT/validation split for the merged canonical corpus.

This module is deliberately independent of torch.  It can therefore be used
in CI to verify pool isolation before a tokenizer/model server is available.
The production CLI uses the Qwen3.5-4B native chat template for the exact
32,768-token audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from ...collection.clean_travel_trajectories import is_sft_eligible
from ...collection.task_pools import TASK_POOL_SELECTION_SEED, TaskPoolError, load_pool_manifest, pool_task_keys


class SFTSplitError(ValueError):
    pass


# Keep gold10 selection aligned with the deterministic task-pool seed.  The
# seed is written to split_manifest.json so a later merge cannot silently
# change the validation task groups.
SFT_SPLIT_SELECTION_SEED = TASK_POOL_SELECTION_SEED


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SFTSplitError(f"invalid JSONL at {path}:{line_number}") from exc
        if not isinstance(value, Mapping):
            raise SFTSplitError(f"non-object row at {path}:{line_number}")
        rows.append(dict(value))
    return rows


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(dict(row), ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    temporary.replace(path)


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def _length_bucket(length: int, all_lengths: list[int]) -> str:
    if not all_lengths:
        return "mid"
    ordered = sorted(all_lengths)
    short_cut = ordered[max(0, (len(ordered) - 1) // 3)]
    long_cut = ordered[min(len(ordered) - 1, (2 * len(ordered)) // 3)]
    if length <= short_cut:
        return "short"
    if length >= long_cut:
        return "long"
    return "mid"


def _task_key_for(index: int, record: Mapping[str, Any], audits: list[Mapping[str, Any]]) -> str | None:
    if index < len(audits):
        value = audits[index].get("task_key")
        if value:
            return str(value)
    value = record.get("task_key") or record.get("trainer_metadata", {}).get("task_key")
    return str(value) if value else None


def _env_from_task_key(task_key: str) -> str:
    return str(task_key).split("::", 1)[0]


def select_validation_tasks(
    records: list[Mapping[str, Any]],
    audits: list[Mapping[str, Any]],
    *,
    count: int = 10,
    token_length_fn: Callable[[Mapping[str, Any]], int] | None = None,
    required_envs: Iterable[str] | None = None,
    selection_seed: int = SFT_SPLIT_SELECTION_SEED,
) -> tuple[list[str], dict[str, Any]]:
    """Select exactly ``count`` strict_gold task groups deterministically."""

    if count < 1:
        raise SFTSplitError("validation task count must be positive")
    default_envs = ("travel22", "travel33", "travel44", "travel233", "travel333", "travel334", "travel444", "travel2222")
    # The formal contract selects gold10 from *all* strict_gold groups.  When
    # callers do not explicitly require a coverage list, an environment with
    # no strict_gold group is recorded as absent rather than making the split
    # impossible.  Explicit ``required_envs`` remains a strict audit mode.
    strict_required_envs = required_envs is not None
    required = list(required_envs) if strict_required_envs else list(default_envs)
    groups: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        metadata = record.get("trainer_metadata", {}) if isinstance(record, Mapping) else {}
        if metadata.get("trajectory_class") != "strict_gold":
            continue
        if not bool(metadata.get("sample_weight", 1.0)):
            continue
        key = _task_key_for(index, record, audits)
        if key:
            groups.setdefault(key, []).append(index)
    if not groups:
        raise SFTSplitError("no strict_gold task groups with private task identities")
    lengths = [int(token_length_fn(records[i])) if token_length_fn else len(records[i].get("messages", [])) for i in range(len(records))]
    env_groups: dict[str, list[str]] = {env: [] for env in required}
    for key in groups:
        env_groups.setdefault(_env_from_task_key(key), []).append(key)
    missing = [env for env in required if not env_groups.get(env)]
    if missing and strict_required_envs:
        raise SFTSplitError(f"strict_gold pool cannot cover environments: {missing}")
    if missing:
        required = [env for env in required if env_groups.get(env)]
    selected: list[str] = []
    # First guarantee all eight environment variants.  A stable hash, rather
    # than source order, keeps the split reproducible after cache merging.
    for env in required:
        candidates = sorted(env_groups[env], key=lambda key: _hash(("env", selection_seed, env, key)))
        selected.append(candidates[0])
    if len(selected) > count:
        raise SFTSplitError(f"validation count {count} is smaller than required environments {len(selected)}")
    remaining = [key for key in groups if key not in selected]
    all_group_lengths = [max(lengths[i] for i in indices) for indices in groups.values()]
    buckets = {_length_bucket(max(lengths[i] for i in groups[key]), all_group_lengths) for key in selected}
    # Prefer one short/mid/long group in the first eight selections when the
    # deterministic environment choice happens to miss a bucket.
    for bucket in ("short", "mid", "long"):
        if bucket in buckets or len(selected) >= count:
            continue
        bucket_candidates = [
            key for key in remaining
            if _length_bucket(max(lengths[i] for i in groups[key]), all_group_lengths) == bucket
        ]
        candidate = min(bucket_candidates, key=lambda key: _hash(("bucket", selection_seed, bucket, key))) if bucket_candidates else None
        if candidate:
            selected.append(candidate)
            remaining.remove(candidate)
            buckets.add(bucket)
    for key in sorted(remaining, key=lambda value: _hash(("extra", selection_seed, value))):
        if len(selected) >= count:
            break
        selected.append(key)
    if len(selected) != count:
        raise SFTSplitError(f"only {len(selected)} strict_gold task groups available; need {count}")
    selected = sorted(selected, key=lambda key: (_env_from_task_key(key), _hash(key)))
    return selected, {
        "required_envs": required,
        "selection_seed": int(selection_seed),
        "selected_task_keys": selected,
        "selected_envs": sorted({_env_from_task_key(key) for key in selected}),
        "length_buckets": {
            key: _length_bucket(max(lengths[i] for i in groups[key]), all_group_lengths)
            for key in selected
        },
        "strict_gold_group_count": len(groups),
    }


def build_sft_split(
    records: list[Mapping[str, Any]],
    audits: list[Mapping[str, Any]],
    *,
    output_dir: str | Path,
    task_pool_manifest: Mapping[str, Any] | None = None,
    tokenizer_name: str | None = None,
    max_length: int = 32768,
    token_length_fn: Callable[[Mapping[str, Any]], int] | None = None,
    require_exact_token_audit: bool = True,
    validation_count: int = 10,
    selection_seed: int = SFT_SPLIT_SELECTION_SEED,
) -> dict[str, Any]:
    """Write train/val JSONL plus a private split manifest."""

    if require_exact_token_audit and token_length_fn is None and not tokenizer_name:
        raise SFTSplitError("formal SFT split requires the Qwen3.5-4B tokenizer for exact token audit")
    if tokenizer_name and token_length_fn is None:
        try:
            from transformers import AutoTokenizer
            from .qwen35_mask import native_template_ids
        except ImportError as exc:  # pragma: no cover - production dependency
            raise SFTSplitError("tokenizer audit requires transformers and qwen35_mask") from exc
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        def token_length_fn(record: Mapping[str, Any]) -> int:
            return len(native_template_ids(tokenizer, record["messages"], tools=record.get("tools"), enable_thinking=True, add_generation_prompt=False))
    lengths = [int(token_length_fn(record)) if token_length_fn else len(record.get("messages", [])) for record in records]
    over = [index for index, value in enumerate(lengths) if value > max_length]
    if over:
        raise SFTSplitError(f"{len(over)} records exceed max_length={max_length}; truncation is error")

    # A formal split is only meaningful when every merged trajectory belongs
    # to the reserved SFT task pool.  This catches accidental concatenation of
    # GRPO/test rows before validation selection and makes the task-level
    # isolation proof explicit rather than relying on source ordering.
    sft_pool_keys: set[str] | None = None
    if task_pool_manifest is not None:
        try:
            sft_pool_keys = pool_task_keys(task_pool_manifest, "sft")
        except TaskPoolError as exc:
            raise SFTSplitError(f"invalid SFT task pool: {exc}") from exc
        if not sft_pool_keys:
            raise SFTSplitError("formal SFT task pool is empty")
        outside: list[tuple[int, str]] = []
        for index, record in enumerate(records):
            key = _task_key_for(index, record, audits)
            if key is not None and key not in sft_pool_keys:
                outside.append((index, key))
        if outside:
            preview = outside[:5]
            raise SFTSplitError(
                "merged SFT corpus contains task(s) outside the formal SFT pool: "
                f"{preview}"
            )

    selected, selection_meta = select_validation_tasks(
        records,
        audits,
        count=validation_count,
        token_length_fn=token_length_fn,
        selection_seed=selection_seed,
    )
    val_keys = set(selected)
    if sft_pool_keys is not None and not val_keys <= sft_pool_keys:
        raise SFTSplitError("validation task selection escaped the formal SFT pool")
    for index, record in enumerate(records):
        if is_sft_eligible(record) and _task_key_for(index, record, audits) is None:
            raise SFTSplitError(
                f"eligible SFT record {index} has no reviewed task identity; "
                "formal train/validation isolation cannot be proven"
            )
    # Validation is task-isolated and ``val_gold10`` is intended to be a
    # clean gold-only view.  Keep only strict_gold rows from the selected task
    # groups; partial or non-SFT audit rows stay in the canonical and
    # per-class artifacts, but must not be emitted to val_gold10.jsonl.
    validation_indices = [
        index
        for index in range(len(records))
        if _task_key_for(index, records[index], audits) in val_keys
        and records[index].get("trainer_metadata", {}).get("trajectory_class") == "strict_gold"
    ]
    train_indices = [
        index for index, record in enumerate(records)
        if _task_key_for(index, record, audits) not in val_keys and is_sft_eligible(record)
    ]
    train = [dict(records[index]) for index in train_indices]
    validation = [dict(records[index]) for index in validation_indices]
    train_keys = {_task_key_for(i, records[i], audits) for i in train_indices}
    if train_keys & val_keys:
        raise SFTSplitError("train/validation task overlap")
    manifest: dict[str, Any] = {
        "schema_version": "travelgym-sft-split-v1",
        "max_length": int(max_length),
        "truncation": "error",
        "tokenizer": tokenizer_name,
        "token_audit_exact": bool(tokenizer_name or token_length_fn),
        "validation_count": int(validation_count),
        "selection_seed": int(selection_seed),
        "selected_task_keys": selected,
        "train_record_count": len(train),
        "validation_record_count": len(validation),
        "train_task_count": len(train_keys),
        "validation_task_count": len(val_keys),
        "source_record_count": len(records),
        "selection": selection_meta,
    }
    output = Path(output_dir)
    _write_jsonl(output / "train.jsonl", train)
    _write_jsonl(output / "val_gold10.jsonl", validation)
    temporary = (output / "split_manifest.json").with_suffix(".json.tmp")
    output.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output / "split_manifest.json")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--task-pool-manifest", type=Path, default=None)
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument("--selection-seed", type=int, default=SFT_SPLIT_SELECTION_SEED)
    args = parser.parse_args()
    records = _read_jsonl(args.input)
    audit_payload = json.loads(args.audit.read_text(encoding="utf-8"))
    audits = audit_payload.get("records", audit_payload) if isinstance(audit_payload, Mapping) else audit_payload
    if not isinstance(audits, list):
        raise SFTSplitError("audit must contain a records list")
    pool = load_pool_manifest(args.task_pool_manifest, require_strict=True) if args.task_pool_manifest else None
    manifest = build_sft_split(
        records,
        audits,
        output_dir=args.output_dir,
        task_pool_manifest=pool,
        tokenizer_name=args.tokenizer,
        max_length=args.max_length,
        selection_seed=args.selection_seed,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()


__all__ = ["SFTSplitError", "build_sft_split", "select_validation_tasks"]
