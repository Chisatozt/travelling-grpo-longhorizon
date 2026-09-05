"""Merge new Teacher trajectories into the canonical TravelGym SFT corpus.

The existing corpus is ``data/sft/travel_sft_public.json`` (currently 244
trajectories).  New inputs may be prepared ShareGPT JSON or raw evaluator
``*_reward_cache.json`` files.  Raw caches are passed through the same public
protocol cleaner as ``prepare_travel_sft.py``; invalid terminal reports and
trajectories without a complete ``<think>...</think>`` block are excluded by
default.  Exact canonical duplicates are removed while preserving source
order: the existing corpus first, then accepted new trajectories.

This script intentionally deduplicates only exact public records.  The
canonical 244-file has no task IDs, and collapsing records merely because
their initial natural-language request is identical could discard legitimate
different TravelGym tasks.  Rows whose labels cannot be recovered, whose
Reward is invalid, or whose required reasoning is missing are retained in the
    private/infrastructure quarantine for audit rather than silently entering SFT;
    the six owner-designated opaque rows are permanently discarded before
    canonical output.  The old ShareGPT helper functions below are compatibility
    shims only.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from .._paths import REPOSITORY_ROOT
from .travel_canonical import (
    SCHEMA_VERSION,
    canonical_hash,
    canonicalize_record,
    iter_source_records,
    render_legacy_record,
    validate_canonical,
)
from .clean_travel_trajectories import clean_trajectory, is_sft_eligible
from .travel_task_resolver import TravelTaskResolver
from .prepare_travel_sft import (
    _is_travel_record,
    _iter_source_records,
    _record_has_think,
    _report_is_trainable,
    _to_sharegpt_record,
    prepare_record,
)

PROJECT_ROOT = REPOSITORY_ROOT


def _canonical_key(record: dict[str, Any]) -> str:
    payload = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"SFT input not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_base(records: Any, path: Path) -> list[dict[str, Any]]:
    if not isinstance(records, list):
        raise ValueError(f"base SFT corpus must be a JSON list: {path}")
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"base SFT record {index} is not an object: {path}")
        if not _is_travel_record(record):
            raise ValueError(f"base SFT record {index} is not a TravelGym record: {path}")
        if not _record_has_think(record):
            raise ValueError(
                f"base SFT record {index} has an assistant turn without <think>...</think>: {path}"
            )
        key = _canonical_key(record)
        if key in seen:
            raise ValueError(f"base SFT corpus contains an exact duplicate at record {index}: {path}")
        seen.add(key)
        output.append(record)
    return output


def prepare_new_records(paths: list[Path], require_think: bool = True) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Clean and validate new inputs, returning public ShareGPT records."""
    stats = {
        "source_records": 0,
        "nontravel_dropped": 0,
        "invalid_reward_dropped": 0,
        "missing_think_dropped": 0,
        "accepted": 0,
    }
    prepared_records: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path)
        source = _load_json(path)
        for record, report in _iter_source_records(source):
            stats["source_records"] += 1
            if not _is_travel_record(record):
                stats["nontravel_dropped"] += 1
                continue
            if not _report_is_trainable(report):
                stats["invalid_reward_dropped"] += 1
                continue
            prepared = _to_sharegpt_record(prepare_record(copy.deepcopy(record)))
            if require_think and not _record_has_think(prepared):
                stats["missing_think_dropped"] += 1
                continue
            prepared_records.append(prepared)
            stats["accepted"] += 1
    return prepared_records, stats


def merge_records(
    base_records: list[dict[str, Any]],
    new_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Append exact public records once, preserving base/new ordering."""
    merged = list(base_records)
    seen = {_canonical_key(record) for record in base_records}
    duplicate_count = 0
    for record in new_records:
        key = _canonical_key(record)
        if key in seen:
            duplicate_count += 1
            continue
        seen.add(key)
        merged.append(record)
    return merged, duplicate_count


def _atomic_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_source(path: Path) -> Any:
    """Read JSON or JSONL without making the CLI depend on pandas."""
    if not path.is_file():
        raise FileNotFoundError(f"SFT input not found: {path}")
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        rows = []
        for line_number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
        return rows


def prepare_canonical_inputs(
    paths: list[Path],
    *,
    resolver: TravelTaskResolver | None = None,
    max_length: int = 32768,
    require_think: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Clean all supplied sources and return records, private audits, stats."""
    resolver = resolver or TravelTaskResolver()
    stats = {
        "source_records": 0,
        "nontravel_dropped": 0,
        "opaque_quarantined": 0,
        "accepted": 0,
        "infrastructure_invalid": 0,
        "strict_gold": 0,
        "recoverable_correct": 0,
        "partial_correct": 0,
        "totally_wrong": 0,
    }
    records: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    from .task_pools import is_permanently_quarantined_opaque
    for raw_path in paths:
        # Public helpers are also used by notebooks/tests that pass strings;
        # normalize once at the I/O boundary so provenance and source identity
        # remain path-stable.
        path = Path(raw_path)
        source = _read_source(path)
        for raw, report, private_meta in iter_source_records(source):
            stats["source_records"] += 1
            if not _is_travel_record(raw):
                stats["nontravel_dropped"] += 1
                continue
            private_meta = {**dict(private_meta), "source_path": str(path)}
            # The six owner-designated opaque historical rows are discarded
            # before canonical cleaning.  Keeping them out of both records
            # and audits is what makes a strict task-pool merge possible; the
            # task-pool manifest remains the audit source for their identities.
            if is_permanently_quarantined_opaque(
                raw,
                source_index=private_meta.get("source_index"),
                source_path=path,
            ):
                stats["opaque_quarantined"] += 1
                continue
            task_id, task_spec = resolver.resolve(raw, private_meta)
            cleaned, audit = clean_trajectory(
                raw,
                task=task_spec,
                task_id=task_id,
                reward_report=report if isinstance(report, dict) else None,
                source=str(path),
                max_length=max_length,
                require_think=require_think,
            )
            # Keep private provenance only in the audit sidecar.
            audit["source_path"] = str(path)
            audit["source_meta"] = dict(private_meta)
            # Task identity is private provenance used by the task-pool
            # guard.  It is deliberately absent from canonical messages.
            if task_id and task_id in resolver.tasks:
                env_name = str(resolver.tasks[task_id].get("env_name") or "")
                if env_name:
                    audit["env_name"] = env_name
                    audit["task_key"] = f"{env_name}::{task_id}"
            if isinstance(report, dict):
                audit["raw_reward_valid"] = bool(report.get("reward_valid_for_training", report.get("reward_valid", True)))
            category = cleaned.get("trainer_metadata", {}).get("trajectory_class", "infrastructure_invalid")
            stats[category] = stats.get(category, 0) + 1
            stats["accepted"] += 1
            records.append(cleaned)
            audits.append(audit)
    return records, audits, stats


def merge_canonical_records(
    base_records: list[dict[str, Any]],
    new_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Deduplicate only exact canonical public content; preserve task variants."""
    merged = list(base_records)
    seen = {canonical_hash(record) for record in merged}
    duplicate_count = 0
    for record in new_records:
        key = canonical_hash(record)
        if key in seen:
            duplicate_count += 1
            continue
        seen.add(key)
        merged.append(record)
    return merged, duplicate_count


def _percentile(values: list[int], percentile: float) -> int | None:
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return int(values[lower])
    return int(round(values[lower] + (values[upper] - values[lower]) * (position - lower)))


def _token_lengths(
    records: list[dict[str, Any]], tokenizer_name: str | None
) -> tuple[list[int], str, list[int] | None, dict[str, Any]]:
    if tokenizer_name:
        from transformers import AutoTokenizer
        from ..training.sft.qwen35_mask import exact_assistant_token_mask, native_template_ids

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        values = []
        supervised_values: list[int] = []
        for record in records:
            full_ids = native_template_ids(
                tokenizer,
                record["messages"],
                tools=record.get("tools"),
                enable_thinking=True,
                add_generation_prompt=False,
            )
            values.append(len(full_ids))
            token_mask = exact_assistant_token_mask(
                record["messages"],
                full_ids,
                lambda prefix, generation: native_template_ids(
                    tokenizer,
                    prefix,
                    tools=record.get("tools"),
                    enable_thinking=True,
                    add_generation_prompt=generation,
                ),
                record.get("assistant_train_mask", [0] * len(record["messages"])),
            )
            # The final input token has no next-token loss.  Count effective
            # supervised prediction positions, not merely assistant messages.
            supervised_values.append(sum(token_mask[:-1]) if token_mask else 0)
        revision = None
        init_kwargs = getattr(tokenizer, "init_kwargs", {})
        if isinstance(init_kwargs, dict):
            revision = init_kwargs.get("_commit_hash") or init_kwargs.get("revision")
        return values, "tokens", supervised_values, {
            "tokenizer": tokenizer_name,
            "tokenizer_revision": revision,
            "enable_thinking": True,
        }
    # A tokenizer is intentionally optional for CPU-only conversion.  Keep the
    # field explicit so a manifest cannot be mistaken for a token audit.
    return (
        [len(json.dumps(record.get("messages", []), ensure_ascii=False)) for record in records],
        "message_chars",
        None,
        {"tokenizer": None, "enable_thinking": True},
    )


def build_manifest(
    records: list[dict[str, Any]],
    audits: list[dict[str, Any]],
    *,
    tokenizer_name: str | None = None,
    max_length: int = 32768,
    duplicate_count: int = 0,
    source_stats: Mapping[str, Mapping[str, int]] | None = None,
) -> dict[str, Any]:
    lengths, unit, supervised_token_counts, template_metadata = _token_lengths(records, tokenizer_name)
    overlength = 0
    effective_lengths: list[int] = []
    for index, length in enumerate(lengths):
        if unit == "tokens" and length > max_length:
            overlength += 1
            records[index].setdefault("trainer_metadata", {})["trajectory_class"] = "overlength_quarantine"
            records[index]["trainer_metadata"]["sample_weight"] = 0.0
            if index < len(audits):
                audits[index]["overlength"] = True
                audits[index]["trajectory_class"] = "overlength_quarantine"
                audits[index]["sample_weight"] = 0.0
        else:
            effective_lengths.append(length)
    by_class: dict[str, dict[str, int]] = {}
    source_distribution: Counter[str] = Counter()
    for index, record in enumerate(records):
        metadata = record.get("trainer_metadata", {})
        category = str(metadata.get("trajectory_class", "unknown"))
        audit = audits[index] if index < len(audits) else {}
        source_path = str(audit.get("source_path", ""))
        source_name = Path(source_path).name.lower()
        if source_name == "travel_sft_public.json":
            source_label = "historical_public"
        elif "teacher" in source_name or "cache" in source_name:
            source_label = "deepseek_teacher"
        elif source_name:
            source_label = source_name
        else:
            source_label = "unknown"
        source_distribution[source_label] += 1
        structural_count = int(sum(1 for value in record.get("assistant_train_mask", []) if value))
        exact_count = (
            int(supervised_token_counts[index])
            if supervised_token_counts is not None and index < len(supervised_token_counts)
            else None
        )
        item = by_class.setdefault(
            category,
            {"records": 0, "supervised_message_turns": 0, "effective_supervised_tokens": 0},
        )
        item["records"] += 1
        item["supervised_message_turns"] += structural_count
        if exact_count is not None:
            item["effective_supervised_tokens"] += exact_count
    effective_supervised_tokens = (
        int(sum(supervised_token_counts)) if supervised_token_counts is not None else None
    )
    eligible_classes = {"strict_gold", "recoverable_correct", "partial_correct"}
    eligible_effective_supervised_tokens = (
        int(
            sum(
                int(supervised_token_counts[index])
                for index, record in enumerate(records)
                if index < len(supervised_token_counts)
                and str(record.get("trainer_metadata", {}).get("trajectory_class", ""))
                in eligible_classes
            )
        )
        if supervised_token_counts is not None
        else None
    )
    exclusion_reasons: Counter[str] = Counter()
    for category in ("totally_wrong", "infrastructure_invalid", "overlength_quarantine"):
        count = int(by_class.get(category, {}).get("records", 0))
        if count:
            exclusion_reasons[category] += count
    if source_stats:
        for stats in source_stats.values():
            if not isinstance(stats, Mapping):
                continue
            for reason in ("nontravel_dropped", "opaque_quarantined"):
                exclusion_reasons[reason] += int(stats.get(reason, 0) or 0)
    return {
        "schema_version": SCHEMA_VERSION,
        "template": template_metadata,
        "max_length": max_length,
        "truncation": "error",
        "length_unit": unit,
        "lengths": {
            "count": len(lengths),
            "p50": _percentile(lengths, 0.50),
            "p90": _percentile(lengths, 0.90),
            "p95": _percentile(lengths, 0.95),
            "p99": _percentile(lengths, 0.99),
            "max": max(lengths) if lengths else None,
        },
        "overlength_count": overlength,
        "duplicate_count": duplicate_count,
        "classification": by_class,
        "source_distribution": dict(sorted(source_distribution.items())),
        "effective_supervised_tokens": effective_supervised_tokens,
        "eligible_effective_supervised_tokens": eligible_effective_supervised_tokens,
        "exclusion_reasons": dict(sorted(exclusion_reasons.items())),
        "note": (
            "effective_supervised_tokens is exact only when --tokenizer is supplied; "
            "otherwise supervised_message_turns is a structural count."
        ),
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    temporary.replace(path)


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write canonical rows to Parquet when the optional pandas engine exists."""
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - training environment only
        raise RuntimeError("--parquet-output requires pandas and a parquet engine") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows).to_parquet(temporary, index=False)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="data/sft/travel_sft_public.json", help="Original 244-record ShareGPT corpus (never overwritten).")
    parser.add_argument("--input", nargs="+", required=True, help="DeepSeek Teacher JSON/JSONL/cache files.")
    parser.add_argument("--output", default="data/sft/travel_sft_canonical.jsonl", help="Merged canonical JSONL output.")
    parser.add_argument("--audit-output", default=None, help="Private cleaning/task audit JSON (default: <output>.audit.json).")
    parser.add_argument("--manifest-output", default=None, help="Manifest JSON (default: <output>.manifest.json).")
    parser.add_argument("--sft-output", default=None, help="SFT-eligible canonical JSONL (default: <output>.train.jsonl).")
    parser.add_argument(
        "--split-output-dir",
        default=None,
        help="Optional formal task-group split directory (train.jsonl, val_gold10.jsonl, split_manifest.json). Requires --tokenizer and a strict task-pool manifest.",
    )
    parser.add_argument("--parquet-output", default=None, help="Optional canonical Parquet mirror (requires pandas/pyarrow).")
    parser.add_argument("--legacy-output", default=None, help="Optional legacy ShareGPT regression renderer output.")
    parser.add_argument("--task-data", nargs="*", default=None, help="Optional TravelGym task JSON files for deterministic ID recovery.")
    parser.add_argument("--task-map", default=None, help="Reviewed JSON sidecar mapping source index/path#source_index to task ID for ambiguous rows.")
    parser.add_argument(
        "--task-pool-manifest",
        default=None,
        help=(
            "Optional travel task-pool manifest. When supplied, every new "
            "Teacher record must belong to the manifest's SFT pool; records "
            "without a resolved task ID fail closed."
        ),
    )
    parser.add_argument("--tokenizer", default=None, help="Optional Qwen3.5 tokenizer for exact 32K length audit.")
    parser.add_argument("--max-length", type=int, default=32768)
    # Compatibility only: canonical conversion always uses enable_thinking=true.
    parser.add_argument(
        "--allow-missing-think",
        action="store_true",
        help="Diagnostic only: do not quarantine assistant tool turns with missing reasoning_content.",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    base_path = Path(args.base)
    # The 244-row ShareGPT corpus is immutable input.  Refuse both an
    # explicit in-place merge and an equivalent path spelling (relative vs.
    # absolute) before any derived artifact is written.
    protected_base = (PROJECT_ROOT / "data" / "sft" / "travel_sft_public.json").resolve()
    if output_path.resolve() in {protected_base, base_path.resolve()}:
        raise ValueError(
            "refusing to overwrite the source SFT corpus; choose a derived output path"
        )
    task_paths = [Path(value) for value in args.task_data] if args.task_data else None
    alignment_map = TravelTaskResolver.load_alignment_map(args.task_map)
    resolver = TravelTaskResolver(
        project_root=PROJECT_ROOT, task_paths=task_paths, explicit_map=alignment_map
    )

    require_think = not bool(args.allow_missing_think)
    base_records, base_audits, base_stats = prepare_canonical_inputs(
        [base_path], resolver=resolver, max_length=args.max_length, require_think=require_think
    )
    new_records, new_audits, new_stats = prepare_canonical_inputs(
        [Path(value) for value in args.input], resolver=resolver, max_length=args.max_length, require_think=require_think
    )
    if args.task_pool_manifest:
        from .task_pools import assert_audits_in_pool, load_pool_manifest
        # A formal merge is a training input.  The six known opaque historical
        # rows have already been discarded by ``prepare_canonical_inputs``;
        # active records must therefore satisfy the strict task-pool contract.
        pool_manifest = load_pool_manifest(args.task_pool_manifest, require_strict=True)
        assert_audits_in_pool([*base_audits, *new_audits], pool_manifest, pool_name="sft")
        for input_path in (Path(value) for value in args.input):
            try:
                payload = _read_source(input_path)
            except Exception:
                continue
            if isinstance(payload, dict) and payload.get("schema_version") == "travelgym-teacher-cache-v1":
                cache_records = payload.get("records", {})
                if not isinstance(cache_records, dict):
                    raise ValueError(f"Teacher cache records must be an object: {input_path}")
                from .teacher_collection import teacher_pass_key
                for pass_key, entry in cache_records.items():
                    if not isinstance(entry, dict):
                        raise ValueError(f"Teacher cache record is not an object: {input_path}::{pass_key}")
                    provenance = entry.get("provenance", {})
                    if not isinstance(provenance, dict):
                        raise ValueError(f"Teacher cache provenance is invalid: {input_path}::{pass_key}")
                    try:
                        expected_pass_key = teacher_pass_key(
                            provenance.get("env_name") or str(provenance.get("task_key", "")).split("::", 1)[0],
                            provenance.get("task_id") or str(provenance.get("task_key", "")).split("::", 1)[-1],
                            int(provenance.get("pass_index")),
                        )
                    except (TypeError, ValueError, IndexError) as exc:
                        raise ValueError(f"Teacher cache pass provenance is invalid: {input_path}::{pass_key}") from exc
                    if str(pass_key) != expected_pass_key:
                        raise ValueError(
                            f"Teacher cache pass key/provenance mismatch: {input_path}::{pass_key} != {expected_pass_key}"
                        )
    merged, duplicate_count = merge_canonical_records(base_records, new_records)
    merged_audits: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record, audit in list(zip(base_records, base_audits)) + list(zip(new_records, new_audits)):
        key = canonical_hash(record)
        if key in seen:
            continue
        seen.add(key)
        merged_audits.append(audit)
    audit_path = Path(args.audit_output) if args.audit_output else output_path.with_suffix(".audit.json")
    manifest_path = Path(args.manifest_output) if args.manifest_output else output_path.with_suffix(".manifest.json")
    sft_path = Path(args.sft_output) if args.sft_output else output_path.with_suffix(".train.jsonl")
    source_paths = {protected_base, base_path.resolve()}
    for label, candidate in (
        ("audit", audit_path),
        ("manifest", manifest_path),
        ("SFT", sft_path),
        ("Parquet", Path(args.parquet_output) if args.parquet_output else None),
        ("legacy", Path(args.legacy_output) if args.legacy_output else None),
    ):
        if candidate is not None and candidate.resolve() in source_paths:
            raise ValueError(f"refusing to overwrite the source SFT corpus via {label} output")
    manifest = build_manifest(
        merged,
        merged_audits,
        tokenizer_name=args.tokenizer,
        max_length=args.max_length,
        duplicate_count=duplicate_count,
        source_stats={"historical_public": base_stats, "deepseek_teacher": new_stats},
    )
    if args.task_pool_manifest:
        manifest["task_pool_manifest"] = str(Path(args.task_pool_manifest).resolve())
    # ``build_manifest`` may quarantine overlength rows and updates the
    # corresponding private audit records; persist the audit only afterwards
    # so canonical, manifest and sidecar classifications cannot drift.
    _atomic_write(audit_path, {"schema_version": SCHEMA_VERSION, "records": merged_audits})
    _atomic_write(manifest_path, manifest)
    # ``build_manifest`` may mark >32K rows as overlength quarantine; write the
    # canonical/parquet artifacts only after that mutation so all outputs agree.
    _write_jsonl(output_path, merged)
    if args.parquet_output:
        _write_parquet(Path(args.parquet_output), merged)
    _write_jsonl(sft_path, [record for record in merged if is_sft_eligible(record)])
    # Classification files are convenient for audit/ablation and remain
    # canonical; only the explicitly generated ``.train.jsonl`` is consumed by
    # the SFT command.
    for category in ("strict_gold", "recoverable_correct", "partial_correct", "totally_wrong", "infrastructure_invalid", "overlength_quarantine"):
        category_path = output_path.with_name(f"{output_path.stem}.{category}.jsonl")
        _write_jsonl(
            category_path,
            [record for record in merged if record.get("trainer_metadata", {}).get("trajectory_class") == category],
        )
    if args.split_output_dir:
        if not args.tokenizer or pool_manifest is None:
            raise ValueError("--split-output-dir requires --tokenizer and --task-pool-manifest with strict active identities")
        from ..training.sft.sft_split import build_sft_split
        build_sft_split(
            merged,
            merged_audits,
            output_dir=args.split_output_dir,
            task_pool_manifest=pool_manifest,
            tokenizer_name=args.tokenizer,
            max_length=args.max_length,
            require_exact_token_audit=True,
        )
    if args.legacy_output:
        _atomic_write(Path(args.legacy_output), [render_legacy_record(record) for record in merged])
    summary = {key: base_stats.get(key, 0) + new_stats.get(key, 0) for key in set(base_stats) | set(new_stats)}
    print(
        f"base={len(base_records)} source={new_stats.get('source_records', 0)} "
        f"opaque_quarantined={base_stats.get('opaque_quarantined', 0) + new_stats.get('opaque_quarantined', 0)} "
        f"duplicates={duplicate_count} merged={len(merged)} sft={sum(is_sft_eligible(record) for record in merged)} "
        f"classes={{strict:{summary.get('strict_gold', 0)}, recoverable:{summary.get('recoverable_correct', 0)}, "
        f"partial:{summary.get('partial_correct', 0)}, wrong:{summary.get('totally_wrong', 0)}, infra:{summary.get('infrastructure_invalid', 0)}}} "
        f"output={output_path}"
    )


if __name__ == "__main__":
    main()
