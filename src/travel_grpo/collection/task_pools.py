"""Deterministic, task-level partitions for the TravelGym pipeline.

The same TravelGym task must never be used as both a supervised example and a
rollout/evaluation example.  This module keeps that rule in one place.  A task
identity is the pair ``(env_name, task_id)`` (not a trajectory hash and not a
human-readable prompt), because the latter two are not stable identities in
the source data.

The generated manifest has three *active* pools:

``sft``
    Tasks represented by the existing corpus plus a deterministic expansion
    reservation from the remaining train split.  Future DeepSeek Teacher
    collection is restricted to this pool; collecting another trajectory for
    an SFT task is allowed.
``sft_smoke``
    A fixed 20-task stratified subset of ``sft`` used for the paid smoke run;
    it is a view, not a fourth disjoint active pool.
``grpo``
    All train-split tasks not reserved by ``sft`` (including its expansion
    reservation).
``validation``
    A deterministic 200-task selection from the test split.  ``smoke`` is a
    small subset of this pool and is intentionally not a fourth active pool.

Six historical rows currently have no reviewed task ID.  They are permanently
isolated in ``quarantined_sft`` and are discarded from every active pool.  They
do not consume the SFT target and cannot be re-introduced by an optional task
map.  The formal manifest is strict for the *active* pools; the quarantine is
an audit-only record of discarded source rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .._paths import REPOSITORY_ROOT

try:
    from .travel_canonical import canonical_hash, canonicalize_record, iter_source_records
    from .travel_task_resolver import TravelTaskResolver
except ImportError:  # pragma: no cover - direct script/module execution
    from travel_canonical import canonical_hash, canonicalize_record, iter_source_records
    from travel_task_resolver import TravelTaskResolver


SCHEMA_VERSION = "travelgym-task-pools-v2"
TASK_KEY_SEPARATOR = "::"
POOL_NAMES = ("sft", "grpo", "validation")
SMOKE_POOL_NAME = "validation_smoke"
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
# Match ``travel_grpo.evaluation.build_test_manifests`` so the validation smoke view is the
# same 20-task set used by the evaluator.
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
# A separate stratified view of the train-side SFT pool used for the paid
# Teacher smoke collection.  It intentionally mirrors the eight-variant
# validation smoke allocation while remaining a subset of ``sft``.
SFT_SMOKE_QUOTAS = dict(SMOKE_QUOTAS)
# Keep task-pool selection aligned with ``travel_grpo.evaluation.build_test_manifests`` so
# the validation pool is exactly the checked-in final200 manifest.
TASK_POOL_SELECTION_SEED = 20260801
# The SFT pool is deliberately larger than the historical 244-record seed.
# This target counts resolved task identities, not trajectories: a task may
# later receive multiple Teacher trajectories while remaining in this pool.
DEFAULT_SFT_TARGET_COUNT = 600

# These are the six historical rows explicitly designated by the experiment
# owner as opaque and unusable.  Keep both the stable source positions and the
# canonical opaque keys: the positions protect the immutable 244-row source,
# while the keys also quarantine an exact duplicate that appears in a derived
# cache.  A reviewed alignment map must never be able to re-introduce them.
QUARANTINED_OPAQUE_RECORD_INDICES = frozenset({97, 144, 159, 180, 206, 208})
QUARANTINED_OPAQUE_TASK_KEYS = frozenset(
    {
        "opaque_sft::3089e2f523836fc496c257b8c66bff348ac3d33016bef110c646a2bf9eaf81da",
        "opaque_sft::d829e87bfd541fa31e31488fd44e5974e04eecc0e54861f2fae1e8ca1762e8df",
        "opaque_sft::e131d0e49687a89cafc50b97622c92ffc0507db6083bf7acae53e75665676482",
        "opaque_sft::78fd3e30451e68f21a334b6503a21964c1fc9d048750f99b2be880c01979ca4c",
        "opaque_sft::1c44e2e96b9457c21b7c803053dab33391baee47ba23bb612881fcca404a2482",
        "opaque_sft::e70679e080acdfd71efab4969b8f6d03185fcfa040a3fba6b55a32d5b78d563d",
    }
)
OPAQUE_TASK_KEY_PREFIX = "opaque_sft::"


def _is_true_flag(value: Any) -> bool:
    """Parse a manifest boolean without treating ``"false"`` as truthy.

    JSON manifests normally store a real boolean, but hand-edited audit files
    sometimes contain strings.  A strict task identity gate must fail closed
    for every spelling other than an explicit true value.
    """

    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value == 1
    return str(value or "").strip().casefold() in {"true", "1", "yes", "y"}


def _norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").casefold()).strip()


class TaskPoolError(ValueError):
    """Raised when a task partition is malformed or violates disjointness."""


@dataclass(frozen=True)
class TaskRef:
    """A task identity plus immutable source provenance."""

    env_name: str
    task_id: str
    split: str
    source_index: int | None = None
    source_path: str | None = None
    role: str | None = None

    @property
    def key(self) -> str:
        return make_task_key(self.env_name, self.task_id)

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "env_name": self.env_name,
            "task_id": self.task_id,
            "task_key": self.key,
            "split": self.split,
        }
        if self.source_index is not None:
            value["source_index"] = int(self.source_index)
        if self.source_path:
            value["source_path"] = self.source_path
        if self.role:
            value["role"] = self.role
        return value


def make_task_key(env_name: Any, task_id: Any) -> str:
    """Return the canonical, collision-resistant task key."""

    env = str(env_name or "").strip()
    task = str(task_id or "").strip()
    if not env or not task:
        raise TaskPoolError("task identity requires non-empty env_name and task_id")
    if TASK_KEY_SEPARATOR in env:
        raise TaskPoolError(f"env_name contains reserved separator: {env!r}")
    return f"{env}{TASK_KEY_SEPARATOR}{task}"


def split_task_key(value: Any) -> tuple[str, str]:
    text = str(value or "")
    env, separator, task = text.partition(TASK_KEY_SEPARATOR)
    if not separator or not env or not task:
        raise TaskPoolError(f"invalid task_key {value!r}")
    return env, task


def _as_task_ref(value: Mapping[str, Any]) -> TaskRef:
    if not isinstance(value, Mapping):
        raise TaskPoolError("task-pool record must be an object")
    env_name = value.get("env_name")
    task_id = value.get("task_id")
    if (not env_name or not task_id) and value.get("task_key"):
        env_name, task_id = split_task_key(value["task_key"])
    if not env_name or not task_id:
        raise TaskPoolError("task-pool record has no task identity")
    return TaskRef(
        env_name=str(env_name),
        task_id=str(task_id),
        split=str(value.get("split", "unknown")),
        source_index=(int(value["source_index"]) if value.get("source_index") is not None else None),
        source_path=(str(value["source_path"]) if value.get("source_path") else None),
        role=(str(value["role"]) if value.get("role") else None),
    )


def _portable_path(path: str | Path, root: Path) -> str:
    """Render provenance paths relative to the repository when possible."""

    value = Path(path).resolve()
    try:
        return str(value.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(value)


def opaque_task_key(record: Mapping[str, Any]) -> str:
    """Return the stable audit key used for an unresolved public transcript."""

    try:
        digest = canonical_hash(canonicalize_record(record))
    except Exception:
        digest = hashlib.sha256(
            json.dumps(record, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
    return f"{OPAQUE_TASK_KEY_PREFIX}{digest}"


def is_permanently_quarantined_opaque(
    record: Mapping[str, Any],
    *,
    source_index: int | None = None,
    source_path: str | Path | None = None,
) -> bool:
    """Whether a source row belongs to the six explicitly discarded rows.

    The index check is scoped to the immutable historical corpus.  The opaque
    digest check additionally catches exact copies in derived caches without
    applying a positional rule to unrelated inputs.
    """

    if opaque_task_key(record) in QUARANTINED_OPAQUE_TASK_KEYS:
        return True
    if source_index is None or int(source_index) not in QUARANTINED_OPAQUE_RECORD_INDICES:
        return False
    if source_path is None:
        return False
    normalized = str(source_path).replace("\\", "/").casefold().rstrip("/")
    return normalized.endswith("/data/sft/travel_sft_public.json") or normalized == "data/sft/travel_sft_public.json"


def _task_rank(seed: int, task_key: str) -> str:
    env_name, task_id = split_task_key(task_key)
    return hashlib.sha256(f"{seed}:{env_name}:{task_id}".encode("utf-8")).hexdigest()


def _read_parquet_rows(path: Path) -> list[Mapping[str, Any]]:
    """Read reward-model metadata without making imports mandatory at module import."""

    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - depends on training env
        raise TaskPoolError(
            "building task pools requires pandas and a parquet engine; "
            "install the project trajectory dependencies first"
        ) from exc
    if not path.is_file():
        raise FileNotFoundError(f"missing TravelGym split: {path}")
    frame = pd.read_parquet(path)
    if "reward_model" not in frame.columns:
        raise TaskPoolError(f"{path} has no reward_model column")
    rows: list[Mapping[str, Any]] = []
    for index, value in enumerate(frame["reward_model"]):
        if not isinstance(value, Mapping) or not value.get("id"):
            raise TaskPoolError(f"{path} row {index} has no reward_model.id")
        rows.append(value)
    return rows


def load_task_inventory(
    project_root: str | Path,
    *,
    include_train: bool = True,
    include_test: bool = True,
) -> dict[str, TaskRef]:
    """Load the authoritative variant parquet IDs and split assignments.

    The aggregate ``travel_multiturn_onechoice`` parquet is deliberately not
    read: it duplicates the eight variant files and could create false pool
    overlaps.
    """

    root = Path(project_root).resolve()
    inventory: dict[str, TaskRef] = {}
    for env_name in SUPPORTED_ENVS:
        for split, enabled in (("train", include_train), ("test", include_test)):
            if not enabled:
                continue
            path = root / "data" / f"{env_name}_multiturn_onechoice" / f"{split}.parquet"
            seen_in_split: set[str] = set()
            for source_index, row in enumerate(_read_parquet_rows(path)):
                task_id = str(row["id"])
                key = make_task_key(env_name, task_id)
                if key in seen_in_split:
                    raise TaskPoolError(f"duplicate task {key!r} in {path}")
                seen_in_split.add(key)
                if key in inventory:
                    previous = inventory[key]
                    raise TaskPoolError(
                        f"task appears in multiple source splits: {key}; "
                        f"{previous.split} and {split}"
                    )
                inventory[key] = TaskRef(
                    env_name=env_name,
                    task_id=task_id,
                    split=split,
                    source_index=source_index,
                    source_path=str(path.relative_to(root)).replace("\\", "/"),
                )
    return inventory


def _read_source(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        rows: list[Any] = []
        for line_number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise TaskPoolError(f"invalid JSONL at {path}:{line_number}") from exc
        return rows


def resolve_sft_task_refs(
    source_path: str | Path,
    *,
    project_root: str | Path,
    task_map: Mapping[str, str] | None = None,
    inventory: Mapping[str, TaskRef] | None = None,
) -> tuple[list[TaskRef], list[dict[str, Any]]]:
    """Resolve historical SFT rows and return quarantined rows separately.

    Opaque rows are never guessed into a real task ID and are not returned as
    active ``TaskRef`` objects.  The second return value is an audit-only
    quarantine list; callers must keep those rows out of SFT, GRPO and
    Validation task selection.
    """

    path = Path(source_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"SFT source not found: {path}")
    resolver = TravelTaskResolver(project_root=project_root, explicit_map=task_map or {})
    inventory = inventory or load_task_inventory(project_root)
    source = _read_source(path)
    refs: list[TaskRef] = []
    unresolved: list[dict[str, Any]] = []
    seen_real: set[str] = set()
    seen_opaque: set[str] = set()
    for source_index, raw in enumerate(iter_source_records(source)):
        record, _report, private_meta = raw
        meta = {**dict(private_meta), "source_path": str(path)}
        # The six owner-designated opaque rows are permanently quarantined.
        # Do this before consulting an explicit map so a stale/reviewed map
        # cannot put a discarded row back into an active pool.
        opaque = opaque_task_key(record)
        permanently_quarantined = is_permanently_quarantined_opaque(
            record,
            source_index=source_index,
            source_path=path,
        )
        if permanently_quarantined:
            task_id, _spec = None, None
        else:
            task_id, _spec = resolver.resolve(record, meta)
        env_name: str | None = None
        if task_id and task_id in resolver.tasks:
            env_name = str(resolver.tasks[task_id].get("env_name") or "") or None
        key = make_task_key(env_name, task_id) if env_name and task_id else None
        if key and key in inventory:
            if key not in seen_real:
                source_ref = inventory[key]
                refs.append(
                    TaskRef(
                        env_name=source_ref.env_name,
                        task_id=source_ref.task_id,
                        split=source_ref.split,
                        source_index=source_ref.source_index,
                        source_path=source_ref.source_path,
                        role="historical_sft",
                    )
                )
                seen_real.add(key)
            continue

        # ``canonical_hash`` only hashes public transcript content, so this
        # opaque identity is stable without copying task/reward labels.
        if opaque in seen_opaque:
            continue
        seen_opaque.add(opaque)
        unresolved.append(
            {
                "record_index": int(source_index),
                "opaque_task_key": opaque,
                "source_path": _portable_path(path, Path(project_root).resolve()),
                "reason": (
                    "explicit_opaque_quarantine"
                    if permanently_quarantined
                    else "task_id_unresolved_or_not_in_inventory"
                ),
                "status": "discarded" if permanently_quarantined else "unresolved",
            }
        )
    return refs, unresolved


def build_sft_candidate_audit(
    source_path: str | Path,
    *,
    project_root: str | Path,
    inventory: Mapping[str, TaskRef] | None = None,
    task_map: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Create a private audit report for unresolved rows.

    The report is deliberately an audit artifact, not an Actor input.  It
    records public-text candidate evidence for historical diagnostics.  The
    six owner-designated opaque rows are marked ``discarded`` and never
    selected, even if a caller supplies a stale explicit map.
    """

    path = Path(source_path).resolve()
    root = Path(project_root).resolve()
    inventory = inventory or load_task_inventory(root)
    resolver = TravelTaskResolver(project_root=root, explicit_map=task_map or {})
    source = _read_source(path)
    report: list[dict[str, Any]] = []
    for source_index, raw in enumerate(iter_source_records(source)):
        record, _reward, private_meta = raw
        meta = {**dict(private_meta), "source_path": str(path)}
        opaque = opaque_task_key(record)
        permanently_quarantined = is_permanently_quarantined_opaque(
            record,
            source_index=source_index,
            source_path=path,
        )
        task_id, _spec = (
            (None, None)
            if permanently_quarantined
            else resolver.resolve(record, meta)
        )
        if task_id and any(ref.key == make_task_key(str(resolver.tasks[task_id].get("env_name")), task_id) for ref in inventory.values()):
            continue
        messages = record.get("messages", record.get("conversations", []))
        user_text = " ".join(
            str(message.get("content", message.get("value", "")))
            for message in messages
            if isinstance(message, Mapping)
            and str(message.get("role", message.get("from", ""))).casefold() in {"user", "human"}
        )
        normalized_user = _norm(user_text)
        candidate_ids: list[str] = []
        evidence: dict[str, list[str]] = {}
        for initial, keys in resolver._by_initial.items():
            if initial and initial in normalized_user:
                for candidate in keys:
                    if candidate not in candidate_ids:
                        candidate_ids.append(candidate)
                    evidence.setdefault(candidate, []).append("initial_description_match")
        # Keep the report useful even if the initial description was not found:
        # explicit IDs are still shown as a rejected/needs-review candidate.
        explicit = record.get("task_id") or record.get("id") or meta.get("gold") or meta.get("task_key")
        if explicit is not None and str(explicit) in resolver.tasks and str(explicit) not in candidate_ids:
            candidate_ids.append(str(explicit))
            evidence.setdefault(str(explicit), []).append("explicit_record_label")
        candidates: list[dict[str, Any]] = []
        for candidate in sorted(candidate_ids):
            task = resolver.tasks.get(candidate, {})
            env_name = str(task.get("env_name") or "")
            key = f"{env_name}{TASK_KEY_SEPARATOR}{candidate}" if env_name else None
            candidates.append({
                "task_id": candidate,
                "env_name": env_name,
                "task_key": key,
                "evidence": sorted(set(evidence.get(candidate, []))),
                "inventory_present": bool(key and key in inventory),
                "requires_human_review": True,
            })
        report.append({
            "record_index": int(source_index),
            "source_path": _portable_path(path, root),
            "opaque_task_key": opaque,
            "public_user_text_preview": user_text[:500],
            # Candidate evidence is retained only as a historical audit.  It
            # is never actionable for the six permanently discarded rows.
            "candidates": [] if permanently_quarantined else candidates,
            "resolved_without_review": False,
            "requires_review": not permanently_quarantined,
            "quarantine_status": "discarded" if permanently_quarantined else None,
        })
    return report


def _stratified_quotas(counts: Mapping[str, int], total: int) -> dict[str, int]:
    if total <= 0 or total > sum(counts.values()):
        raise TaskPoolError(f"invalid validation size {total}; available={sum(counts.values())}")
    raw = {env: total * int(count) / sum(counts.values()) for env, count in counts.items()}
    quotas = {env: math.floor(value) for env, value in raw.items()}
    remaining = total - sum(quotas.values())
    order = sorted(counts, key=lambda env: (raw[env] - quotas[env], env), reverse=True)
    for env in order[:remaining]:
        quotas[env] += 1
    return quotas


def _select_stratified(
    inventory: Mapping[str, TaskRef],
    *,
    split: str,
    count: int,
    seed: int,
    quotas: Mapping[str, int] | None = None,
    exclude_keys: set[str] | None = None,
) -> list[TaskRef]:
    excluded = exclude_keys or set()
    by_env: dict[str, list[TaskRef]] = {env: [] for env in SUPPORTED_ENVS}
    for ref in inventory.values():
        if ref.split == split and ref.key not in excluded:
            by_env.setdefault(ref.env_name, []).append(ref)
    for env in by_env:
        by_env[env].sort(key=lambda ref: _task_rank(seed, ref.key))
    selected_quotas = (
        {env: int(value) for env, value in quotas.items()}
        if quotas is not None
        else _stratified_quotas({env: len(items) for env, items in by_env.items()}, count)
    )
    selected: list[TaskRef] = []
    for env in SUPPORTED_ENVS:
        selected.extend(by_env[env][: selected_quotas.get(env, 0)])
    selected.sort(key=lambda ref: (SUPPORTED_ENVS.index(ref.env_name), _task_rank(seed, ref.key)))
    return selected


def build_task_pool_manifest(
    *,
    project_root: str | Path,
    sft_source: str | Path = "data/sft/travel_sft_public.json",
    task_map: Mapping[str, str] | None = None,
    seed: int = TASK_POOL_SELECTION_SEED,
    sft_target_count: int = DEFAULT_SFT_TARGET_COUNT,
    validation_size: int = 200,
    smoke_size: int = 20,
) -> dict[str, Any]:
    """Build the complete disjoint partition manifest in memory.

    ``sft_target_count`` is a task count, not a trajectory count.  Historical
    rows that cannot be mapped to an authoritative task are quarantined and
    do not consume this target.  The expansion reservation is sampled from
    the remaining *train* tasks only; test tasks are never eligible for SFT.
    """

    if sft_target_count <= 0:
        raise TaskPoolError("sft_target_count must be positive")

    root = Path(project_root).resolve()
    sft_path = Path(sft_source)
    if not sft_path.is_absolute():
        sft_path = root / sft_path
    inventory = load_task_inventory(root)
    historical_refs, quarantined = resolve_sft_task_refs(
        sft_path,
        project_root=root,
        task_map=task_map,
        inventory=inventory,
    )
    historical_refs = [ref for ref in historical_refs if ref.key in inventory]
    historical_refs.sort(key=lambda ref: (SUPPORTED_ENVS.index(ref.env_name), ref.task_id))
    historical_keys = {ref.key for ref in historical_refs}
    if any(ref.split != "train" for ref in historical_refs):
        raise TaskPoolError("historical SFT task resolved to a non-train source split")

    train_inventory_refs = [ref for ref in inventory.values() if ref.split == "train"]
    if sft_target_count > len(train_inventory_refs):
        raise TaskPoolError(
            f"sft_target_count={sft_target_count} exceeds available train tasks "
            f"{len(train_inventory_refs)}"
        )
    if sft_target_count < len(historical_refs):
        raise TaskPoolError(
            f"sft_target_count={sft_target_count} is smaller than resolved historical "
            f"SFT task count {len(historical_refs)}"
        )

    # Make the expansion fill the composition variants that are under-
    # represented by the historical corpus.  First allocate the final SFT
    # target proportionally over all train tasks, then select only each
    # variant's deficit after subtracting historical SFT tasks.
    train_counts = {
        env_name: sum(ref.env_name == env_name for ref in train_inventory_refs)
        for env_name in SUPPORTED_ENVS
    }
    target_quotas = _stratified_quotas(train_counts, sft_target_count)
    historical_counts = {
        env_name: sum(ref.env_name == env_name for ref in historical_refs)
        for env_name in SUPPORTED_ENVS
    }
    over_target = {
        env_name: (historical_counts[env_name], target_quotas.get(env_name, 0))
        for env_name in SUPPORTED_ENVS
        if historical_counts[env_name] > target_quotas.get(env_name, 0)
    }
    if over_target:
        raise TaskPoolError(
            "sft_target_count is too small for historical composition quotas: "
            f"{over_target}"
        )
    expansion_quotas = {
        env_name: target_quotas.get(env_name, 0) - historical_counts[env_name]
        for env_name in SUPPORTED_ENVS
    }
    expansion_count = sum(expansion_quotas.values())
    expected_expansion_count = sft_target_count - len(historical_refs)
    if expansion_count != expected_expansion_count:
        raise TaskPoolError(
            "internal SFT quota error: "
            f"expansion={expansion_count}, expected={expected_expansion_count}"
        )
    expansion_refs = _select_stratified(
        inventory,
        split="train",
        count=expansion_count,
        seed=seed,
        quotas=expansion_quotas,
        exclude_keys=historical_keys,
    )
    expansion_refs = [
        TaskRef(
            env_name=ref.env_name,
            task_id=ref.task_id,
            split=ref.split,
            source_index=ref.source_index,
            source_path=ref.source_path,
            role="teacher_expansion",
        )
        for ref in expansion_refs
    ]
    expansion_keys = {ref.key for ref in expansion_refs}
    if historical_keys & expansion_keys:
        raise TaskPoolError("SFT expansion selected a historical SFT task")
    sft_refs = sorted(
        [*historical_refs, *expansion_refs],
        key=lambda ref: (SUPPORTED_ENVS.index(ref.env_name), _task_rank(seed, ref.key)),
    )
    sft_keys = historical_keys | expansion_keys

    # Select a deterministic, composition-stratified 20-task smoke view from
    # the already-reserved SFT pool.  The view is kept inside the formal
    # manifest so smoke and full collection share one task-pool hash/cache
    # provenance while the full pool remains exactly 600 tasks.
    sft_by_env: dict[str, list[TaskRef]] = {env: [] for env in SUPPORTED_ENVS}
    for ref in sft_refs:
        sft_by_env.setdefault(ref.env_name, []).append(ref)
    for env in sft_by_env:
        sft_by_env[env].sort(key=lambda ref: _task_rank(seed, f"sft-smoke:{ref.key}"))
    sft_smoke_refs: list[TaskRef] = []
    for env in SUPPORTED_ENVS:
        quota = int(SFT_SMOKE_QUOTAS.get(env, 0))
        if quota > len(sft_by_env.get(env, [])):
            raise TaskPoolError(
                f"SFT smoke quota for {env}={quota} exceeds reserved SFT tasks "
                f"{len(sft_by_env.get(env, []))}"
            )
        sft_smoke_refs.extend(sft_by_env[env][:quota])
    sft_smoke_refs.sort(key=lambda ref: (SUPPORTED_ENVS.index(ref.env_name), _task_rank(seed, f"sft-smoke:{ref.key}")))
    if len(sft_smoke_refs) != sum(SFT_SMOKE_QUOTAS.values()):
        raise TaskPoolError(
            f"SFT smoke selection returned {len(sft_smoke_refs)} tasks; "
            f"expected {sum(SFT_SMOKE_QUOTAS.values())}"
        )
    sft_smoke_keys = {ref.key for ref in sft_smoke_refs}
    if not sft_smoke_keys <= sft_keys:
        raise TaskPoolError("SFT smoke selection escaped the formal SFT pool")

    train_refs = [ref for ref in train_inventory_refs if ref.key not in sft_keys]
    train_refs.sort(key=lambda ref: (SUPPORTED_ENVS.index(ref.env_name), _task_rank(seed, ref.key)))
    validation_refs = _select_stratified(
        inventory,
        split="test",
        count=validation_size,
        seed=seed,
    )
    validation_keys = {ref.key for ref in validation_refs}
    # Select smoke with the same stratified hash policy (rather than taking
    # the first 20 manifest rows, which would over-represent travel22).
    smoke_refs = _select_stratified(
        inventory,
        split="test",
        count=smoke_size,
        seed=seed,
        quotas=SMOKE_QUOTAS if smoke_size == sum(SMOKE_QUOTAS.values()) else None,
    )
    if smoke_size <= 0 or smoke_size > validation_size:
        raise TaskPoolError("smoke_size must be positive and no larger than validation_size")
    unexpected_unresolved = [
        item
        for item in quarantined
        if str(item.get("opaque_task_key", "")) not in QUARANTINED_OPAQUE_TASK_KEYS
    ]

    pools = {
        "sft": {
            "pool": "sft",
            "split": "train",
            "purpose": "historical SFT plus reserved DeepSeek Teacher expansion",
            "historical_task_count": len(historical_refs),
            "expansion_task_count": len(expansion_refs),
            "records": [ref.to_dict() for ref in sft_refs],
        },
        "sft_smoke": {
            "pool": "sft_smoke",
            "parent_pool": "sft",
            "split": "train",
            "purpose": "20-task stratified smoke collection for DeepSeek Teacher",
            "records": [ref.to_dict() for ref in sft_smoke_refs],
        },
        "grpo": {
            "pool": "grpo",
            "split": "train",
            "purpose": "Actor rollout tasks for GRPO",
            "records": [ref.to_dict() for ref in train_refs],
        },
        "validation": {
            "pool": "validation",
            "split": "test",
            "purpose": "final validation; smoke is a subset",
            "records": [ref.to_dict() for ref in validation_refs],
        },
        "validation_smoke": {
            "pool": "validation_smoke",
            "parent_pool": "validation",
            "split": "test",
            "purpose": "small validation smoke run",
            "records": [ref.to_dict() for ref in smoke_refs],
        },
    }
    source_files: dict[str, dict[str, Any]] = {}
    for ref in inventory.values():
        if not ref.source_path:
            continue
        split_meta = source_files.setdefault(ref.source_path, {"split": ref.split})
        split_meta["task_count"] = int(split_meta.get("task_count", 0)) + 1
    manifest: dict[str, Any] = {
        "manifest_version": SCHEMA_VERSION,
        "name": "travelgym-task-pools",
        "task_key_format": "<env_name>::<task_id>",
        "selection_strategy": "stratified_hash_v1",
        "selection_seed": int(seed),
        "sft_selection_strategy": "historical_plus_composition_deficit_v1",
        "sft_target_count": int(sft_target_count),
        "sft_historical_count": len(historical_refs),
        "sft_expansion_count": len(expansion_refs),
        "sft_smoke_size": len(sft_smoke_refs),
        "sft_smoke_quotas": {env: int(SFT_SMOKE_QUOTAS.get(env, 0)) for env in SUPPORTED_ENVS},
        "validation_size": int(validation_size),
        "smoke_size": int(smoke_size),
        # The active pools contain only authoritative env::task identities.
        # The six owner-designated rows are intentionally discarded and do not
        # make the manifest non-strict; any *other* unresolved source row still
        # requires review before a formal manifest may be consumed.
        "strict_task_identity": not bool(unexpected_unresolved),
        "task_alignment": {
            "reviewed_map_supplied": bool(task_map),
            "reviewed_map_entry_count": int(len(task_map or {})),
            "unresolved_count": int(len(quarantined)),
            "quarantine_policy": "isolate_discard",
            "unexpected_unresolved_count": int(len(unexpected_unresolved)),
        },
        "sft_source": _portable_path(sft_path, root),
        "source_files": source_files,
        # Keep the old field as a compatibility/audit alias, but explicitly
        # state that these rows are excluded from every active pool.
        "unresolved_sft": quarantined,
        "quarantined_sft": quarantined,
        "quarantined_sft_count": len(quarantined),
        "pools": pools,
        "source_task_count": len(inventory),
        "source_split_counts": {
            "train": sum(ref.split == "train" for ref in inventory.values()),
            "test": sum(ref.split == "test" for ref in inventory.values()),
        },
        "pool_task_counts": {
            name: len(value["records"]) for name, value in pools.items()
        },
    }
    assert_task_pools_disjoint(manifest)
    if unexpected_unresolved:
        manifest["disjointness_note"] = (
            "The six owner-designated opaque historical rows are permanently "
            "isolated/discarded. Additional unresolved source rows remain "
            "outside the formal active-pool contract and require review."
        )
    elif quarantined:
        manifest["disjointness_note"] = (
            "Six owner-designated opaque historical rows are permanently "
            "isolated/discarded and excluded from SFT, Teacher collection, "
            "GRPO and Validation. Active pools have strict authoritative task "
            "identities; no reviewed map is required for the quarantine."
        )
    else:
        manifest["disjointness_note"] = "All historical SFT rows resolved to authoritative task IDs."
    return manifest


def _records_for(manifest: Mapping[str, Any], pool_name: str) -> list[dict[str, Any]]:
    pools = manifest.get("pools")
    if not isinstance(pools, Mapping) or pool_name not in pools:
        raise TaskPoolError(f"task pool {pool_name!r} is absent")
    value = pools[pool_name]
    if not isinstance(value, Mapping) or not isinstance(value.get("records"), list):
        raise TaskPoolError(f"task pool {pool_name!r} has no records")
    return [dict(item) for item in value["records"] if isinstance(item, Mapping)]


def pool_task_keys(manifest: Mapping[str, Any], pool_name: str) -> set[str]:
    keys: set[str] = set()
    for item in _records_for(manifest, pool_name):
        ref = _as_task_ref(item)
        if ref.env_name == "opaque_sft" or ref.key.startswith(OPAQUE_TASK_KEY_PREFIX):
            raise TaskPoolError(
                f"opaque task {ref.key!r} must remain in quarantined_sft and cannot "
                f"appear in active pool {pool_name!r}"
            )
        if ref.key in keys:
            raise TaskPoolError(f"duplicate task in pool {pool_name}: {ref.key}")
        keys.add(ref.key)
    return keys


def assert_task_pools_disjoint(manifest: Mapping[str, Any], *, require_strict: bool = False) -> None:
    """Validate pairwise disjointness and smoke-subset invariants."""

    if manifest.get("manifest_version") != SCHEMA_VERSION:
        raise TaskPoolError("unsupported task-pool manifest version")
    active = {name: pool_task_keys(manifest, name) for name in POOL_NAMES}
    for left_index, left in enumerate(POOL_NAMES):
        for right in POOL_NAMES[left_index + 1 :]:
            overlap = active[left] & active[right]
            if overlap:
                preview = sorted(overlap)[:5]
                raise TaskPoolError(f"task-pool overlap {left}/{right}: {preview}")
    smoke = pool_task_keys(manifest, SMOKE_POOL_NAME)
    if not smoke <= active["validation"]:
        raise TaskPoolError("validation_smoke contains task(s) outside validation pool")
    if "sft_smoke" in (manifest.get("pools") or {}):
        sft_smoke = pool_task_keys(manifest, "sft_smoke")
        if not sft_smoke <= active["sft"]:
            raise TaskPoolError("sft_smoke contains task(s) outside the SFT pool")
    if require_strict and not _is_true_flag(manifest.get("strict_task_identity", False)):
        quarantined = manifest.get("quarantined_sft", manifest.get("unresolved_sft", []))
        raise TaskPoolError(
            "strict task disjointness is unavailable for this manifest; "
            f"quarantined_or_unresolved_count={len(quarantined) if isinstance(quarantined, list) else '?'}"
        )


def load_pool_manifest(
    path: str | Path,
    *,
    require_strict: bool = False,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    manifest_path = Path(path).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"task-pool manifest not found: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TaskPoolError(f"invalid task-pool manifest: {manifest_path}") from exc
    if not isinstance(manifest, Mapping):
        raise TaskPoolError("task-pool manifest must be a JSON object")
    assert_task_pools_disjoint(manifest, require_strict=require_strict)
    return dict(manifest)


def task_keys_from_audits(
    audits: Iterable[Mapping[str, Any]],
    *,
    inventory: Mapping[str, TaskRef],
) -> tuple[set[str], list[dict[str, Any]]]:
    """Extract private task identities from merge audits for pool enforcement."""

    keys: set[str] = set()
    unresolved: list[dict[str, Any]] = []
    for index, audit in enumerate(audits):
        env_name = audit.get("env_name")
        task_id = audit.get("task_id")
        if env_name and task_id:
            key = make_task_key(env_name, task_id)
            if key in inventory:
                keys.add(key)
                continue
        unresolved.append({"audit_index": index, "task_id": task_id, "env_name": env_name})
    return keys, unresolved


def assert_audits_in_pool(
    audits: Iterable[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    *,
    pool_name: str = "sft",
    inventory: Mapping[str, TaskRef] | None = None,
) -> None:
    """Fail closed when a Teacher/cache record is outside its declared pool."""

    allowed = pool_task_keys(manifest, pool_name)
    if inventory is None:
        # Audits generated by the merge script carry an authoritative
        # ``task_key`` when available; this path is useful for unit tests.
        inventory = {}
    observed: set[str] = set()
    unresolved: list[int] = []
    for index, audit in enumerate(audits):
        task_key = audit.get("task_key")
        if not task_key:
            env_name, task_id = audit.get("env_name"), audit.get("task_id")
            if env_name and task_id:
                task_key = make_task_key(env_name, task_id)
        if not task_key:
            unresolved.append(index)
            continue
        task_key = str(task_key)
        observed.add(task_key)
        if task_key not in allowed:
            raise TaskPoolError(
                f"record {index} uses task {task_key!r}, outside {pool_name} task pool"
            )
    if unresolved:
        raise TaskPoolError(
            f"{len(unresolved)} record(s) have no resolvable task identity; "
            "provide a reviewed task alignment map before pool-constrained collection"
        )


def filter_records_to_pool(
    records: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    *,
    pool_name: str,
) -> list[dict[str, Any]]:
    """Filter parquet-like records by reward_model.id and env_name."""

    allowed = pool_task_keys(manifest, pool_name)
    allowed_by_id: dict[str, set[str]] = {}
    for key in allowed:
        _env, task_id = split_task_key(key)
        allowed_by_id.setdefault(task_id, set()).add(key)
    kept: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in records:
        reward_model = row.get("reward_model") if isinstance(row, Mapping) else None
        if not isinstance(reward_model, Mapping) or not reward_model.get("id"):
            raise TaskPoolError("dataset row has no reward_model.id")
        env_name = str(reward_model.get("task_env_name") or row.get("env_name") or "")
        # The parquet schema stores TravelGym in reward_model.env_name; the
        # composition is encoded by the task ID and is supplied by callers
        # while iterating variant files.
        if not env_name or env_name == "TravelGym":
            env_name = str(row.get("_pool_env_name") or "")
        task_id = str(reward_model["id"])
        if env_name:
            key = make_task_key(env_name, task_id)
        else:
            candidates = allowed_by_id.get(task_id, set())
            if len(candidates) != 1:
                raise TaskPoolError(
                    f"dataset row task {task_id!r} has no unique composition env_name"
                )
            key = next(iter(candidates))
        if key in allowed and key not in seen:
            kept.append(dict(row))
            seen.add(key)
    return kept


def write_manifest(manifest: Mapping[str, Any], output_path: str | Path) -> Path:
    target = Path(output_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--sft-source", type=Path, default=Path("data/sft/travel_sft_public.json"))
    parser.add_argument("--task-map", type=Path, default=None)
    parser.add_argument(
        "--candidate-audit-output",
        type=Path,
        default=Path("data/task_pools/sft_task_alignment_candidates.json"),
        help="Private opaque-row audit; discarded rows are never reintroduced.",
    )
    parser.add_argument("--output", type=Path, default=Path("data/task_pools/travel_task_pools.json"))
    parser.add_argument("--seed", type=int, default=TASK_POOL_SELECTION_SEED)
    parser.add_argument(
        "--sft-target-count",
        type=int,
        default=DEFAULT_SFT_TARGET_COUNT,
        help=(
            "Number of resolved train tasks reserved for SFT (historical plus "
            f"deterministic Teacher expansion; default: {DEFAULT_SFT_TARGET_COUNT})."
        ),
    )
    parser.add_argument("--validation-size", type=int, default=200)
    parser.add_argument("--smoke-size", type=int, default=20)
    parser.add_argument(
        "--require-strict",
        action="store_true",
        help="Fail unless the active pools have strict task identities.",
    )
    args = parser.parse_args()
    root = args.project_root.resolve()
    task_map = TravelTaskResolver.load_alignment_map(args.task_map) if args.task_map else {}
    source_path = (root / args.sft_source).resolve() if not args.sft_source.is_absolute() else args.sft_source.resolve()
    inventory = load_task_inventory(root)
    candidate_report = build_sft_candidate_audit(
        source_path,
        project_root=root,
        inventory=inventory,
        task_map=task_map,
    )
    manifest = build_task_pool_manifest(
        project_root=root,
        sft_source=source_path,
        task_map=task_map,
        seed=args.seed,
        sft_target_count=args.sft_target_count,
        validation_size=args.validation_size,
        smoke_size=args.smoke_size,
    )
    audit_output = (root / args.candidate_audit_output).resolve() if not args.candidate_audit_output.is_absolute() else args.candidate_audit_output.resolve()
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    temporary_audit = audit_output.with_suffix(audit_output.suffix + ".tmp")
    temporary_audit.write_text(
        json.dumps(
            {
                "schema_version": "travelgym-task-alignment-audit-v1",
                "source_path": _portable_path(source_path, root),
                "requires_review": any(bool(row.get("requires_review")) for row in candidate_report),
                "quarantine_policy": "isolate_discard",
                "records": candidate_report,
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    temporary_audit.replace(audit_output)
    # Validate after persisting the candidate report: a failed strict build
    # still leaves the exact artifact a reviewer needs to create the map.
    assert_task_pools_disjoint(manifest, require_strict=args.require_strict)
    output = (root / args.output).resolve() if not args.output.is_absolute() else args.output.resolve()
    write_manifest(manifest, output)
    print(
        f"wrote {output} "
        f"sft={len(_records_for(manifest, 'sft'))} "
        f"sft_smoke={len(_records_for(manifest, 'sft_smoke'))} "
        f"grpo={len(_records_for(manifest, 'grpo'))} "
        f"validation={len(_records_for(manifest, 'validation'))} "
        f"smoke={len(_records_for(manifest, 'validation_smoke'))} "
        f"sft_expansion={manifest.get('sft_expansion_count', 0)} "
        f"quarantined_sft={len(manifest.get('quarantined_sft', []))} "
        f"strict={manifest.get('strict_task_identity', False)} candidate_audit={audit_output}"
    )


if __name__ == "__main__":  # pragma: no cover
    main()


__all__ = [
    "POOL_NAMES",
    "SCHEMA_VERSION",
    "SMOKE_POOL_NAME",
    "SMOKE_QUOTAS",
    "SFT_SMOKE_QUOTAS",
    "SUPPORTED_ENVS",
    "DEFAULT_SFT_TARGET_COUNT",
    "OPAQUE_TASK_KEY_PREFIX",
    "QUARANTINED_OPAQUE_RECORD_INDICES",
    "QUARANTINED_OPAQUE_TASK_KEYS",
    "TASK_POOL_SELECTION_SEED",
    "TASK_KEY_SEPARATOR",
    "TaskPoolError",
    "TaskRef",
    "assert_audits_in_pool",
    "assert_task_pools_disjoint",
    "build_task_pool_manifest",
    "build_sft_candidate_audit",
    "filter_records_to_pool",
    "load_pool_manifest",
    "load_task_inventory",
    "make_task_key",
    "opaque_task_key",
    "is_permanently_quarantined_opaque",
    "pool_task_keys",
    "resolve_sft_task_refs",
    "split_task_key",
    "task_keys_from_audits",
    "write_manifest",
]
