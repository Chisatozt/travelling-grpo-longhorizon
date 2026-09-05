"""Offline-safe DeepSeek Teacher collection cache and provenance helpers.

The network client lives in :mod:`travel_grpo.evaluation.eval`; this module intentionally has no
OpenAI/DeepSeek dependency.  It gives the collector an idempotent key
``env_name::task_id::pass_index`` and a small atomic cache so a failed process
can resume without billing a completed pass again.
"""

from __future__ import annotations

import json
import hashlib
import os
import re
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .._paths import REPOSITORY_ROOT


class TeacherCacheError(ValueError):
    pass


def teacher_pass_key(env_name: str, task_id: str, pass_index: int) -> str:
    env = str(env_name).strip()
    task = str(task_id).strip()
    index = int(pass_index)
    if not env or not task or index not in (0, 1):
        raise TeacherCacheError("teacher pass key requires env/task and pass_index 0 or 1")
    return f"{env}::{task}::{index}"


def code_revision(default: str = "unknown") -> str:
    """Return an immutable code revision without requiring a Git checkout.

    Collection runs are often executed from an exported workspace rather than
    a Git worktree.  In that case we derive a short content hash from the
    source files that define collection, TravelGym, and SFT canonicalisation.
    This keeps the SwanLab label useful while avoiding any secret or machine
    specific values.
    """

    value = os.environ.get("CODE_REVISION")
    if value:
        return value
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass

    # The repository may intentionally not contain ``.git`` (for example when
    # copied to a training server).  Hash only source files relevant to the
    # trajectory pipeline; generated caches, datasets, and credentials are
    # deliberately excluded.
    root = REPOSITORY_ROOT
    source_roots = ("src/travel_grpo", "environments/TravelGym", "verl")
    source_files = []
    for relative_root in source_roots:
        candidate = root / relative_root
        if candidate.is_dir():
            source_files.extend(path for path in candidate.rglob("*.py") if path.is_file())
    if not source_files:
        return default
    digest = hashlib.sha256()
    for path in sorted(source_files):
        try:
            relative = path.relative_to(root).as_posix().encode("utf-8")
            digest.update(relative + b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
        except OSError:
            continue
    return f"workspace-{digest.hexdigest()[:16]}"


def make_provenance(
    *,
    env_name: str,
    task_id: str,
    pass_index: int,
    model: str,
    collection_run_id: str,
    thinking: str = "enabled",
    reasoning_effort: str = "high",
    max_turns: int = 25,
    revision: str | None = None,
    task_pool_hash: str | None = None,
    api_endpoint_label: str | None = None,
) -> dict[str, Any]:
    """Build private cache provenance; never put this object in Actor context."""

    result: dict[str, Any] = {
        "task_key": f"{str(env_name).strip()}::{str(task_id).strip()}",
        "env_name": str(env_name).strip(),
        "task_id": str(task_id).strip(),
        "pass_index": int(pass_index),
        "model": str(model),
        "thinking": str(thinking),
        "reasoning_effort": str(reasoning_effort),
        "max_turns": int(max_turns),
        "code_revision": revision or code_revision(),
        "collection_run_id": str(collection_run_id),
    }
    if task_pool_hash:
        result["task_pool_hash"] = str(task_pool_hash)
    if api_endpoint_label:
        result["api_endpoint_label"] = str(api_endpoint_label)
    return result


def sanitize_tracking_payload(payload: Any) -> Any:
    """Remove private labels, credentials and raw reasoning from tracking.

    The private cache still stores reasoning_content for replay.  This helper
    is only for SwanLab/other external tracking payloads.
    """

    forbidden = {
        "api_key", "apikey", "authorization", "token", "correct_id", "correct_ids",
        "best_id", "best_ids", "preference_id", "preference_ids", "hidden_ids",
        "hidden_labels", "reward_ledger", "raw_reward", "reasoning_content", "cot",
        "messages", "transcript",
    }
    if isinstance(payload, Mapping):
        result: dict[str, Any] = {}
        for key, value in payload.items():
            lowered = str(key).casefold()
            if lowered in forbidden or any(part in lowered for part in ("api_key", "preference_id", "correct_id", "best_id", "reward_ledger", "reasoning_content")):
                continue
            result[str(key)] = sanitize_tracking_payload(value)
        return result
    if isinstance(payload, (list, tuple)):
        return [sanitize_tracking_payload(value) for value in payload]
    if isinstance(payload, str):
        text = re.sub(
            r"<(?:think|analysis|reasoning)>.*?</(?:think|analysis|reasoning)>",
            "[REASONING_REDACTED]",
            payload,
            flags=re.IGNORECASE | re.DOTALL,
        )
        text = re.sub(
            r"(?:reasoning_content|chain_of_thought|scratchpad)\s*[:=].*?(?=\n\s*[A-Za-z_][\w -]*\s*[:=]|$)",
            "[REASONING_REDACTED]",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        text = re.sub(
            r"^\s*(?:preference[_ ]?ids?|correct[_ ]?ids?|best[_ ]?ids?|"
            r"ground[_ ]?truth|reward[_ ]?report|diagnostics?)\s*[:=].*$",
            "[PRIVATE_REDACTED]",
            text,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        text = re.sub(r"(?<![A-Za-z0-9])P\d+(?![A-Za-z0-9])", "[PRIVATE_ID]", text, flags=re.IGNORECASE)
        if "sk-" in text or "api_key" in text.casefold():
            return "[REDACTED]"
        return text
    return payload


@dataclass
class TeacherCache:
    """Atomic JSON cache with deterministic pass-level idempotency."""

    path: Path
    collection_run_id: str
    model: str = "unknown"
    thinking: str = "enabled"
    reasoning_effort: str = "high"
    max_turns: int = 25
    revision: str | None = None
    task_pool_hash: str | None = None
    api_endpoint_label: str | None = None

    SCHEMA_VERSION = "travelgym-teacher-cache-v1"

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.records: dict[str, dict[str, Any]] = {}
        self.stats: dict[str, Any] = {
            "completed_tasks": 0,
            "valid_trajectories": 0,
            "invalid_trajectories": 0,
            "abandoned_trajectories": 0,
            "api_errors": 0,
            "missing_reasoning": 0,
            "tokens": 0,
            "elapsed_seconds": 0.0,
            "estimated_cost": 0.0,
        }
        self._load()
        self._refresh_completed_tasks()

    def _default_provenance(self, *, env_name: str, task_id: str, pass_index: int) -> dict[str, Any]:
        """Build provenance for in-flight/error records as well as successes."""

        return make_provenance(
            env_name=env_name,
            task_id=task_id,
            pass_index=pass_index,
            model=self.model,
            collection_run_id=self.collection_run_id,
            thinking=self.thinking,
            reasoning_effort=self.reasoning_effort,
            max_turns=self.max_turns,
            revision=self.revision,
            task_pool_hash=self.task_pool_hash,
            api_endpoint_label=self.api_endpoint_label,
        )

    def _load(self) -> None:
        if not self.path.is_file():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TeacherCacheError(f"invalid teacher cache: {self.path}") from exc
        if not isinstance(payload, Mapping) or payload.get("schema_version") != self.SCHEMA_VERSION:
            raise TeacherCacheError("unsupported teacher cache schema")
        existing_run_id = str(payload.get("collection_run_id", ""))
        if existing_run_id and existing_run_id != str(self.collection_run_id):
            raise TeacherCacheError(
                f"teacher cache collection run mismatch: {existing_run_id} != {self.collection_run_id}"
            )
        records = payload.get("records", {})
        if isinstance(records, Mapping):
            loaded: dict[str, dict[str, Any]] = {}
            for raw_key, raw_value in records.items():
                if not isinstance(raw_value, Mapping):
                    raise TeacherCacheError("teacher cache record must be an object")
                key = str(raw_key)
                provenance = raw_value.get("provenance", {})
                if not isinstance(provenance, Mapping):
                    raise TeacherCacheError(f"teacher cache provenance is invalid for {key}")
                try:
                    pass_index = int(provenance.get("pass_index", key.rsplit("::", 1)[-1]))
                    env_task = str(provenance.get("task_key", key.rsplit("::", 1)[0]))
                    env_name, task_id = env_task.split("::", 1)
                    expected_key = teacher_pass_key(env_name, task_id, pass_index)
                except (TypeError, ValueError, TeacherCacheError) as exc:
                    raise TeacherCacheError(f"teacher cache record key is invalid: {key}") from exc
                if expected_key != key:
                    raise TeacherCacheError(
                        f"teacher cache record key/provenance mismatch: {key} != {expected_key}"
                    )
                # Keep the duplicated convenience fields honest.  A copied
                # cache entry must not be able to change env/task provenance
                # while leaving the stable task_key untouched.
                if provenance.get("env_name") not in (None, "", env_name):
                    raise TeacherCacheError(f"teacher cache env_name mismatch: {key}")
                if provenance.get("task_id") not in (None, "", task_id):
                    raise TeacherCacheError(f"teacher cache task_id mismatch: {key}")
                if self.task_pool_hash and provenance.get("task_pool_hash") not in (None, "", self.task_pool_hash):
                    raise TeacherCacheError(f"teacher cache task-pool hash mismatch: {key}")
                loaded[key] = dict(raw_value)
            self.records = loaded
        if isinstance(payload.get("stats"), Mapping):
            self.stats.update(dict(payload["stats"]))

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": self.SCHEMA_VERSION,
            "collection_run_id": self.collection_run_id,
            "records": self.records,
            "stats": self.stats,
        }
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.path)

    def has_success(self, env_name: str, task_id: str, pass_index: int) -> bool:
        value = self.records.get(teacher_pass_key(env_name, task_id, pass_index))
        return bool(isinstance(value, Mapping) and value.get("status") == "success")

    def pending(self, tasks: Iterable[Mapping[str, Any]], pass_k: int = 2) -> list[dict[str, Any]]:
        if int(pass_k) != 2:
            raise TeacherCacheError("the SFT Teacher protocol requires pass_k=2")
        output: list[dict[str, Any]] = []
        for task in tasks:
            env_name, task_id = task.get("env_name"), task.get("task_id")
            if not env_name or not task_id:
                raise TeacherCacheError("task pool row has no env_name/task_id")
            for index in (0, 1):
                key = teacher_pass_key(str(env_name), str(task_id), index)
                existing = self.records.get(key)
                # ``in_flight`` means a previous process may already have
                # been billed.  Never automatically retry it; a human/API
                # reconciliation can explicitly clear the marker.
                # ``invalid`` means the provider returned a complete pass but
                # the trajectory failed the local protocol/Thinking check.
                # It is retained for audit and is not silently re-issued on
                # resume (which would bill a second request for the same
                # fixed pass).  A retryable transport ``error`` may be
                # retried because no provider response was committed.
                if not self.has_success(str(env_name), str(task_id), index) and not (
                    isinstance(existing, Mapping)
                    and existing.get("status") in {"in_flight", "invalid", "abandoned"}
                ):
                    output.append({"env_name": str(env_name), "task_id": str(task_id), "pass_index": index, "task_key": teacher_pass_key(str(env_name), str(task_id), index)})
        return output

    def claim(
        self,
        *,
        env_name: str,
        task_id: str,
        pass_index: int,
        provenance: Mapping[str, Any] | None = None,
    ) -> str | None:
        """Atomically mark a pass in-flight before the paid request starts."""
        key = teacher_pass_key(env_name, task_id, pass_index)
        if self.has_success(env_name, task_id, pass_index):
            return None
        existing = self.records.get(key)
        if isinstance(existing, Mapping) and existing.get("status") in {"in_flight", "invalid", "abandoned"}:
            return None
        request_id = str(uuid.uuid4())
        claim_provenance = self._default_provenance(
            env_name=env_name, task_id=task_id, pass_index=pass_index
        )
        if provenance is not None:
            claim_provenance.update(dict(provenance))
        self.records[key] = {
            "status": "in_flight",
            "request_id": request_id,
            "provenance": claim_provenance,
            "claimed_at": time.time(),
        }
        self._refresh_completed_tasks()
        self._write()
        return request_id

    def record_success(
        self,
        *,
        env_name: str,
        task_id: str,
        pass_index: int,
        trajectory: Mapping[str, Any],
        provenance: Mapping[str, Any],
        token_count: int = 0,
        elapsed_seconds: float = 0.0,
        estimated_cost: float = 0.0,
        valid: bool = True,
        missing_reasoning_count: int = 0,
    ) -> bool:
        key = teacher_pass_key(env_name, task_id, pass_index)
        if self.has_success(env_name, task_id, pass_index):
            return False
        expected_task_key = f"{str(env_name).strip()}::{str(task_id).strip()}"
        if str(provenance.get("task_key", "")) != expected_task_key:
            raise TeacherCacheError(
                "provenance task key does not match the cache record: "
                f"{provenance.get('task_key')!r} != {expected_task_key!r}"
            )
        try:
            provenance_pass_index = int(provenance.get("pass_index"))
        except (TypeError, ValueError) as exc:
            raise TeacherCacheError("provenance pass_index is invalid") from exc
        if provenance_pass_index != int(pass_index):
            raise TeacherCacheError(
                "provenance pass_index does not match the cache record: "
                f"{provenance_pass_index} != {int(pass_index)}"
            )
        if not provenance.get("reasoning_effort") or provenance.get("thinking") != "enabled":
            raise TeacherCacheError("Teacher provenance must record thinking=enabled and reasoning_effort")
        messages = trajectory.get("messages", []) if isinstance(trajectory, Mapping) else []
        missing_reasoning = any(
            isinstance(message, Mapping)
            and message.get("role") == "assistant"
            and message.get("tool_calls")
            and not str(message.get("reasoning_content") or "").strip()
            for message in messages
        )
        if missing_reasoning:
            self.stats["missing_reasoning"] = int(self.stats.get("missing_reasoning", 0)) + 1
        missing_reasoning_count = max(int(missing_reasoning_count), int(missing_reasoning))
        if missing_reasoning_count:
            self.stats["missing_reasoning"] = int(self.stats.get("missing_reasoning", 0)) + max(
                0, missing_reasoning_count - int(missing_reasoning)
            )
        now = time.time()
        self.records[key] = {
            "status": "success" if valid else "invalid",
            "trajectory": dict(trajectory),
            "provenance": dict(provenance),
            "token_count": int(token_count),
            "elapsed_seconds": float(elapsed_seconds),
            "estimated_cost": float(estimated_cost),
            "missing_reasoning": bool(missing_reasoning),
            "written_at": now,
        }
        self.stats["valid_trajectories"] = int(self.stats.get("valid_trajectories", 0)) + int(bool(valid))
        self.stats["invalid_trajectories"] = int(self.stats.get("invalid_trajectories", 0)) + int(not valid)
        self.stats["tokens"] = int(self.stats.get("tokens", 0)) + int(token_count)
        self.stats["elapsed_seconds"] = float(self.stats.get("elapsed_seconds", 0.0)) + float(elapsed_seconds)
        self.stats["estimated_cost"] = float(self.stats.get("estimated_cost", 0.0)) + float(estimated_cost)
        self._refresh_completed_tasks()
        self._write()
        return True

    def reconcile_in_flight(self, *, env_name: str, task_id: str, pass_index: int, retry: bool) -> None:
        """Explicitly clear a paid-request marker after provider reconciliation."""
        key = teacher_pass_key(env_name, task_id, pass_index)
        value = self.records.get(key)
        if not isinstance(value, Mapping) or value.get("status") != "in_flight":
            return
        if retry:
            self.records.pop(key, None)
        else:
            self.records[key] = {**dict(value), "status": "error", "reconcile_required": False}
        self._refresh_completed_tasks()
        self._write()

    def abandon_in_flight(
        self,
        *,
        env_name: str,
        task_id: str,
        pass_index: int,
        reason: str = "abandoned_by_user",
    ) -> bool:
        """Mark an unresolved paid-request marker as a terminal failure.

        This is intentionally separate from ``invalid``: an ``invalid`` pass
        has a provider response that failed local validation, whereas an
        abandoned pass has no trustworthy response and must never be silently
        re-issued on resume.  The original request/provenance fields remain
        intact for audit and billing reconciliation.
        """

        key = teacher_pass_key(env_name, task_id, pass_index)
        value = self.records.get(key)
        if not isinstance(value, Mapping) or value.get("status") != "in_flight":
            return False
        record = dict(value)
        record.update(
            {
                "status": "abandoned",
                "abandoned_at": time.time(),
                "abandon_reason": str(reason),
                "reconcile_required": False,
            }
        )
        self.records[key] = record
        self.stats["abandoned_trajectories"] = int(self.stats.get("abandoned_trajectories", 0)) + 1
        self._refresh_completed_tasks()
        self._write()
        return True

    def record_error(
        self,
        *,
        env_name: str,
        task_id: str,
        pass_index: int,
        error: str,
        retryable: bool = True,
        provenance: Mapping[str, Any] | None = None,
    ) -> None:
        key = teacher_pass_key(env_name, task_id, pass_index)
        if self.has_success(env_name, task_id, pass_index):
            return
        # Error entries are resumable and never count as completed/billable.
        existing = self.records.get(key)
        record_provenance = (
            dict(existing.get("provenance", {}))
            if isinstance(existing, Mapping) and isinstance(existing.get("provenance"), Mapping)
            else self._default_provenance(env_name=env_name, task_id=task_id, pass_index=pass_index)
        )
        if provenance is not None:
            record_provenance.update(dict(provenance))
        self.records[key] = {
            "status": "error" if retryable else "in_flight",
            "error": str(error),
            "reconcile_required": not retryable,
            "provenance": record_provenance,
            "written_at": time.time(),
        }
        self.stats["api_errors"] = int(self.stats.get("api_errors", 0)) + 1
        self._refresh_completed_tasks()
        self._write()

    def _refresh_completed_tasks(self) -> None:
        task_pairs = set()
        for value in self.records.values():
            if value.get("status") != "success":
                continue
            provenance = value.get("provenance", {})
            task_pairs.add(str(provenance.get("task_key", "")))
        completed = 0
        for task_key in task_pairs:
            if all(any(v.get("status") == "success" and v.get("provenance", {}).get("task_key") == task_key and int(v.get("provenance", {}).get("pass_index", -1)) == index for v in self.records.values()) for index in (0, 1)):
                completed += 1
        self.stats["completed_tasks"] = completed
        self.stats["in_flight_passes"] = sum(
            int(isinstance(value, Mapping) and value.get("status") == "in_flight")
            for value in self.records.values()
        )


def summarize_pass_stats(records: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Return pass-indexed success/invalid/error counts for collection reports."""

    summary: dict[str, dict[str, Any]] = {}
    for pass_index in (0, 1):
        counts = {
            "attempted": 0,
            "success": 0,
            "invalid": 0,
            "abandoned": 0,
            "error": 0,
            "in_flight": 0,
        }
        for value in records.values():
            if not isinstance(value, Mapping):
                continue
            provenance = value.get("provenance", {})
            if not isinstance(provenance, Mapping):
                continue
            try:
                current_pass = int(provenance.get("pass_index", -1))
            except (TypeError, ValueError):
                continue
            if current_pass != pass_index:
                continue
            status = str(value.get("status", "error"))
            if status not in counts:
                status = "error"
            counts[status] += 1
            counts["attempted"] += 1
        counts["success_rate"] = (
            float(counts["success"]) / float(counts["attempted"])
            if counts["attempted"] else 0.0
        )
        summary[str(pass_index)] = counts
    return summary


__all__ = [
    "TeacherCache",
    "TeacherCacheError",
    "code_revision",
    "make_provenance",
    "sanitize_tracking_payload",
    "summarize_pass_stats",
    "teacher_pass_key",
]
