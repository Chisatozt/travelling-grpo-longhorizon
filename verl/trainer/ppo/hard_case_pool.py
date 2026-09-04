"""Trainer-private append-only Hard Case Pool for TravelGym GRPO.

The observer is deliberately passive: it receives completed rollout-group
metadata, writes an audit file on rank 0, and never changes sampling, rewards,
advantages, or Actor observations.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


def compose_task_key(task_id: Any, env_name: Any | None = None) -> str:
    """Build a private, collision-resistant ``env_name::task_id`` key."""

    task = str(task_id or "").strip()
    if not task:
        return ""
    if "::" in task:
        return task
    env = str(env_name or "").strip()
    return f"{env}::{task}" if env else task


class HardCasePool:
    """Observe valid all-zero groups and persist three-group admissions."""

    def __init__(self, path: str | Path, *, threshold: int = 3, reward_version: str = "travelgym-terminal-v2", enabled: bool = True, rank: int = 0):
        if threshold <= 0:
            raise ValueError("threshold must be positive")
        self.path = Path(path)
        self.threshold = int(threshold)
        self.reward_version = str(reward_version)
        self.enabled = bool(enabled)
        self.rank = int(rank)
        self.streaks: dict[str, int] = {}
        self.admitted: dict[str, dict[str, Any]] = {}
        self.groups_seen = 0
        self._load()

    def _load(self) -> None:
        if not self.path.is_file():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"invalid Hard Case Pool state: {self.path}") from exc
        if not isinstance(payload, Mapping):
            raise ValueError("Hard Case Pool state must be an object")
        if payload.get("threshold") is not None and int(payload.get("threshold")) != self.threshold:
            raise ValueError("Hard Case Pool threshold mismatch")
        if payload.get("reward_version") is not None and str(payload.get("reward_version")) != self.reward_version:
            raise ValueError("Hard Case Pool reward version mismatch")
        self.streaks = {str(key): int(value) for key, value in (payload.get("streaks", {}) or {}).items()}
        self.admitted = {str(key): dict(value) for key, value in (payload.get("admitted", {}) or {}).items() if isinstance(value, Mapping)}
        self.groups_seen = int(payload.get("groups_seen", 0) or 0)

    def state_dict(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "reward_version": self.reward_version,
            "streaks": dict(self.streaks),
            "admitted": dict(self.admitted),
            "groups_seen": self.groups_seen,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping):
            raise ValueError("Hard Case Pool checkpoint state must be an object")
        if state.get("threshold") is not None and int(state.get("threshold")) != self.threshold:
            raise ValueError("Hard Case Pool checkpoint threshold mismatch")
        if state.get("reward_version") is not None and str(state.get("reward_version")) != self.reward_version:
            raise ValueError("Hard Case Pool checkpoint reward version mismatch")
        self.streaks = {str(key): int(value) for key, value in (state.get("streaks", {}) or {}).items()}
        self.admitted = {str(key): dict(value) for key, value in (state.get("admitted", {}) or {}).items() if isinstance(value, Mapping)}
        self.groups_seen = int(state.get("groups_seen", 0) or 0)

    def _persist(self) -> None:
        if not self.enabled or self.rank != 0:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.state_dict()
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, self.path)

    def observe_group(
        self,
        *,
        task_id: str,
        source: str = "unknown",
        reward_valid: Sequence[bool],
        correct_completion: Sequence[float],
        group_size: int,
        step: int,
    ) -> dict[str, Any]:
        """Record one complete group; invalid groups never count as hard cases."""
        if not self.enabled:
            return {"qualified": False, "reason": "disabled"}
        task_key = str(task_id)
        valid = len(reward_valid) == int(group_size) and all(bool(value) for value in reward_valid)
        all_zero = valid and len(correct_completion) == int(group_size) and all(float(value) <= 0.0 for value in correct_completion)
        self.groups_seen += 1
        if not all_zero:
            # A non-zero valid group breaks consecutive zero streak.  An
            # invalid group also breaks it, but is never counted itself.
            self.streaks[task_key] = 0
            self._persist()
            return {"qualified": False, "reason": "invalid_or_nonzero", "streak": 0}
        streak = self.streaks.get(task_key, 0) + 1
        self.streaks[task_key] = streak
        qualified = streak >= self.threshold
        if qualified and task_key not in self.admitted:
            self.admitted[task_key] = {
                "task_id": task_key,
                "data_source": str(source),
                "zero_group_streak": streak,
                "first_admitted_step": int(step),
                "group_size": int(group_size),
                "reward_version": self.reward_version,
                "admitted_at": time.time(),
            }
        elif task_key in self.admitted:
            self.admitted[task_key]["zero_group_streak"] = streak
        self._persist()
        return {"qualified": qualified, "reason": "all_zero_valid", "streak": streak, "admitted": qualified}

    def observe_output(
        self,
        output: Any,
        *,
        task_ids: Sequence[Any] | None = None,
        sources: Sequence[Any] | None = None,
        group_size: int,
        step: int,
    ) -> list[dict[str, Any]]:
        """Best-effort extraction from VERL non-tensor output metadata."""
        if not self.enabled:
            return []
        raw = getattr(output, "non_tensor_batch", {}).get("reward_scores")
        if raw is None:
            return []
        if hasattr(raw, "tolist"):
            raw = raw.tolist()
        rows = list(raw) if isinstance(raw, (list, tuple)) else [raw]
        if group_size <= 1 or len(rows) < group_size or len(rows) % group_size:
            # A trailing partial group is not an independent complete rollout
            # group and must never advance a task's zero streak.
            return []
        for values_name, values_seq in (
            ("task_ids", task_ids),
            ("sources", sources),
        ):
            if values_seq is not None and len(values_seq) != len(rows):
                return []
        # Reward scores are keyed by tool name in the current rollout worker.
        def values(key: str, default: float = 0.0) -> list[float]:
            result = []
            for row in rows:
                if isinstance(row, Mapping):
                    try:
                        result.append(float(row.get(key, default)))
                    except (TypeError, ValueError):
                        result.append(default)
                else:
                    result.append(default)
            return result
        if not all(
            isinstance(row, Mapping)
            and "interact_with_env_reward_valid" in row
            and "interact_with_env_correct_completion" in row
            and "interact_with_env_terminal_only" in row
            for row in rows
        ):
            return []
        rewards_valid = [bool(value) for value in values("interact_with_env_reward_valid", 0.0)]
        correctness = values("interact_with_env_correct_completion", 0.0)
        terminal_only = [bool(value) for value in values("interact_with_env_terminal_only", 0.0)]
        # Hard cases are admitted only from a genuinely terminal-only Travel
        # score.  A mixed/non-terminal rollout must never be turned into a
        # zero-reward training target by this passive observer.
        if not all(terminal_only):
            return []
        if task_ids is None:
            task_ids = [f"unknown-{index // group_size}" for index in range(len(rows))]
        if sources is None:
            sources = ["unknown"] * len(rows)
        outcomes = []
        for start in range(0, len(rows) - group_size + 1, group_size):
            task = str(task_ids[start] if start < len(task_ids) else f"unknown-{start // group_size}")
            source = str(sources[start] if start < len(sources) else "unknown")
            outcomes.append(
                self.observe_group(
                    task_id=task,
                    source=source,
                    reward_valid=rewards_valid[start : start + group_size],
                    correct_completion=correctness[start : start + group_size],
                    group_size=group_size,
                    step=step,
                )
            )
        return outcomes


__all__ = ["HardCasePool", "compose_task_key"]
