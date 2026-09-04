"""Small, framework-independent helpers for task-level validation passes.

The training validation path historically treated repeated validation rows as
an ordinary flat batch.  That is not the same protocol as pass@k: attempts
belong to one task, and a successful attempt removes that task from later
attempts.  This module keeps the accounting pure so it can be tested without
Ray, CUDA, SGLang, or an API client.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


PUBLIC_VALIDATION_METRICS = (
    "terminal_reward",
    "completion_success",
    "correct_completion",
    "answer_quality",
    "best_answer_rate",
    "answer_coverage",
    "legal_chain_rate",
    "coverage_adjusted_answer_quality",
    "coverage_adjusted_legal_chain_rate",
    "hidden_preference_hit_rate",
    "efficiency",
    "unanswered_count",
    "agent_elicited_preference_count",
    "proactive_preference_count",
    "useful_action_count",
    "no_gain_action_count",
    "redundant_action_count",
    "duplicate_action_count",
    "invalid_call_count",
    "wrong_answer_count",
    "policy_penalty",
    "redundant_action_penalty",
    "incomplete_penalty",
    "zero_answer_penalty",
    "max_steps_reached",
    "max_steps_penalty",
    "total_penalty",
    "raw_terminal_reward",
    "user_api_calls",
    "user_api_errors",
    "user_retries",
    "user_cache_hits",
    "user_judge_api_calls",
    "user_response_api_calls",
    "user_prompt_tokens",
    "user_completion_tokens",
    "user_total_tokens",
    "user_reasoning_tokens",
    "user_wall_time_seconds",
)


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(result):
        return float(default)
    return result


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    values = [_finite_float(row.get(key, 0.0)) for row in rows]
    return sum(values) / len(values) if values else 0.0


def _is_success(row: Mapping[str, Any]) -> bool:
    # TravelGym's public pass predicate is intentionally exact: only the
    # completion_success metric, equal to one, ends a task's pass sequence.
    return _finite_float(row.get("completion_success")) == 1.0


def _is_valid(row: Mapping[str, Any]) -> bool:
    # A missing validity field is treated as valid for generic injected
    # runners.  Native TravelGym validation supplies the field explicitly and
    # therefore still reports malformed/invalid rows correctly.
    return _finite_float(row.get("reward_valid"), default=1.0) != 0.0


def aggregate_validation_attempts(
    attempts_by_task: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    pass_k: int,
) -> dict[str, float | int]:
    """Aggregate task-level attempts using the public pass@k contract.

    ``attempts_by_task`` contains only the attempts that were actually run.
    Consequently ``attempt_count`` and ``valid_attempt_rate`` reflect early
    stopping instead of pretending every task always consumed ``k`` attempts.
    ``mean@1`` is the first-attempt view used for comparison with step-0
    baselines; ``mean_attempt`` describes all generated attempts; and
    ``mean_best`` is the per-task maximum over valid attempts.
    """

    try:
        k = int(pass_k)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"pass_k must be a positive integer, got {pass_k!r}") from exc
    if k < 1:
        raise ValueError(f"pass_k must be a positive integer, got {pass_k!r}")

    normalized: list[tuple[str, list[Mapping[str, Any]]]] = []
    for task_key, rows in attempts_by_task.items():
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            raise ValueError(f"attempts for task {task_key!r} must be a sequence")
        task_rows = [row for row in rows if isinstance(row, Mapping)]
        if not task_rows:
            raise ValueError(f"task {task_key!r} has no validation attempt")
        if len(task_rows) > k:
            raise ValueError(
                f"task {task_key!r} has {len(task_rows)} attempts, exceeding pass_k={k}"
            )
        normalized.append((str(task_key), task_rows))

    task_count = len(normalized)
    all_rows = [row for _task_key, rows in normalized for row in rows]
    first_rows = [rows[0] for _task_key, rows in normalized]
    success_count = sum(any(_is_success(row) for row in rows) for _key, rows in normalized)
    first_success_count = sum(_is_success(row) for row in first_rows)
    valid_count = sum(_is_valid(row) for row in all_rows)
    early_stopped_count = sum(
        len(rows) < k and any(_is_success(row) for row in rows)
        for _key, rows in normalized
    )

    summary: dict[str, float | int] = {
        "task_count": task_count,
        "pass_k": k,
        "pass_at_k": success_count / task_count if task_count else 0.0,
        f"pass@{k}": success_count / task_count if task_count else 0.0,
        "pass_count": success_count,
        "pass@1": first_success_count / task_count if task_count else 0.0,
        "attempt_count": len(all_rows),
        "attempts_per_task": len(all_rows) / task_count if task_count else 0.0,
        "early_stopped_tasks": early_stopped_count,
        "valid_attempt_count": valid_count,
        "invalid_attempt_count": len(all_rows) - valid_count,
        "valid_attempt_rate": valid_count / len(all_rows) if all_rows else 0.0,
    }

    for metric in PUBLIC_VALIDATION_METRICS:
        summary[f"{metric}/mean@1"] = _mean(first_rows, metric)
        summary[f"{metric}/mean_attempt"] = _mean(all_rows, metric)

        best_values: list[float] = []
        for _task_key, rows in normalized:
            valid_rows = [row for row in rows if _is_valid(row)]
            candidates = valid_rows or rows
            best_values.append(max(_finite_float(row.get(metric)) for row in candidates))
        summary[f"{metric}/mean_best"] = (
            sum(best_values) / len(best_values) if best_values else 0.0
        )

    return summary


__all__ = ["PUBLIC_VALIDATION_METRICS", "aggregate_validation_attempts"]
