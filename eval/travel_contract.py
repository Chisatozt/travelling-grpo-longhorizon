"""Shared TravelGym evaluation and leakage contract.

The evaluator and training rollout both consume this small, dependency-free
contract.  It intentionally distinguishes public observations from private
terminal diagnostics so adding a metric cannot silently change the Actor's
prompt.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any


REWARD_VERSION = "travelgym-terminal-v1"
PUBLIC_CONTROL_KEYS = {
    "current_aspect",
    "searched_aspects",
    "visible_option_ids",
    "action_count_by_aspect",
    "answered_aspects",
    "public_conversation_history",
}
FORBIDDEN_PUBLIC_KEYS = {
    # Simulator labels and derived terminal diagnostics.  Keep this deny-list
    # broader than the current observation schema so a future field cannot
    # accidentally turn a reward/label metric into an Actor side channel.
    "preference_id",
    "preference_ids",
    "ground_truth",
    "gold",
    "task_id",
    "current_task",
    "task",
    "data_source",
    "scenario",
    "preferences",
    "all_options",
    "state_list",
    "active_preference_ids",
    "hit_preference_ids",
    "correct_ids",
    "best_id",
    "best_ids",
    "remaining_preferences",
    "remaining_correct_options",
    "remaining_best_options",
    "elicited_preferences",
    "elicitation_ratio",
    "correct_completion",
    "completion_success",
    "answer_coverage",
    "best_answer_rate",
    "answer_quality",
    "legal_chain_rate",
    "hidden_preference_hit_rate",
    "efficiency",
    "reward",
    "step_reward",
    "terminal_only",
    "reward_version",
    "termination_reason",
    "terminal_reward",
    "raw_terminal_reward",
    "reward_valid",
    "reward_valid_for_training",
    "policy_penalty",
    "penalty_components",
    "quality_by_aspect",
    "chain_by_aspect",
    "correct_by_aspect",
    "best_by_aspect",
    "answers",
    "reward_report",
    "diagnostics",
}


class TravelContractError(ValueError):
    """Raised when an evaluation/training contract is violated."""


def _walk_keys(value: Any):
    if isinstance(value, Mapping):
        for key, nested in value.items():
            yield str(key)
            yield from _walk_keys(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            yield from _walk_keys(nested)


def assert_public_observation(observation: Mapping[str, Any]) -> None:
    """Reject hidden labels before an observation is sent to an Actor."""
    if not isinstance(observation, Mapping):
        raise TravelContractError("public observation must be a mapping")
    missing = PUBLIC_CONTROL_KEYS - set(observation)
    if missing:
        raise TravelContractError(f"public observation is missing control keys: {sorted(missing)}")
    forbidden = {key for key in _walk_keys(observation) if key.casefold() in {item.casefold() for item in FORBIDDEN_PUBLIC_KEYS}}
    if forbidden:
        raise TravelContractError(f"public observation contains forbidden keys: {sorted(forbidden)}")
    serialized = json.dumps(observation, ensure_ascii=False)
    if re.search(r"\b(?:terminal[_ ]?reward|reward)\s*[:=]", serialized, flags=re.IGNORECASE):
        raise TravelContractError("public observation contains a reward side channel")
    if re.search(r"(?<![A-Za-z0-9])P\d+(?![A-Za-z0-9])", serialized, flags=re.IGNORECASE):
        raise TravelContractError("public observation contains a preference identifier")
    if re.search(
        r"^\s*(?:preference[_ ]?ids?|correct[_ ]?ids?|best[_ ]?ids?|"
        r"ground[_ ]?truth|reward[_ ]?report|diagnostics?)\s*[:=]",
        serialized,
        flags=re.IGNORECASE | re.MULTILINE,
    ):
        raise TravelContractError("public observation contains a private label line")


def make_eval_manifest(*, data_path: str, config: Mapping[str, Any], code_revision: str | None = None) -> dict[str, Any]:
    """Create a reproducible manifest for offline/shadow/train evaluation."""
    return {
        "contract_version": "travelgym-public-v1",
        "reward_version": REWARD_VERSION,
        "data_path": str(data_path),
        "config": dict(config),
        "code_revision": code_revision,
    }


def summarize_terminal_report(report: Mapping[str, Any]) -> dict[str, float | str | bool]:
    """Return stable, non-label metrics for evaluation tables."""
    if report.get("reward_version") != REWARD_VERSION:
        raise TravelContractError("unexpected TravelGym reward version")
    return {
        "terminal_reward": float(report.get("terminal_reward", 0.0)),
        "reward_valid": bool(report.get("reward_valid_for_training", report.get("reward_valid", False))),
        "correct_completion": float(report.get("correct_completion", 0.0)),
        "completion_success": float(report.get("completion_success", 0.0)),
        "answer_coverage": float(report.get("answer_coverage", 0.0)),
        "best_answer_rate": float(report.get("best_answer_rate", 0.0)),
        "answer_quality": float(report.get("answer_quality", 0.0)),
        "legal_chain_rate": float(report.get("legal_chain_rate", 0.0)),
        "hidden_preference_hit_rate": float(report.get("hidden_preference_hit_rate", 0.0)),
        "efficiency": float(report.get("efficiency", 0.0)),
    }


__all__ = [
    "FORBIDDEN_PUBLIC_KEYS",
    "PUBLIC_CONTROL_KEYS",
    "REWARD_VERSION",
    "TravelContractError",
    "assert_public_observation",
    "make_eval_manifest",
    "summarize_terminal_report",
]
