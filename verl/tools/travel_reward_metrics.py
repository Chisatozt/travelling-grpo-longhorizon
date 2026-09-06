"""Canonical trainer-side metrics emitted by the TravelGym interaction tool."""

from __future__ import annotations


TRAVEL_QUALITY_METRIC_NAMES = (
    "correct_completion",
    "answer_quality",
    "legal_chain_rate",
    "coverage_adjusted_answer_quality",
    "coverage_adjusted_legal_chain_rate",
    "hidden_preference_hit_rate",
    "efficiency",
    "completion_success",
    "answer_coverage",
    "best_answer_rate",
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
)

TRAVEL_USER_METRIC_NAMES = (
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

# Telemetry and parser diagnostics are carried alongside reward metadata for
# audit/cost accounting; they are deliberately not independent entries in
# TRAVEL_QUALITY_METRIC_NAMES or the terminal component-attribution routing.
# The small plan-error penalty, when present, is folded into the environment's
# existing policy_penalty handling rather than trained as a separate signal.
TRAVEL_ACTOR_ASPECT_METRIC_NAMES = (
    "actor_aspect_extraction_calls",
    "actor_aspect_extraction_errors",
    "actor_aspect_extraction_retries",
    "actor_aspect_extraction_prompt_tokens",
    "actor_aspect_extraction_completion_tokens",
    "actor_aspect_extraction_total_tokens",
    "actor_aspect_extraction_reasoning_tokens",
    "actor_aspect_extraction_wall_time_seconds",
    "actor_aspect_extraction_error_count",
    "actor_aspect_plan_penalty",
)

TRAVEL_REWARD_METRIC_NAMES = (
    *TRAVEL_QUALITY_METRIC_NAMES,
    *TRAVEL_USER_METRIC_NAMES,
    *TRAVEL_ACTOR_ASPECT_METRIC_NAMES,
)


__all__ = [
    "TRAVEL_QUALITY_METRIC_NAMES",
    "TRAVEL_REWARD_METRIC_NAMES",
    "TRAVEL_ACTOR_ASPECT_METRIC_NAMES",
    "TRAVEL_USER_METRIC_NAMES",
]
