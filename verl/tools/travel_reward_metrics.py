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

TRAVEL_REWARD_METRIC_NAMES = (
    *TRAVEL_QUALITY_METRIC_NAMES,
    *TRAVEL_USER_METRIC_NAMES,
)


__all__ = [
    "TRAVEL_QUALITY_METRIC_NAMES",
    "TRAVEL_REWARD_METRIC_NAMES",
    "TRAVEL_USER_METRIC_NAMES",
]
