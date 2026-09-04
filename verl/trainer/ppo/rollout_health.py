from __future__ import annotations

from collections.abc import Sequence


class InitialRolloutHealthError(RuntimeError):
    """Raised when the step-0 rollout is clearly unusable for training."""


def _is_character_collapse(output: str) -> bool:
    compact = "".join(output.split())
    return len(compact) >= 64 and len(set(compact)) <= 2


def validate_initial_rollout_health(
    outputs: Sequence[str],
    *,
    response_token_limit: int,
) -> None:
    """Reject unmistakably collapsed initial generations before step 1.

    This is intentionally conservative: normal short answers are accepted,
    including answers without tool calls. It catches the two signatures seen
    when SGLang did not receive the actor weights: repeated punctuation and a
    full-budget response with no tool protocol marker.
    """
    if not outputs:
        raise InitialRolloutHealthError("Initial validation produced no outputs")

    normalized = [str(output or "").strip() for output in outputs]
    if all(_is_character_collapse(output) for output in normalized):
        raise InitialRolloutHealthError(
            "Initial rollout health check failed: every validation output "
            "collapsed to one or two repeated characters"
        )

    tool_markers = ("<tool_call", "<|tool_call", '"name":', "'name':")
    no_tool_protocol = all(
        not any(marker in output for marker in tool_markers) for output in normalized
    )
    all_at_budget = all(len(output) >= response_token_limit for output in normalized)
    if no_tool_protocol and all_at_budget:
        raise InitialRolloutHealthError(
            "Initial rollout health check failed: every validation output "
            "exhausted the response budget without a tool-call marker"
        )
