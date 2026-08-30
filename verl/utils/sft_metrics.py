"""Dependency-light SFT metric aggregation and checkpoint selection."""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence


def masked_token_statistics(
    token_losses: Sequence[float],
    loss_mask: Sequence[float | int | bool],
    sample_weight: Sequence[float] | None = None,
    trajectory_ids: Sequence[Any] | None = None,
) -> dict[str, float]:
    if len(token_losses) != len(loss_mask):
        raise ValueError("token_losses and loss_mask must have equal length")
    weights = list(sample_weight or [1.0] * len(token_losses))
    if len(weights) not in {len(token_losses), 0}:
        raise ValueError("sample_weight must be token-aligned or omitted")
    if not weights:
        weights = [1.0] * len(token_losses)
    numerator = 0.0
    denominator = 0.0
    for loss, mask, weight in zip(token_losses, loss_mask, weights):
        if bool(mask) and float(weight) > 0:
            numerator += float(loss) * float(weight)
            denominator += float(weight)
    nll = numerator / denominator if denominator else 0.0
    result = {
        "masked_token_nll": float(nll),
        "perplexity": float(math.exp(min(50.0, nll))) if denominator else 1.0,
        "supervised_tokens": float(sum(bool(mask) and float(weight) > 0 for mask, weight in zip(loss_mask, weights))),
        "weighted_supervised_tokens": float(denominator),
    }
    if trajectory_ids is not None:
        groups: dict[Any, list[float]] = {}
        for loss, mask, weight, trajectory_id in zip(token_losses, loss_mask, weights, trajectory_ids):
            if bool(mask) and float(weight) > 0:
                groups.setdefault(trajectory_id, []).append(float(loss))
        means = [sum(values) / len(values) for values in groups.values() if values]
        result["trajectory_macro_loss"] = float(sum(means) / len(means)) if means else 0.0
    return result


def span_nll_metrics(
    token_losses: Sequence[float],
    loss_mask: Sequence[float | int | bool],
    reasoning_mask: Sequence[float | int | bool] | None = None,
    tool_call_mask: Sequence[float | int | bool] | None = None,
) -> dict[str, float]:
    def mean_for(mask):
        values = [float(loss) for loss, selected, supervised in zip(token_losses, mask, loss_mask) if bool(selected) and bool(supervised)]
        return sum(values) / len(values) if values else 0.0
    return {
        "reasoning_nll": mean_for(reasoning_mask or [0] * len(token_losses)),
        "tool_call_nll": mean_for(tool_call_mask or [0] * len(token_losses)),
    }


def structured_action_metrics(records: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    total = parsed = valid_choice = valid_content = 0
    for record in records:
        for message in record.get("messages", []) if isinstance(record, Mapping) else []:
            if not isinstance(message, Mapping) or message.get("role") != "assistant":
                continue
            calls = message.get("tool_calls") or []
            if not calls:
                continue
            total += 1
            call = calls[0] if isinstance(calls[0], Mapping) else {}
            function = call.get("function", call) if isinstance(call, Mapping) else {}
            arguments = function.get("arguments") if isinstance(function, Mapping) else None
            if isinstance(arguments, Mapping):
                parsed += 1
                valid_choice += int(str(arguments.get("choice", "")).casefold() in {"search", "action", "answer"})
                valid_content += int(bool(str(arguments.get("content", "")).strip()))
    return {
        "tool_parse_rate": parsed / total if total else 0.0,
        "structured_choice_rate": valid_choice / parsed if parsed else 0.0,
        "structured_content_rate": valid_content / parsed if parsed else 0.0,
        "tool_call_count": float(total),
    }


def protocol_non_degraded(current: Mapping[str, Any], previous: Mapping[str, Any] | None) -> bool:
    """Whether protocol health has not regressed from the prior epoch."""
    if previous is None:
        return True
    for key in ("tool_parse_rate", "structured_choice_rate", "structured_content_rate"):
        if float(current.get(key, 0.0)) + 1e-12 < float(previous.get(key, 0.0)):
            return False
    return True


def select_best_checkpoint(history: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """Choose lowest NLL among protocol-non-degraded validation epochs."""
    best = None
    best_nll = float("inf")
    for index, item in enumerate(history):
        previous = history[index - 1] if index else None
        if not protocol_non_degraded(item, previous):
            continue
        nll = float(item.get("masked_token_nll", float("inf")))
        if nll < best_nll:
            best_nll = nll
            best = item
    return best


__all__ = [
    "masked_token_statistics",
    "protocol_non_degraded",
    "select_best_checkpoint",
    "span_nll_metrics",
    "structured_action_metrics",
]
