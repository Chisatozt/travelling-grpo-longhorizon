"""Bounded, terminal-reward-aware GRPO sampling helpers.

The sampler works on rollout metadata only. It never reads candidate
attributes or simulator preferences. When enabled, a generation call is
retried a bounded number of times and one varying, valid group is selected for
each input UID. If no complete group can be built, the last output is marked
``travel_skip_update`` and the downstream advantage mask makes the update a
no-op. This keeps retry behavior finite and UID preserving.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


DEFAULT_NUMERICAL_EPSILON = 1.0e-6
DEFAULT_MIN_REWARD_SPREAD = 5.0e-3


def _as_python(value: Any) -> Any:
    return value.item() if hasattr(value, "item") else value


def _ordered_unique(values: Sequence[Any]) -> list[Any]:
    ordered: list[Any] = []
    seen: set[Any] = set()
    for raw in values:
        value = _as_python(raw)
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def resolve_reward_spread_thresholds(
    config: Mapping[str, Any] | None,
) -> tuple[float, float]:
    """Resolve numerical equality and semantic reward-spread thresholds.

    ``reward_tolerance`` is retained as a compatibility alias for the old
    numerical-only setting. New configurations should use the two explicit
    names so floating-point equality is not confused with useful GRPO signal.
    """

    config = config or {}
    numerical_epsilon = float(
        config.get(
            "numerical_epsilon",
            config.get("reward_tolerance", DEFAULT_NUMERICAL_EPSILON),
        )
    )
    min_reward_spread = float(
        config.get("min_reward_spread", DEFAULT_MIN_REWARD_SPREAD)
    )
    if numerical_epsilon < 0 or not math.isfinite(numerical_epsilon):
        raise ValueError("numerical_epsilon must be finite and non-negative")
    if min_reward_spread < 0 or not math.isfinite(min_reward_spread):
        raise ValueError("min_reward_spread must be finite and non-negative")
    return numerical_epsilon, min_reward_spread


def _reward_spread_reason(
    spread: float,
    *,
    numerical_epsilon: float,
    min_reward_spread: float,
) -> str | None:
    if spread <= numerical_epsilon:
        return "constant_reward"
    if spread + numerical_epsilon < min_reward_spread:
        return "insufficient_reward_spread"
    return None


def select_reward_varying_groups(
    uids: Sequence[Hashable],
    terminal_rewards: Sequence[float],
    *,
    reward_valid: Sequence[bool] | None = None,
    expected_group_size: int = 2,
    numerical_epsilon: float = DEFAULT_NUMERICAL_EPSILON,
    min_reward_spread: float = DEFAULT_MIN_REWARD_SPREAD,
) -> tuple[list[int], dict[str, Any]]:
    """Select rows from complete, valid groups with useful reward spread.

    Rows with an invalid sibling remain available for diagnostics and for a
    later bounded retry. Numerically constant groups cannot produce a GRPO
    relative advantage. Groups below ``min_reward_spread`` are also dropped so
    standard-deviation normalization cannot amplify tiny shaping differences
    into order-one advantages. No environment state is changed by this helper.
    """

    if len(uids) != len(terminal_rewards):
        raise ValueError("uids and terminal_rewards must have equal length")
    if expected_group_size <= 1:
        raise ValueError("expected_group_size must be greater than one")
    numerical_epsilon, min_reward_spread = resolve_reward_spread_thresholds(
        {
            "numerical_epsilon": numerical_epsilon,
            "min_reward_spread": min_reward_spread,
        }
    )
    if reward_valid is None:
        invalid = [False] * len(uids)
    else:
        if len(reward_valid) != len(uids):
            raise ValueError("reward_valid must align with uids")
        invalid = [not bool(value) for value in reward_valid]

    grouped: dict[Hashable, list[int]] = defaultdict(list)
    values: list[float] = []
    for row, (uid, reward) in enumerate(zip(uids, terminal_rewards)):
        try:
            hash(uid)
        except TypeError as exc:
            raise ValueError(f"uid at index {row} is not hashable") from exc
        try:
            value = float(reward)
        except (TypeError, ValueError):
            value = 0.0
            invalid[row] = True
        if not math.isfinite(value):
            value = 0.0
            invalid[row] = True
        values.append(value)
        grouped[uid].append(row)

    kept: list[int] = []
    groups: list[dict[str, Any]] = []
    reasons = Counter()
    for uid, rows in grouped.items():
        clean = [row for row in rows if not invalid[row]]
        reward_min = min(values[row] for row in rows)
        reward_max = max(values[row] for row in rows)
        reward_spread = reward_max - reward_min
        if len(rows) < expected_group_size:
            reason = "incomplete_group"
        elif any(invalid[row] for row in rows):
            reason = "reward_invalid"
        else:
            reason = _reward_spread_reason(
                reward_spread,
                numerical_epsilon=numerical_epsilon,
                min_reward_spread=min_reward_spread,
            )
        if reason is None:
            kept.extend(rows)
        elif reason in {"incomplete_group", "reward_invalid"}:
            # Clean rows are retained for diagnostics; the terminal advantage
            # reducer still excludes invalid/incomplete siblings.
            kept.extend(clean)
        if reason is not None:
            reasons[reason] += 1
        groups.append(
            {
                "uid": uid,
                "indices": tuple(rows),
                "reward_min": reward_min,
                "reward_max": reward_max,
                "reward_spread": reward_spread,
                "reason": reason,
                "trainable": reason is None,
            }
        )
    stats = {
        "num_trajectories": len(uids),
        "num_groups": len(grouped),
        "kept_rows": len(kept),
        "trainable_group_count": sum(group["trainable"] for group in groups),
        "skipped_group_count": sum(not group["trainable"] for group in groups),
        "numerical_epsilon": numerical_epsilon,
        "min_reward_spread": min_reward_spread,
        "skip_reason_counts": dict(sorted(reasons.items())),
        "groups": tuple(groups),
    }
    return sorted(set(kept)), stats


@dataclass
class BoundedSamplingState:
    """Track bounded generation retries and consecutive skipped updates."""

    required_groups: int = 1
    max_generation_batches: int = 3
    max_consecutive_skips: int = 10
    generation_batches: int = 0
    accepted_groups: int = 0
    consecutive_skips: int = 0
    diagnostics: list[dict[str, Any]] = field(default_factory=list)

    def record_batch(self, stats: Mapping[str, Any]) -> bool:
        if self.generation_batches >= self.max_generation_batches:
            raise RuntimeError("bounded sampler exceeded its generation-batch limit")
        self.generation_batches += 1
        self.accepted_groups += int(
            stats.get("trainable_group_count", stats.get("kept_group_count", 0))
        )
        self.diagnostics.append(dict(stats))
        return self.accepted_groups >= self.required_groups

    @property
    def may_generate(self) -> bool:
        return (
            self.accepted_groups < self.required_groups
            and self.generation_batches < self.max_generation_batches
        )

    def finish_update(self) -> bool:
        train = self.accepted_groups >= self.required_groups
        if train:
            self.consecutive_skips = 0
        else:
            self.consecutive_skips += 1
            if self.consecutive_skips > self.max_consecutive_skips:
                raise RuntimeError("bounded sampler exceeded consecutive skipped-update limit")
        self.generation_batches = 0
        self.accepted_groups = 0
        self.diagnostics.clear()
        return train


@dataclass(frozen=True)
class _RolloutCandidate:
    uid: Hashable
    row: Any = field(repr=False, compare=False)
    reward: float
    generation_batch: int
    row_index: int


def _select_candidate_group(
    candidates: Sequence[_RolloutCandidate],
    *,
    expected_group_size: int,
    numerical_epsilon: float,
    min_reward_spread: float,
) -> tuple[_RolloutCandidate, ...] | None:
    """Choose a varied group without a combinatorial search."""

    if len(candidates) < expected_group_size:
        return None
    ordered = sorted(candidates, key=lambda item: (item.reward, item.generation_batch, item.row_index))
    reason = _reward_spread_reason(
        ordered[-1].reward - ordered[0].reward,
        numerical_epsilon=numerical_epsilon,
        min_reward_spread=min_reward_spread,
    )
    if reason is not None:
        return None

    selected = [ordered[0]]
    if expected_group_size > 1:
        selected.append(ordered[-1])
    remaining = expected_group_size - len(selected)
    middle = ordered[1:-1]
    if remaining > 0:
        if len(middle) <= remaining:
            selected.extend(middle)
        else:
            for slot in range(remaining):
                position = round((slot + 1) * (len(middle) - 1) / (remaining + 1))
                selected.append(middle[position])

    seen = set()
    unique: list[_RolloutCandidate] = []
    for item in selected + list(ordered):
        marker = (item.generation_batch, item.row_index)
        if marker not in seen:
            seen.add(marker)
            unique.append(item)
        if len(unique) >= expected_group_size:
            break
    return tuple(unique) if len(unique) == expected_group_size else None


def _extract_rollout_signals(output: Any) -> tuple[list[Any], list[float], list[bool]] | None:
    """Read UID/reward metadata emitted by InteractTool rollouts."""

    raw = output.non_tensor_batch.get("reward_scores")
    if raw is None:
        return None
    if hasattr(raw, "tolist"):
        raw = raw.tolist()
    if not isinstance(raw, (list, tuple)):
        raw = [raw]
    uids = output.non_tensor_batch.get("uid")
    if uids is None:
        return None
    if hasattr(uids, "tolist"):
        uids = uids.tolist()
    if len(uids) != len(raw):
        return None
    rewards: list[float] = []
    invalid: list[bool] = []
    normalized_uids = []
    for uid, item in zip(uids, raw):
        while isinstance(item, (list, tuple)) and len(item) == 1:
            item = item[0]
        if not isinstance(item, Mapping) or "interact_with_env" not in item:
            return None
        try:
            value = float(item["interact_with_env"])
        except (TypeError, ValueError):
            value = 0.0
            bad = True
        else:
            bad = not math.isfinite(value)
        try:
            valid = bool(float(item.get("interact_with_env_reward_valid", 1.0)))
        except (TypeError, ValueError):
            valid = False
        normalized_uids.append(_as_python(uid))
        rewards.append(value if math.isfinite(value) else 0.0)
        invalid.append(bad or not valid)
    return normalized_uids, rewards, invalid


def install_verl_bounded_sampler(
    manager: Any,
    config: Mapping[str, Any] | None,
    *,
    group_size: int,
) -> None:
    """Wrap a rollout manager with finite UID-preserving resampling.

    ``manager.generate_sequences`` is called at most
    ``max_generation_batches`` times per update. Validation and deterministic
    baseline generations bypass the wrapper.
    """

    if getattr(manager, "_travel_bounded_sampler_installed", False):
        return
    config = config or {}
    if not bool(config.get("enable", False)) or int(group_size) <= 1:
        return
    expected_group_size = int(config.get("group_size", group_size))
    max_batches = int(config.get("max_generation_batches", 3))
    max_skips = int(config.get("max_consecutive_skips", 10))
    numerical_epsilon, min_reward_spread = resolve_reward_spread_thresholds(config)
    if expected_group_size != int(group_size):
        raise ValueError("dynamic sampler group_size must match rollout.n")
    if expected_group_size <= 1 or max_batches <= 0 or max_skips < 0:
        raise ValueError("invalid bounded-sampling configuration")
    original = manager.generate_sequences
    state_by_batch_size: dict[int, BoundedSamplingState] = {}

    def generate_sequences(batch: Any) -> Any:
        if batch.meta_info.get("validate", False) or batch.meta_info.get("do_sample") is False:
            return original(batch)
        raw_uids = batch.non_tensor_batch.get("uid")
        if raw_uids is None:
            return original(batch)
        if hasattr(raw_uids, "tolist"):
            raw_uids = raw_uids.tolist()
        prompt_uids = _ordered_unique(raw_uids)
        if not prompt_uids:
            return original(batch)
        state = state_by_batch_size.setdefault(
            len(prompt_uids),
            BoundedSamplingState(
                required_groups=len(prompt_uids),
                max_generation_batches=max_batches,
                max_consecutive_skips=max_skips,
            ),
        )
        candidates: dict[Hashable, list[_RolloutCandidate]] = {uid: [] for uid in prompt_uids}
        last_output = None
        aggregate = Counter()
        for generation_batch in range(max_batches):
            output = original(batch)
            last_output = output
            output_rows = int(output.batch["responses"].shape[0])
            expected_rows = len(raw_uids) * expected_group_size
            output_uids = output.non_tensor_batch.get("uid")
            if output_uids is None or len(output_uids) == 0:
                if output_rows != expected_rows:
                    return output
                import numpy as np
                output.non_tensor_batch["uid"] = np.repeat(
                    np.asarray(raw_uids, dtype=object), expected_group_size, axis=0
                )
            else:
                if hasattr(output_uids, "tolist"):
                    output_uids = output_uids.tolist()
                if len(output_uids) != output_rows:
                    return output
                import numpy as np
                output.non_tensor_batch["uid"] = np.asarray(
                    [_as_python(value) for value in output_uids], dtype=object
                )
            signals = _extract_rollout_signals(output)
            if signals is None:
                return output
            uids, rewards, invalid = signals
            if tuple(uids) != tuple(output.non_tensor_batch["uid"].tolist()):
                return output
            _, stats = select_reward_varying_groups(
                uids,
                rewards,
                reward_valid=[not value for value in invalid],
                expected_group_size=expected_group_size,
                numerical_epsilon=numerical_epsilon,
                min_reward_spread=min_reward_spread,
            )
            aggregate["constant_reward_group_count"] += stats["skip_reason_counts"].get("constant_reward", 0)
            aggregate["insufficient_reward_spread_group_count"] += stats["skip_reason_counts"].get(
                "insufficient_reward_spread", 0
            )
            aggregate["invalid_group_count"] += stats["skip_reason_counts"].get("reward_invalid", 0)
            aggregate["incomplete_group_count"] += stats["skip_reason_counts"].get("incomplete_group", 0)
            for index, (uid, reward, bad) in enumerate(zip(uids, rewards, invalid)):
                if bad or uid not in candidates:
                    continue
                candidates[uid].append(
                    _RolloutCandidate(
                        uid=uid,
                        row=output.select_idxs([index]),
                        reward=float(reward),
                        generation_batch=generation_batch,
                        row_index=index,
                    )
                )
            selections: dict[Hashable, tuple[_RolloutCandidate, ...]] = {}
            for uid in prompt_uids:
                selection = _select_candidate_group(
                    candidates[uid],
                    expected_group_size=expected_group_size,
                    numerical_epsilon=numerical_epsilon,
                    min_reward_spread=min_reward_spread,
                )
                if selection is not None:
                    selections[uid] = selection
            state.record_batch({"trainable_group_count": 0, **dict(aggregate)})
            if len(selections) != len(prompt_uids):
                continue

            from verl import DataProto
            rows = []
            for uid in prompt_uids:
                rows.extend(candidate.row for candidate in selections[uid])
            for row in rows:
                row.meta_info = {}
            merged = DataProto.concat(rows)
            merged.meta_info.update(output.meta_info)
            state.accepted_groups = len(prompt_uids)
            state.finish_update()
            merged.meta_info["travel_dynamic_sampling"] = {
                "sampled_batches": generation_batch + 1,
                "accepted_groups": len(selections),
                "candidate_count": sum(len(value) for value in candidates.values()),
                "constant_reward_group_count": int(aggregate["constant_reward_group_count"]),
                "insufficient_reward_spread_group_count": int(
                    aggregate["insufficient_reward_spread_group_count"]
                ),
                "invalid_group_count": int(aggregate["invalid_group_count"]),
                "incomplete_group_count": int(aggregate["incomplete_group_count"]),
                "numerical_epsilon": numerical_epsilon,
                "min_reward_spread": min_reward_spread,
                "bounded": True,
            }
            return merged

        assert last_output is not None
        state.accepted_groups = 0
        state.finish_update()
        last_output.meta_info["travel_skip_update"] = True
        last_output.meta_info["travel_dynamic_sampling"] = {
            "sampled_batches": max_batches,
            "accepted_groups": 0,
            "candidate_count": sum(len(value) for value in candidates.values()),
            "constant_reward_group_count": int(aggregate["constant_reward_group_count"]),
            "insufficient_reward_spread_group_count": int(
                aggregate["insufficient_reward_spread_group_count"]
            ),
            "invalid_group_count": int(aggregate["invalid_group_count"]),
            "incomplete_group_count": int(aggregate["incomplete_group_count"]),
            "numerical_epsilon": numerical_epsilon,
            "min_reward_spread": min_reward_spread,
            "bounded": True,
        }
        return last_output

    manager.generate_sequences = generate_sequences
    manager._travel_bounded_sampler_installed = True
    manager._travel_bounded_sampling_state = state_by_batch_size


__all__ = [
    "BoundedSamplingState",
    "DEFAULT_MIN_REWARD_SPREAD",
    "DEFAULT_NUMERICAL_EPSILON",
    "install_verl_bounded_sampler",
    "resolve_reward_spread_thresholds",
    "select_reward_varying_groups",
]
