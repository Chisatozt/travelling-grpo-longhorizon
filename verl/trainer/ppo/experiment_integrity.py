"""Training-step policy and RNG helpers for TravelGym GRPO."""

from __future__ import annotations

import random
from typing import Any, Mapping


class ExperimentIntegrityError(ValueError):
    pass


def validate_run_until_step(run_until_step: int | None, total_training_steps: int = 200) -> int:
    value = total_training_steps if run_until_step is None else int(run_until_step)
    if value < 0 or value > int(total_training_steps):
        raise ExperimentIntegrityError(
            f"run_until_step must be in [0,{total_training_steps}], got {value}"
        )
    return value


def validate_process_run_until_step(
    run_until_step: int | None,
    *,
    total_training_steps: int = 200,
    allowed_steps: tuple[int, ...] = (5, 50, 200),
) -> int:
    """Validate the process-local GRPO stop points.

    ``total_training_steps`` is the experiment horizon used by the optimizer
    and scheduler.  A process may stop only at the explicitly supported
    milestones so that the documented 5 -> 50 -> 200 resume protocol cannot
    accidentally be replaced by a partially trained, untracked horizon.
    ``None`` means the full experiment horizon (200 by default).
    """

    value = validate_run_until_step(run_until_step, total_training_steps)
    allowed = tuple(int(step) for step in allowed_steps)
    if value not in allowed:
        raise ExperimentIntegrityError(
            f"run_until_step must be one of {allowed}, got {value}"
        )
    return value


def validate_total_training_steps(value: int | None, *, expected: int = 200) -> int:
    """Require the fixed optimizer/scheduler horizon for this experiment."""

    resolved = expected if value is None else int(value)
    if resolved != int(expected):
        raise ExperimentIntegrityError(
            f"total_training_steps is fixed at {expected}, got {resolved}"
        )
    return resolved


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {"python": random.getstate()}
    try:
        import numpy as np
        state["numpy"] = np.random.get_state()
    except ImportError:
        pass
    try:
        import torch
        state["torch_cpu"] = torch.get_rng_state()
        if torch.cuda.is_available():
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
    except ImportError:
        pass
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    if not isinstance(state, Mapping):
        raise ExperimentIntegrityError("RNG state must be a mapping")
    if "python" in state:
        random.setstate(state["python"])
    try:
        import numpy as np
        if "numpy" in state:
            np.random.set_state(state["numpy"])
    except ImportError:
        pass
    try:
        import torch
        if "torch_cpu" in state:
            torch.set_rng_state(state["torch_cpu"])
        if "torch_cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["torch_cuda"])
    except ImportError:
        pass


__all__ = [
    "ExperimentIntegrityError",
    "capture_rng_state",
    "restore_rng_state",
    "validate_process_run_until_step",
    "validate_run_until_step",
    "validate_total_training_steps",
]
