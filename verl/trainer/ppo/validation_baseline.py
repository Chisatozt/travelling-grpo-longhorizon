"""Load the immutable SFT validation aggregate used at GRPO step 0."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path


BASELINE_SCHEMA_VERSION = "travelgym-grpo-validation-baseline-v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


class ValidationBaselineError(ValueError):
    """Raised when a baseline artifact is not safe for trainer logging."""


def resolve_validation_baseline_path(value: str | Path) -> Path:
    """Resolve a configured path from the launch cwd or repository root."""
    raw = Path(value).expanduser()
    if raw.is_absolute():
        return raw.resolve()
    for candidate in (Path.cwd() / raw, REPOSITORY_ROOT / raw):
        if candidate.is_file():
            return candidate.resolve()
    return (REPOSITORY_ROOT / raw).resolve()


def load_step0_validation_metrics(value: str | Path) -> dict[str, float]:
    """Load and validate canonical ``grpo/val/*`` scalar metrics."""
    path = resolve_validation_baseline_path(value)
    if not path.is_file():
        raise FileNotFoundError(f"validation baseline is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationBaselineError(f"cannot read validation baseline: {path}") from exc
    if not isinstance(payload, Mapping):
        raise ValidationBaselineError("validation baseline root must be an object")
    if payload.get("schema_version") != BASELINE_SCHEMA_VERSION:
        raise ValidationBaselineError(
            f"unsupported validation baseline schema: {payload.get('schema_version')!r}"
        )
    protocol = payload.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValidationBaselineError("validation baseline has no protocol")
    required_protocol = {
        "source": "native_validation",
        "split": "validation_smoke",
        "native_two_stage": True,
        "template_prefill": "<think>",
        "reasoning_max_tokens": 2560,
        "tool_call_max_tokens": 512,
        "forced_reasoning_end_loss_mask": 0,
        "pass_k": 3,
        "task_level_early_stop": True,
        "validation_retry_attempts": 0,
    }
    if payload.get("source") != required_protocol["source"]:
        raise ValidationBaselineError("validation baseline source is not native_validation")
    for key, expected in required_protocol.items():
        if key == "source":
            continue
        if protocol.get(key) != expected:
            raise ValidationBaselineError(
                f"validation baseline protocol mismatch for {key}: "
                f"expected {expected!r}, got {protocol.get(key)!r}"
            )
    metrics = payload.get("step0_metrics")
    if not isinstance(metrics, Mapping) or not metrics:
        raise ValidationBaselineError("validation baseline has no step0_metrics")
    if "grpo/val/smoke20/pass@3" not in metrics:
        raise ValidationBaselineError(
            "validation baseline is missing grpo/val/smoke20/pass@3"
        )
    result: dict[str, float] = {}
    for key, value in metrics.items():
        name = str(key)
        if not name.startswith("grpo/val/"):
            raise ValidationBaselineError(f"baseline metric is outside validation namespace: {name}")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValidationBaselineError(f"baseline metric is not numeric: {name}")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValidationBaselineError(f"baseline metric is not finite: {name}")
        result[name] = numeric
    return result
