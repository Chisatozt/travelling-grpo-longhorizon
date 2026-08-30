"""Unified final200 evaluation plan (offline-safe).

The evaluator intentionally separates planning/manifest checks from model
execution.  Running this module with ``--dry-run`` never creates an API
client, starts Ray/SGLang, or loads a model.  Execution is left behind an
explicit ``SELECTED_GRPO_CHECKPOINT`` guard so step-200 cannot silently pick a
checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from collections.abc import Callable, Iterable, Mapping
from typing import Any

try:
    from .travel_contract import REWARD_VERSION
except ImportError:  # direct script execution
    from travel_contract import REWARD_VERSION
try:
    from sft.task_pools import load_pool_manifest
except ModuleNotFoundError:  # direct ``python eval/final200.py`` execution
    # Python puts ``eval/`` (rather than the repository root) on ``sys.path``
    # for a file invocation.  Add the root only for this import fallback; the
    # package invocation (``python -m eval.final200``) remains unchanged.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from sft.task_pools import load_pool_manifest


class Final200Error(ValueError):
    pass


_EVAL_METRICS = (
    "terminal_reward",
    "completion_success",
    "correct_completion",
    "best_answer_rate",
    "answer_coverage",
    "legal_chain_rate",
    "efficiency",
)
_TRACKED_METRICS = (
    "terminal_reward",
    "completion_success",
    "correct_completion",
    "best_answer_rate",
    "answer_coverage",
    "legal_chain_rate",
    "efficiency",
)


def _as_finite_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if result == result and abs(result) != float("inf") else float(default)


def _summarize_reports(reports: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    """Aggregate only public scalar evaluator outputs.

    A runner may return a richer private reward report, but this function
    intentionally copies no per-task transcript, option ID, preference label,
    or reward ledger into the result/tracking payload.
    """

    rows = [row for row in reports if isinstance(row, Mapping)]
    summary: dict[str, float] = {"episodes": float(len(rows))}
    for key in _EVAL_METRICS:
        if not rows:
            summary[f"{key}_mean"] = 0.0
            continue
        summary[f"{key}_mean"] = sum(_as_finite_float(row.get(key, 0.0)) for row in rows) / len(rows)
    summary["completion_success_count"] = float(
        sum(_as_finite_float(row.get("completion_success", 0.0)) >= 1.0 for row in rows)
    )
    return summary


def _tracker_log(tracker: Any, metrics: Mapping[str, Any], step: int = 0) -> None:
    if tracker is None:
        return
    log = getattr(tracker, "log", None)
    if not callable(log):
        raise Final200Error("tracking factory must return an object with log()")
    log(data=dict(metrics), step=int(step))


def _tracker_finish(tracker: Any) -> None:
    if tracker is None:
        return
    finish = getattr(tracker, "finish", None)
    if callable(finish):
        finish()


def evaluate_final200(
    plan: Mapping[str, Any],
    *,
    runner: Callable[..., Mapping[str, Any]],
    tracking_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Run a planned final200 evaluation through an injected model runner.

    This is the execution seam for the real TravelGym/SGLang/API adapters.
    Keeping the runner injected makes the repository's CPU fixture fully
    offline while enforcing that all four models receive the same protocol,
    task keys, seed, simulator and tool schema.  The function never performs
    model/API discovery itself and never returns raw reports.

    ``runner`` is called as ``runner(model_name=<ref>, task_key=<key>,
    protocol=<plan protocol>)`` and must return a mapping containing public
    scalar fields such as ``terminal_reward`` and ``completion_success``.
    ``tracking_factory`` (when supplied) is called once per independent model
    run and once for the comparison run with ``name``, ``run_spec`` and
    ``protocol`` keyword arguments.  A factory can construct the repository's
    :class:`verl.utils.tracking.Tracking` SwanLab wrapper on a real server.
    """

    if not isinstance(plan, Mapping):
        raise Final200Error("final200 plan must be a mapping")
    if bool(plan.get("selection_pending", False)):
        raise Final200Error("final200 selection is pending; provide SELECTED_GRPO_CHECKPOINT")
    protocol = plan.get("protocol")
    models = plan.get("models")
    splits = plan.get("splits")
    runs = plan.get("runs")
    if not isinstance(protocol, Mapping) or not isinstance(models, Mapping):
        raise Final200Error("final200 plan is missing protocol/models")
    if not isinstance(splits, Mapping) or not isinstance(runs, Mapping):
        raise Final200Error("final200 plan is missing splits/runs")
    model_names = ("base", "sft", "grpo", "deepseek")
    if set(models) != set(model_names):
        raise Final200Error("final200 plan must contain base, sft, grpo and deepseek")
    all_keys = [str(value) for value in splits.get("all200", [])]
    seen_keys = {str(value) for value in splits.get("smoke20_seen", [])}
    unseen_keys = {str(value) for value in splits.get("unseen180", [])}
    if len(all_keys) != 200 or len(set(all_keys)) != 200:
        raise Final200Error("final200 execution requires exactly 200 unique all200 tasks")
    if len(seen_keys) != 20 or not seen_keys <= set(all_keys):
        raise Final200Error("final200 execution requires a 20-task seen subset")
    if unseen_keys != set(all_keys) - seen_keys:
        raise Final200Error("final200 unseen180 is not the complement of seen20")
    if not callable(runner):
        raise Final200Error("final200 runner must be callable")

    result: dict[str, Any] = {
        "protocol": dict(protocol),
        "models": {},
        "comparison": {},
        "task_count": {"all200": 200, "smoke20_seen": 20, "unseen180": 180},
    }
    reports_by_model: dict[str, dict[str, Mapping[str, Any]]] = {}
    for model_name in model_names:
        model_reports: dict[str, Mapping[str, Any]] = {}
        for task_key in all_keys:
            report = runner(model_name=str(models[model_name]), task_key=task_key, protocol=dict(protocol))
            if not isinstance(report, Mapping):
                raise Final200Error(f"runner returned a non-mapping report for {model_name}/{task_key}")
            # ``report`` remains process-local and is reduced immediately;
            # no raw trajectory is retained in the returned result.
            model_reports[task_key] = report
        reports_by_model[model_name] = model_reports
        split_keys = {
            "all": set(all_keys),
            "smoke20_seen": seen_keys,
            "unseen180": unseen_keys,
        }
        model_result: dict[str, Any] = {}
        for split_name, keys in split_keys.items():
            summary = _summarize_reports(model_reports[key] for key in all_keys if key in keys)
            model_result[split_name] = summary
        result["models"][model_name] = model_result

        tracker = None
        try:
            if tracking_factory is not None:
                tracker = tracking_factory(
                    name=model_name,
                    run_spec=dict(runs.get(model_name, {})),
                    protocol=dict(protocol),
                )
                tracked: dict[str, float] = {}
                for split_name, summary in model_result.items():
                    namespace = "all" if split_name == "all" else split_name
                    for metric in _TRACKED_METRICS:
                        tracked[f"final200/{namespace}/{metric}"] = float(summary.get(f"{metric}_mean", 0.0))
                _tracker_log(tracker, tracked, step=0)
            
        finally:
            _tracker_finish(tracker)

    # Comparison is a separate run and contains only aggregate public scalar
    # differences, never a per-task report or a model transcript.
    comparison: dict[str, float] = {}
    for split_name in ("all", "smoke20_seen", "unseen180"):
        for metric in _TRACKED_METRICS:
            values = {
                model_name: float(result["models"][model_name][split_name].get(f"{metric}_mean", 0.0))
                for model_name in model_names
            }
            for model_name, value in values.items():
                comparison[f"{split_name}/{model_name}_{metric}"] = value
            best_model = max(values, key=values.get)
            comparison[f"{split_name}/best_{metric}_model_index"] = float(model_names.index(best_model))
    result["comparison"] = comparison
    tracker = None
    try:
        if tracking_factory is not None:
            tracker = tracking_factory(
                name="comparison",
                run_spec=dict(runs.get("comparison", {})),
                protocol=dict(protocol),
            )
            _tracker_log(
                tracker,
                {f"final200/comparison/{key}": float(value) for key, value in comparison.items()},
                step=0,
            )
    finally:
        _tracker_finish(tracker)
    return result


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise Final200Error(f"manifest must be an object: {path}")
    return dict(value)


def _keys(manifest: Mapping[str, Any]) -> set[str]:
    records = manifest.get("records", [])
    return {f"{item.get('env_name')}::{item.get('task_id')}" for item in records if isinstance(item, Mapping)}


def build_final200_plan(
    *,
    task_pool_manifest: Mapping[str, Any],
    smoke_manifest: Mapping[str, Any],
    models: Mapping[str, str],
    seed: int,
    max_turns: int,
    reward_version: str = REWARD_VERSION,
    selected_grpo_checkpoint: str | None = None,
) -> dict[str, Any]:
    if set(models) != {"base", "sft", "grpo", "deepseek"}:
        raise Final200Error("models must contain exactly base, sft, grpo and deepseek")
    if str(reward_version) != REWARD_VERSION:
        raise Final200Error(
            "final200 must use the repository reward version "
            f"{REWARD_VERSION!r}, got {reward_version!r}"
        )
    if str(task_pool_manifest.get("strict_task_identity", False)).casefold() not in {"true", "1"}:
        raise Final200Error("final200 requires a strict active task-pool manifest")
    validation = task_pool_manifest.get("pools", {}).get("validation", {})
    all_keys = {f"{item.get('env_name')}::{item.get('task_id')}" for item in validation.get("records", []) if isinstance(item, Mapping)}
    if len(all_keys) != 200:
        raise Final200Error(f"validation pool must contain 200 tasks, got {len(all_keys)}")
    seen_keys = _keys(smoke_manifest)
    if len(seen_keys) != 20 or not seen_keys <= all_keys:
        raise Final200Error("smoke20 must contain exactly 20 tasks inside validation200")
    if selected_grpo_checkpoint is None:
        selected_grpo_checkpoint = os.environ.get("SELECTED_GRPO_CHECKPOINT")
    if not selected_grpo_checkpoint:
        raise Final200Error("step-200 is paused: provide SELECTED_GRPO_CHECKPOINT manually")
    models = dict(models)
    models["grpo"] = str(selected_grpo_checkpoint)
    protocol = {
        "reward_version": REWARD_VERSION,
        "max_turns": int(max_turns),
        "seed": int(seed),
        "user_simulator": "TravelGym-public-v1",
        "tool_schema": "interact_with_env-v1",
    }
    return {
        "schema_version": "travelgym-final200-plan-v1",
        "protocol": protocol,
        "models": models,
        "selection_pending": str(selected_grpo_checkpoint).startswith("[pending:"),
        "runs": {
            "base": {"namespace": "final200/all", "run_id": "final200-base"},
            "sft": {"namespace": "final200/all", "run_id": "final200-sft"},
            "grpo": {"namespace": "final200/all", "run_id": "final200-grpo"},
            "deepseek": {"namespace": "final200/all", "run_id": "final200-deepseek"},
            "comparison": {"namespace": "final200/comparison", "run_id": "final200-comparison"},
        },
        "splits": {
            "all200": sorted(all_keys),
            "smoke20_seen": sorted(seen_keys),
            "unseen180": sorted(all_keys - seen_keys),
        },
        "metrics_namespaces": ["final200/all", "final200/smoke20_seen", "final200/unseen180"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-pool-manifest", required=True, type=Path)
    parser.add_argument("--smoke-manifest", required=True, type=Path)
    parser.add_argument("--base", required=True)
    parser.add_argument("--sft", required=True)
    parser.add_argument("--grpo", default=None)
    parser.add_argument("--deepseek", required=True)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--max-turns", type=int, default=25)
    parser.add_argument("--selected-grpo-checkpoint", default=None)
    parser.add_argument("--output", type=Path, default=Path("final200_plan.json"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    selected = args.selected_grpo_checkpoint or os.environ.get("SELECTED_GRPO_CHECKPOINT")
    if not args.dry_run and not selected:
        raise Final200Error("refusing to execute final200 without SELECTED_GRPO_CHECKPOINT")
    if args.dry_run and not selected:
        selected = "[pending:SELECTED_GRPO_CHECKPOINT]"
    pool = load_pool_manifest(args.task_pool_manifest, require_strict=True)
    smoke = _load_json(args.smoke_manifest)
    plan = build_final200_plan(
        task_pool_manifest=pool,
        smoke_manifest=smoke,
        models={"base": args.base, "sft": args.sft, "grpo": args.grpo or "[manual-selection-required]", "deepseek": args.deepseek},
        seed=args.seed,
        max_turns=args.max_turns,
        selected_grpo_checkpoint=selected,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    if not args.dry_run:
        raise Final200Error("execution adapter is intentionally disabled in this offline code-change phase")


if __name__ == "__main__":  # pragma: no cover - exercised by the CLI
    main()


__all__ = ["Final200Error", "build_final200_plan", "evaluate_final200"]
