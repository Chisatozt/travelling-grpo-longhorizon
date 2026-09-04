#!/usr/bin/env python3
"""Merge SFT-186, preflight, run only 1-task GRPO, record, then power off."""

from __future__ import annotations
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STOP_REQUESTED = False


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def nonempty(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def merged_model_complete(path: Path) -> tuple[bool, list[str]]:
    errors: list[str] = []
    weights = sorted(path.glob("model*.safetensors")) if path.is_dir() else []
    if not weights or not all(nonempty(item) for item in weights):
        errors.append(f"missing or empty merged weights under {path}")
    for name in ("config.json", "tokenizer_config.json", "merge_metadata.json"):
        if not nonempty(path / name):
            errors.append(f"missing or empty merged-model file: {path / name}")
    return not errors, errors


def verify_overfit_artifacts(checkpoint_root: Path, artifact_root: Path) -> dict[str, Any]:
    errors: list[str] = []
    step_dir = checkpoint_root / "global_step_10"
    latest_file = checkpoint_root / "latest_checkpointed_iteration.txt"
    latest = latest_file.read_text(encoding="utf-8").strip() if nonempty(latest_file) else None
    if latest != "10":
        errors.append(f"latest checkpoint marker is {latest!r}, expected '10'")
    actor_dir = step_dir / "actor"
    actor_files = sorted(item for item in actor_dir.rglob("*") if nonempty(item)) if actor_dir.is_dir() else []
    if not actor_files:
        errors.append(f"no non-empty Actor checkpoint files under {actor_dir}")
    for name in ("data.pt", "rng_state.pt"):
        if not nonempty(step_dir / name):
            errors.append(f"missing or empty state: {step_dir / name}")
    validation = artifact_root / "validation/10.jsonl"
    rollout = artifact_root / "rollouts/10.jsonl"
    if not nonempty(validation):
        errors.append(f"missing final validation generations: {validation}")
    if not nonempty(rollout):
        errors.append(f"missing final training rollout: {rollout}")
    return {
        "complete": not errors,
        "errors": errors,
        "checkpoint_root": str(checkpoint_root),
        "artifact_root": str(artifact_root),
        "latest_checkpointed_iteration": latest,
        "actor_files": [str(item) for item in actor_files],
        "validation_file": str(validation),
        "rollout_file": str(rollout),
    }


def gpu_status() -> dict[str, Any]:
    try:
        import torch
        available = torch.cuda.is_available()
        count = torch.cuda.device_count()
        names = [torch.cuda.get_device_name(index) for index in range(count)] if available else []
        return {"available": available, "count": count, "names": names, "torch": torch.__version__}
    except Exception as exc:
        return {"available": False, "count": 0, "error": repr(exc)}


def poweroff() -> tuple[bool, list[dict[str, Any]]]:
    attempts: list[dict[str, Any]] = []
    for command in (
        ["/usr/bin/shutdown", "-h", "now"],
        ["/usr/bin/systemctl", "poweroff"],
        ["/usr/sbin/poweroff"],
    ):
        if not Path(command[0]).exists():
            continue
        try:
            result = subprocess.run(command, text=True, capture_output=True, timeout=20, check=False)
            attempt = {
                "command": command,
                "returncode": result.returncode,
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip(),
            }
        except (OSError, subprocess.SubprocessError) as exc:
            attempt = {"command": command, "error": repr(exc)}
        attempts.append(attempt)
        if attempt.get("returncode") == 0:
            return True, attempts
    return False, attempts


def request_stop(_signum: int, _frame: Any) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=root)
    parser.add_argument("--base-model", type=Path, default=Path("/root/autodl-tmp/models/Qwen3.5-4B"))
    parser.add_argument("--merged-model", type=Path, default=root / "checkpoints/TravelGym/qwen35_4b_canonical_sft/merged_step_186")
    parser.add_argument("--shutdown-delay-seconds", type=int, default=60)
    parser.add_argument("--arm", action="store_true", help="Power off after success or failure.")
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.repository_root.resolve()
    merged = args.merged_model.resolve()
    checkpoint = root / "checkpoints/TravelGym/travelgym_grpo_overfit_1task_sft186"
    artifacts_root = root / "outputs/travelgym_grpo_overfit_1task_sft186"
    run_root = root / "outputs/grpo_one_shutdown"
    report_file = run_root / "report.json"
    state_file = run_root / "state.json"
    log_file = run_root / "workflow.log"

    merged_ok, merged_errors = merged_model_complete(merged)
    status = {
        "captured_at": utc_now(),
        "gpu": gpu_status(),
        "merged_model": {"path": str(merged), "complete": merged_ok, "errors": merged_errors},
        "overfit": verify_overfit_artifacts(checkpoint, artifacts_root),
    }
    if args.check_only:
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return 0
    if run_root.exists() and any(run_root.iterdir()):
        print(f"refusing to overwrite previous workflow records: {run_root}", file=sys.stderr)
        return 2

    run_root.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    report: dict[str, Any] = {
        "schema_version": "travelgym-grpo-one-shutdown-v1",
        "started_at": utc_now(),
        "finished_at": None,
        "outcome": "running",
        "failed_stage": None,
        "armed": bool(args.arm),
        "stages": [],
        "artifacts": None,
        "log_file": str(log_file),
        "policy": "merge -> preflight -> overfit-one only -> record -> poweroff",
        "forbidden_automatic_stages": ["overfit-four", "production"],
        "poweroff": {"requested_at": None, "accepted": False, "attempts": []},
    }
    atomic_json(state_file, report)

    def run_stage(name: str, command: list[str]) -> bool:
        stage = {"name": name, "command": command, "started_at": utc_now(), "finished_at": None, "returncode": None}
        report["stages"].append(stage)
        atomic_json(state_file, report)
        with log_file.open("a", encoding="utf-8") as log:
            log.write(f"\n[{utc_now()}] START {name}: {' '.join(command)}\n")
            log.flush()
            result = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=False)
            stage["returncode"] = result.returncode
            stage["finished_at"] = utc_now()
            log.write(f"[{utc_now()}] END {name}: returncode={result.returncode}\n")
            log.flush()
            os.fsync(log.fileno())
        atomic_json(state_file, report)
        if result.returncode:
            report["failed_stage"] = name
            return False
        return True

    ok = True
    gpu = gpu_status()
    if not gpu.get("available"):
        report["failed_stage"] = "gpu-check"
        report["failure_reason"] = f"no usable CUDA GPU: {gpu}"
        ok = False
    merged_ok, _ = merged_model_complete(merged)
    if ok and not merged_ok:
        ok = run_stage("merge-sft-step-186", [
            sys.executable, str(root / "scripts/merge_sft_lora.py"),
            "--base-model", str(args.base_model.resolve()),
            "--output", str(merged), "--device", "cuda",
        ])
    if ok:
        merged_ok, errors = merged_model_complete(merged)
        if not merged_ok:
            report["failed_stage"] = "verify-merged-model"
            report["failure_reason"] = "; ".join(errors)
            ok = False
    stage_script = str(root / "examples/sglang_multiturn/run_grpo_stage.sh")
    if ok:
        ok = run_stage("grpo-preflight", ["bash", stage_script, "check"])
    if ok:
        ok = run_stage("overfit-one", ["bash", stage_script, "overfit-one"])

    artifacts = verify_overfit_artifacts(checkpoint, artifacts_root)
    report["artifacts"] = artifacts
    if ok and not artifacts["complete"]:
        report["failed_stage"] = "verify-overfit-one-artifacts"
        report["failure_reason"] = "; ".join(artifacts["errors"])
        ok = False
    report["finished_at"] = utc_now()
    report["outcome"] = "success" if ok else "incomplete_or_failed"
    if not ok and "failure_reason" not in report:
        report["failure_reason"] = f"stage {report['failed_stage']} returned non-zero; see {log_file}"
    atomic_json(report_file, report)
    atomic_json(state_file, report)

    if not args.arm:
        print(f"workflow outcome={report['outcome']}; dry run; report={report_file}")
        return 0 if ok else 3
    deadline = time.monotonic() + max(0, args.shutdown_delay_seconds)
    while time.monotonic() < deadline:
        if STOP_REQUESTED:
            report["poweroff"]["cancelled_at"] = utc_now()
            atomic_json(report_file, report)
            return 130
        time.sleep(1)
    subprocess.run(["/usr/bin/sync"], check=False)
    report["poweroff"]["requested_at"] = utc_now()
    atomic_json(report_file, report)
    accepted, attempts = poweroff()
    report["poweroff"]["accepted"] = accepted
    report["poweroff"]["attempts"] = attempts
    atomic_json(report_file, report)
    return 0 if accepted else 5


if __name__ == "__main__":
    raise SystemExit(main())
