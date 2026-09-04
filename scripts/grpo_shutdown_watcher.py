#!/usr/bin/env python3
"""Watch an active GRPO training or native-validation PID, then power off.

The exact Linux PID/start-time pair is monitored to avoid PID-reuse races.
Training and val-only evaluation use separate artifact contracts; an early
exit is recorded as incomplete. Armed mode powers off after either outcome.
"""

from __future__ import annotations

import argparse
import fcntl
import glob
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger("grpo-shutdown-watcher")
STOP_REQUESTED = False


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True, help="PID of the active main_ppo process")
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--training-log", type=Path, required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument(
        "--task-kind",
        choices=("training", "training_no_validation", "validation"),
        default="training",
        help="Artifact contract to verify after the target exits.",
    )
    parser.add_argument("--command-substring", default="verl.trainer.main_ppo")
    parser.add_argument("--poll-seconds", type=int, default=20)
    parser.add_argument("--heartbeat-seconds", type=int, default=300)
    parser.add_argument("--post-exit-grace-seconds", type=int, default=30)
    parser.add_argument("--shutdown-delay-seconds", type=int, default=30)
    parser.add_argument("--log-file", type=Path, required=True)
    parser.add_argument("--report-file", type=Path, required=True)
    parser.add_argument("--state-file", type=Path, required=True)
    parser.add_argument("--lock-file", type=Path, required=True)
    parser.add_argument("--arm", action="store_true", help="Power off after any target exit.")
    parser.add_argument(
        "--allow-platform-shutdown-wrapper",
        action="store_true",
        help=(
            "Deprecated compatibility flag; AutoDL's /usr/bin/shutdown is now "
            "always preferred when available."
        ),
    )
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def configure_logging(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)sZ %(levelname)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S"
    )
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    for handler in (logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(formatter)
        LOGGER.addHandler(handler)


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


def process_identity(pid: int) -> dict[str, Any] | None:
    proc = Path("/proc") / str(pid)
    try:
        stat_text = (proc / "stat").read_text(encoding="utf-8")
        fields = stat_text[stat_text.rfind(")") + 2 :].split()
        start_ticks = int(fields[19])
        cmdline = " ".join(
            part.decode("utf-8", errors="replace")
            for part in (proc / "cmdline").read_bytes().split(b"\0")
            if part
        )
    except (FileNotFoundError, ProcessLookupError):
        return None
    except (OSError, ValueError, IndexError) as exc:
        return {"pid": pid, "error": str(exc)}
    return {"pid": pid, "start_ticks": start_ticks, "cmdline": cmdline}


def same_process(pid: int, start_ticks: int) -> bool:
    identity = process_identity(pid)
    return bool(identity and identity.get("start_ticks") == start_ticks)


def numeric_steps(paths: list[Path], prefix: str = "") -> list[int]:
    steps: list[int] = []
    for path in paths:
        try:
            steps.append(int(path.name.removeprefix(prefix).removesuffix(".jsonl")))
        except ValueError:
            pass
    return sorted(set(steps))


def _validation_step_numbers(paths: list[Path]) -> list[int]:
    steps: list[int] = []
    for path in paths:
        match = re.fullmatch(r"(\d+)(?:_pass\d+)?\.jsonl", path.name)
        if match:
            steps.append(int(match.group(1)))
    return sorted(set(steps))


def _validation_artifacts(artifact_root: Path, expected_step: int) -> tuple[list[Path], list[Path]]:
    validation_root = artifact_root / "validation"
    generations = sorted(
        path
        for path in (
            list(validation_root.glob(f"{expected_step}.jsonl"))
            + list(validation_root.glob(f"{expected_step}_pass*.jsonl"))
        )
        if nonempty(path)
    )
    summaries = sorted(
        path
        for path in (
            list(validation_root.glob(f"{expected_step}_summary.json"))
            + list(validation_root.glob(f"{expected_step}_pass*_summary.json"))
        )
        if nonempty(path)
    )
    return generations, summaries


def grpo_artifact_status(
    checkpoint_root: Path,
    artifact_root: Path,
    expected_step: int,
    task_kind: str = "training",
) -> dict[str, Any]:
    """Check the small set of artifacts that distinguishes a clean exit.

    Training needs a checkpoint, rollout dump, and ordinary validation dump.
    A training_no_validation job intentionally omits the validation dump.
    Native val-only evaluation has no optimizer checkpoint, so it instead needs
    its pass-generation JSONL and summary JSON.
    """
    if task_kind not in {"training", "training_no_validation", "validation"}:
        raise ValueError(f"unknown GRPO task kind: {task_kind!r}")

    errors: list[str] = []
    final_dir = checkpoint_root / f"global_step_{expected_step}"
    latest_file = checkpoint_root / "latest_checkpointed_iteration.txt"
    latest = latest_file.read_text(encoding="utf-8").strip() if nonempty(latest_file) else None
    actor_dir = final_dir / "actor"
    actor_files = sorted(path for path in actor_dir.rglob("*") if nonempty(path)) if actor_dir.is_dir() else []
    validation_root = artifact_root / "validation"
    validation_file = artifact_root / "validation" / f"{expected_step}.jsonl"
    validation_summary = None
    validation_generations: list[Path] = []
    validation_summaries: list[Path] = []
    rollout_file = artifact_root / "rollouts" / f"{expected_step}.jsonl"

    if task_kind in {"training", "training_no_validation"}:
        if latest != str(expected_step):
            errors.append(f"latest checkpoint marker is {latest!r}, expected {expected_step!r}")
        if not actor_files:
            errors.append(f"no non-empty Actor checkpoint files under {actor_dir}")
        for name in ("data.pt", "rng_state.pt"):
            if not nonempty(final_dir / name):
                errors.append(f"missing or empty state: {final_dir / name}")
        if task_kind == "training" and not nonempty(validation_file):
            errors.append(f"missing final validation generations: {validation_file}")
        if not nonempty(rollout_file):
            errors.append(f"missing final training rollout: {rollout_file}")
        completed_validation_steps = numeric_steps(
            list((artifact_root / "validation").glob("*.jsonl"))
        )
    else:
        validation_generations, validation_summaries = _validation_artifacts(
            artifact_root, expected_step
        )
        if not validation_generations:
            errors.append(
                f"missing validation generations for step {expected_step} under {validation_root}"
            )
        if not validation_summaries:
            errors.append(
                f"missing validation summary for step {expected_step} under {validation_root}"
            )
        if validation_generations:
            validation_file = validation_generations[-1]
        if validation_summaries:
            validation_summary = validation_summaries[-1]
        completed_validation_steps = _validation_step_numbers(
            list(validation_root.glob("*.jsonl"))
        )

    return {
        "complete": not errors,
        "errors": errors,
        "task_kind": task_kind,
        "expected_step": expected_step,
        "latest_checkpointed_iteration": latest,
        "checkpoint_root": str(checkpoint_root),
        "artifact_root": str(artifact_root),
        "completed_checkpoint_steps": numeric_steps(
            list(checkpoint_root.glob("global_step_*")), "global_step_"
        ),
        "completed_validation_steps": completed_validation_steps,
        "completed_rollout_steps": numeric_steps(
            list((artifact_root / "rollouts").glob("*.jsonl"))
        ),
        "final_actor_files": [str(path) for path in actor_files],
        "final_validation_file": str(validation_file),
        "final_validation_summary": str(validation_summary) if validation_summary else None,
        "final_rollout_file": str(rollout_file),
    }


def training_log_status(path: Path, max_bytes: int = 2_000_000) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes))
            text = handle.read().decode("utf-8", errors="replace")
        stat = path.stat()
    except FileNotFoundError:
        return {"path": str(path), "exists": False, "latest_step": None, "failure_lines": []}
    steps = [int(value) for value in re.findall(r"\bstep:(\d+)\s+-", text)]
    markers = (
        "traceback", "cuda out of memory", "actordiederror", "runtimeerror",
        "valueerror", "system_error", "response length", "rollout failed", "killed",
    )
    failures = [
        line[-2000:] for line in text.splitlines()
        if any(marker in line.casefold() for marker in markers)
    ][-20:]
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": size,
        "mtime_ns": stat.st_mtime_ns,
        "latest_step": max(steps) if steps else None,
        "failure_lines": failures,
    }


def run_capture(command: list[str]) -> dict[str, Any]:
    try:
        result = subprocess.run(command, text=True, capture_output=True, timeout=20, check=False)
        return {
            "command": command,
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except (OSError, subprocess.SubprocessError) as exc:
        return {"command": command, "error": repr(exc)}


def matching_processes(substring: str) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for item in glob.glob("/proc/[0-9]*/cmdline"):
        pid = int(Path(item).parent.name)
        if pid == os.getpid():
            continue
        try:
            cmdline = " ".join(
                part.decode("utf-8", errors="replace")
                for part in Path(item).read_bytes().split(b"\0") if part
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if substring in cmdline:
            matches.append({"pid": pid, "cmdline": cmdline})
    return sorted(matches, key=lambda value: value["pid"])


def system_snapshot(workspace: Path, command_substring: str) -> dict[str, Any]:
    disk = shutil.disk_usage(workspace)
    snapshot: dict[str, Any] = {
        "captured_at": utc_now(),
        "load_average": list(os.getloadavg()),
        "disk": {"path": str(workspace), "total_bytes": disk.total, "used_bytes": disk.used, "free_bytes": disk.free},
        "remaining_training_processes": matching_processes(command_substring),
    }
    if Path("/usr/bin/nvidia-smi").exists():
        snapshot["gpu"] = run_capture([
            "/usr/bin/nvidia-smi",
            "--query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu,power.draw",
            "--format=csv,noheader",
        ])
    snapshot["kernel_warnings"] = run_capture(["/usr/bin/dmesg", "--ctime", "--level=err,warn"])
    output = snapshot["kernel_warnings"].get("stdout", "")
    if isinstance(output, str) and len(output) > 20000:
        snapshot["kernel_warnings"]["stdout"] = output[-20000:]
    return snapshot


def request_stop(signum: int, _frame: Any) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    LOGGER.warning("received signal %s; watcher cancelled without powering off", signum)


def poweroff_commands(allow_platform_shutdown_wrapper: bool = False) -> list[list[str]]:
    """Use AutoDL's shutdown command first, then retain host fallbacks."""
    commands: list[list[str]] = []
    shutdown = Path("/usr/bin/shutdown")
    if shutdown.exists():
        # AutoDL supplies this as an executable shell fragment rather than a
        # shebang script; invoke it through bash and do not require an opt-in.
        commands.append(["/bin/bash", str(shutdown)])
    for command in (["/usr/bin/systemctl", "poweroff"], ["/usr/sbin/poweroff"]):
        if Path(command[0]).exists():
            commands.append(command)
    return commands


def main() -> int:
    args = parse_args()
    configure_logging(args.log_file.resolve())
    workspace = Path.cwd().resolve()
    checkpoint_root = args.checkpoint_root.resolve()
    artifact_root = args.artifact_root.resolve()
    training_log = args.training_log.resolve()
    identity = process_identity(args.pid)
    artifacts = grpo_artifact_status(
        checkpoint_root, artifact_root, args.expected_step, args.task_kind
    )
    log_status = training_log_status(training_log)
    if args.check_only:
        print(json.dumps({"process": identity, "artifacts": artifacts, "training_log": log_status}, ensure_ascii=False, indent=2))
        return 0
    if not identity or identity.get("error") or args.command_substring not in str(identity.get("cmdline", "")):
        LOGGER.error("target PID %s is absent or does not match %r; watcher was not armed", args.pid, args.command_substring)
        return 2

    args.lock_file.parent.mkdir(parents=True, exist_ok=True)
    lock = args.lock_file.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        LOGGER.error("another watcher already holds %s", args.lock_file)
        return 3
    lock.seek(0)
    lock.truncate()
    lock.write(f"pid={os.getpid()}\ntarget_pid={args.pid}\nstarted_at={utc_now()}\n")
    lock.flush()
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    started_at = utc_now()
    start_ticks = int(identity["start_ticks"])
    state: dict[str, Any] = {
        "schema_version": "grpo-shutdown-state-v1",
        "watcher_pid": os.getpid(), "target": identity, "started_at": started_at,
        "last_seen_alive_at": started_at, "armed": bool(args.arm),
        "policy": "record_then_poweroff_after_any_target_exit",
        "expected_step": args.expected_step, "task_kind": args.task_kind,
        "checkpoint_root": str(checkpoint_root),
        "artifact_root": str(artifact_root), "training_log": str(training_log),
    }
    atomic_json(args.state_file.resolve(), state)
    LOGGER.info("watching PID %s start_ticks=%s expected_step=%s armed=%s", args.pid, start_ticks, args.expected_step, args.arm)

    last_heartbeat = 0.0
    while same_process(args.pid, start_ticks):
        if STOP_REQUESTED:
            state["cancelled_at"] = utc_now()
            atomic_json(args.state_file.resolve(), state)
            return 130
        now = time.monotonic()
        state["last_seen_alive_at"] = utc_now()
        if now - last_heartbeat >= args.heartbeat_seconds:
            live_artifacts = grpo_artifact_status(
                checkpoint_root, artifact_root, args.expected_step, args.task_kind
            )
            live_log = training_log_status(training_log)
            state.update({
                "latest_step": live_log["latest_step"],
                "completed_checkpoint_steps": live_artifacts["completed_checkpoint_steps"],
                "completed_validation_steps": live_artifacts["completed_validation_steps"],
                "completed_rollout_steps": live_artifacts["completed_rollout_steps"],
                "training_log_size_bytes": live_log.get("size_bytes"),
            })
            atomic_json(args.state_file.resolve(), state)
            LOGGER.info("target alive; latest_step=%s checkpoints=%s validation=%s rollouts=%s", live_log["latest_step"], live_artifacts["completed_checkpoint_steps"], live_artifacts["completed_validation_steps"], live_artifacts["completed_rollout_steps"])
            last_heartbeat = now
        time.sleep(max(1, args.poll_seconds))

    ended_at = utc_now()
    LOGGER.warning("target process ended or identity changed at %s", ended_at)
    time.sleep(max(0, args.post_exit_grace_seconds))
    artifacts = grpo_artifact_status(
        checkpoint_root, artifact_root, args.expected_step, args.task_kind
    )
    log_status = training_log_status(training_log)
    outcome = "success" if artifacts["complete"] else "incomplete_or_failed"
    if outcome == "success":
        if args.task_kind == "training":
            reason = f"step-{args.expected_step} training, validation, rollout, and checkpoint completed"
        elif args.task_kind == "training_no_validation":
            reason = f"step-{args.expected_step} training, rollout, and checkpoint completed (validation disabled)"
        else:
            reason = f"step-{args.expected_step} native validation generations and summary completed"
    else:
        reason = "target exited before final artifacts completed: " + "; ".join(artifacts["errors"])
        if log_status["failure_lines"]:
            reason += "; detected errors: " + " | ".join(log_status["failure_lines"][-5:])

    commands = poweroff_commands(args.allow_platform_shutdown_wrapper)
    report: dict[str, Any] = {
        "schema_version": "grpo-shutdown-report-v1", "outcome": outcome, "reason": reason,
        "watch_started_at": started_at, "target_ended_at": ended_at,
        "task_kind": args.task_kind,
        "last_seen_alive_at": state["last_seen_alive_at"], "target": identity,
        "artifacts": artifacts, "training_log": log_status,
        "system": system_snapshot(workspace, args.command_substring),
        "poweroff": {
            "armed": bool(args.arm),
            "allow_platform_shutdown_wrapper": bool(args.allow_platform_shutdown_wrapper),
            "requested_at": None,
            "planned_commands": commands,
            "attempts": [],
        },
    }
    atomic_json(args.report_file.resolve(), report)
    atomic_json(args.state_file.resolve(), {**state, "finished_at": utc_now(), "outcome": outcome})
    LOGGER.warning("final outcome=%s; reason=%s", outcome, reason)
    if not args.arm:
        return 0 if outcome == "success" else 4

    LOGGER.warning("poweroff in %s seconds; SIGTERM watcher PID %s to cancel", args.shutdown_delay_seconds, os.getpid())
    deadline = time.monotonic() + max(0, args.shutdown_delay_seconds)
    while time.monotonic() < deadline:
        if STOP_REQUESTED:
            report["poweroff"]["cancelled_at"] = utc_now()
            atomic_json(args.report_file.resolve(), report)
            return 130
        time.sleep(1)
    subprocess.run(["/usr/bin/sync"], check=False)
    report["poweroff"]["requested_at"] = utc_now()
    atomic_json(args.report_file.resolve(), report)
    attempts: list[dict[str, Any]] = []
    for command in commands:
        result = run_capture(command)
        attempts.append(result)
        if result.get("returncode") == 0:
            report["poweroff"].update({"attempts": attempts, "command_accepted": True})
            atomic_json(args.report_file.resolve(), report)
            return 0
    report["poweroff"].update({"attempts": attempts, "command_accepted": False})
    atomic_json(args.report_file.resolve(), report)
    LOGGER.error("all poweroff commands failed: %s", attempts)
    return 5


if __name__ == "__main__":
    raise SystemExit(main())
