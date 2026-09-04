#!/usr/bin/env python3
"""Watch one already-running SFT job and power off after it exits.

The watcher treats the run as successful only when the final validation and
checkpoint markers are complete.  An early process exit is recorded as an
incomplete/failed run.  In armed mode both outcomes end in a machine poweroff,
which is useful for unattended cloud instances.
"""

from __future__ import annotations

import argparse
import fcntl
import glob
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger("sft-shutdown-watcher")
STOP_REQUESTED = False


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Power off after a specific SFT process exits, recording success or failure first."
    )
    parser.add_argument("--pid", type=int, required=True, help="PID of the torchrun parent process")
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--expected-epoch", type=int, required=True)
    parser.add_argument("--command-substring", default="verl.trainer.fsdp_sft_trainer")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--heartbeat-seconds", type=int, default=600)
    parser.add_argument("--post-exit-grace-seconds", type=int, default=60)
    parser.add_argument("--shutdown-delay-seconds", type=int, default=30)
    parser.add_argument("--log-file", type=Path, required=True)
    parser.add_argument("--report-file", type=Path, required=True)
    parser.add_argument("--state-file", type=Path, required=True)
    parser.add_argument("--lock-file", type=Path, required=True)
    parser.add_argument(
        "--arm",
        action="store_true",
        help="Actually power off. Without this flag the watcher only records the outcome.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Print current process/artifact status and exit without monitoring or poweroff.",
    )
    return parser.parse_args()


def configure_logging(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)sZ %(levelname)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S")
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    LOGGER.addHandler(stream_handler)


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, f"missing: {path}"
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"unreadable JSON: {path}: {exc}"
    if not isinstance(value, dict):
        return None, f"JSON root is not an object: {path}"
    return value, None


def process_identity(pid: int) -> dict[str, Any] | None:
    proc = Path("/proc") / str(pid)
    try:
        stat_text = (proc / "stat").read_text(encoding="utf-8")
        right_paren = stat_text.rfind(")")
        fields_after_comm = stat_text[right_paren + 2 :].split()
        start_ticks = int(fields_after_comm[19])  # /proc stat field 22
        raw_cmdline = (proc / "cmdline").read_bytes()
        cmdline = " ".join(part.decode("utf-8", errors="replace") for part in raw_cmdline.split(b"\0") if part)
    except (FileNotFoundError, ProcessLookupError):
        return None
    except (OSError, ValueError, IndexError) as exc:
        return {"pid": pid, "error": str(exc)}
    return {"pid": pid, "start_ticks": start_ticks, "cmdline": cmdline}


def same_process(pid: int, expected_start_ticks: int) -> bool:
    identity = process_identity(pid)
    return bool(identity and identity.get("start_ticks") == expected_start_ticks)


def matching_training_processes(command_substring: str) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for item in glob.glob("/proc/[0-9]*/cmdline"):
        try:
            raw = Path(item).read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        cmdline = " ".join(part.decode("utf-8", errors="replace") for part in raw.split(b"\0") if part)
        if command_substring in cmdline:
            matches.append({"pid": int(Path(item).parent.name), "cmdline": cmdline})
    return sorted(matches, key=lambda value: value["pid"])


def checkpoint_status(root: Path, expected_step: int, expected_epoch: int) -> dict[str, Any]:
    errors: list[str] = []
    final_dir = root / f"global_step_{expected_step}"
    expected_alias = f"epoch_{expected_epoch}"

    metadata, error = read_json(final_dir / "checkpoint_metadata.json")
    if error:
        errors.append(error)
    elif metadata:
        if int(metadata.get("step", -1)) != expected_step:
            errors.append(f"final checkpoint metadata step is {metadata.get('step')}, expected {expected_step}")
        if metadata.get("alias") != expected_alias:
            errors.append(f"final checkpoint alias is {metadata.get('alias')!r}, expected {expected_alias!r}")

    trainer_state = final_dir / "trainer_state.pt"
    if not trainer_state.is_file() or trainer_state.stat().st_size <= 0:
        errors.append(f"missing or empty trainer state: {trainer_state}")

    weight_files = [
        path
        for pattern in ("*.safetensors", "pytorch_model*.bin", "adapter_model*.bin")
        for path in final_dir.glob(pattern)
        if path.is_file() and path.stat().st_size > 0
    ]
    if not weight_files:
        errors.append(f"no non-empty model/adapter weight file in {final_dir}")

    alias, error = read_json(root / f"{expected_alias}.json")
    if error:
        errors.append(error)
    elif alias and int(alias.get("step", -1)) != expected_step:
        errors.append(f"{expected_alias}.json step is {alias.get('step')}, expected {expected_step}")

    alias_reference, error = read_json(root / expected_alias / "checkpoint_reference.json")
    if error:
        errors.append(error)
    elif alias_reference and int(alias_reference.get("step", -1)) != expected_step:
        errors.append(f"{expected_alias} reference step is {alias_reference.get('step')}, expected {expected_step}")

    last, error = read_json(root / "last_checkpoint.json")
    if error:
        errors.append(error)
    elif last:
        metrics = last.get("metrics") if isinstance(last.get("metrics"), dict) else {}
        if int(last.get("step", -1)) != expected_step:
            errors.append(f"last checkpoint step is {last.get('step')}, expected {expected_step}")
        if int(metrics.get("step", -1)) != expected_step:
            errors.append(f"last validation step is {metrics.get('step')}, expected {expected_step}")
        if float(metrics.get("sft/val/epoch", -1)) != float(expected_epoch):
            errors.append(
                f"last validation epoch is {metrics.get('sft/val/epoch')}, expected {expected_epoch}"
            )
        for key in ("sft/val/masked_token_nll", "sft/val/trajectory_macro_loss"):
            if key not in metrics:
                errors.append(f"last validation metrics missing {key}")

    best, error = read_json(root / "best_checkpoint.json")
    if error:
        errors.append(error)
    elif best:
        best_step = int(best.get("step", -1))
        if best_step <= 0 or best_step > expected_step:
            errors.append(f"best checkpoint step {best_step} is outside 1..{expected_step}")
        elif not (root / f"global_step_{best_step}" / "checkpoint_metadata.json").is_file():
            errors.append(f"best checkpoint target global_step_{best_step} is missing")

    completed_steps: list[int] = []
    for metadata_path in root.glob("global_step_*/checkpoint_metadata.json"):
        value, metadata_error = read_json(metadata_path)
        if not metadata_error and value:
            try:
                completed_steps.append(int(value["step"]))
            except (KeyError, TypeError, ValueError):
                pass

    return {
        "complete": not errors,
        "errors": errors,
        "checkpoint_root": str(root),
        "expected_step": expected_step,
        "expected_epoch": expected_epoch,
        "completed_checkpoint_steps": sorted(set(completed_steps)),
        "final_weight_files": [str(path) for path in sorted(weight_files)],
        "last_checkpoint": last,
        "best_checkpoint": best,
    }


def run_capture(command: list[str]) -> dict[str, Any]:
    try:
        result = subprocess.run(command, text=True, capture_output=True, timeout=15, check=False)
        return {
            "command": command,
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except (OSError, subprocess.SubprocessError) as exc:
        return {"command": command, "error": str(exc)}


def system_snapshot(workspace: Path, command_substring: str) -> dict[str, Any]:
    disk = shutil.disk_usage(workspace)
    snapshot: dict[str, Any] = {
        "captured_at": utc_now(),
        "load_average": list(os.getloadavg()),
        "disk": {
            "path": str(workspace),
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
        },
        "remaining_training_processes": matching_training_processes(command_substring),
    }
    if Path("/usr/bin/nvidia-smi").exists():
        snapshot["gpu"] = run_capture(
            [
                "/usr/bin/nvidia-smi",
                "--query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu,power.draw",
                "--format=csv,noheader",
            ]
        )
    snapshot["kernel_warnings"] = run_capture(
        ["/usr/bin/dmesg", "--ctime", "--level=err,warn"]
    )
    output = snapshot.get("kernel_warnings", {}).get("stdout", "")
    if isinstance(output, str) and len(output) > 20000:
        snapshot["kernel_warnings"]["stdout"] = output[-20000:]
    return snapshot


def request_stop(signum: int, _frame: Any) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    LOGGER.warning("received signal %s; watcher cancelled without powering off", signum)


def poweroff() -> tuple[bool, list[dict[str, Any]]]:
    attempts: list[dict[str, Any]] = []
    commands = [
        ["/usr/bin/shutdown", "-h", "now"],
        ["/usr/bin/systemctl", "poweroff"],
        ["/usr/sbin/poweroff"],
    ]
    for command in commands:
        if not Path(command[0]).exists():
            continue
        result = run_capture(command)
        attempts.append(result)
        if result.get("returncode") == 0:
            return True, attempts
    return False, attempts


def main() -> int:
    args = parse_args()
    configure_logging(args.log_file.resolve())
    checkpoint_root = args.checkpoint_root.resolve()
    workspace = Path.cwd().resolve()

    identity = process_identity(args.pid)
    artifacts = checkpoint_status(checkpoint_root, args.expected_step, args.expected_epoch)
    if args.check_only:
        print(json.dumps({"process": identity, "artifacts": artifacts}, ensure_ascii=False, indent=2))
        return 0
    if identity is None:
        LOGGER.error("target PID %s does not exist; watcher was not armed", args.pid)
        return 2
    if identity.get("error"):
        LOGGER.error("cannot read target process identity: %s", identity["error"])
        return 2
    if args.command_substring not in str(identity.get("cmdline", "")):
        LOGGER.error("PID %s does not match required command substring %r", args.pid, args.command_substring)
        return 2

    args.lock_file.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = args.lock_file.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        LOGGER.error("another SFT shutdown watcher already holds %s", args.lock_file)
        return 3
    lock_handle.seek(0)
    lock_handle.truncate()
    lock_handle.write(f"pid={os.getpid()}\ntarget_pid={args.pid}\nstarted_at={utc_now()}\n")
    lock_handle.flush()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    started_at = utc_now()
    start_ticks = int(identity["start_ticks"])
    state = {
        "watcher_pid": os.getpid(),
        "target": identity,
        "started_at": started_at,
        "last_seen_alive_at": started_at,
        "armed": bool(args.arm),
        "policy": "poweroff_after_any_target_exit",
        "expected_step": args.expected_step,
        "expected_epoch": args.expected_epoch,
        "checkpoint_root": str(checkpoint_root),
    }
    atomic_write_json(args.state_file.resolve(), state)
    LOGGER.info(
        "watching PID %s (start_ticks=%s), expected epoch=%s step=%s, armed=%s",
        args.pid,
        start_ticks,
        args.expected_epoch,
        args.expected_step,
        args.arm,
    )

    last_heartbeat = 0.0
    while same_process(args.pid, start_ticks):
        if STOP_REQUESTED:
            state["cancelled_at"] = utc_now()
            atomic_write_json(args.state_file.resolve(), state)
            return 130
        now = time.monotonic()
        state["last_seen_alive_at"] = utc_now()
        if now - last_heartbeat >= args.heartbeat_seconds:
            live_artifacts = checkpoint_status(checkpoint_root, args.expected_step, args.expected_epoch)
            LOGGER.info(
                "target alive; completed checkpoints=%s; final_complete=%s",
                live_artifacts["completed_checkpoint_steps"],
                live_artifacts["complete"],
            )
            state["completed_checkpoint_steps"] = live_artifacts["completed_checkpoint_steps"]
            atomic_write_json(args.state_file.resolve(), state)
            last_heartbeat = now
        time.sleep(max(1, args.poll_seconds))

    ended_at = utc_now()
    LOGGER.warning("target process ended or its identity changed at %s", ended_at)
    time.sleep(max(0, args.post_exit_grace_seconds))

    artifacts = checkpoint_status(checkpoint_root, args.expected_step, args.expected_epoch)
    outcome = "success" if artifacts["complete"] else "incomplete_or_failed"
    if outcome == "success":
        reason = (
            f"training, epoch-{args.expected_epoch} validation and global_step_{args.expected_step} "
            "checkpoint completed before the target process exited"
        )
    else:
        reason = "target process exited before all required final artifacts were complete: " + "; ".join(
            artifacts["errors"]
        )

    report = {
        "schema_version": "sft-shutdown-report-v1",
        "outcome": outcome,
        "reason": reason,
        "watch_started_at": started_at,
        "target_ended_at": ended_at,
        "last_seen_alive_at": state["last_seen_alive_at"],
        "target": identity,
        "artifacts": artifacts,
        "system": system_snapshot(workspace, args.command_substring),
        "poweroff": {
            "armed": bool(args.arm),
            "requested_at": None,
            "attempts": [],
        },
    }
    atomic_write_json(args.report_file.resolve(), report)
    LOGGER.warning("final outcome=%s; reason=%s", outcome, reason)

    if not args.arm:
        LOGGER.warning("dry-run mode: report written, machine will remain online")
        return 0 if outcome == "success" else 4

    LOGGER.warning("machine poweroff in %s seconds; send SIGTERM to watcher PID %s to cancel", args.shutdown_delay_seconds, os.getpid())
    deadline = time.monotonic() + max(0, args.shutdown_delay_seconds)
    while time.monotonic() < deadline:
        if STOP_REQUESTED:
            report["poweroff"]["cancelled_at"] = utc_now()
            atomic_write_json(args.report_file.resolve(), report)
            LOGGER.warning("poweroff cancelled")
            return 130
        time.sleep(1)

    subprocess.run(["/usr/bin/sync"], check=False)
    report["poweroff"]["requested_at"] = utc_now()
    atomic_write_json(args.report_file.resolve(), report)
    success, attempts = poweroff()
    report["poweroff"]["attempts"] = attempts
    report["poweroff"]["command_accepted"] = success
    atomic_write_json(args.report_file.resolve(), report)
    if not success:
        LOGGER.error("all poweroff commands failed: %s", attempts)
        return 5
    LOGGER.warning("poweroff command accepted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
