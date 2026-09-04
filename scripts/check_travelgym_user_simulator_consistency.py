#!/usr/bin/env python3
"""Compare the restored DeepSeek User Simulator with Teacher SFT turns."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gyms" / "TravelGym"))

from dotenv import load_dotenv  # noqa: E402

import travelgym  # noqa: E402
from travelgym.env.user_simulator import (  # noqa: E402
    UNAVAILABLE_RESPONSE,
    VAGUE_RESPONSE,
    evaluate_user_question,
    initialize_preferences,
    new_user_telemetry,
)


PRIVATE_ID = re.compile(r"\bP\d+\b", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def load_tasks() -> dict[str, dict[str, Any]]:
    tasks: dict[str, dict[str, Any]] = {}
    for path in sorted((ROOT / "gyms" / "TravelGym" / "travelgym" / "data").glob("travelgym_data_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        tasks.update({str(key): value for key, value in payload.items()})
    return tasks


def tool_arguments(message: Mapping[str, Any]) -> dict[str, str] | None:
    calls = message.get("tool_calls")
    if not isinstance(calls, list) or not calls:
        return None
    arguments = calls[0].get("function", {}).get("arguments")
    return dict(arguments) if isinstance(arguments, Mapping) else None


def clean_tool_text(value: Any) -> str:
    return str(value or "").split("\n\nPublic control:", 1)[0].strip()


def expected_type(baseline: str) -> str:
    if baseline.startswith(VAGUE_RESPONSE):
        return "4"
    if baseline.startswith(UNAVAILABLE_RESPONSE):
        return "3"
    return "2"


def select_samples(
    audit_records: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    tasks: Mapping[str, dict[str, Any]],
    count: int,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for row_index, (audit, row) in enumerate(zip(audit_records, rows, strict=True)):
        source_meta = audit.get("source_meta", {})
        is_teacher = isinstance(source_meta, Mapping) and bool(source_meta.get("teacher_cache_key"))
        task_id = str(audit.get("task_id", ""))
        task = tasks.get(task_id)
        if not task:
            continue
        messages = row.get("messages", [])
        events = audit.get("events", [])
        first_action = next(
            (
                event for event in events
                if event.get("choice") == "action"
                and (
                    event.get("accepted") is True
                    or event.get("error_kind") == "vague-action"
                )
            ),
            None,
        )
        if not first_action:
            continue
        assistant_index = int(first_action["assistant_index"])
        if assistant_index + 1 >= len(messages):
            continue
        arguments = tool_arguments(messages[assistant_index])
        if not arguments or arguments.get("choice") != "action":
            continue
        baseline = clean_tool_text(messages[assistant_index + 1].get("content"))
        kind = expected_type(baseline)
        candidates.append({
            "row_index": row_index,
            "source_kind": "deepseek_teacher" if is_teacher else "historical_sft",
            "task_id": task_id,
            "aspect": str(first_action.get("aspect", "")),
            "assistant_index": assistant_index,
            "question": str(arguments.get("content", "")),
            "teacher_response": baseline,
            "expected_type": kind,
            "task": task,
            "messages": messages,
            "events": events,
        })

    chosen: list[dict[str, Any]] = []
    vague = next((item for item in candidates if item["expected_type"] == "4"), None)
    if vague:
        chosen.append(vague)
    used_aspects = {item["aspect"] for item in chosen}
    specific_candidates = [
        item for item in candidates if item["source_kind"] == "deepseek_teacher"
    ]
    for item in specific_candidates:
        if item["expected_type"] != "2" or item["aspect"] in used_aspects:
            continue
        chosen.append(item)
        used_aspects.add(item["aspect"])
        if len(chosen) >= count:
            break
    for item in candidates:
        if len(chosen) >= count:
            break
        if item not in chosen:
            chosen.append(item)
    if len(chosen) < count:
        raise RuntimeError(f"only {len(chosen)} usable SFT action turns were found")
    return chosen[:count]


def simulator_state(sample: Mapping[str, Any]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    task = sample["task"]
    preferences = initialize_preferences(task)
    history: list[dict[str, str]] = []
    assistant_index = int(sample["assistant_index"])
    messages = sample["messages"]
    event_by_index = {
        int(event["assistant_index"]): event
        for event in sample["events"]
        if "assistant_index" in event
    }
    for index in range(assistant_index):
        message = messages[index]
        if message.get("role") != "assistant":
            continue
        arguments = tool_arguments(message)
        event = event_by_index.get(index)
        if not arguments or not event or event.get("accepted") is not True:
            continue
        choice = arguments.get("choice")
        content = str(arguments.get("content", ""))
        if choice == "search":
            aspect = str(event.get("aspect", ""))
            history.extend([
                {"role": "agent", "content": content},
                {
                    "role": "database",
                    "content": f"Search results for {aspect}; all candidates are shown. ... (skip detailed results here) ...",
                },
            ])
        elif choice == "answer":
            aspect = str(event.get("aspect", ""))
            preferences = [item for item in preferences if item["aspect"] != aspect]
    return history, preferences


async def run(args: argparse.Namespace) -> dict[str, Any]:
    load_dotenv(ROOT / ".env", override=False)
    if args.samples < 3:
        raise ValueError("--samples must be at least 3")
    output_dir = args.output_dir or (
        ROOT
        / "outputs"
        / "travelgym_user_simulator_consistency"
        / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    os.environ["TRAVELGYM_USER_CACHE_PATH"] = str(output_dir / "responses.sqlite3")
    os.environ["TRAVELGYM_USER_API_LOG_PATH"] = str(output_dir / "api.events.jsonl")

    config = travelgym.get_default_config()
    config.user_simulator_mode = "deepseek_api"
    config.model_name = os.environ.get("USER_MODEL_NAME", config.model_name)
    config.timeout = args.timeout
    config.validate()
    if "deepseek" not in config.model_name.casefold():
        raise RuntimeError("USER_MODEL_NAME must select the DeepSeek Teacher/User Simulator model")

    audit = json.loads((ROOT / "sft" / "travel_sft_qwen35_merged.audit.json").read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in (ROOT / "sft" / "travel_sft_qwen35_merged.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    samples = select_samples(audit["records"], rows, load_tasks(), args.samples)
    telemetry = new_user_telemetry()
    results: list[dict[str, Any]] = []
    import random

    for index, sample in enumerate(samples):
        history, preferences = simulator_state(sample)
        turn = await evaluate_user_question(
            sample["question"],
            task_id=sample["task_id"],
            aspect=sample["aspect"],
            scenario=str(sample["task"].get("scenario", "")),
            history=history,
            available_preferences=preferences,
            nonpreference_times=0,
            elicitation_interval=3,
            config=config,
            telemetry=telemetry,
            rng=random.Random(20260902 + index),
            max_attempts=args.max_attempts,
        )
        structurally_consistent = (
            bool(turn.response.strip())
            and not PRIVATE_ID.search(turn.response)
            and (
                turn.judgment_type == sample["expected_type"]
                if sample["expected_type"] in {"3", "4"}
                else turn.judgment_type == "2" and bool(turn.elicited_preference_ids)
            )
        )
        results.append({
            "row_index": sample["row_index"],
            "source_kind": sample["source_kind"],
            "task_id": sample["task_id"],
            "aspect": sample["aspect"],
            "question": sample["question"],
            "teacher_response": sample["teacher_response"],
            "restored_response": turn.response,
            "expected_type": sample["expected_type"],
            "actual_type": turn.judgment_type,
            "elicited_preference_count": len(turn.elicited_preference_ids),
            "structurally_consistent": structurally_consistent,
        })

    report = {
        "schema_version": "travelgym-user-simulator-consistency-v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": config.model_name,
        "sample_count": len(results),
        "passed": all(item["structurally_consistent"] for item in results),
        "telemetry": telemetry,
        "samples": results,
        "cache_path": str(output_dir / "responses.sqlite3"),
        "event_log_path": str(output_dir / "api.events.jsonl"),
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "passed": report["passed"],
        "sample_count": report["sample_count"],
        "model": report["model"],
        "api_calls": telemetry["user_api_calls"],
        "total_tokens": telemetry["user_total_tokens"],
        "report": str(report_path),
    }, ensure_ascii=False, indent=2))
    return report


def main() -> int:
    args = parse_args()
    report = asyncio.run(run(args))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
