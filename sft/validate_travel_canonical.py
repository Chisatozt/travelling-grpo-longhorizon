"""Validate a canonical TravelGym JSONL/JSON corpus without loading a model.

The validator is intentionally strict about the public schema and message
boundary, while leaving task IDs/reward labels to the private audit sidecar.
It is suitable for CI and for checking a generated corpus before a GPU job.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Mapping

try:
    from .travel_canonical import SCHEMA_VERSION, CanonicalError, validate_canonical
except ImportError:  # direct script execution
    from travel_canonical import SCHEMA_VERSION, CanonicalError, validate_canonical


PRIVATE_TEXT = re.compile(
    r"(?:terminal[_ ]?reward|step[_ ]?reward|reward\s*[:=]|preference\s*id\s*[:#]?\s*P\d+|\b(?:correct_ids?|best_ids?|preference_ids?)\s*[:=])",
    re.IGNORECASE,
)
ALLOWED_METADATA = {"source", "data_source", "trajectory_class", "sample_weight", "enable_thinking"}


def _rows(path: Path) -> list[Mapping[str, Any]]:
    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(payload, Mapping):
        payload = payload.get("data", [])
    if not isinstance(payload, list):
        raise ValueError("canonical input must be a JSON list or JSONL")
    return [row for row in payload if isinstance(row, Mapping)]


def validate_rows(rows: list[Mapping[str, Any]], *, max_length: int = 32768) -> dict[str, Any]:
    classes: dict[str, int] = {}
    total_supervised_turns = 0
    for index, row in enumerate(rows):
        try:
            validate_canonical(row)
        except CanonicalError as exc:
            raise ValueError(f"row {index}: {exc}") from exc
        messages = row["messages"]
        mask = row["assistant_train_mask"]
        if len(messages) != len(mask):
            raise ValueError(f"row {index}: assistant_train_mask is not message-aligned")
        for message, value in zip(messages, mask):
            role = message.get("role")
            if role != "assistant" and int(value):
                raise ValueError(f"row {index}: non-assistant message has a training mask")
            if role == "assistant" and message.get("tool_calls") and int(value):
                if not str(message.get("reasoning_content") or "").strip():
                    raise ValueError(f"row {index}: supervised tool turn has no reasoning_content")
        metadata = row.get("trainer_metadata", {})
        if isinstance(metadata, Mapping):
            unknown = set(metadata) - ALLOWED_METADATA
            if unknown:
                raise ValueError(f"row {index}: private metadata leaked into trainer_metadata: {sorted(unknown)}")
            category = str(metadata.get("trajectory_class", "unknown"))
            classes[category] = classes.get(category, 0) + 1
        else:
            raise ValueError(f"row {index}: trainer_metadata must be an object")
        total_supervised_turns += sum(int(value) for value in mask)
        public = json.dumps(messages, ensure_ascii=False)
        if PRIVATE_TEXT.search(public):
            raise ValueError(f"row {index}: private reward/label text in public messages")
    return {
        "schema_version": SCHEMA_VERSION,
        "records": len(rows),
        "classes": classes,
        "supervised_message_turns": total_supervised_turns,
        "max_length": max_length,
        "length_check": "requires --tokenizer manifest for exact token lengths",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--max-length", type=int, default=32768)
    args = parser.parse_args()
    report = validate_rows(_rows(Path(args.input)), max_length=args.max_length)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

