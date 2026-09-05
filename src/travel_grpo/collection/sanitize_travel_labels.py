"""Remove historical TravelGym best-ID labels from rollout parquet files.

Only TravelGym rows are accepted by the active data contract.  The script is
safe to rerun and writes through a sibling temporary file before replacement;
rows from an accidentally mixed input are left untouched and should be
removed before merging the training split.
"""

from __future__ import annotations

import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .._paths import REPOSITORY_ROOT


PUBLIC_TRAVEL_POLICY = (
    "\n\nTravelGym public protocol: Search returns the complete candidate list for "
    "the current aspect. Action only asks for or records natural-language "
    "preference evidence; it never filters or shrinks that list. The Actor "
    "must compare the evidence implicitly and submit exactly one option ID "
    "that appeared in Search."
)


def sanitize_file(path: Path) -> int:
    table = pq.read_table(path)
    column_index = table.schema.get_field_index("reward_model")
    if column_index < 0:
        return 0
    column = table.column("reward_model")
    rows = column.to_pylist()
    prompt_column = table.column("prompt") if "prompt" in table.column_names else None
    prompt_rows = prompt_column.to_pylist() if prompt_column is not None else None
    changed = 0
    for row_index, row in enumerate(rows):
        if isinstance(row, dict) and row.get("env_name") == "TravelGym":
            row_changed = False
            if row.get("ground_truth") or row.get("style") != "terminal":
                row["ground_truth"] = ""
                row["style"] = "terminal"
                row_changed = True
            if prompt_rows is not None and isinstance(prompt_rows[row_index], list):
                prompt_messages = prompt_rows[row_index]
                already_has_policy = any(
                    isinstance(message, dict)
                    and message.get("role") == "system"
                    and isinstance(message.get("content"), str)
                    and "TravelGym public protocol:" in message["content"]
                    for message in prompt_messages
                )
                for message in prompt_messages:
                    if not isinstance(message, dict):
                        continue
                    content = message.get("content")
                    if message.get("role") != "system" and isinstance(content, str) and PUBLIC_TRAVEL_POLICY in content:
                        message["content"] = content.replace(PUBLIC_TRAVEL_POLICY, "")
                        content = message["content"]
                        row_changed = True
                    if (
                        not already_has_policy
                        and message.get("role") == "system"
                        and isinstance(content, str)
                    ):
                        message["content"] = content + PUBLIC_TRAVEL_POLICY
                        row_changed = True
                        break
            changed += int(row_changed)
    if not changed:
        return 0
    sanitized = table.set_column(column_index, table.schema.field(column_index).name, pa.array(rows, type=column.type))
    if prompt_column is not None:
        prompt_index = table.schema.get_field_index("prompt")
        sanitized = sanitized.set_column(prompt_index, table.schema.field(prompt_index).name, pa.array(prompt_rows, type=prompt_column.type))
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(sanitized, temporary, compression="zstd")
    os.replace(temporary, path)
    return changed


def main() -> None:
    root = REPOSITORY_ROOT / "data"
    total = 0
    for path in sorted(root.rglob("*.parquet")):
        changed = sanitize_file(path)
        if changed:
            print(f"{path}: sanitized {changed} TravelGym rows")
            total += changed
    print(f"sanitized_rows={total}")


if __name__ == "__main__":
    main()
