"""Prepare a canonical, public-only TravelGym trajectory corpus.

The command-line path accepts ShareGPT JSON/JSONL or evaluator reward caches,
canonicalizes every record, replays the public reducer, applies the complete
trajectory ``assistant_train_mask`` and writes derived JSONL/audit files.  The
legacy :func:`prepare_record` helper is retained only for old regression tests;
it is not the authoritative SFT pipeline.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path
from typing import Any

try:
    from .travel_canonical import (
        CanonicalError,
        canonical_hash,
        canonicalize_record,
        render_legacy_record,
        validate_canonical,
    )
    from .clean_travel_trajectories import TaskSpec, clean_trajectory, is_sft_eligible
except ImportError:  # direct module loading or ``python sft/prepare_travel_sft.py``
    try:
        from sft.travel_canonical import (
            CanonicalError,
            canonical_hash,
            canonicalize_record,
            render_legacy_record,
            validate_canonical,
        )
        from sft.clean_travel_trajectories import TaskSpec, clean_trajectory, is_sft_eligible
    except ImportError:
        from travel_canonical import (
            CanonicalError,
            canonical_hash,
            canonicalize_record,
            render_legacy_record,
            validate_canonical,
        )
        from clean_travel_trajectories import TaskSpec, clean_trajectory, is_sft_eligible


PRIVATE_ID = re.compile(r"\bP\d+\b", re.IGNORECASE)
REWARD_LINE = re.compile(
    r"^\s*(?:reward|step[_ ]?reward|terminal[_ ]?reward)\s*[:=].*$",
    re.IGNORECASE | re.MULTILINE,
)
OPTION_ID = re.compile(r"(?<![A-Za-z0-9])([ACFHR]\d+)(?![A-Za-z0-9])", re.IGNORECASE)
ASPECT_BY_PREFIX = {"F": "flight", "H": "hotel", "A": "apartment", "C": "rental_car", "R": "restaurant"}
ASPECT_HINTS = {
    "flight": ("flight", "airline", "carrier"),
    "hotel": ("hotel",),
    "apartment": ("apartment",),
    "rental_car": ("rental car", "car rental", "rental vehicle"),
    "restaurant": ("restaurant", "dining"),
}
PUBLIC_TRAVEL_POLICY = (
    "\n\nTravelGym public protocol: Search returns the complete candidate list for "
    "the current aspect. Action only asks for or records natural-language "
    "preference evidence; it never filters or shrinks that list. The Actor "
    "must compare the evidence implicitly and submit exactly one option ID "
    "that appeared in Search."
)
PUBLIC_TOOL_SCHEMA = json.dumps(
    [
        {
            "name": "interact_with_env",
            "description": (
                "TravelGym public protocol: Search returns the complete candidate list; "
                "Action gathers natural-language evidence without filtering; Answer submits "
                "exactly one option ID visible in the current Search result."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "choice": {
                        "type": "string",
                        "enum": ["search", "action", "answer"],
                    },
                    "content": {
                        "type": "string",
                        "description": (
                            "Search query, natural-language evidence question, "
                            "or exactly one visible option ID."
                        ),
                    },
                },
                "required": ["choice", "content"],
            },
        }
    ],
    ensure_ascii=False,
)


def _public_text(value: Any) -> Any:
    if isinstance(value, str):
        text = PRIVATE_ID.sub("that preference", value)
        text = REWARD_LINE.sub("", text)
        return text.strip()
    if isinstance(value, list):
        return [_public_text(item) for item in value]
    if isinstance(value, dict):
        return {key: _public_text(item) for key, item in value.items()}
    return value


def _role(message: dict[str, Any]) -> str:
    role = str(message.get("role", message.get("from", ""))).casefold()
    return {"human": "user", "gpt": "assistant", "observation": "tool"}.get(role, role)


def _choice_from_message(message: dict[str, Any]) -> str | None:
    arguments = _tool_arguments(message)
    choice = arguments.get("choice") if isinstance(arguments, dict) else None
    if isinstance(choice, str) and choice.casefold() in {"search", "action", "answer"}:
        return choice.casefold()
    # Legacy ShareGPT rows wrap the JSON tool call in model reasoning and a
    # ``<tool_call>`` marker, so the complete message is not valid JSON.
    text = message.get("value") or message.get("content")
    if isinstance(text, str):
        match = re.search(r'"choice"\s*:\s*"(search|action|answer)"', text, re.IGNORECASE)
        if match:
            return match.group(1).casefold()
    return None


def _tool_arguments(message: dict[str, Any]) -> dict[str, Any]:
    """Read tool arguments from ShareGPT strings or evaluator role messages."""
    value = message.get("value") if "value" in message else message.get("content")
    if isinstance(value, dict):
        arguments = value.get("arguments", value)
        return arguments if isinstance(arguments, dict) else {}
    if not isinstance(value, str):
        return {}
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return {}
    if not isinstance(parsed, dict):
        return {}
    arguments = parsed.get("arguments", parsed)
    return arguments if isinstance(arguments, dict) else {}


def _tool_content(message: dict[str, Any]) -> str:
    """Read the actual tool argument instead of model-thought mentions."""
    arguments = _tool_arguments(message)
    if "content" in arguments:
        return str(arguments["content"])
    text = message.get("value") or message.get("content")
    if not isinstance(text, str):
        return ""
    match = re.search(r'"content"\s*:\s*"((?:\\.|[^"\\])*)"', text, re.IGNORECASE)
    if not match:
        return text
    try:
        return json.loads('"' + match.group(1) + '"')
    except (TypeError, ValueError):
        return match.group(1)


def _aspect_from_text(choice: str | None, text: str) -> str | None:
    normalized = str(text).casefold()
    if choice == "answer":
        match = OPTION_ID.search(text)
        if match:
            return ASPECT_BY_PREFIX.get(match.group(1)[0].upper())
    for aspect, hints in ASPECT_HINTS.items():
        if any(hint in normalized for hint in hints):
            return aspect
    return None


def _public_tools(value: Any) -> Any:
    """Update a serialized interact tool schema to the public contract."""
    if not isinstance(value, str):
        return value
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return _public_text(value)
    if isinstance(parsed, dict):
        tools = [parsed]
    elif isinstance(parsed, list):
        tools = parsed
    else:
        return value
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("name") != "interact_with_env":
            continue
        tool["description"] = (
            "TravelGym public protocol: Search returns the complete candidate list; "
            "Action gathers natural-language evidence without filtering; Answer submits "
            "exactly one option ID visible in the current Search result."
        )
        parameters = tool.setdefault("parameters", {})
        properties = parameters.setdefault("properties", {})
        if isinstance(properties.get("choice"), dict):
            properties["choice"]["description"] = "Use search, action, or answer in the public protocol."
        if isinstance(properties.get("content"), dict):
            properties["content"]["description"] = "Search query, natural-language evidence question, or exactly one visible option ID."
    output = tools[0] if isinstance(parsed, dict) else tools
    return json.dumps(output, ensure_ascii=False)


def _has_think_block(value: Any) -> bool:
    return isinstance(value, str) and bool(
        re.search(r"<think>\s*.*?\s*</think>", value, flags=re.IGNORECASE | re.DOTALL)
    )


def _record_has_think(record: dict[str, Any]) -> bool:
    """Require a think block on every serialized assistant tool turn."""
    conversations = record.get("conversations", record.get("messages", []))
    assistant_turns = []
    for message in conversations if isinstance(conversations, list) else []:
        if not isinstance(message, dict) or _role(message) != "assistant":
            continue
        if _choice_from_message(message) is None:
            continue
        value = message.get("value", message.get("content", ""))
        assistant_turns.append(value)
    return bool(assistant_turns) and all(_has_think_block(value) for value in assistant_turns)


def prepare_record(record: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(record)
    system = str(result.get("system", ""))
    messages = result.get("conversations", result.get("messages", []))
    is_travel = "TravelGym" in system or "TravelGym" in json.dumps(result, ensure_ascii=False)
    if not is_travel or not isinstance(messages, list):
        return result
    if isinstance(result.get("system"), str):
        result["system"] = _public_text(result["system"])
        if "TravelGym public protocol:" not in result["system"]:
            result["system"] += PUBLIC_TRAVEL_POLICY
    searched_aspects: set[str] = set()
    answered_aspects: set[str] = set()
    # Mirror the public reducer's single-current-aspect state.  The previous
    # cleaner only tracked the set of searches, which could preserve an
    # invalid cross-aspect Search before the first Answer.
    current_aspect: str | None = None
    unknown_search_seen = False
    cleaned = []
    keep_observations = False
    previous_choice: str | None = None
    for message in messages:
        if not isinstance(message, dict):
            continue
        item = copy.deepcopy(message)
        field = "value" if "value" in item else "content"
        item[field] = _public_text(item.get(field, ""))
        if item.get("role") == "system" and isinstance(item.get(field), str) and "TravelGym public protocol:" not in item[field]:
            item[field] += PUBLIC_TRAVEL_POLICY
        role = _role(item)
        choice = _choice_from_message(item) if role == "assistant" else None
        if role == "assistant":
            aspect = _aspect_from_text(choice, _tool_content(item))
            keep = choice in {"search", "action", "answer"}
            if choice == "search":
                if aspect is None:
                    keep = not unknown_search_seen and current_aspect is None
                    if keep:
                        unknown_search_seen = True
                        current_aspect = "__unknown__"
                        searched_aspects.add(current_aspect)
                else:
                    if current_aspect is None:
                        current_aspect = aspect
                    keep = aspect == current_aspect and aspect not in searched_aspects
                    if keep:
                        searched_aspects.add(aspect)
            elif choice == "action":
                # Action is evidence gathering only and is legal only after
                # at least one Search.  If an aspect is explicit, require its
                # own Search result to be present.
                keep = (
                    current_aspect is not None
                    and current_aspect in searched_aspects
                    and current_aspect not in answered_aspects
                    and (aspect is None or aspect == current_aspect)
                )
            elif choice == "answer":
                # Answers must be a single visible-looking option ID.  The
                # actual visibility is checked by the environment; SFT only
                # enforces the public protocol shape and current aspect.
                body = _tool_content(item).strip()
                option_ids = [value.upper() for value in OPTION_ID.findall(body)]
                exact_single_id = len(option_ids) == 1 and body.upper() == option_ids[0]
                if current_aspect == "__unknown__" and aspect is not None:
                    current_aspect = aspect
                    searched_aspects.add(aspect)
                keep = (
                    exact_single_id
                    and aspect is not None
                    and current_aspect == aspect
                    and aspect in searched_aspects
                    and aspect not in answered_aspects
                )
                if keep:
                    answered_aspects.add(aspect)
                    current_aspect = None
            # A skipped assistant invalidates all following observations until
            # the next kept assistant call; this removes stale duplicate-search
            # and hidden answer-quality feedback from old transcripts.
            keep_observations = keep
            previous_choice = choice if keep else None
            if keep:
                cleaned.append(item)
            continue
        if role in {"tool", "observation"}:
            if not keep_observations:
                continue
            if previous_choice == "answer":
                item[field] = "Your answer was recorded."
            cleaned.append(item)
            continue
        # Keep the initial system/user context and any other public context.
        cleaned.append(item)
    if isinstance(result.get("tools"), str):
        result["tools"] = _public_tools(result["tools"])
    elif "tools" not in result:
        result["tools"] = PUBLIC_TOOL_SCHEMA
    if "conversations" in result:
        result["conversations"] = cleaned
    else:
        result["messages"] = cleaned
    for private_field in ("reward_model", "gold", "env_name", "reward", "reward_report"):
        result.pop(private_field, None)
    return result


def _is_travel_record(record: dict[str, Any]) -> bool:
    """Recognize both legacy ShareGPT rows and evaluator cache records."""
    if record.get("schema_version") == "travelgym-canonical-v1":
        return True
    if str(record.get("env_name", "")).casefold() == "travelgym":
        return True
    serialized = json.dumps(record, ensure_ascii=False)
    if "TravelGym" in serialized or "interact_with_env" in serialized:
        return True
    return False


def _iter_source_records(source: Any):
    """Yield ``(record, terminal_report)`` from lists or evaluator caches."""
    if isinstance(source, list):
        for record in source:
            if isinstance(record, dict):
                yield record, None
        return
    if not isinstance(source, dict):
        raise ValueError("SFT input must be a JSON list or evaluator cache object")
    if isinstance(source.get("data"), list):
        for record in source["data"]:
            if isinstance(record, dict):
                yield record, None
        return
    # Evaluator caches are keyed by environment and task hash.  Manifests are
    # metadata, not training records.
    for env_name, task_entries in source.items():
        if str(env_name).startswith("_") or not isinstance(task_entries, dict):
            continue
        for entry in task_entries.values():
            if not isinstance(entry, dict):
                continue
            records = entry.get("data", [])
            reports = entry.get("reward_report", [])
            if not isinstance(records, list):
                records = [records]
            if not isinstance(reports, list):
                reports = [reports]
            for index, record in enumerate(records):
                report = reports[index] if index < len(reports) else None
                if isinstance(record, dict):
                    yield record, report


def _report_is_trainable(report: Any) -> bool:
    if not isinstance(report, dict):
        return True
    return bool(report.get("reward_valid_for_training", report.get("reward_valid", True)))


def prepare_canonical_record(
    record: dict[str, Any],
    *,
    task: dict[str, Any] | TaskSpec | None = None,
    task_id: str | None = None,
    reward_report: dict[str, Any] | None = None,
    source: str = "unknown",
    max_length: int = 32768,
    require_think: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Canonicalize and replay one record for the new SFT pipeline.

    ``prepare_record`` below remains a legacy ShareGPT projection for old
    callers/tests.  New merge/build commands must call this function so the
    canonical mask and post-clean metrics are authoritative.
    """

    return clean_trajectory(
        record,
        task=task,
        task_id=task_id,
        reward_report=reward_report,
        source=source,
        max_length=max_length,
        require_think=require_think,
    )


def canonical_to_legacy(record: dict[str, Any]) -> dict[str, Any]:
    """Render canonical data only for regression comparisons."""

    return render_legacy_record(record)


def _to_sharegpt_record(record: dict[str, Any]) -> dict[str, Any]:
    """Normalize evaluator role messages to the ShareGPT SFT schema."""
    if "conversations" in record:
        return record
    messages = record.pop("messages", [])
    conversations = []
    system = str(record.get("system", ""))
    for message in messages if isinstance(messages, list) else []:
        if not isinstance(message, dict):
            continue
        role = _role(message)
        value = message.get("content", message.get("value", ""))
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)
        value = str(value)
        if role == "system":
            system = value
        elif role in {"user", "assistant", "tool"}:
            conversations.append(
                {
                    "from": "human" if role == "user" else "gpt" if role == "assistant" else "observation",
                    "value": value,
                }
            )
    record["system"] = system
    record["conversations"] = conversations
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="One or more ShareGPT JSON/JSONL files or evaluator reward caches.",
    )
    parser.add_argument(
        "--output",
        default="sft/travel_sft_prepared.jsonl",
        help="New canonical JSONL output (never overwrite the source corpus).",
    )
    parser.add_argument("--audit-output", default=None, help="Private audit sidecar (default: <output>.audit.json).")
    parser.add_argument("--manifest-output", default=None, help="Manifest JSON (default: <output>.manifest.json).")
    parser.add_argument("--parquet-output", default=None, help="Optional canonical Parquet mirror (requires pandas/pyarrow).")
    parser.add_argument("--legacy-output", default=None, help="Optional old ShareGPT renderer for regression only.")
    parser.add_argument("--task-data", nargs="*", default=None, help="Optional TravelGym task JSON files for ID recovery.")
    parser.add_argument("--task-map", default=None, help="Reviewed JSON sidecar mapping source index/path#source_index to task ID.")
    parser.add_argument("--tokenizer", default=None, help="Optional Qwen3.5 tokenizer used for exact length audit.")
    parser.add_argument("--max-length", type=int, default=32768)
    think_group = parser.add_mutually_exclusive_group()
    think_group.add_argument(
        "--require-think",
        dest="require_think",
        action="store_true",
        help="Quarantine assistant tool turns without reasoning_content (default).",
    )
    think_group.add_argument(
        "--allow-missing-think",
        dest="require_think",
        action="store_false",
        help="Diagnostic-only: allow missing reasoning_content; do not use for SFT.",
    )
    parser.set_defaults(require_think=True)
    args = parser.parse_args()

    output_path = Path(args.output).resolve()
    input_paths = [Path(value).resolve() for value in args.input]
    # The historical 244-record file is immutable input.  Refuse an accidental
    # in-place conversion even when the caller passes a relative path.
    protected = (Path(__file__).resolve().parent / "travel_sft_public.json").resolve()
    if output_path == protected or output_path in set(input_paths):
        raise ValueError("refusing to overwrite the source SFT corpus; choose a derived output path")
    task_paths = [Path(value) for value in args.task_data] if args.task_data else None
    # Import lazily so legacy helper functions remain usable in lightweight
    # environments and to avoid a module cycle during direct script execution.
    try:
        from .merge_travel_sft import (
            _atomic_write,
            _write_jsonl,
            _write_parquet,
            build_manifest,
            prepare_canonical_inputs,
        )
        from .travel_task_resolver import TravelTaskResolver
    except ImportError:
        from sft.merge_travel_sft import (
            _atomic_write,
            _write_jsonl,
            _write_parquet,
            build_manifest,
            prepare_canonical_inputs,
        )
        from sft.travel_task_resolver import TravelTaskResolver

    resolver = TravelTaskResolver(
        project_root=Path(__file__).resolve().parents[1],
        task_paths=task_paths,
        explicit_map=TravelTaskResolver.load_alignment_map(args.task_map),
    )
    records, audits, stats = prepare_canonical_inputs(
        input_paths,
        resolver=resolver,
        max_length=args.max_length,
        require_think=bool(args.require_think),
    )
    # This command prepares/cleans inputs but does not silently deduplicate
    # distinct records.  Exact canonical duplicates are handled consistently
    # with the merge command and counted in the manifest; audits follow the
    # kept rows one-for-one.
    deduped_records: list[dict[str, Any]] = []
    deduped_audits: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    duplicate_count = 0
    for record, audit in zip(records, audits):
        key = canonical_hash(record)
        if key in seen_hashes:
            duplicate_count += 1
            continue
        seen_hashes.add(key)
        deduped_records.append(record)
        deduped_audits.append(audit)
    records, audits = deduped_records, deduped_audits
    audit_path = Path(args.audit_output).resolve() if args.audit_output else output_path.with_suffix(".audit.json")
    manifest_path = Path(args.manifest_output).resolve() if args.manifest_output else output_path.with_suffix(".manifest.json")
    source_paths = {protected, *input_paths}
    for label, candidate in (
        ("audit", audit_path),
        ("manifest", manifest_path),
        ("Parquet", Path(args.parquet_output).resolve() if args.parquet_output else None),
        ("legacy", Path(args.legacy_output).resolve() if args.legacy_output else None),
    ):
        if candidate is not None and candidate.resolve() in source_paths:
            raise ValueError(f"refusing to overwrite the source SFT corpus via {label} output")
    manifest = build_manifest(
        records,
        audits,
        tokenizer_name=args.tokenizer,
        max_length=args.max_length,
        duplicate_count=duplicate_count,
    )
    _atomic_write(audit_path, {"schema_version": "travelgym-canonical-v1", "records": audits})
    _atomic_write(manifest_path, manifest)
    _write_jsonl(output_path, records)
    if args.parquet_output:
        _write_parquet(Path(args.parquet_output).resolve(), records)
    if args.legacy_output:
        _atomic_write(Path(args.legacy_output).resolve(), [render_legacy_record(record) for record in records])
    for category in ("strict_gold", "recoverable_correct", "partial_correct", "totally_wrong", "infrastructure_invalid", "overlength_quarantine"):
        _write_jsonl(
            output_path.with_name(f"{output_path.stem}.{category}.jsonl"),
            [record for record in records if record.get("trainer_metadata", {}).get("trajectory_class") == category],
        )
    print(
        f"source={stats.get('source_records', 0)} merged={len(records)} duplicates={duplicate_count} "
        f"strict={stats.get('strict_gold', 0)} recoverable={stats.get('recoverable_correct', 0)} "
        f"partial={stats.get('partial_correct', 0)} wrong={stats.get('totally_wrong', 0)} "
        f"infra={stats.get('infrastructure_invalid', 0)} output={output_path}"
    )


if __name__ == "__main__":
    main()
