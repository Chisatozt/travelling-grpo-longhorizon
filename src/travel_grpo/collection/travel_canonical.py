"""Model-independent TravelGym trajectory representation.

This module is deliberately independent from a model's chat template.  The
training and serving layers can therefore use the native Qwen3.5 template
without making the old ShareGPT representation part of the environment
protocol.

The public transcript contains only the six public roles and the public
``interact_with_env`` contract.  Reward/task labels are accepted by the
conversion helpers as *sidecar* values and are never copied into messages.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, MutableMapping, Sequence


SCHEMA_VERSION = "travelgym-canonical-v1"
TOOL_NAME = "interact_with_env"
ENABLE_THINKING = True

_PRIVATE_ID = re.compile(r"\bP\d+\b", re.IGNORECASE)
_REWARD_LINE = re.compile(
    r"^\s*(?:reward|step[_ ]?reward|terminal[_ ]?reward)\s*[:=].*$",
    re.IGNORECASE | re.MULTILINE,
)
_PRIVATE_TEXT_LINE = re.compile(
    r"^\s*(?:preference[_ ]?ids?|correct[_ ]?ids?|best[_ ]?ids?|"
    r"ground[_ ]?truth|reward[_ ]?report|diagnostics?)\s*[:=].*$",
    re.IGNORECASE | re.MULTILINE,
)
_THINK_RE = re.compile(r"<think>(?P<body>.*?)</think>", re.IGNORECASE | re.DOTALL)
_TOOL_RE = re.compile(r"<tool_call>(?P<body>.*?)</tool_call>", re.IGNORECASE | re.DOTALL)
_OPTION_ID = re.compile(r"(?<![A-Za-z0-9])([ACFHR]\d+)(?![A-Za-z0-9])", re.IGNORECASE)
_PRIVATE_KEYS = {
    "preference_id",
    "preference_ids",
    "correct_id",
    "correct_ids",
    "best_id",
    "best_ids",
    "reward",
    "step_reward",
    "terminal_reward",
    "reward_report",
    "ground_truth",
}


class CanonicalError(ValueError):
    """An input cannot be represented without losing protocol information."""


def public_text(value: Any) -> Any:
    """Preserve the existing public scrubber without making it a drop rule.

    Hidden strings in old data are scrubbed from the public transcript, but a
    trajectory is *not* deleted merely because such a string was present.
    """

    if isinstance(value, str):
        text = _REWARD_LINE.sub("", value)
        text = _PRIVATE_TEXT_LINE.sub("", text)
        return _PRIVATE_ID.sub("that preference", text).strip()
    if isinstance(value, list):
        return [public_text(item) for item in value]
    if isinstance(value, dict):
        return {
            key: public_text(item)
            for key, item in value.items()
            if str(key).casefold() not in _PRIVATE_KEYS
        }
    return value


def _json_loads(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str):
        raise CanonicalError("tool_call_json_invalid")
    try:
        return json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise CanonicalError("tool_call_json_invalid") from exc


def _arguments(value: Any) -> dict[str, Any]:
    parsed = _json_loads(value)
    if isinstance(parsed, dict) and isinstance(parsed.get("arguments"), (dict, str)):
        parsed = parsed["arguments"]
        if isinstance(parsed, str):
            parsed = _json_loads(parsed)
    if not isinstance(parsed, dict):
        raise CanonicalError("tool_call_arguments_not_object")
    return copy.deepcopy(parsed)


def normalize_arguments(arguments: Any) -> dict[str, Any]:
    """Normalize one tool argument payload at the internal boundary.

    ``function.arguments`` is a dict in canonical records.  API adapters may
    call :func:`arguments_for_api` immediately before sending it to an
    OpenAI-compatible endpoint.
    """

    parsed = _arguments(arguments)
    choice = parsed.get("choice")
    content = parsed.get("content")
    if not isinstance(choice, str) or choice.casefold() not in {"search", "action", "answer"}:
        raise CanonicalError("invalid_choice")
    if not isinstance(content, str):
        # The environment accepts a string only.  Do not silently stringify
        # dictionaries because that would change the action semantics.
        raise CanonicalError("invalid_content")
    return {"choice": choice.casefold(), "content": content}


def arguments_for_api(arguments: Mapping[str, Any]) -> str:
    """Serialize canonical arguments at an OpenAI-compatible API boundary."""

    return json.dumps(normalize_arguments(dict(arguments)), ensure_ascii=False, separators=(",", ":"))


def _tool_schema_from_any(value: Any) -> list[dict[str, Any]]:
    """Return the one canonical OpenAI function schema."""

    description = (
        "TravelGym public protocol: Search returns the complete candidate list; "
        "Action gathers natural-language evidence without filtering; Answer "
        "submits exactly one option ID visible in the current Search result."
    )
    schema = {
        "type": "function",
        "function": {
            "name": TOOL_NAME,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "choice": {
                        "type": "string",
                        "enum": ["search", "action", "answer"],
                        "description": "Use search, action, or answer in the public protocol.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Search query, natural-language evidence question, or exactly one visible option ID.",
                    },
                },
                "required": ["choice", "content"],
                "additionalProperties": False,
            },
        },
    }
    # The schema is fixed intentionally.  User-provided descriptions are not
    # allowed to introduce hidden task/reward fields into Actor context.
    _ = value
    return [schema]


def canonical_tools_schema() -> list[dict[str, Any]]:
    """Return a fresh copy of the single public TravelGym tool schema.

    SFT rows, evaluator requests and SGLang tool configuration should all be
    byte-equivalent after JSON normalisation.  Keeping this constructor
    public also gives tests and conversion utilities one source of truth
    without importing a model-specific chat template.
    """

    return copy.deepcopy(_tool_schema_from_any(None))


def _legacy_tools_schema() -> str:
    """Serialize the historical ShareGPT *bare* function schema.

    The canonical schema is OpenAI-shaped (``type=function`` plus a nested
    ``function`` object), whereas the 244-record file stores a JSON string of
    bare functions.  Keep that distinction confined to the regression
    renderer; training and serving consume :func:`canonical_tools_schema`.
    """

    function = copy.deepcopy(canonical_tools_schema()[0]["function"])
    # Preserve the historical enum order so renderer comparisons are stable.
    properties = function.get("parameters", {}).get("properties", {})
    if isinstance(properties.get("choice"), Mapping):
        properties["choice"]["enum"] = ["action", "answer", "search"]
    return json.dumps([function], ensure_ascii=False)


def _message_role(message: Mapping[str, Any]) -> str:
    role = str(message.get("role", message.get("from", ""))).casefold()
    return {"human": "user", "gpt": "assistant", "observation": "tool"}.get(role, role)


def _message_value(message: Mapping[str, Any]) -> Any:
    return message.get("content", message.get("value", ""))


def _extract_legacy_call(text: str) -> tuple[str, dict[str, Any], bool]:
    """Extract ``<think>`` and legacy ``<tool_call>`` payloads.

    The bool indicates whether a think block was present.  Unbalanced markers
    and malformed JSON are explicit canonical errors so the cleaner can
    truncate the complete suffix from that Assistant turn.
    """

    if "<think>" in text.lower() and not _THINK_RE.search(text):
        raise CanonicalError("assistant_think_truncated")
    if "<tool_call>" in text.lower() and not _TOOL_RE.search(text):
        raise CanonicalError("assistant_tool_call_truncated")
    think_match = _THINK_RE.search(text)
    reasoning = think_match.group("body").strip() if think_match else ""
    tool_match = _TOOL_RE.search(text)
    if tool_match:
        payload = _json_loads(tool_match.group("body").strip())
        return reasoning, payload, bool(think_match)
    # A few evaluator versions emitted a bare JSON object after the reasoning.
    remainder = _THINK_RE.sub("", text).strip()
    if remainder.startswith("{"):
        return reasoning, _json_loads(remainder), bool(think_match)
    raise CanonicalError("assistant_tool_call_missing")


def _strip_think_wrapper(value: Any) -> str:
    """Keep reasoning as an independent field without nested Qwen markers."""

    text = str(value or "").strip()
    match = _THINK_RE.fullmatch(text)
    return match.group("body").strip() if match else text


def _normalize_call(call: Any, index: int) -> dict[str, Any]:
    if not isinstance(call, Mapping):
        raise CanonicalError("tool_call_not_object")
    function = call.get("function") if isinstance(call.get("function"), Mapping) else call
    if not isinstance(function, Mapping):
        raise CanonicalError("tool_call_function_not_object")
    name = str(function.get("name", call.get("name", TOOL_NAME)))
    if name != TOOL_NAME:
        raise CanonicalError("unknown_tool")
    raw_args = function.get("arguments", call.get("arguments"))
    # Canonicalisation must distinguish malformed JSON (unrecoverable) from a
    # syntactically valid but semantically invalid public call (for example an
    # empty ``content`` or an unknown ``choice``).  Keep the latter as a
    # dict-valued call so the reducer can pair it with the environment's
    # explicit rejection and train a later repair.  The strict API adapter
    # still calls ``normalize_arguments`` immediately before execution.
    # Keep the canonical public call deliberately narrow.  A malformed
    # ``choice``/``content`` is still retained so the offline reducer can
    # classify an explicit public rejection as recoverable, but arbitrary
    # fields (including legacy preference/reward labels) never reach the
    # Actor-facing transcript.
    raw_arguments = public_text(_arguments(raw_args))
    args = {
        key: raw_arguments[key]
        for key in ("choice", "content")
        if key in raw_arguments
    }
    call_id = call.get("id") or function.get("id") or f"call_{index:04d}"
    return {
        "id": str(call_id),
        "type": "function",
        "function": {"name": TOOL_NAME, "arguments": args},
    }


def _assistant_to_canonical(message: Mapping[str, Any], index: int) -> tuple[dict[str, Any], list[str]]:
    """Normalize one assistant message and return canonical errors separately."""

    errors: list[str] = []
    supplied_errors = message.get("_canonical_errors", [])
    if isinstance(supplied_errors, str):
        supplied_errors = [supplied_errors]
    if isinstance(supplied_errors, Sequence):
        errors.extend(str(code) for code in supplied_errors)
    value = _message_value(message)
    reasoning = message.get("reasoning_content")
    if reasoning is None:
        reasoning = ""
    if not isinstance(reasoning, str):
        reasoning = str(reasoning)
    reasoning = _strip_think_wrapper(reasoning)
    calls: Any = message.get("tool_calls")
    if calls is None and isinstance(value, Mapping):
        # DeepSeek cache format: assistant.content = {name, arguments}.
        calls = [value]
    if calls is None and isinstance(value, list):
        calls = value
    if calls is None and isinstance(value, str):
        try:
            legacy_reasoning, payload, had_think = _extract_legacy_call(value)
            reasoning = reasoning or legacy_reasoning
            reasoning = _strip_think_wrapper(reasoning)
            calls = [payload]
            if had_think and not reasoning:
                reasoning = legacy_reasoning
        except CanonicalError as exc:
            errors.append(str(exc))
            calls = []
    if calls is None:
        # Plain assistant text is allowed as context, but TravelGym training
        # targets are tool turns.  Keep it loss-masked by the cleaner.
        calls = []
        if isinstance(value, str):
            content = public_text(value)
        elif value is None:
            content = ""
        else:
            content = public_text(str(value))
    else:
        content = ""
    canonical_calls: list[dict[str, Any]] = []
    for call_offset, call in enumerate(calls if isinstance(calls, list) else [calls]):
        try:
            canonical_calls.append(_normalize_call(call, index * 100 + call_offset))
        except CanonicalError as exc:
            errors.append(str(exc))
    result: dict[str, Any] = {
        "role": "assistant",
        "content": public_text(content) if isinstance(content, str) else "",
        "reasoning_content": public_text(reasoning) if reasoning else "",
    }
    if canonical_calls:
        result["tool_calls"] = canonical_calls
    if message.get("finish_reason") == "length" or message.get("truncated") is True:
        errors.append("assistant_content_truncated")
    return result, errors


def _tool_to_canonical(message: Mapping[str, Any], pending_ids: Sequence[str]) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    supplied_errors = message.get("_canonical_errors", [])
    if isinstance(supplied_errors, str):
        supplied_errors = [supplied_errors]
    if isinstance(supplied_errors, Sequence):
        errors.extend(str(code) for code in supplied_errors)
    content = _message_value(message)
    if isinstance(content, (dict, list)):
        # Search/API adapters occasionally return a structured diagnostic
        # object.  Scrub private report keys before serialising it as the
        # public tool observation; this is a boundary defence, not a cleaner
        # drop rule.
        content = json.dumps(public_text(content), ensure_ascii=False)
    if content is None:
        content = ""
    if not isinstance(content, str):
        content = str(content)
    explicit_id = message.get("tool_call_id")
    if explicit_id is None and len(pending_ids) == 1:
        call_id = pending_ids[0]
    elif explicit_id is not None:
        call_id = str(explicit_id)
        if call_id not in pending_ids:
            errors.append("tool_call_id_mismatch")
    else:
        errors.append("tool_observation_misaligned")
        call_id = ""
    result: dict[str, Any] = {
        "role": "tool",
        "tool_call_id": call_id,
        "name": str(message.get("name", TOOL_NAME)),
        "content": public_text(content),
    }
    return result, errors


def canonicalize_messages(messages: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Convert legacy/OpenAI/DeepSeek messages.

    Returns ``(messages, errors)``.  Errors carry the message index and are
    intentionally kept outside the public transcript so the trajectory
    cleaner can apply the suffix rule deterministically.
    """

    output: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    pending_ids: list[str] = []
    pending_origin_index: int | None = None
    for index, raw in enumerate(messages):
        if not isinstance(raw, Mapping):
            errors.append({"index": index, "code": "message_not_object"})
            continue
        role = _message_role(raw)
        if role in {"system", "user"}:
            if pending_ids:
                errors.append({"index": pending_origin_index if pending_origin_index is not None else max(0, len(output) - 1), "source_index": index, "code": "tool_observation_misaligned"})
            content = _message_value(raw)
            if isinstance(content, (dict, list)):
                content = json.dumps(content, ensure_ascii=False)
            item = {"role": role, "content": public_text("" if content is None else str(content))}
            output.append(item)
            pending_ids = []
            pending_origin_index = None
        elif role == "assistant":
            if pending_ids:
                errors.append({"index": pending_origin_index if pending_origin_index is not None else max(0, len(output) - 1), "source_index": index, "code": "tool_observation_misaligned"})
            item, item_errors = _assistant_to_canonical(raw, index)
            output.append(item)
            pending_ids = [call["id"] for call in item.get("tool_calls", [])]
            pending_origin_index = len(output) - 1 if pending_ids else None
            for code in item_errors:
                errors.append({"index": len(output) - 1, "source_index": index, "code": code})
        elif role == "tool":
            item, item_errors = _tool_to_canonical(raw, pending_ids)
            output.append(item)
            for code in item_errors:
                errors.append({"index": len(output) - 1, "source_index": index, "code": code})
            # Consume exactly one pending call ID.  A missing result or a
            # duplicate result is therefore caught deterministically instead
            # of leaving an apparently valid but misaligned transcript.
            call_id = item.get("tool_call_id")
            if call_id in pending_ids and not item_errors:
                pending_ids.remove(call_id)
            if not pending_ids:
                pending_origin_index = None
        else:
            # Unknown role is an unrecoverable protocol issue, but retain a
            # neutral user context so audit output remains inspectable.
            errors.append({"index": index, "code": "unknown_role"})
    if pending_ids:
        errors.append({
            "index": pending_origin_index if pending_origin_index is not None else max(0, len(output) - 1),
            "code": "tool_observation_misaligned",
        })
    return output, errors


def _legacy_assistant(message: Mapping[str, Any]) -> str:
    reasoning = str(message.get("reasoning_content") or "").strip()
    calls = message.get("tool_calls") or []
    if not calls:
        return (f"<think>{reasoning}</think>\n\n" if reasoning else "") + str(message.get("content") or "")
    call = calls[0]
    function = call.get("function", call)
    payload = {
        "name": function.get("name", TOOL_NAME),
        "arguments": function.get("arguments", {}),
    }
    # Legacy format intentionally serializes dict arguments as JSON; this is a
    # renderer only and is never fed to the native Qwen3.5 template.
    return (
        (f"<think>{reasoning}</think>\n\n" if reasoning else "")
        + "<tool_call>\n"
        + json.dumps(payload, ensure_ascii=False)
        + "\n</tool_call>"
    )


def render_legacy_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Render canonical data to the old 244-record ShareGPT shape."""

    result: dict[str, Any] = {
        "system": "",
        "conversations": [],
        "tools": _legacy_tools_schema(),
    }
    for message in record.get("messages", []):
        role = message.get("role")
        if role == "system":
            result["system"] = message.get("content", "")
        elif role in {"user", "assistant", "tool"}:
            result["conversations"].append(
                {
                    "from": "human" if role == "user" else "gpt" if role == "assistant" else "observation",
                    "value": _legacy_assistant(message) if role == "assistant" else str(message.get("content", "")),
                }
            )
    return result


def canonicalize_record(record: Mapping[str, Any], *, source: str = "unknown") -> dict[str, Any]:
    """Canonicalize one complete record without copying private labels."""

    if not isinstance(record, Mapping):
        raise CanonicalError("record_not_object")
    raw_messages = record.get("messages", record.get("conversations", []))
    if not isinstance(raw_messages, Sequence) or isinstance(raw_messages, (str, bytes)):
        raise CanonicalError("messages_not_list")
    # Legacy ShareGPT stores the system prompt beside ``conversations``.
    # Promote it to a standard system message before any pairing/replay.
    raw_messages = list(raw_messages)
    if isinstance(record.get("system"), str) and not any(
        isinstance(message, Mapping) and _message_role(message) == "system" for message in raw_messages
    ):
        raw_messages.insert(0, {"role": "system", "content": record["system"]})
    messages, errors = canonicalize_messages(raw_messages)
    # ``_canonical_errors`` can be attached to individual messages by an
    # upstream converter.  Hoist those audit markers to the top-level sidecar
    # and strip every private/underscore key before the record is exposed to a
    # tokenizer or an Actor.  The reducer still sees the top-level indices and
    # can therefore apply the whole-suffix fatal rule deterministically.
    public_messages: list[dict[str, Any]] = []
    for message_index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            continue
        raw_errors = message.get("_canonical_errors", [])
        if isinstance(raw_errors, str):
            raw_errors = [raw_errors]
        if isinstance(raw_errors, Sequence):
            for code in raw_errors:
                errors.append({"index": message_index, "code": str(code)})
        public_messages.append(
            {key: value for key, value in message.items() if not str(key).startswith("_")}
        )
    messages = public_messages
    # Preserve externally supplied audit errors when a canonical record is
    # cleaned again, but never copy the private annotations into messages.
    supplied_errors = record.get("_canonical_errors", [])
    if isinstance(supplied_errors, Mapping):
        supplied_errors = [supplied_errors]
    if isinstance(supplied_errors, Sequence) and not isinstance(supplied_errors, (str, bytes)):
        for error in supplied_errors:
            if isinstance(error, Mapping) and "index" in error and "code" in error:
                errors.append({"index": int(error["index"]), "code": str(error["code"])})
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "messages": messages,
        "tools": canonical_tools_schema(),
        "enable_thinking": True,
        "assistant_train_mask": [0] * len(messages),
        "trainer_metadata": {"source": source},
    }
    if errors:
        result["_canonical_errors"] = errors
    # Preserve an explicit public data-source marker, but never task/reward
    # values.  The caller writes those to a separate audit sidecar.
    if isinstance(record.get("data_source"), str):
        result["trainer_metadata"]["data_source"] = record["data_source"]
    return result


def validate_canonical(record: Mapping[str, Any], *, require_mask: bool = True) -> None:
    """Validate schema and the assistant/tool pairing contract."""

    if record.get("schema_version") != SCHEMA_VERSION:
        raise CanonicalError("schema_version_mismatch")
    messages = record.get("messages")
    if not isinstance(messages, list):
        raise CanonicalError("messages_not_list")
    if require_mask:
        mask = record.get("assistant_train_mask")
        if not isinstance(mask, list) or len(mask) != len(messages):
            raise CanonicalError("assistant_train_mask_length")
    valid_ids: set[str] = set()
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise CanonicalError(f"message_{index}_not_object")
        role = message.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            raise CanonicalError(f"message_{index}_role")
        if role == "assistant":
            calls = message.get("tool_calls") or []
            if not isinstance(calls, list):
                raise CanonicalError(f"message_{index}_tool_calls")
            valid_ids = {str(call.get("id")) for call in calls if isinstance(call, Mapping)}
            for call_index, call in enumerate(calls):
                if not isinstance(call, Mapping):
                    raise CanonicalError(f"message_{index}_tool_call_{call_index}_not_object")
                function = call.get("function")
                if not isinstance(function, Mapping):
                    raise CanonicalError(f"message_{index}_tool_function_{call_index}")
                arguments = function.get("arguments")
                if not isinstance(arguments, Mapping):
                    raise CanonicalError(f"message_{index}_tool_arguments_{call_index}")
                private = {
                    str(key)
                    for key in arguments
                    if str(key).casefold() in _PRIVATE_KEYS
                }
                if private:
                    raise CanonicalError(
                        f"message_{index}_tool_arguments_{call_index}_private_keys"
                    )
        if role == "tool":
            call_id = str(message.get("tool_call_id", ""))
            if not call_id or call_id not in valid_ids:
                raise CanonicalError(f"message_{index}_tool_call_id")
    if not isinstance(record.get("tools"), list) or not record["tools"]:
        raise CanonicalError("tools_schema_missing")


def canonical_hash(record: Mapping[str, Any]) -> str:
    """Stable hash over public canonical content (excluding masks/metadata)."""

    messages = []
    for message in record.get("messages", []):
        if isinstance(message, Mapping):
            # ``_canonical_errors`` is an audit-only annotation.  It must not
            # make two byte-identical public transcripts look different to
            # the exact-content merger.
            messages.append({key: value for key, value in message.items() if not str(key).startswith("_")})
        else:
            messages.append(message)
    payload = {
        "schema_version": record.get("schema_version", SCHEMA_VERSION),
        "messages": messages,
        "tools": record.get("tools", []),
        "enable_thinking": bool(record.get("enable_thinking", True)),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def iter_source_records(source: Any) -> Iterator[tuple[dict[str, Any], Any, dict[str, Any]]]:
    """Yield ``(record, report, private_meta)`` from lists and reward caches."""

    if isinstance(source, list):
        for index, record in enumerate(source):
            if isinstance(record, Mapping):
                yield dict(record), None, {"source_index": index}
        return
    if not isinstance(source, Mapping):
        raise CanonicalError("source_not_object_or_list")
    if isinstance(source.get("data"), list):
        for index, record in enumerate(source["data"]):
            if isinstance(record, Mapping):
                yield dict(record), None, {"source_index": index}
        return
    # Private pass-indexed Teacher cache.  Reasoning_content is intentionally
    # preserved in the trajectory for SFT replay; provenance remains in the
    # audit sidecar and is never copied into canonical messages.
    if source.get("schema_version") == "travelgym-teacher-cache-v1" and isinstance(source.get("records"), Mapping):
        for index, (pass_key, entry) in enumerate(source["records"].items()):
            if not isinstance(entry, Mapping) or entry.get("status") != "success":
                continue
            trajectory = entry.get("trajectory")
            if not isinstance(trajectory, Mapping):
                continue
            provenance = entry.get("provenance", {})
            meta = {
                "source_index": index,
                "task_key": provenance.get("task_key"),
                "env_name": provenance.get("env_name"),
                "gold": provenance.get("task_id"),
                "pass_index": provenance.get("pass_index"),
                "teacher_cache_key": str(pass_key),
            }
            yield dict(trajectory), None, meta
        return
    # Evaluator cache: env -> task -> {gold, data, reward_report}.
    for env_name, task_entries in source.items():
        if str(env_name).startswith("_") or not isinstance(task_entries, Mapping):
            continue
        for task_key, entry in task_entries.items():
            if not isinstance(entry, Mapping):
                continue
            records = entry.get("data", [])
            reports = entry.get("reward_report", [])
            if not isinstance(records, list):
                records = [records]
            if not isinstance(reports, list):
                reports = [reports]
            for index, record in enumerate(records):
                if isinstance(record, Mapping):
                    meta = {
                        "env_name": str(env_name),
                        "task_key": str(task_key),
                        "gold": entry.get("gold"),
                        "source_index": index,
                    }
                    yield dict(record), reports[index] if index < len(reports) else None, meta


def option_ids(text: Any) -> list[str]:
    return [match.upper() for match in _OPTION_ID.findall(str(text or ""))]


def tool_call_arguments(message: Mapping[str, Any]) -> dict[str, Any] | None:
    calls = message.get("tool_calls") or []
    if not calls:
        return None
    call = calls[0]
    if not isinstance(call, Mapping):
        return None
    function = call.get("function", call)
    if not isinstance(function, Mapping):
        return None
    raw_arguments = function.get("arguments", {})
    if isinstance(raw_arguments, Mapping):
        # Preserve parseable semantic errors for the offline reducer.  Do not
        # apply the strict execution validator here: an invalid public call
        # can be a recoverable training context when the tool returned a
        # visible rejection and the Actor subsequently repaired it.
        return dict(raw_arguments)
    try:
        return normalize_arguments(raw_arguments)
    except CanonicalError:
        try:
            parsed = _arguments(raw_arguments)
        except CanonicalError:
            return None
        return dict(parsed)


__all__ = [
    "CanonicalError",
    "SCHEMA_VERSION",
    "TOOL_NAME",
    "ENABLE_THINKING",
    "arguments_for_api",
    "canonical_tools_schema",
    "canonical_hash",
    "canonicalize_messages",
    "canonicalize_record",
    "iter_source_records",
    "normalize_arguments",
    "option_ids",
    "public_text",
    "render_legacy_record",
    "tool_call_arguments",
    "validate_canonical",
]
