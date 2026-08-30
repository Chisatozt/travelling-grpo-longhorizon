"""One parameter adapter shared by Qwen3.5 serving and TravelGym."""

from __future__ import annotations

import json
import re
from typing import Any, Mapping


TOOL_NAME = "interact_with_env"
_PRIVATE_ID = re.compile(r"\bP\d+\b", re.IGNORECASE)
_REWARD_LINE = re.compile(r"^\s*(?:reward|step[_ ]?reward|terminal[_ ]?reward)\s*[:=].*$", re.IGNORECASE | re.MULTILINE)
_PRIVATE_TEXT_LINE = re.compile(
    r"^\s*(?:preference[_ ]?ids?|correct[_ ]?ids?|best[_ ]?ids?|"
    r"ground[_ ]?truth|reward[_ ]?report|diagnostics?)\s*[:=].*$",
    re.IGNORECASE | re.MULTILINE,
)
_PRIVATE_KEYS = {
    "preference_id", "preference_ids", "correct_id", "correct_ids",
    "best_id", "best_ids", "reward", "step_reward", "terminal_reward",
    "reward_report", "ground_truth",
}


class TravelToolAdapterError(ValueError):
    pass


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        value = value.get("arguments", value)
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise TravelToolAdapterError("tool arguments JSON is invalid") from exc
    elif isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise TravelToolAdapterError("tool arguments JSON is invalid") from exc
    if not isinstance(value, Mapping):
        raise TravelToolAdapterError("tool arguments must be an object")
    return dict(value)


def normalize_tool_call(call: Any) -> dict[str, Any]:
    """Parse OpenAI, Qwen, DeepSeek or legacy call into public parameters."""
    if not isinstance(call, Mapping):
        raise TravelToolAdapterError("tool call must be an object")
    function = call.get("function") if isinstance(call.get("function"), Mapping) else call
    name = str(function.get("name", call.get("name", TOOL_NAME)))
    if name != TOOL_NAME:
        raise TravelToolAdapterError(f"unsupported tool {name!r}")
    args = _as_dict(function.get("arguments", call.get("arguments", {})))
    choice = args.get("choice")
    content = args.get("content")
    if not isinstance(choice, str) or choice.casefold() not in {"search", "action", "answer"}:
        raise TravelToolAdapterError("choice must be search, action or answer")
    if not isinstance(content, str):
        raise TravelToolAdapterError("content must be a string")
    return {"choice": choice.casefold(), "content": content}


def parse_assistant_tool_calls(message: Mapping[str, Any]) -> list[dict[str, Any]]:
    calls = message.get("tool_calls")
    if calls is None and isinstance(message.get("content"), Mapping):
        calls = [message["content"]]
    if calls is None:
        content = message.get("content", "")
        if isinstance(content, str):
            match = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", content, re.IGNORECASE | re.DOTALL)
            if match:
                try:
                    calls = [json.loads(match.group(1))]
                except (TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise TravelToolAdapterError("tool call JSON is invalid") from exc
    if calls is None:
        return []
    if not isinstance(calls, list):
        calls = [calls]
    return [normalize_tool_call(call) for call in calls]


def format_environment_action(parameters: Mapping[str, Any]) -> str:
    """Convert normalized parameters to TravelGym's public wire strings."""
    normalized = normalize_tool_call({"name": TOOL_NAME, "arguments": dict(parameters)})
    return f"[{normalized['choice']}] {normalized['content']}"


def sanitize_public_feedback(value: Any) -> str:
    """Keep neutral public feedback; never expose reward/private labels."""
    def scrub(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {
                key: scrub(child)
                for key, child in item.items()
                if str(key).casefold() not in _PRIVATE_KEYS
            }
        if isinstance(item, list):
            return [scrub(child) for child in item]
        if isinstance(item, tuple):
            return [scrub(child) for child in item]
        return item

    value = scrub(value)
    text = value if isinstance(value, str) else str(value or "")
    text = _REWARD_LINE.sub("", text)
    text = _PRIVATE_TEXT_LINE.sub("", text)
    text = _PRIVATE_ID.sub("that preference", text).strip()
    # These are private report fields occasionally appended by an evaluator.
    text = re.sub(r"\b(?:correct_ids?|best_ids?|preference_ids?)\s*[:=].*$", "", text, flags=re.IGNORECASE | re.MULTILINE).strip()
    return text


def canonical_internal_tool_call(parameters: Mapping[str, Any], *, call_id: str = "call_0000") -> dict[str, Any]:
    """Build the canonical in-process call (``arguments`` is a dict)."""
    normalized = normalize_tool_call({"name": TOOL_NAME, "arguments": dict(parameters)})
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": TOOL_NAME,
            "arguments": normalized,
        },
    }


def canonical_api_tool_call(parameters: Mapping[str, Any], *, call_id: str = "call_0000") -> dict[str, Any]:
    """Build an OpenAI-compatible call (``arguments`` is a JSON string)."""
    normalized = normalize_tool_call({"name": TOOL_NAME, "arguments": dict(parameters)})
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": TOOL_NAME,
            "arguments": json.dumps(normalized, ensure_ascii=False, separators=(",", ":")),
        },
    }


__all__ = [
    "TOOL_NAME",
    "TravelToolAdapterError",
    "canonical_api_tool_call",
    "canonical_internal_tool_call",
    "format_environment_action",
    "normalize_tool_call",
    "parse_assistant_tool_calls",
    "sanitize_public_feedback",
]
