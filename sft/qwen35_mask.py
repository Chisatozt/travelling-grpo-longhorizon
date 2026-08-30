"""Dependency-free exact token-span construction for Qwen3.5 training.

The tokenizer callback is the model's native ``apply_chat_template``.  This
module only aligns token IDs; it never slices rendered strings or changes the
template, so the resulting ``input_ids`` are byte-for-byte/token-for-token
identical to inference.
"""

from __future__ import annotations

import copy
import json
from typing import Any, Callable, Mapping, Sequence


def template_messages(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Prepare canonical messages for Qwen's native chat template.

    Canonical storage keeps ``function.arguments`` as a structured mapping so
    protocol validation is deterministic.  Qwen3.5's Jinja templates and most
    OpenAI-compatible tokenizers expect that field as a JSON string.  Convert
    only at the tokenizer boundary; the canonical record and environment wire
    format remain unchanged.
    """

    prepared = copy.deepcopy(list(messages))
    for message in prepared:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        calls = message.get("tool_calls")
        if not isinstance(calls, list):
            continue
        for call in calls:
            if not isinstance(call, dict):
                continue
            function = call.get("function")
            if not isinstance(function, dict):
                continue
            arguments = function.get("arguments")
            if isinstance(arguments, Mapping):
                function["arguments"] = json.dumps(
                    dict(arguments), ensure_ascii=False, separators=(",", ":")
                )
    return prepared


def _prefix_len(full_ids: Sequence[int], candidate: Sequence[int]) -> int:
    # A loss span is meaningful only when the native generation rendering is
    # an exact token prefix of the completed rendering.  Longest-common-prefix
    # recovery is deliberately forbidden: it can silently mask a template or
    # tool-schema mismatch and shift supervision onto the wrong tokens.
    if len(full_ids) < len(candidate) or list(full_ids[: len(candidate)]) != list(candidate):
        raise ValueError(
            "Qwen3.5 native template prefix mismatch; check tokenizer revision, "
            "tools schema and enable_thinking settings."
        )
    return len(candidate)


def assert_template_equivalence(official_ids: Sequence[int], training_ids: Sequence[int]) -> None:
    """Assert that training and inference render the exact same token stream.

    The loss mask is metadata layered on top of this stream; it must never be
    produced by re-tokenising a hand-sliced string or by silently accepting a
    template mismatch.
    """

    if list(official_ids) != list(training_ids):
        raise ValueError(
            "Qwen3.5 training template diverges from the official template "
            f"({len(official_ids)} vs {len(training_ids)} tokens)"
        )


def exact_assistant_token_mask(
    messages: Sequence[Mapping[str, Any]],
    full_ids: Sequence[int],
    render_ids: Callable[[Sequence[Mapping[str, Any]], bool], Sequence[int]],
    assistant_train_mask: Sequence[int],
) -> list[int]:
    """Return a token mask while preserving the official full input IDs.

    ``render_ids(messages, add_generation_prompt)`` must call the same native
    Qwen3.5 template/tokenizer as the full render.  Generation prompt tokens
    are excluded; all reasoning and tool-call tokens emitted by a supervised
    Assistant message are included.
    """
    if len(messages) != len(assistant_train_mask):
        raise ValueError("assistant_train_mask must be message-aligned")
    mask = [0] * len(full_ids)
    for index, message in enumerate(messages):
        if message.get("role") != "assistant" or not int(assistant_train_mask[index]):
            continue
        before = list(render_ids(messages[:index], True))
        after = list(render_ids(messages[: index + 1], False))
        try:
            start = _prefix_len(full_ids, before)
        except ValueError:
            # A completed transcript is allowed to omit the template's
            # generation-only marker (the tiny fake tokenizer used in the
            # CPU tests models this case).  Fall back only to the exact
            # completed prefix—not to an arbitrary longest-common prefix.
            completed_before = list(render_ids(messages[:index], False))
            start = _prefix_len(full_ids, completed_before)
        end = _prefix_len(full_ids, after)
        if end < start:
            raise ValueError(f"assistant span inversion at message {index}")
        for position in range(start, min(end, len(mask))):
            mask[position] = 1
    return mask


def exact_assistant_span_masks(
    messages: Sequence[Mapping[str, Any]],
    full_ids: Sequence[int],
    render_ids: Callable[[Sequence[Mapping[str, Any]], bool], Sequence[int]],
    assistant_train_mask: Sequence[int],
) -> tuple[list[int], list[int], list[int]]:
    """Return loss, reasoning and tool/content masks on one native stream.

    The ordinary Assistant mask is authoritative.  The two span masks are
    diagnostics layered on top of it: when a Qwen template can render the
    reasoning-only prefix as an exact prefix of the complete Assistant turn,
    those tokens are tagged as reasoning; the remaining supervised tokens are
    tagged as tool/content.  If a provider's template does not expose a
    separately alignable reasoning field, we fail soft by assigning the whole
    Assistant span to tool/content rather than shifting token boundaries.
    """

    if len(messages) != len(assistant_train_mask):
        raise ValueError("assistant_train_mask must be message-aligned")
    loss_mask = exact_assistant_token_mask(messages, full_ids, render_ids, assistant_train_mask)
    reasoning_mask = [0] * len(full_ids)
    tool_mask = [0] * len(full_ids)

    for index, message in enumerate(messages):
        if message.get("role") != "assistant" or not int(assistant_train_mask[index]):
            continue
        before = list(render_ids(messages[:index], True))
        after = list(render_ids(messages[: index + 1], False))
        try:
            start = _prefix_len(full_ids, before)
        except ValueError:
            start = _prefix_len(full_ids, render_ids(messages[:index], False))
        end = _prefix_len(full_ids, after)
        if end < start:
            raise ValueError(f"assistant span inversion at message {index}")

        reasoning_text = str(message.get("reasoning_content") or "").strip()
        calls = message.get("tool_calls")
        has_tool_or_content = bool(calls) or bool(str(message.get("content") or "").strip())
        reasoning_end = start
        if reasoning_text:
            reasoning_message = copy.deepcopy(dict(message))
            reasoning_message["content"] = ""
            reasoning_message.pop("tool_calls", None)
            try:
                reasoning_after = list(
                    render_ids(messages[:index] + [reasoning_message], False)
                )
                reasoning_end = _prefix_len(full_ids, reasoning_after)
                reasoning_end = max(start, min(end, reasoning_end))
            except ValueError:
                # Some chat templates only expose a combined Assistant span;
                # preserve exact loss supervision and leave reasoning tagged
                # unavailable instead of guessing a character/token offset.
                reasoning_end = start

        for position in range(start, min(end, len(loss_mask))):
            if not loss_mask[position]:
                continue
            if position < reasoning_end:
                reasoning_mask[position] = 1
            elif has_tool_or_content:
                tool_mask[position] = 1
            else:
                # A plain Assistant text turn with no explicit tool/content is
                # still a supervised public response; count it as content.
                tool_mask[position] = 1

    return loss_mask, reasoning_mask, tool_mask


def native_template_ids(tokenizer: Any, messages: Sequence[Mapping[str, Any]], tools: Any = None, *, enable_thinking: bool = True, add_generation_prompt: bool = False) -> list[int]:
    """Call the native template and normalize tensor/list return values."""
    messages = template_messages(messages)
    kwargs = {
        "tools": tools,
        "tokenize": True,
        "add_generation_prompt": add_generation_prompt,
        "enable_thinking": enable_thinking,
    }
    try:
        output = tokenizer.apply_chat_template(messages, return_tensors=None, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        output = tokenizer.apply_chat_template(messages, **kwargs)
    if hasattr(output, "tolist") and not isinstance(output, list):
        output = output.tolist()
    if output and isinstance(output[0], list):
        output = output[0]
    return [int(value) for value in output]


__all__ = [
    "assert_template_equivalence",
    "exact_assistant_token_mask",
    "exact_assistant_span_masks",
    "native_template_ids",
    "template_messages",
]
