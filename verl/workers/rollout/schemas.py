# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import os
from enum import Enum
from collections.abc import Mapping
from typing import Any, Dict, List, Optional

import torch
from pydantic import BaseModel, model_validator
from transformers import PreTrainedTokenizer

from verl.tools.schemas import OpenAIFunctionToolCall, OpenAIFunctionToolSchema
from verl.utils.model import compute_position_id_with_mask

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class RolloutTemplateAlignmentError(ValueError):
    """Raised when an append-only rollout ledger diverges from native text."""


def _native_template_ids(tokenizer: PreTrainedTokenizer, messages, *, tools=None, add_generation_prompt=False) -> list[int]:
    """Tokenize with the model-native Qwen template (never slice strings)."""
    kwargs = {
        "tools": tools,
        "add_generation_prompt": bool(add_generation_prompt),
        "tokenize": True,
        "enable_thinking": True,
    }
    try:
        value = tokenizer.apply_chat_template(messages, return_tensors=None, **kwargs)
    except TypeError:
        value = tokenizer.apply_chat_template(messages, **kwargs)
    if isinstance(value, Mapping) and "input_ids" in value:
        # Transformers 5 returns a BatchEncoding when tokenization is enabled.
        value = value["input_ids"]
    if isinstance(value, torch.Tensor):
        value = value.tolist()
    elif hasattr(value, "tolist") and not isinstance(value, list):
        value = value.tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return [int(token) for token in value]


def _native_template_text(tokenizer: PreTrainedTokenizer, messages, *, tools=None, add_generation_prompt=False) -> str:
    """Render native Qwen text so append boundaries are checked before BPE."""
    value = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        add_generation_prompt=bool(add_generation_prompt),
        tokenize=False,
        enable_thinking=True,
    )
    if not isinstance(value, str):
        raise TypeError(f"native chat template returned {type(value).__name__}, expected str")
    return value


def _decode_ledger(tokenizer: PreTrainedTokenizer, token_ids: list[int]) -> str:
    return tokenizer.decode(
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def _incremental_template_suffix_ids(
    tokenizer: PreTrainedTokenizer,
    current_ids: list[int],
    target_text: str,
    *,
    label: str,
) -> list[int]:
    """Encode only newly appended native-template text.

    Token-level prefix checks are invalid at a BPE boundary: for example,
    Qwen3.5 encodes one trailing newline as token 198 but retokenizes two
    adjacent newlines as token 271.  An online generation ledger cannot
    rewrite an already-consumed token, so validate the decoded text and encode
    only the suffix that is actually appended.
    """
    current_text = _decode_ledger(tokenizer, current_ids)
    if not target_text.startswith(current_text):
        common = 0
        for current_char, target_char in zip(current_text, target_text):
            if current_char != target_char:
                break
            common += 1
        raise RolloutTemplateAlignmentError(
            f"native Qwen text mismatch while appending {label}: "
            f"common_chars={common}, ledger_chars={len(current_text)}, "
            f"target_chars={len(target_text)}"
        )
    suffix_text = target_text[len(current_text) :]
    if not suffix_text:
        return []
    suffix_ids = tokenizer.encode(suffix_text, add_special_tokens=False)
    combined_text = _decode_ledger(tokenizer, list(current_ids) + list(suffix_ids))
    if combined_text != target_text:
        raise RolloutTemplateAlignmentError(
            f"native Qwen suffix failed round-trip while appending {label}: "
            f"ledger_chars={len(current_text)}, suffix_chars={len(suffix_text)}, "
            f"target_chars={len(target_text)}"
        )
    return [int(token) for token in suffix_ids]


def _suffix_after_prefix(full_ids: list[int], prefix_ids: list[int], *, label: str) -> list[int]:
    """Return a native-template suffix, failing on any token mismatch."""
    if len(full_ids) < len(prefix_ids) or full_ids[: len(prefix_ids)] != prefix_ids:
        raise RolloutTemplateAlignmentError(f"native Qwen template prefix mismatch while appending {label}")
    return full_ids[len(prefix_ids) :]


class FinishReasonTypeEnum(str, Enum):
    """The enum for finish reason type."""

    LENGTH = "length"
    STOP = "stop"
    TOOL_CALL = "tool_calls"

    @classmethod
    def from_str(cls, value: str) -> "FinishReasonTypeEnum":
        if value == "stop":
            return cls.STOP
        elif value == "length":
            return cls.LENGTH
        elif value == "tool_calls":
            return cls.TOOL_CALL
        else:
            raise ValueError(f"Unsupported finish reason type: {value}")


class Message(BaseModel):
    role: str
    content: str
    tool_calls: Optional[List[OpenAIFunctionToolCall]] = None
    # Qwen3.5/DeepSeek thinking is a first-class field.  Keeping it separate
    # prevents duplicated <think> blocks when the native chat template is
    # applied.
    reasoning_content: Optional[str] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def normalize_null_content(cls, values):
        """OpenAI tool turns may carry ``content: null``.

        Qwen's native template and the internal request ledger use an empty
        string for a tool-only Assistant message.  Normalising this at the
        boundary keeps the canonical message contract strict without making
        every rollout backend special-case ``None``.
        """
        if isinstance(values, dict) and values.get("content") is None:
            values = dict(values)
            values["content"] = ""
        return values


class AsyncRolloutRequestStateEnum(str, Enum):
    """The enum for async rollout request state."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TOOL_CALLING = "tool_calling"


class RolloutLengthExceededError(ValueError):
    """Raised when one trajectory cannot fit its configured token budget."""


class RolloutProtocolError(ValueError):
    """Raised when a generated tool call violates the advertised protocol."""


def compute_generation_budget(
    *,
    generation_prompt_tokens: int,
    current_response_tokens: int,
    max_model_len: int,
    max_response_len: int,
    max_new_tokens_per_turn: int = 2048,
    tool_response_token_reserve: int = 6144,
    template_token_reserve: int = 32,
    has_tools: bool = True,
) -> tuple[int, bool]:
    """Compute a safe per-turn generation cap from cumulative budgets.

    current_response_tokens includes all assistant turns, native template
    markers, and serialized tool observations already appended to the request.
    The reserve leaves room for the next tool observation (or just the native
    template/EOS suffix for a no-tool request), so a successful generation does
    not immediately make the following turn impossible.
    """
    generation_prompt_tokens = int(generation_prompt_tokens)
    current_response_tokens = int(current_response_tokens)
    max_model_len = int(max_model_len)
    max_response_len = int(max_response_len)
    max_new_tokens_per_turn = int(max_new_tokens_per_turn)
    reserve = int(tool_response_token_reserve if has_tools else template_token_reserve)
    if min(
        generation_prompt_tokens,
        current_response_tokens,
        max_model_len,
        max_response_len,
        max_new_tokens_per_turn,
        reserve,
    ) < 0:
        raise ValueError("rollout token budgets must be non-negative")
    remaining_response = max_response_len - current_response_tokens
    remaining_context = max_model_len - generation_prompt_tokens
    max_new_tokens = min(
        max_new_tokens_per_turn,
        remaining_response - reserve,
        remaining_context - reserve - 1,
    )
    if max_new_tokens <= 0:
        raise RolloutLengthExceededError(
            "no generation budget remains after reserving "
            f"{reserve} tokens (generation_prompt={generation_prompt_tokens}, "
            f"current_response={current_response_tokens}, "
            f"max_model_len={max_model_len}, max_response_len={max_response_len})"
        )
    return int(max_new_tokens), bool(max_new_tokens < max_new_tokens_per_turn)


class AsyncRolloutRequest(BaseModel):
    """The data model for async rollout."""

    batch_data_id: int = 0
    rollout_offset: int = 0
    request_id: str
    state: AsyncRolloutRequestStateEnum
    messages: List[Message]
    tool_schemas: Optional[List[OpenAIFunctionToolSchema]] = None
    tools_kwargs: Dict[str, Any] = {}
    input_ids: List[int]
    prompt_ids: List[int]
    response_ids: List[int]
    attention_mask: List[int]
    prompt_attention_mask: List[int]
    response_attention_mask: List[int]
    position_ids: List[int]
    prompt_position_ids: List[int]
    response_position_ids: List[int]
    loss_mask: List[int]
    prompt_loss_mask: List[int]
    response_loss_mask: List[int]
    reward_scores: Dict[str, float]
    max_prompt_len: int
    max_response_len: int = 8192
    max_model_len: int = 32768
    finish_reason: Optional[str] = None
    metrics: Dict[str, List[Any]] = {}
    length_events: List[Dict[str, Any]] = []

    turn_boundaries: List[int] = []
    conversation_histories: List[Dict[str, Any]] = []

    def record_length_event(self, phase: str, **values: int | float | bool) -> None:
        """Record bounded, numeric length diagnostics for trainer-side logging."""
        event: Dict[str, Any] = {
            "phase": str(phase),
            "context_tokens": int(len(self.input_ids)),
            "response_tokens": int(max(0, len(self.input_ids) - len(self.prompt_ids))),
        }
        event.update(values)
        self.length_events.append(event)

    def length_summary(self, *, finish_reason: str | None = None) -> Dict[str, float]:
        """Return scalar length diagnostics without copying transcript text."""
        events = self.length_events
        context_values = [int(event.get("context_tokens", 0)) for event in events]
        response_values = [int(event.get("response_tokens", 0)) for event in events]
        generation_values = [int(event.get("generation_prompt_tokens", 0)) for event in events]
        generated_values = [int(event.get("max_new_tokens", 0)) for event in events]
        tool_response_values = [int(event.get("tool_response_tokens", 0)) for event in events]
        return {
            "max_context_tokens": float(max(context_values, default=len(self.input_ids))),
            "max_response_tokens": float(max(response_values, default=max(0, len(self.input_ids) - len(self.prompt_ids)))),
            "max_generation_prompt_tokens": float(max(generation_values, default=0)),
            "max_new_tokens": float(max(generated_values, default=0)),
            "max_tool_response_tokens": float(max(tool_response_values, default=0)),
            "generation_turns": float(sum(event.get("phase") == "generation" for event in events)),
            "reasoning_generation_turns": float(sum(event.get("phase") == "reasoning_generation" for event in events)),
            "tool_call_generation_turns": float(sum(event.get("phase") == "tool_call_generation" for event in events)),
            "forced_reasoning_end_count": float(sum(event.get("phase") == "forced_reasoning_end" for event in events)),
            "tool_response_turns": float(sum(event.get("phase") == "tool_response" for event in events)),
            "budget_clamp_count": float(sum(bool(event.get("budget_clamped", False)) for event in events)),
            "length_limited": float(str(finish_reason or "") == "length"),
            "invalid_rollout": float(any(bool(event.get("invalid", False)) for event in events)),
        }

    use_inference_chat_template: bool
    enable_tokenization_sanity_check: bool
    generation_prompt_ids: List[int]
    base_conv_wo_gen_prompt_end_pos: int
    base_conv_with_gen_prompt_end_pos: int

    @property
    def tools(self) -> Optional[List[OpenAIFunctionToolSchema]]:
        """Compatibility alias used by the legacy async rollout worker."""
        return self.tool_schemas

    @model_validator(mode="before")
    @classmethod
    def initialize_request(cls, values):
        # Older async rollout call sites still pass ``tools=...``.  Accept
        # that spelling at the boundary, but keep one canonical field inside
        # the request so native-template rendering always sees the same tool
        # schema.
        if isinstance(values, dict) and "tool_schemas" not in values and "tools" in values:
            values = dict(values)
            values["tool_schemas"] = values.pop("tools")
        if not (messages := values.get("messages")):
            raise ValueError("messages is required for AsyncRolloutRequest initialization")
        if not (max_prompt_len := values.get("max_prompt_len")):
            raise ValueError("max_prompt_len is required for AsyncRolloutRequest initialization")
        if not (tokenizer := values.pop("tokenizer", None)):
            raise ValueError("tokenizer is required for AsyncRolloutRequest initialization")

        values["messages"] = [Message.model_validate(msg) for msg in messages]

        if (tool_schemas := values.get("tool_schemas", [])):
            # Pydantic callers provide OpenAIFunctionToolSchema instances,
            # whereas a few API-facing paths already carry plain dicts.  Both
            # represent the same canonical schema and must render identically.
            normalized_tool_schemas = [
                OpenAIFunctionToolSchema.model_validate(tool) for tool in tool_schemas
            ]
            values["tool_schemas"] = normalized_tool_schemas
            tools = [tool.model_dump(exclude_none=True) for tool in normalized_tool_schemas]
        else:
            tools = None
        normalized_messages = [msg.model_dump(exclude_none=True) for msg in values["messages"]]
        tokens_without_prompt = _native_template_ids(tokenizer, normalized_messages, tools=tools, add_generation_prompt=False)
        if not values.get("input_ids") or not values.get("attention_mask"):
            prompt_ids = _native_template_ids(tokenizer, normalized_messages, tools=tools, add_generation_prompt=True)
            values["input_ids"], values["attention_mask"] = prompt_ids, [1] * len(prompt_ids)
        if len(values["input_ids"]) > max_prompt_len:
            # Never silently truncate a native Qwen3.5 prompt.  A cut in
            # the system/tool schema changes the protocol and invalidates
            # the token-level alignment, so fail before generation and
            # let the caller increase prompt_length or quarantine the row.
            raise ValueError(
                f"Prompt {values.get('batch_data_id', 0)} length "
                f"{len(values['input_ids'])} exceeds max_prompt_len {max_prompt_len}; "
                "truncation=error"
            )

        values["prompt_ids"], values["prompt_attention_mask"] = values["input_ids"], values["attention_mask"]
        values["position_ids"] = values["prompt_position_ids"] = compute_position_id_with_mask(torch.tensor(values["attention_mask"])).tolist()
        values["loss_mask"] = values["prompt_loss_mask"] = [0] * len(values["input_ids"])
        # Keep these legacy bookkeeping fields tied to the *actual* transcript
        # rather than a synthetic two-message history.  The latter could make
        # a Qwen3.5 generation suffix look aligned while silently dropping
        # real system/tool tokens.
        values["generation_prompt_ids"] = _suffix_after_prefix(
            values["input_ids"], tokens_without_prompt, label="initial generation prompt"
        )
        values["base_conv_wo_gen_prompt_end_pos"] = len(tokens_without_prompt)
        values["base_conv_with_gen_prompt_end_pos"] = len(values["input_ids"])
        return values

    def _update_input_ids(self, new_input_ids: List[int], attention_mask: bool, loss_mask: bool) -> None:
        """
        Update the input_ids, attention_mask, position_ids, and loss_mask of the request in additive manner.
        """
        self.input_ids += new_input_ids
        attention_mask = [int(attention_mask)] * len(new_input_ids)
        self.attention_mask += attention_mask
        self.loss_mask += [int(loss_mask)] * len(new_input_ids)
        self.position_ids += (compute_position_id_with_mask(torch.tensor(attention_mask)) + (self.position_ids[-1] + 1)).tolist()

        assert len(self.input_ids) == len(self.attention_mask) == len(self.position_ids) == len(self.loss_mask), f"""Request {self.request_id} has different length of {len(self.input_ids)=}, 
            {len(self.attention_mask)=}, {len(self.position_ids)=}, {len(self.loss_mask)=}"""

    format_config: dict = {
        "chatml": {
            "assistant_prefix_msg": "\n<|im_start|>assistant\n",
            "assistant_suffix_msg": "<|im_end|>",
            "tool_prefix_msg": "\n<|im_start|>tool\n",
            "tool_suffix_msg": "<|im_end|>",
        },
        "qwen": {
            "assistant_prefix_msg": "\n<|im_start|>assistant\n",
            "assistant_suffix_msg": "<|im_end|>",
            "merge_tool_response": True,
            "tool_prefix_msg": "\n<|im_start|>user",
            "tool_suffix_msg": "<|im_end|>",
            "tool_response_prefix_msg": "\n<tool_response>\n",
            "tool_response_suffix_msg": "\n</tool_response>",
        }
    }

    def _message_dicts(self) -> list[dict[str, Any]]:
        return [message.model_dump(exclude_none=True) for message in self.messages]

    def _tool_dicts(self) -> list[dict[str, Any]] | None:
        return [tool.model_dump(exclude_none=True) for tool in self.tool_schemas] if self.tool_schemas else None

    def _replace_input_ids(self, token_ids: list[int], *, loss_mask: int | None = None) -> None:
        """Keep the request ledger exactly aligned with native template IDs."""
        if loss_mask is None:
            loss_mask = 0
        self.input_ids = list(token_ids)
        self.attention_mask = [1] * len(self.input_ids)
        self.position_ids = compute_position_id_with_mask(torch.tensor(self.attention_mask)).tolist()
        self.loss_mask = [int(loss_mask)] * len(self.input_ids)

    def inject_initial_user_context(
        self,
        tokenizer: PreTrainedTokenizer,
        context: str,
    ) -> None:
        """Add reset-time public environment state before the first generation."""
        context = str(context or "").strip()
        if not context:
            return
        if self.state != AsyncRolloutRequestStateEnum.PENDING or self.response_ids:
            raise ValueError("initial environment context must be injected before generation")

        tools = self._tool_dicts()
        previous_generation_ids = _native_template_ids(
            tokenizer, self._message_dicts(), tools=tools, add_generation_prompt=True
        )
        if self.input_ids != previous_generation_ids:
            raise ValueError("request input_ids diverged before initial context injection")

        context_block = f"[TravelGym public state]\n{context}"
        if self.messages and self.messages[-1].role == "user":
            prior = self.messages[-1].content.rstrip()
            self.messages[-1].content = f"{prior}\n\n{context_block}"
        else:
            self.messages.append(Message(role="user", content=context_block))

        completed_ids = _native_template_ids(
            tokenizer, self._message_dicts(), tools=tools, add_generation_prompt=False
        )
        generation_ids = _native_template_ids(
            tokenizer, self._message_dicts(), tools=tools, add_generation_prompt=True
        )
        if len(generation_ids) > self.max_prompt_len:
            raise ValueError(
                f"Initial environment context expands prompt to {len(generation_ids)} "
                f"tokens, exceeding max_prompt_len={self.max_prompt_len}"
            )

        self._replace_input_ids(generation_ids, loss_mask=0)
        self.prompt_ids = list(generation_ids)
        self.prompt_attention_mask = list(self.attention_mask)
        self.prompt_position_ids = list(self.position_ids)
        self.prompt_loss_mask = list(self.loss_mask)
        self.generation_prompt_ids = _suffix_after_prefix(
            generation_ids, completed_ids, label="initial environment generation prompt"
        )
        self.base_conv_wo_gen_prompt_end_pos = len(completed_ids)
        self.base_conv_with_gen_prompt_end_pos = len(generation_ids)

    def get_generation_prompt_ids(self, tokenizer: PreTrainedTokenizer) -> list[int]:
        """Render the complete current transcript with Qwen's native prompt.

        Older code appended a cached generation marker using a synthetic
        placeholder conversation.  That can silently mis-tokenize a real
        TravelGym transcript (especially after tool responses).  We instead
        render the actual message list and assert that the completed stream is
        a prefix of the generation stream; only the native suffix is appended.
        """
        if not self.use_inference_chat_template:
            return self.input_ids
        tools = self._tool_dicts()
        completed_text = _native_template_text(
            tokenizer, self._message_dicts(), tools=tools, add_generation_prompt=False
        )
        generation_text = _native_template_text(
            tokenizer, self._message_dicts(), tools=tools, add_generation_prompt=True
        )
        current_text = _decode_ledger(tokenizer, self.input_ids)
        # A prior caller may already have appended the generation prompt.
        if current_text == generation_text:
            return self.input_ids
        if current_text != completed_text:
            raise RolloutTemplateAlignmentError(
                "request ledger diverged from native Qwen3.5 text before generation"
            )
        generation_suffix = _incremental_template_suffix_ids(
            tokenizer,
            self.input_ids,
            generation_text,
            label="generation prompt",
        )
        if generation_suffix:
            self._update_input_ids(generation_suffix, attention_mask=True, loss_mask=False)
        return self.input_ids

    # Legacy async workers use the singular method name.  Keep it as a thin
    # alias so both rollout implementations share the native-template path.
    def get_generation_prompt(self, tokenizer: PreTrainedTokenizer) -> list[int]:
        return self.get_generation_prompt_ids(tokenizer)

    def build_tool_call_prompt_ids(
        self,
        tokenizer: PreTrainedTokenizer,
        reasoning_content: str,
    ) -> list[int]:
        """Build the native continuation immediately after ``</think>``.

        The request ledger remains unchanged until the complete assistant turn
        is committed.  This lets rollout generate reasoning and the tool call
        in separate engine requests without teaching a synthetic transcript.
        """
        if not self.use_inference_chat_template:
            raise RolloutTemplateAlignmentError(
                "two-stage tool generation requires the native chat template"
            )
        previous_messages = self._message_dicts()
        tools = self._tool_dicts()
        generation_prefix_text = _native_template_text(
            tokenizer, previous_messages, tools=tools, add_generation_prompt=True
        )
        if _decode_ledger(tokenizer, self.input_ids) != generation_prefix_text:
            raise RolloutTemplateAlignmentError(
                "request ledger diverged before building the tool-call prompt"
            )

        reasoning_message = Message(
            role="assistant",
            content="",
            reasoning_content=str(reasoning_content or "").strip(),
        )
        completed_text = _native_template_text(
            tokenizer,
            previous_messages + [reasoning_message.model_dump(exclude_none=True)],
            tools=tools,
            add_generation_prompt=False,
        )
        assistant_end = f"{tokenizer.eos_token}\n"
        if not completed_text.endswith(assistant_end):
            raise RolloutTemplateAlignmentError(
                "native assistant turn does not end with the expected EOS marker"
            )
        tool_call_prompt_text = completed_text[: -len(assistant_end)]
        if not tool_call_prompt_text.endswith("</think>\n\n"):
            raise RolloutTemplateAlignmentError(
                "native assistant reasoning does not end at the tool-call boundary"
            )
        suffix_ids = _incremental_template_suffix_ids(
            tokenizer,
            self.input_ids,
            tool_call_prompt_text,
            label="reasoning close before tool call",
        )
        return list(self.input_ids) + suffix_ids

    def add_assistant_message(
        self,
        tokenizer: PreTrainedTokenizer,
        content: str,
        tool_calls: Optional[List[OpenAIFunctionToolCall]] = None,
        reasoning_content: Optional[str] = None,
        already_over_long: bool = False,
        format: Optional[str] = None,
        forced_reasoning_end: bool = False,
    ) -> None:
        # Some SGLang parsers return <think>...</think> in ``content`` while
        # Qwen3.5 expects ``reasoning_content``.  Normalize at this boundary.
        import re

        raw_content = str(content or "")
        if reasoning_content is None:
            match = re.search(r"<think>\s*(.*?)\s*</think>", raw_content, re.IGNORECASE | re.DOTALL)
            if match:
                reasoning_content = match.group(1).strip()
                raw_content = (raw_content[: match.start()] + raw_content[match.end() :]).strip()
            elif self.use_inference_chat_template and raw_content:
                # Qwen3.5's generation prompt already ends in ``<think>\n``.
                # SGLang normally strips that control marker from the returned
                # text, so a plain completion is the body of the open thinking
                # block rather than ordinary assistant content.  Rendering it
                # as ``content`` would make the template insert an empty
                # ``<think>...</think>`` pair first; BPE merges around the
                # resulting newlines then break the token-prefix invariant.
                # Keep the text as reasoning_content so native rendering is
                # exactly the same stream that was generated.
                reasoning_content = raw_content
                raw_content = ""
        del already_over_long, format  # compatibility flags; native template is authoritative
        previous_messages = self._message_dicts()
        tools = self._tool_dicts()
        # ``input_ids`` includes the generation prompt while the engine is
        # producing this Assistant turn.  Render that exact prefix before
        # appending the completed Assistant message.
        generation_prefix_text = _native_template_text(
            tokenizer, previous_messages, tools=tools, add_generation_prompt=True
        )
        if _decode_ledger(tokenizer, self.input_ids) != generation_prefix_text:
            raise RolloutTemplateAlignmentError(
                "request ledger diverged before appending an assistant turn"
            )
        assistant_message = Message(
            role="assistant",
            content=raw_content,
            tool_calls=tool_calls,
            reasoning_content=reasoning_content,
        )
        full_text = _native_template_text(
            tokenizer,
            previous_messages + [assistant_message.model_dump(exclude_none=True)],
            tools=tools,
            add_generation_prompt=False,
        )
        content_ids = _incremental_template_suffix_ids(
            tokenizer,
            self.input_ids,
            full_text,
            label="assistant",
        )
        self.messages.append(assistant_message)
        assistant_start = len(self.input_ids)
        self._update_input_ids(content_ids, attention_mask=True, loss_mask=True)
        if forced_reasoning_end:
            closing_ids = tokenizer.encode("</think>", add_special_tokens=False)
            closing_start = -1
            for index in range(len(content_ids) - len(closing_ids), -1, -1):
                if content_ids[index : index + len(closing_ids)] == closing_ids:
                    closing_start = index
                    break
            if closing_start < 0:
                raise RolloutTemplateAlignmentError(
                    "forced </think> marker was not found in the assistant suffix"
                )
            for index in range(closing_start, closing_start + len(closing_ids)):
                self.loss_mask[assistant_start + index] = 0

    def add_tool_response_messages(
        self,
        tokenizer: PreTrainedTokenizer,
        contents: list[str],
        tool_call_ids: Optional[List[str]] = None,
        names: Optional[List[str]] = None,
    ) -> None:
        if not contents:
            return
        previous_messages = self._message_dicts()
        tools = self._tool_dicts()
        completed_prefix_text = _native_template_text(
            tokenizer, previous_messages, tools=tools, add_generation_prompt=False
        )
        if _decode_ledger(tokenizer, self.input_ids) != completed_prefix_text:
            raise RolloutTemplateAlignmentError(
                "request ledger diverged before appending a tool response"
            )
        if tool_call_ids is not None and len(tool_call_ids) != len(contents):
            raise ValueError("tool_call_ids must align with tool response contents")
        if names is not None and len(names) != len(contents):
            raise ValueError("tool response names must align with contents")
        tool_messages = [
            Message(
                role="tool",
                content=str(content),
                tool_call_id=str(tool_call_ids[index]) if tool_call_ids is not None else None,
                name=str(names[index]) if names is not None else None,
            )
            for index, content in enumerate(contents)
        ]
        full_text = _native_template_text(
            tokenizer,
            previous_messages + [message.model_dump(exclude_none=True) for message in tool_messages],
            tools=tools,
            add_generation_prompt=False,
        )
        content_ids = _incremental_template_suffix_ids(
            tokenizer,
            self.input_ids,
            full_text,
            label="tool response",
        )
        self.messages.extend(tool_messages)
        self._update_input_ids(content_ids, attention_mask=True, loss_mask=False)

    def add_tool_response_message(
        self,
        tokenizer: PreTrainedTokenizer,
        content: str,
        is_last: bool = True,
        format: Optional[str] = None,
        tool_call_id: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        """Compatibility wrapper for the legacy async worker."""
        del is_last, format
        ids = [tool_call_id] if tool_call_id is not None else None
        names = [name] if name is not None else None
        self.add_tool_response_messages(tokenizer, [content], tool_call_ids=ids, names=names)

    def update_metrics(self, metrics: Any, tool_id: str) -> None:
        """
        metrics: should be a dict of tools_name -> Any
        """
        if self.metrics.get(tool_id) is None:
            self.metrics[tool_id] = []
        self.metrics[tool_id].append(metrics)

    def finalize(
        self,
        tokenizer: PreTrainedTokenizer,
        reward_scores: Dict[str, float],
        turn_boundaries: List[int] = [],
        conversation_histories: List[Dict[str, Any]] = [],
        finish_reason_type: FinishReasonTypeEnum = FinishReasonTypeEnum.STOP,
    ) -> None:
        self.state = AsyncRolloutRequestStateEnum.COMPLETED
        self.finish_reason = str(finish_reason_type.value)
        self.reward_scores = reward_scores
        self.turn_boundaries = turn_boundaries
        self.conversation_histories = conversation_histories
        self.response_ids = self.input_ids[len(self.prompt_ids) :]
        if finish_reason_type == FinishReasonTypeEnum.STOP:
            pass
        elif finish_reason_type == FinishReasonTypeEnum.LENGTH:
            pass
        else:
            raise ValueError(f"Unsupported finalize finish reason type: {finish_reason_type}")
        self.truncate_output_ids(tokenizer)
        assert len(self.input_ids) == len(self.attention_mask) == len(self.position_ids) == len(self.loss_mask), f"""Request {self.request_id} has different length of {len(self.input_ids)=}, 
            {len(self.attention_mask)=}, {len(self.position_ids)=}, {len(self.loss_mask)=}"""

    def truncate_output_ids(self, tokenizer: PreTrainedTokenizer) -> None:
        # Historical code sliced overlong sequences here.  That can cut a
        # Qwen3.5 tool call/reasoning block and create a transcript that never
        # existed in the environment.  Generation is configured with the
        # remaining native context budget, so an overrun is a hard protocol
        # error rather than a recoverable truncation.
        if len(self.input_ids) > self.max_model_len:
            raise RolloutLengthExceededError(
                f"request {self.request_id} length {len(self.input_ids)} exceeds "
                f"max_model_len={self.max_model_len}; truncation=error"
            )
        response_start = len(self.prompt_ids)
        response_end = len(self.input_ids)
        response_length = response_end - response_start
        if response_length > self.max_response_len:
            raise RolloutLengthExceededError(
                f"request {self.request_id} response length {response_length} exceeds "
                f"max_response_len={self.max_response_len}; truncation=error"
            )
        # Keep the explicit assignments for callers that expect these views;
        # no slicing is performed because the bounds were checked above.
        self.response_ids = self.input_ids[response_start:response_end]
        self.response_attention_mask = self.attention_mask[len(self.prompt_attention_mask):]
        self.response_position_ids = self.position_ids[len(self.prompt_position_ids):]
        self.response_loss_mask = self.loss_mask[len(self.prompt_loss_mask):]
