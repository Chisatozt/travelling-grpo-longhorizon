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
from copy import deepcopy
from enum import Enum
from collections.abc import Mapping
from typing import Any, Dict, List, Optional

import torch
from pydantic import BaseModel, Field, model_validator
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

    # ``messages``/``input_ids`` are the mutable model-visible context.  The
    # archive is append-only so cleanup can remove old context without
    # deleting training targets or the complete interaction transcript.
    archive_messages: List[Message] = Field(default_factory=list)
    archive_model_outputs: List[Dict[str, Any]] = Field(default_factory=list)
    archive_segment_records: List[Dict[str, Any]] = Field(default_factory=list)
    archive_turns: List[Dict[str, Any]] = Field(default_factory=list)
    segment_records: List[Dict[str, Any]] = Field(default_factory=list)
    cleanup_events: List[Dict[str, Any]] = Field(default_factory=list)
    active_turn: Dict[str, Any] = Field(default_factory=dict)
    active_segment: Optional[Dict[str, Any]] = None
    initial_archive_message_count: int = 0
    archive_token_count: int = 0
    active_context_peak_tokens: int = 0

    def record_length_event(self, phase: str, **values: int | float | bool) -> None:
        """Record bounded, numeric length diagnostics for trainer-side logging."""
        event: Dict[str, Any] = {
            "phase": str(phase),
            "context_tokens": int(len(self.input_ids)),
            "response_tokens": int(self.active_response_tokens),
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
            "max_response_tokens": float(max(response_values, default=self.active_response_tokens)),
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
            "cleanup_attempts": float(len(self.cleanup_events)),
            "cleanup_successes": float(
                sum(bool(event.get("success", False)) for event in self.cleanup_events)
            ),
            "cleanup_released_tokens": float(
                sum(float(event.get("released_tokens", 0.0)) for event in self.cleanup_events)
            ),
            "archive_tokens": float(self.archive_token_count),
            "active_context_peak_tokens": float(self.active_context_peak_tokens),
            "segment_count": float(len(self.segment_records)),
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
        if values.get("archive_messages"):
            values["archive_messages"] = [
                Message.model_validate(msg) for msg in values["archive_messages"]
            ]
        else:
            values["archive_messages"] = deepcopy(values["messages"])

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
        values["initial_archive_message_count"] = len(values["archive_messages"])
        # The archive is rendered without the pending generation marker;
        # keep this diagnostic on the same native-template basis from the
        # first request onward.
        values["archive_token_count"] = len(tokens_without_prompt)
        values["active_context_peak_tokens"] = len(values["input_ids"])
        return values

    @property
    def active_response_tokens(self) -> int:
        """Number of tokens in the current fragment after its prompt."""
        if self.active_segment is not None:
            start = int(self.active_segment.get("start_input", len(self.prompt_ids)))
            return max(0, len(self.input_ids) - start)
        return max(0, len(self.input_ids) - len(self.prompt_ids))

    def _update_input_ids(self, new_input_ids: List[int], attention_mask: bool, loss_mask: bool) -> None:
        """
        Update the input_ids, attention_mask, position_ids, and loss_mask of the request in additive manner.
        """
        self.input_ids += new_input_ids
        attention_mask = [int(attention_mask)] * len(new_input_ids)
        self.attention_mask += attention_mask
        self.loss_mask += [int(loss_mask)] * len(new_input_ids)
        self.position_ids += (compute_position_id_with_mask(torch.tensor(attention_mask)) + (self.position_ids[-1] + 1)).tolist()
        self.active_context_peak_tokens = max(
            int(self.active_context_peak_tokens), len(self.input_ids)
        )

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
        self.active_context_peak_tokens = max(
            int(self.active_context_peak_tokens), len(self.input_ids)
        )

    def _archive_message_dicts(self) -> list[dict[str, Any]]:
        return [message.model_dump(exclude_none=True) for message in self.archive_messages]

    def _update_archive_token_count(self, tokenizer: PreTrainedTokenizer) -> None:
        """Count the complete archive with the native template when possible."""
        try:
            rendered = _native_template_ids(
                tokenizer,
                self._archive_message_dicts(),
                tools=self._tool_dicts(),
                add_generation_prompt=False,
            )
        except Exception:
            # The active ledger remains authoritative; a diagnostic count must
            # not hide a template error that the caller needs to handle.
            return
        # The archive is append-only, but native BPE merges can make the
        # rendered length non-monotonic by a token or two.  This field is the
        # current complete-archive length; the independent peak metric tracks
        # the largest active context seen during the rollout.
        self.archive_token_count = len(rendered)

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

        # Reset state is public and belongs to the complete archive too.  It
        # is copied before any later aspect cleanup can occur.
        if self.archive_messages and self.archive_messages[-1].role == "user":
            prior = self.archive_messages[-1].content.rstrip()
            self.archive_messages[-1].content = f"{prior}\n\n{context_block}"
        else:
            self.archive_messages.append(Message(role="user", content=context_block))
        self.initial_archive_message_count = len(self.archive_messages)

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
        self._update_archive_token_count(tokenizer)

    def begin_turn(self, global_turn_idx: int) -> None:
        """Open one global turn and record its boundary in the active segment."""
        if self.active_turn:
            raise ValueError("cannot begin a new turn before completing the previous turn")
        self.active_turn = {
            "global_turn": int(global_turn_idx),
            "archive_message_indices": [],
            "assistant_message_index": None,
            "tool_message_indices": [],
            "model_output_indices": [],
            "actor_end_input_len": None,
            "segment_id": None,
        }
        if self.active_segment is not None:
            start = int(self.active_segment.get("start_input", len(self.input_ids)))
            self.active_segment.setdefault("turn_boundaries", []).append(
                max(0, len(self.input_ids) - start)
            )

    def ensure_active_segment(self, prompt_ids: list[int]) -> None:
        """Start a fragment at the exact native prompt sent to the engine."""
        if list(prompt_ids) != list(self.input_ids):
            raise RolloutTemplateAlignmentError(
                "segment prompt must equal the native generation input"
            )
        if self.active_segment is None:
            self.active_segment = {
                "segment_id": len(self.segment_records),
                "global_turn_indices": [],
                "turn_boundaries": [0],
                "conversation_history": [],
                "prompt_ids": list(prompt_ids),
                "prompt_attention_mask": list(self.attention_mask),
                "prompt_position_ids": list(self.position_ids),
                "prompt_loss_mask": list(self.loss_mask),
                "start_input": len(self.input_ids),
            }
        if self.active_turn:
            self.active_turn["segment_id"] = int(self.active_segment["segment_id"])

    def _append_archive_message(self, message: Message, *, kind: str) -> int:
        index = len(self.archive_messages)
        self.archive_messages.append(deepcopy(message))
        if self.active_turn:
            self.active_turn.setdefault("archive_message_indices", []).append(index)
            if kind == "assistant":
                self.active_turn["assistant_message_index"] = index
            elif kind == "tool":
                self.active_turn.setdefault("tool_message_indices", []).append(index)
        return index

    def record_model_output(
        self,
        phase: str,
        text: str,
        *,
        finish_reason: str | None = None,
        output_ids: list[int] | None = None,
    ) -> int:
        """Keep the raw model completion in the append-only rollout archive."""
        record: Dict[str, Any] = {
            "output_index": len(self.archive_model_outputs),
            "phase": str(phase),
            "global_turn": (
                int(self.active_turn["global_turn"])
                if self.active_turn and self.active_turn.get("global_turn") is not None
                else None
            ),
            "text": str(text or ""),
        }
        if finish_reason is not None:
            record["finish_reason"] = str(finish_reason)
        if output_ids is not None:
            record["output_ids"] = [int(token_id) for token_id in output_ids]
        self.archive_model_outputs.append(record)
        if self.active_turn:
            self.active_turn.setdefault("model_output_indices", []).append(
                int(record["output_index"])
            )
        return int(record["output_index"])

    def _finalize_active_segment(self, *, end_input: int | None = None) -> bool:
        """Archive one non-empty fragment without retokenizing its target."""
        segment = self.active_segment
        if segment is None:
            return False
        start = int(segment.get("start_input", len(self.input_ids)))
        end = len(self.input_ids) if end_input is None else int(end_input)
        if end < start or end > len(self.input_ids):
            raise RolloutLengthExceededError(
                f"invalid segment bounds for request {self.request_id}: {start}:{end}"
            )
        if end == start:
            self.active_segment = None
            return False
        prompt_ids = list(segment.get("prompt_ids", []))
        response_ids = list(self.input_ids[start:end])
        actual_length = len(prompt_ids) + len(response_ids)
        if actual_length > self.max_model_len:
            raise RolloutLengthExceededError(
                f"segment {segment.get('segment_id', 0)} for request {self.request_id} "
                f"has {actual_length} tokens; max_model_len={self.max_model_len}"
            )
        response_attention_mask = list(self.attention_mask[start:end])
        response_position_ids = list(self.position_ids[start:end])
        response_loss_mask = list(self.loss_mask[start:end])
        response_length = len(response_ids)
        boundaries = sorted(
            {
                max(0, min(response_length - 1, int(value)))
                for value in segment.get("turn_boundaries", [])
                if response_length and int(value) < response_length
            }
        )
        if response_length and not boundaries:
            boundaries = [0]
        turns = list(segment.get("conversation_history", []))
        self.segment_records.append(
            {
                "segment_id": int(segment.get("segment_id", len(self.segment_records))),
                "global_turn_indices": [
                    int(value) for value in segment.get("global_turn_indices", [])
                ],
                "turn_boundaries": boundaries,
                "conversation_history": deepcopy(turns),
                "prompt_ids": prompt_ids,
                "prompt_attention_mask": list(segment.get("prompt_attention_mask", [])),
                "prompt_position_ids": list(segment.get("prompt_position_ids", [])),
                "prompt_loss_mask": list(segment.get("prompt_loss_mask", [])),
                "response_ids": response_ids,
                "response_attention_mask": response_attention_mask,
                "response_position_ids": response_position_ids,
                "response_loss_mask": response_loss_mask,
                "actual_input_ids": prompt_ids + response_ids,
                "actual_input_tokens": actual_length,
            }
        )
        self.active_segment = None
        return True

    def complete_turn(
        self,
        turn_events: list[dict[str, Any]] | None = None,
        conversation_history: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Close a global turn using only structured public tool events."""
        if not self.active_turn:
            raise ValueError("cannot complete a turn that was not started")
        events = [
            dict(event) for event in (turn_events or []) if isinstance(event, dict)
        ]
        event = events[0] if len(events) == 1 else {}
        raw_aspect = event.get("aspect")
        aspect = raw_aspect.strip() if isinstance(raw_aspect, str) else ""
        cleanable = bool(
            len(events) == 1
            and str(event.get("choice", "")).casefold() == "answer"
            and event.get("accepted") is True
            and event.get("completed_aspect") is True
            and aspect
        )
        history = deepcopy(conversation_history or {})
        record = {
            "global_turn": int(
                self.active_turn.get("global_turn", len(self.archive_turns))
            ),
            "archive_message_indices": list(
                self.active_turn.get("archive_message_indices", [])
            ),
            "assistant_message_index": self.active_turn.get("assistant_message_index"),
            "tool_message_indices": list(
                self.active_turn.get("tool_message_indices", [])
            ),
            "model_output_indices": list(
                self.active_turn.get("model_output_indices", [])
            ),
            "actor_end_input_len": self.active_turn.get("actor_end_input_len"),
            "turn_events": events,
            "history": history,
            "aspect": aspect,
            "accepted": bool(event.get("accepted", False)) if event else False,
            "completed_aspect": bool(event.get("completed_aspect", False))
            if event
            else False,
            "cleanable": cleanable,
            "cleaned": False,
            "segment_id": self.active_turn.get("segment_id"),
        }
        self.archive_turns.append(record)
        record_index = len(self.archive_turns) - 1
        if cleanable:
            # A completed answer makes the *whole* aspect history eligible,
            # not just its final answer call.  If any earlier turn cannot be
            # assigned to this aspect from one structured public event, keep
            # the group intact rather than guessing at message ownership.
            group_indices = [
                index
                for index, prior in enumerate(self.archive_turns)
                if int(prior.get("global_turn", -1)) <= record["global_turn"]
                and str(prior.get("aspect", "")) == aspect
            ]
            group_turns = [self.archive_turns[index] for index in group_indices]
            prior_turns_reliably_owned = all(
                len(prior.get("turn_events", [])) == 1
                and bool(prior.get("turn_events", [{}])[0].get("aspect", ""))
                for prior in self.archive_turns
                if int(prior.get("global_turn", -1)) <= record["global_turn"]
            )
            group_reliable = bool(group_turns) and prior_turns_reliably_owned and all(
                len(prior.get("turn_events", [])) == 1
                and str(prior.get("turn_events", [{}])[0].get("aspect", "")) == aspect
                for prior in group_turns
            )
            if group_reliable:
                record["cleanup_turn_indices"] = group_indices
                record["cleanup_archive_message_indices"] = sorted(
                    {
                        int(message_index)
                        for prior in group_turns
                        for message_index in prior.get("archive_message_indices", [])
                    }
                )
                record["cleanup_global_turns"] = [
                    int(prior.get("global_turn", -1)) for prior in group_turns
                ]
            else:
                record["cleanable"] = False
                record["cleanup_blocked_reason"] = "unreliable_aspect_message_ownership"
                record["cleanup_turn_indices"] = [record_index]
                record["cleanup_archive_message_indices"] = list(
                    record.get("archive_message_indices", [])
                )
                record["cleanup_global_turns"] = [record["global_turn"]]
        if self.active_segment is not None:
            self.active_segment.setdefault("global_turn_indices", []).append(
                record["global_turn"]
            )
            self.active_segment.setdefault("conversation_history", []).append(history)
        self.active_turn = {}
        return record

    def _cleanup_group_turns(self, turn: dict[str, Any]) -> list[dict[str, Any]]:
        indices = turn.get("cleanup_turn_indices", [])
        if not isinstance(indices, (list, tuple)):
            indices = []
        group = [
            self.archive_turns[int(index)]
            for index in indices
            if 0 <= int(index) < len(self.archive_turns)
        ]
        return sorted(
            group or [turn],
            key=lambda item: int(item.get("global_turn", 0)),
        )

    def _memory_message_for_turn(self, turn: dict[str, Any]) -> Message:
        """Make labelled public memory, never a synthetic tool result."""
        group_turns = self._cleanup_group_turns(turn)
        public_feedback: list[str] = []
        action_feedback: list[str] = []
        answer_feedback: list[str] = []
        answer_ids: list[str] = []
        seen_message_indices: set[int] = set()
        reliable_group = all(
            len(source.get("turn_events", [])) == 1
            for source in group_turns
        )
        for source in group_turns:
            events = source.get("turn_events", [])
            event = events[0] if reliable_group and events else {}
            choice = str(event.get("choice", "")).casefold()
            for index in source.get("tool_message_indices", []):
                index = int(index)
                if index in seen_message_indices or not (0 <= index < len(self.archive_messages)):
                    continue
                seen_message_indices.add(index)
                message = self.archive_messages[index]
                if message.role != "tool" or not message.content:
                    continue
                if choice == "action":
                    action_feedback.append(message.content)
                elif choice == "answer":
                    answer_feedback.append(message.content)
                elif not reliable_group:
                    public_feedback.append(message.content)
            for assistant_index in (
                [source.get("assistant_message_index")]
                if source.get("assistant_message_index") is not None
                else []
            ):
                assistant_index = int(assistant_index)
                if not (0 <= assistant_index < len(self.archive_messages)):
                    continue
                assistant = self.archive_messages[assistant_index]
                for tool_call in assistant.tool_calls or []:
                    arguments = getattr(tool_call.function, "arguments", {}) or {}
                    if not isinstance(arguments, Mapping):
                        continue
                    if str(arguments.get("choice", "")).casefold() == "answer":
                        answer_ids.append(str(arguments.get("content", "")))

        # Preserve action-user replies verbatim.  The final answer feedback is
        # also kept because it contains the latest public current-aspect,
        # visible-ID and counter state after the completed aspect advances.
        if reliable_group:
            public_feedback = action_feedback + answer_feedback
            if not public_feedback:
                # A structured event without a tool response is unusual; keep
                # the group conservative rather than manufacturing state.
                public_feedback = [
                    self.archive_messages[int(index)].content
                    for source in group_turns
                    for index in source.get("tool_message_indices", [])
                    if 0 <= int(index) < len(self.archive_messages)
                    and self.archive_messages[int(index)].content
                ]
        lines = [
            "[TravelGym public memory; historical completed aspect]",
            f"Source aspect: {str(turn.get('cleanup_aspect', turn.get('aspect', '')))}",
            "Source global turns: "
            + ", ".join(
                str(int(source.get("global_turn", -1))) for source in group_turns
            ),
        ]
        if answer_ids:
            lines.append("Submitted answer ID(s): " + ", ".join(answer_ids))
        if public_feedback:
            lines.append(
                "Public feedback (original text):\n" + "\n\n".join(public_feedback)
            )
        else:
            lines.append("Public feedback (original text): unavailable")
        return Message(role="system", content="\n".join(lines))

    def _compacted_messages(
        self, turns_to_clean: list[dict[str, Any]]
    ) -> tuple[list[Message], list[Message]]:
        ordered_turns: list[dict[str, Any]] = []
        seen_sources: set[tuple[int, tuple[int, ...]]] = set()
        for turn in sorted(
            turns_to_clean,
            key=lambda item: int(item.get("global_turn", 0)),
        ):
            source_indices = tuple(
                int(index) for index in turn.get("archive_message_indices", [])
            )
            source_key = (int(turn.get("global_turn", -1)), source_indices)
            if source_key in seen_sources:
                continue
            seen_sources.add(source_key)
            ordered_turns.append(turn)
        clean_ids = {
            int(index)
            for turn in ordered_turns
            for index in turn.get(
                "cleanup_archive_message_indices",
                turn.get("archive_message_indices", []),
            )
        }
        base_count = max(
            0, min(int(self.initial_archive_message_count), len(self.archive_messages))
        )
        base = [deepcopy(message) for message in self.archive_messages[:base_count]]
        memory_by_first_source: dict[int, list[Message]] = {}
        trailing_memory: list[Message] = []
        memory_messages: list[Message] = []
        for turn in ordered_turns:
            memory = self._memory_message_for_turn(turn)
            memory_messages.append(memory)
            source_indices = sorted(
                int(index)
                for index in turn.get(
                    "cleanup_archive_message_indices",
                    turn.get("archive_message_indices", []),
                )
                if int(index) >= base_count
            )
            if source_indices:
                memory_by_first_source.setdefault(source_indices[0], []).append(memory)
            else:
                trailing_memory.append(memory)

        retained: list[Message] = []
        for index, message in enumerate(self.archive_messages):
            if index < base_count:
                continue
            retained.extend(memory_by_first_source.get(index, []))
            if index not in clean_ids:
                retained.append(deepcopy(message))
        retained.extend(trailing_memory)
        return base + retained, memory_messages

    def maybe_cleanup_context(
        self,
        tokenizer: PreTrainedTokenizer,
        *,
        enabled: bool,
        target_context_tokens: int,
        next_turn_reserve: int,
        reason: str,
        force: bool = False,
        require_next_turn: bool = True,
    ) -> dict[str, Any]:
        """Compact old completed aspects only when the next-round budget is unsafe."""
        before_tokens = len(self.input_ids)
        reserve = max(0, int(next_turn_reserve))
        target = max(1, min(int(target_context_tokens), int(self.max_model_len)))
        danger = bool(force or before_tokens + reserve > int(self.max_model_len))
        result = {
            "attempted": False,
            "success": False,
            "reason": str(reason),
            "before_context_tokens": before_tokens,
            "after_context_tokens": before_tokens,
            "released_tokens": 0,
            "cleaned_aspects": [],
            "global_turn_ranges": [],
            "memory_tokens": 0,
            "recovered_next_turn_budget": bool(
                before_tokens + reserve <= self.max_model_len
            ),
        }
        if not enabled or not danger:
            return result
        result["attempted"] = True
        candidates = [
            turn
            for turn in sorted(
                self.archive_turns,
                key=lambda item: int(item.get("global_turn", 0)),
            )
            if bool(turn.get("cleanable", False))
            and not bool(turn.get("cleaned", False))
        ]
        if not candidates:
            result["failure"] = "no_completed_aspect_history"
            self.cleanup_events.append(deepcopy(result))
            return result

        snapshot = {
            "messages": deepcopy(self.messages),
            "input_ids": list(self.input_ids),
            "attention_mask": list(self.attention_mask),
            "position_ids": list(self.position_ids),
            "loss_mask": list(self.loss_mask),
            "active_turn": deepcopy(self.active_turn),
            "active_segment": deepcopy(self.active_segment),
            "segment_records": deepcopy(self.segment_records),
            "archive_turns": deepcopy(self.archive_turns),
        }
        selected: list[dict[str, Any]] = []
        previously_cleaned = [
            turn
            for turn in sorted(
                self.archive_turns,
                key=lambda item: int(item.get("global_turn", 0)),
            )
            if bool(turn.get("cleaned", False))
        ]
        candidate_messages: list[Message] | None = None
        candidate_memory: list[Message] = []
        candidate_ids: list[int] = []
        try:
            for turn in candidates:
                selected.append(turn)
                candidate_messages, candidate_memory = self._compacted_messages(
                    previously_cleaned + selected
                )
                candidate_ids = _native_template_ids(
                    tokenizer,
                    [
                        message.model_dump(exclude_none=True)
                        for message in candidate_messages
                    ],
                    tools=self._tool_dicts(),
                    add_generation_prompt=False,
                )
                safe_now = len(candidate_ids) + reserve <= int(self.max_model_len)
                if (
                    (len(candidate_ids) <= target and (safe_now or not require_next_turn))
                    or (safe_now and not require_next_turn)
                ):
                    break
            if candidate_messages is None:
                raise RolloutLengthExceededError("no compacted context candidate was built")
            safe_now = len(candidate_ids) + reserve <= int(self.max_model_len)
            within_model = len(candidate_ids) <= int(self.max_model_len)
            recovered = safe_now if require_next_turn else within_model
            if not recovered:
                raise RolloutLengthExceededError(
                    f"compacted context remains unsafe: {len(candidate_ids)} tokens, "
                    f"reserve={reserve}, max_model_len={self.max_model_len}"
                )

            # A tool response may be the token that pushed the old fragment
            # over max_model_len. Keep the last complete Actor output in the
            # old fragment and carry the public response through memory.
            end_input = before_tokens
            if before_tokens > int(self.max_model_len) and self.active_segment is not None:
                actor_end = (
                    self.active_turn.get("actor_end_input_len")
                    if self.active_turn
                    else None
                )
                if actor_end is None:
                    actor_end = next(
                        (
                            turn.get("actor_end_input_len")
                            for turn in reversed(self.archive_turns)
                            if turn.get("actor_end_input_len") is not None
                        ),
                        None,
                    )
                if actor_end is not None:
                    end_input = int(actor_end)
            if self.active_segment is not None:
                self._finalize_active_segment(end_input=end_input)
            for turn in selected:
                turn["cleaned"] = True
            self.messages = candidate_messages
            self._replace_input_ids(candidate_ids, loss_mask=0)
            self.active_segment = None
            memory_token_count = 0
            for message in candidate_memory:
                memory_token_count += len(
                    _native_template_ids(
                        tokenizer,
                        [message.model_dump(exclude_none=True)],
                        tools=None,
                        add_generation_prompt=False,
                    )
                )
            result.update(
                {
                    "success": True,
                    "after_context_tokens": len(candidate_ids),
                    "released_tokens": max(0, before_tokens - len(candidate_ids)),
                    "cleaned_aspects": [
                        str(turn.get("aspect", "")) for turn in selected
                    ],
                    "global_turn_ranges": [
                        [
                            min(
                                [int(value) for value in turn.get("cleanup_global_turns", [])]
                                or [int(turn.get("global_turn", -1))]
                            ),
                            max(
                                [int(value) for value in turn.get("cleanup_global_turns", [])]
                                or [int(turn.get("global_turn", -1))]
                            ),
                        ]
                        for turn in selected
                    ],
                    "memory_tokens": memory_token_count,
                    "recovered_next_turn_budget": bool(safe_now),
                    "archive_tokens": int(self.archive_token_count),
                    "active_context_peak_tokens": int(self.active_context_peak_tokens),
                    "segment_count": len(self.segment_records),
                }
            )
            self.cleanup_events.append(deepcopy(result))
            return result
        except Exception as exc:
            self.messages = snapshot["messages"]
            self.input_ids = snapshot["input_ids"]
            self.attention_mask = snapshot["attention_mask"]
            self.position_ids = snapshot["position_ids"]
            self.loss_mask = snapshot["loss_mask"]
            self.active_turn = snapshot["active_turn"]
            self.active_segment = snapshot["active_segment"]
            self.segment_records = snapshot["segment_records"]
            self.archive_turns = snapshot["archive_turns"]
            result["failure"] = f"{type(exc).__name__}: {exc}"
            self.cleanup_events.append(deepcopy(result))
            return result

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
        self._append_archive_message(assistant_message, kind="assistant")
        assistant_start = len(self.input_ids)
        self._update_input_ids(content_ids, attention_mask=True, loss_mask=True)
        if self.active_turn:
            self.active_turn["actor_end_input_len"] = len(self.input_ids)
        self._update_archive_token_count(tokenizer)
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
        for message in tool_messages:
            self._append_archive_message(message, kind="tool")
        self._update_input_ids(content_ids, attention_mask=True, loss_mask=False)
        self._update_archive_token_count(tokenizer)

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
        if finish_reason_type == FinishReasonTypeEnum.STOP:
            pass
        elif finish_reason_type == FinishReasonTypeEnum.LENGTH:
            pass
        else:
            raise ValueError(f"Unsupported finalize finish reason type: {finish_reason_type}")
        if self.active_segment is not None:
            self._finalize_active_segment()
        if self.segment_records:
            # ``response_ids`` is a compatibility view for callers that only
            # inspect the last fragment.  The complete set of targets lives in
            # ``segment_records`` and is what the segmented trainer consumes.
            last = self.segment_records[-1]
            self.response_ids = list(last.get("response_ids", []))
            self.response_attention_mask = list(last.get("response_attention_mask", []))
            self.response_position_ids = list(last.get("response_position_ids", []))
            self.response_loss_mask = list(last.get("response_loss_mask", []))
            if len(self.input_ids) > self.max_model_len:
                raise RolloutLengthExceededError(
                    f"request {self.request_id} active context length {len(self.input_ids)} "
                    f"exceeds max_model_len={self.max_model_len}"
                )
        else:
            self.response_ids = self.input_ids[len(self.prompt_ids) :]
            self.truncate_output_ids(tokenizer)
        self._update_archive_token_count(tokenizer)
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
