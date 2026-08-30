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
from typing import Any, Dict, List, Optional

import torch
from pydantic import BaseModel, model_validator
from transformers import PreTrainedTokenizer

from verl.tools.schemas import OpenAIFunctionToolCall, OpenAIFunctionToolSchema
from verl.utils.model import compute_position_id_with_mask

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

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
        kwargs.pop("enable_thinking", None)
        value = tokenizer.apply_chat_template(messages, **kwargs)
    if isinstance(value, torch.Tensor):
        value = value.tolist()
    elif hasattr(value, "tolist") and not isinstance(value, list):
        value = value.tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return [int(token) for token in value]


def _suffix_after_prefix(full_ids: list[int], prefix_ids: list[int], *, label: str) -> list[int]:
    """Return a native-template suffix, failing on any token mismatch."""
    if len(full_ids) < len(prefix_ids) or full_ids[: len(prefix_ids)] != prefix_ids:
        raise ValueError(f"native Qwen template prefix mismatch while appending {label}")
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
    metrics: Dict[str, List[Any]] = {}

    turn_boundaries: List[int] = []
    conversation_histories: List[Dict[str, Any]] = []

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
            tools = [
                tool.model_dump() if hasattr(tool, "model_dump") else dict(tool)
                for tool in tool_schemas
            ]
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
        completed_ids = _native_template_ids(
            tokenizer, self._message_dicts(), tools=tools, add_generation_prompt=False
        )
        generation_ids = _native_template_ids(
            tokenizer, self._message_dicts(), tools=tools, add_generation_prompt=True
        )
        # At this point input_ids should represent the completed transcript.
        # A prior caller may already have appended the generation prompt; in
        # that case leave it untouched.  Any other mismatch is unsafe.
        if self.input_ids == generation_ids:
            return generation_ids
        if self.input_ids != completed_ids:
            raise ValueError("request input_ids diverged from native Qwen3.5 transcript before generation")
        generation_suffix = _suffix_after_prefix(generation_ids, completed_ids, label="generation prompt")
        if generation_suffix:
            self._update_input_ids(generation_suffix, attention_mask=True, loss_mask=False)
        return generation_ids

    # Legacy async workers use the singular method name.  Keep it as a thin
    # alias so both rollout implementations share the native-template path.
    def get_generation_prompt(self, tokenizer: PreTrainedTokenizer) -> list[int]:
        return self.get_generation_prompt_ids(tokenizer)

    def add_assistant_message(
        self,
        tokenizer: PreTrainedTokenizer,
        content: str,
        tool_calls: Optional[List[OpenAIFunctionToolCall]] = None,
        reasoning_content: Optional[str] = None,
        already_over_long: bool = False,
        format: Optional[str] = None,
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
        del already_over_long, format  # compatibility flags; native template is authoritative
        previous_messages = self._message_dicts()
        tools = self._tool_dicts()
        # ``input_ids`` includes the generation prompt while the engine is
        # producing this Assistant turn.  Render that exact prefix before
        # appending the completed Assistant message.
        generation_prefix = _native_template_ids(
            tokenizer, previous_messages, tools=tools, add_generation_prompt=True
        )
        if self.input_ids != generation_prefix:
            raise ValueError("request input_ids diverged before appending an assistant turn")
        self.messages.append(
            Message(
                role="assistant",
                content=raw_content,
                tool_calls=tool_calls,
                reasoning_content=reasoning_content,
            )
        )
        full_ids = _native_template_ids(tokenizer, self._message_dicts(), tools=tools, add_generation_prompt=False)
        content_ids = _suffix_after_prefix(full_ids, generation_prefix, label="assistant")
        self._update_input_ids(content_ids, attention_mask=True, loss_mask=True)

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
        completed_prefix = _native_template_ids(
            tokenizer, previous_messages, tools=tools, add_generation_prompt=False
        )
        if self.input_ids != completed_prefix:
            raise ValueError("request input_ids diverged before appending a tool response")
        if tool_call_ids is not None and len(tool_call_ids) != len(contents):
            raise ValueError("tool_call_ids must align with tool response contents")
        if names is not None and len(names) != len(contents):
            raise ValueError("tool response names must align with contents")
        self.messages.extend(
            [
                Message(
                    role="tool",
                    content=str(content),
                    tool_call_id=str(tool_call_ids[index]) if tool_call_ids is not None else None,
                    name=str(names[index]) if names is not None else None,
                )
                for index, content in enumerate(contents)
            ]
        )
        full_ids = _native_template_ids(tokenizer, self._message_dicts(), tools=tools, add_generation_prompt=False)
        content_ids = _suffix_after_prefix(full_ids, completed_prefix, label="tool response")
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
            raise ValueError(
                f"request {self.request_id} length {len(self.input_ids)} exceeds "
                f"max_model_len={self.max_model_len}; truncation=error"
            )
        response_start = len(self.prompt_ids)
        response_end = len(self.input_ids)
        response_length = response_end - response_start
        if response_length > self.max_response_len:
            raise ValueError(
                f"request {self.request_id} response length {response_length} exceeds "
                f"max_response_len={self.max_response_len}; truncation=error"
            )
        # Keep the explicit assignments for callers that expect these views;
        # no slicing is performed because the bounds were checked above.
        self.response_ids = self.input_ids[response_start:response_end]
        self.response_attention_mask = self.attention_mask[len(self.prompt_attention_mask):]
        self.response_position_ids = self.position_ids[len(self.prompt_position_ids):]
        self.response_loss_mask = self.loss_mask[len(self.prompt_loss_mask):]
