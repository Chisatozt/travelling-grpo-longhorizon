# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2025 ModelBest Inc. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Multi-turn SFT dataset that supports training on conversation data with multiple turns
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
try:
    import pandas as pd
except ImportError:  # keep mask helpers importable in CPU-only test envs
    pd = None
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from verl.utils import hf_tokenizer
from verl.utils.fs import copy_local_path_from_hdfs
from sft.qwen35_mask import exact_assistant_span_masks, exact_assistant_token_mask, native_template_ids


def convert_nested_value_to_list_recursive(data_item):
    if isinstance(data_item, dict):
        return {k: convert_nested_value_to_list_recursive(v) for k, v in data_item.items()}
    elif isinstance(data_item, list):
        return [convert_nested_value_to_list_recursive(elem) for elem in data_item]
    elif isinstance(data_item, np.ndarray):
        # Convert to list, then recursively process the elements of the new list
        return convert_nested_value_to_list_recursive(data_item.tolist())
    else:
        # Base case: item is already a primitive type (int, str, float, bool, etc.)
        return data_item


def message_has_tool_call(message: Dict[str, Any]) -> bool:
    """Return whether an assistant message contains a tool invocation."""
    if message.get("tool_calls"):
        return True
    content = message.get("content")
    if isinstance(content, str):
        return "<tool_call>" in content or '"interact_with_env"' in content
    return False


class MultiTurnSFTDataset(Dataset):
    """
    Dataset for multi-turn conversations where each assistant response should be trained
    """

    def __init__(self, parquet_files: Union[str, List[str]], tokenizer, config=None):
        # Set defaults and extract parameters from config if provided
        config = config or {}
        self.truncation = config.get("truncation", "error")
        self.max_length = int(config.get("max_length", 32768))
        # Get messages_key from the new multiturn config structure
        multiturn_config = config.get("multiturn", {})
        self.messages_key = multiturn_config.get("messages_key", "messages")
        self.tools_key = multiturn_config.get("tools_key", "tools")
        self.enable_thinking_key = multiturn_config.get("enable_thinking_key", "enable_thinking")
        self.assistant_train_mask_key = multiturn_config.get("assistant_train_mask_key", "assistant_train_mask")
        self.sample_weight_key = multiturn_config.get("sample_weight_key", "sample_weight")
        self.require_assistant_train_mask = bool(multiturn_config.get("require_assistant_train_mask", False))
        self.dynamic_padding = bool(multiturn_config.get("dynamic_padding", False))
        self.length_bucketing = bool(multiturn_config.get("length_bucketing", False))
        self._sequence_lengths: list[int] | None = None
        # Legacy fallback for non-Travel datasets. Canonical Travel records use
        # the explicit per-message mask instead.
        self.train_last_assistant_only = bool(multiturn_config.get("train_last_assistant_only", False))
        assert self.truncation in ["error", "left", "right"]

        if not isinstance(parquet_files, List):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer

        self._download()
        self._read_files_and_process()

    def _download(self):
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_local_path_from_hdfs(parquet_file, verbose=True)

    def _read_files_and_process(self):
        if pd is None:
            raise ImportError("pandas is required to load MultiTurnSFTDataset files")
        def series_to_item(ls):
            import numpy
            import pandas

            while isinstance(ls, (pandas.core.series.Series, numpy.ndarray)) and len(ls) == 1:
                ls = ls[0]
            return ls

        dataframes = []
        for parquet_file in self.parquet_files:
            path = str(parquet_file)
            if path.lower().endswith((".json", ".jsonl")):
                import json

                with open(path, "r", encoding="utf-8") as handle:
                    if path.lower().endswith(".jsonl"):
                        rows = [json.loads(line) for line in handle if line.strip()]
                    else:
                        rows = json.load(handle)
                if isinstance(rows, dict):
                    rows = rows.get("data", [])
                dataframe = pd.DataFrame(rows if isinstance(rows, list) else [])
            else:
                dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)

        # Extract messages list from dataframe
        self.messages = self.dataframe[self.messages_key].apply(series_to_item).tolist()

        # Extract tools list from dataframe
        if self.tools_key in self.dataframe.columns:
            self.tools = self.dataframe[self.tools_key].apply(convert_nested_value_to_list_recursive).tolist()
        else:
            self.tools = None
        # Extract enable_thinking list from dataframe
        if self.enable_thinking_key in self.dataframe.columns:
            self.enable_thinking = self.dataframe[self.enable_thinking_key].tolist()
        else:
            self.enable_thinking = None
        if self.assistant_train_mask_key in self.dataframe.columns:
            self.assistant_train_mask = self.dataframe[self.assistant_train_mask_key].apply(series_to_item).tolist()
        else:
            self.assistant_train_mask = None
        if self.sample_weight_key in self.dataframe.columns:
            self.sample_weight = self.dataframe[self.sample_weight_key].tolist()
        elif "trainer_metadata" in self.dataframe.columns:
            metadata = self.dataframe["trainer_metadata"].apply(series_to_item).tolist()
            self.sample_weight = [
                float(item.get("sample_weight", 1.0)) if isinstance(item, dict)
                else float(__import__("json").loads(item).get("sample_weight", 1.0)) if isinstance(item, str) and item.strip().startswith("{")
                else 1.0
                for item in metadata
            ]
        else:
            self.sample_weight = [1.0] * len(self.messages)

    def __len__(self):
        return len(self.messages)

    def _template_ids(self, messages, *, add_generation_prompt=False, tools=None, enable_thinking=True):
        """Tokenize the native tokenizer template without text slicing."""
        return native_template_ids(
            self.tokenizer,
            messages,
            tools=tools,
            enable_thinking=bool(enable_thinking),
            add_generation_prompt=add_generation_prompt,
        )

    def get_sequence_lengths(self) -> list[int]:
        """Return exact native-template lengths for length-grouped sampling."""

        if self._sequence_lengths is None:
            lengths = []
            for item, messages in enumerate(self.messages):
                tools = self.tools[item] if self.tools is not None else None
                enable_thinking = self.enable_thinking[item] if self.enable_thinking is not None else True
                lengths.append(len(self._template_ids(
                    list(messages), tools=tools, enable_thinking=enable_thinking,
                )))
            self._sequence_lengths = lengths
        return list(self._sequence_lengths)

    @staticmethod
    def _starts_with(sequence: list[int], prefix: list[int]) -> bool:
        return len(sequence) >= len(prefix) and sequence[: len(prefix)] == prefix

    @staticmethod
    def _longest_common_prefix(left: list[int], right: list[int]) -> int:
        length = 0
        for first, second in zip(left, right):
            if first != second:
                break
            length += 1
        return length

    def _assistant_token_spans(self, messages, full_ids, tools, enable_thinking, message_mask):
        """Build exact loss spans from token streams emitted by the native template.

        This intentionally avoids character-prefix subtraction.  Prefix and
        end streams are tokenized independently, then aligned to the official
        full input IDs.  Thus Qwen3.5's special tokens, reasoning markers and
        structured tool calls are treated exactly as they are in inference.
        """
        return exact_assistant_token_mask(
            messages,
            full_ids,
            lambda prefix, generation: self._template_ids(
                prefix, add_generation_prompt=generation, tools=tools, enable_thinking=enable_thinking
            ),
            message_mask,
        )

    def _assistant_token_masks(self, messages, full_ids, tools, enable_thinking, message_mask):
        """Return the authoritative loss mask plus diagnostic span masks."""

        return exact_assistant_span_masks(
            messages,
            full_ids,
            lambda prefix, generation: self._template_ids(
                prefix, add_generation_prompt=generation, tools=tools, enable_thinking=enable_thinking
            ),
            message_mask,
        )

    def _process_message_tokens(
        self,
        messages: List[Dict[str, Any]],
        start_idx: int,
        end_idx: int,
        is_assistant: bool = False,
        enable_thinking: Optional[bool] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        Process tokens for a single message or a group of messages.

        Args:
            messages: List of message dictionaries
            start_idx: Start index in messages list
            end_idx: End index in messages list
            is_assistant: Whether this is an assistant message
            enable_thinking: Whether to enable thinking mode

        Returns:
            Tuple of (tokens, loss_mask, attention_mask)
        """
        if not 0 <= start_idx <= end_idx <= len(messages):
            raise ValueError(f"invalid message span: {start_idx=} {end_idx=}")

        # Keep this compatibility helper on the same token-level path as the
        # canonical dataset.  In particular, never subtract character
        # offsets from ``apply_chat_template(..., tokenize=False)``: Qwen3.5
        # special/reasoning/tool markers are not guaranteed to have stable
        # character prefixes.  The returned span is aligned to the native
        # token stream instead.
        completed_prefix = self._template_ids(
            messages[:start_idx],
            add_generation_prompt=False,
            tools=tools,
            enable_thinking=enable_thinking,
        )
        end_ids = self._template_ids(
            messages[:end_idx],
            add_generation_prompt=False,
            tools=tools,
            enable_thinking=enable_thinking,
        )

        if is_assistant:
            generation_prefix = self._template_ids(
                messages[:start_idx],
                add_generation_prompt=True,
                tools=tools,
                enable_thinking=enable_thinking,
            )
            generation_prefix_start = self._longest_common_prefix(generation_prefix, completed_prefix)
            if generation_prefix_start != len(completed_prefix):
                raise ValueError("native Qwen3.5 generation prefix is not token-aligned")
            generation_prompt_tokens = generation_prefix[len(completed_prefix) :]
            end_start = self._longest_common_prefix(end_ids, generation_prefix)
            if end_start != len(generation_prefix):
                raise ValueError("native Qwen3.5 assistant span is not token-aligned")
            content_tokens = end_ids[end_start:]
            message_tokens = list(generation_prompt_tokens) + list(content_tokens)
            loss_mask = [0] * len(generation_prompt_tokens) + [1] * len(content_tokens)
        else:
            end_start = self._longest_common_prefix(end_ids, completed_prefix)
            if end_start != len(completed_prefix):
                raise ValueError("native Qwen3.5 message span is not token-aligned")
            message_tokens = end_ids[end_start:]
            loss_mask = [0] * len(message_tokens)

        attention_mask = [1] * len(message_tokens)
        return message_tokens, loss_mask, attention_mask

    def _validate_and_convert_tokens(
        self,
        full_tokens: torch.Tensor,
        concat_tokens: List[int],
        concat_loss_mask: List[int],
        concat_attention_mask: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Validate tokenization and convert to tensors.

        Args:
            full_tokens: Full conversation tokens
            concat_tokens: Concatenated tokens
            concat_loss_mask: Concatenated loss mask
            concat_attention_mask: Concatenated attention mask

        Returns:
            Tuple of (input_ids, loss_mask, attention_mask) as tensors
        """
        full_tokens_list = full_tokens.tolist()

        if len(concat_tokens) != len(full_tokens_list) or not all(a == b for a, b in zip(concat_tokens, full_tokens_list)):
            logging.warning(
                f"Token mismatch detected! Full tokenization length: {len(full_tokens_list)}, Concatenated tokens length: {len(concat_tokens)}. Using concatenated version."
                # f"full tokens text: {self.tokenizer.decode(full_tokens_list)}"
                # f"concat tokens text: {self.tokenizer.decode(concat_tokens)}"
            )
            return (
                torch.tensor(concat_tokens, dtype=torch.long),
                torch.tensor(concat_loss_mask, dtype=torch.long),
                torch.tensor(concat_attention_mask, dtype=torch.long),
            )

        return full_tokens, torch.tensor(concat_loss_mask, dtype=torch.long), torch.tensor(concat_attention_mask, dtype=torch.long)

    def __getitem__(self, item):
        messages = self.messages[item]
        if not isinstance(messages, list):
            messages = list(messages)
        tools = self.tools[item] if self.tools is not None else None
        # Travel canonical records always enable Qwen3.5 thinking.  A legacy
        # non-Travel caller can still provide a per-row value.
        enable_thinking = self.enable_thinking[item] if self.enable_thinking is not None else True
        if self.assistant_train_mask is not None:
            message_mask = self.assistant_train_mask[item]
            if hasattr(message_mask, "tolist"):
                message_mask = message_mask.tolist()
            message_mask = list(message_mask)
            if len(message_mask) != len(messages):
                raise ValueError(f"assistant_train_mask length {len(message_mask)} != messages length {len(messages)}")
        elif self.require_assistant_train_mask:
            raise ValueError("canonical Travel SFT row is missing assistant_train_mask")
        else:
            # Backwards-compatible fallback for old generic multi-turn data.
            message_mask = [
                1 if isinstance(message, dict) and message.get("role") == "assistant" else 0
                for message in messages
            ]
            if self.train_last_assistant_only:
                targets = [i for i, message in enumerate(messages) if isinstance(message, dict) and message.get("role") == "assistant"]
                target = targets[-1] if targets else None
                message_mask = [int(i == target) for i in range(len(messages))]

        full_ids = self._template_ids(messages, add_generation_prompt=False, tools=tools, enable_thinking=enable_thinking)
        token_mask, reasoning_token_mask, tool_call_token_mask = self._assistant_token_masks(
            messages, full_ids, tools, enable_thinking, message_mask
        )
        if not any(token_mask) and self.require_assistant_train_mask:
            raise ValueError("all-zero assistant_train_mask cannot enter SFT")
        input_ids = torch.tensor(full_ids, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        loss_mask = torch.tensor(token_mask, dtype=torch.long)
        reasoning_loss_mask = torch.tensor(reasoning_token_mask, dtype=torch.long)
        tool_call_loss_mask = torch.tensor(tool_call_token_mask, dtype=torch.long)
        sequence_length = input_ids.shape[0]
        if sequence_length > self.max_length:
            # Canonical TravelGym rows must never be silently cut: a native
            # Qwen3.5 reasoning/tool block is one indivisible trajectory and
            # truncating it would manufacture a transcript that the
            # environment never observed.  Legacy generic multi-turn data
            # may still opt into the historical left/right behavior when it
            # does not require the canonical message-aligned mask.
            if self.require_assistant_train_mask or self.truncation == "error":
                raise ValueError(f"{sequence_length=} is larger than {self.max_length=} (overlength_quarantine)")
            if self.truncation == "left":
                input_ids, attention_mask, loss_mask = input_ids[-self.max_length :], attention_mask[-self.max_length :], loss_mask[-self.max_length :]
                reasoning_loss_mask, tool_call_loss_mask = reasoning_loss_mask[-self.max_length :], tool_call_loss_mask[-self.max_length :]
            elif self.truncation == "right":
                input_ids, attention_mask, loss_mask = input_ids[: self.max_length], attention_mask[: self.max_length], loss_mask[: self.max_length]
                reasoning_loss_mask, tool_call_loss_mask = reasoning_loss_mask[: self.max_length], tool_call_loss_mask[: self.max_length]
            else:
                raise ValueError(f"Unknown truncation method {self.truncation}")
        elif sequence_length < self.max_length and not self.dynamic_padding:
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            pad = self.max_length - sequence_length
            input_ids = torch.cat((input_ids, torch.full((pad,), pad_token_id, dtype=input_ids.dtype)))
            attention_mask = torch.cat((attention_mask, torch.zeros((pad,), dtype=attention_mask.dtype)))
            loss_mask = torch.cat((loss_mask, torch.zeros((pad,), dtype=loss_mask.dtype)))
            reasoning_loss_mask = torch.cat((reasoning_loss_mask, torch.zeros((pad,), dtype=reasoning_loss_mask.dtype)))
            tool_call_loss_mask = torch.cat((tool_call_loss_mask, torch.zeros((pad,), dtype=tool_call_loss_mask.dtype)))
        position_ids = torch.arange(len(input_ids), dtype=torch.long) * attention_mask
        try:
            sample_weight = float(self.sample_weight[item])
        except (TypeError, ValueError, IndexError):
            sample_weight = 1.0
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
            "reasoning_loss_mask": reasoning_loss_mask,
            "tool_call_loss_mask": tool_call_loss_mask,
            "sample_weight": torch.tensor(sample_weight, dtype=torch.float32),
        }
