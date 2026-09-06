"""Utilities for representing a context-cleaned rollout as trainable fragments.

The rollout worker keeps an append-only transcript and records the exact native
prompt/response ids used before each cleanup.  This module converts those
records into ordinary fixed-width ``DataProto`` rows for old-log-prob,
reference-log-prob, and actor update code.  It deliberately does not create a
new trajectory: all rows retain the source UID and terminal metadata.
"""

from __future__ import annotations

from copy import deepcopy
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from tensordict import TensorDict

from verl import DataProto
from verl.protocol import make_1d_object_array


_ROLLOUT_TENSOR_KEYS = {
    "prompts",
    "responses",
    "input_ids",
    "attention_mask",
    "position_ids",
    "loss_mask",
    "response_mask",
    "turn_boundaries",
    "token_level_scores",
    "token_level_rewards",
    "advantages",
    "returns",
    "old_log_probs",
    "ref_log_prob",
    "entropys",
}


def _plain(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def _records_for_row(value: Any) -> list[dict[str, Any]]:
    value = _plain(value)
    if isinstance(value, Mapping):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def has_segmented_rollouts(data: DataProto) -> bool:
    """Return whether the rollout output contains the fragment ledger."""
    raw = data.non_tensor_batch.get("segment_records")
    if raw is None:
        return False
    return any(bool(_records_for_row(item)) for item in raw)


def _as_int_list(value: Any, default_length: int = 0, default_start: int = 0) -> list[int]:
    value = _plain(value)
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    return list(range(default_start, default_start + max(0, int(default_length))))


def _as_mask(value: Any, length: int, default: int = 1) -> list[int]:
    value = _plain(value)
    if isinstance(value, (list, tuple)) and len(value) == length:
        return [int(item) for item in value]
    return [int(default)] * length


def _record_layout(record: Mapping[str, Any], *, max_model_len: int, pad_token_id: int) -> dict[str, Any]:
    prompt_ids = _as_int_list(record.get("prompt_ids", []))
    response_ids = _as_int_list(record.get("response_ids", []))
    prompt_len = len(prompt_ids)
    response_len = len(response_ids)
    actual_len = prompt_len + response_len
    if actual_len <= 0:
        raise ValueError("a segmented rollout record cannot be empty")
    if actual_len > max_model_len:
        raise ValueError(
            f"segmented rollout input has {actual_len} tokens, exceeds "
            f"max_model_len={max_model_len}"
        )

    prompt_attention = _as_mask(record.get("prompt_attention_mask"), prompt_len)
    response_attention = _as_mask(record.get("response_attention_mask"), response_len)
    prompt_loss = _as_mask(record.get("prompt_loss_mask"), prompt_len, default=0)
    response_loss = _as_mask(record.get("response_loss_mask"), response_len, default=0)
    prompt_positions = _as_int_list(
        record.get("prompt_position_ids"), prompt_len, default_start=0
    )
    response_positions = _as_int_list(
        record.get("response_position_ids"), response_len, default_start=prompt_len
    )
    if len(prompt_positions) != prompt_len:
        prompt_positions = list(range(prompt_len))
    if len(response_positions) != response_len:
        response_positions = list(range(prompt_len, actual_len))

    left_pad = max_model_len - actual_len
    input_ids = [pad_token_id] * left_pad + prompt_ids + response_ids
    attention_mask = [0] * left_pad + prompt_attention + response_attention
    position_ids = [0] * left_pad + prompt_positions + response_positions
    loss_mask = [0] * left_pad + prompt_loss + response_loss

    prompts = [pad_token_id] * max_model_len
    prompts[left_pad : left_pad + prompt_len] = prompt_ids
    responses = [pad_token_id] * max_model_len
    response_start = left_pad + prompt_len
    responses[response_start : response_start + response_len] = response_ids
    response_mask = [0] * max_model_len
    response_mask[response_start : response_start + response_len] = response_attention

    turn_boundaries = [0] * max_model_len
    boundaries = _plain(record.get("turn_boundaries", []))
    if not isinstance(boundaries, (list, tuple)):
        boundaries = []
    for boundary in boundaries:
        boundary = int(boundary)
        if 0 <= boundary < response_len:
            turn_boundaries[response_start + boundary] = 1
    if response_len and not any(turn_boundaries) and any(response_attention):
        first_response = next(
            index for index, value in enumerate(response_attention) if value
        )
        turn_boundaries[response_start + first_response] = 1

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "loss_mask": loss_mask,
        "prompts": prompts,
        "responses": responses,
        "segment_response_mask": response_mask,
        "turn_boundaries": turn_boundaries,
        "response_start": response_start,
        "response_end": response_start + response_len,
        "actual_input_ids": prompt_ids + response_ids,
        "actual_input_tokens": actual_len,
        "target_positions": [
            response_start + index
            for index, value in enumerate(response_loss)
            if int(value)
        ],
    }


def _fallback_record(data: DataProto, row: int) -> dict[str, Any]:
    """Represent a quarantined/legacy row without inventing trainable tokens."""
    input_ids = data.batch["input_ids"][row].detach().cpu().tolist()
    attention = data.batch["attention_mask"][row].detach().cpu().tolist()
    position_ids = data.batch["position_ids"][row].detach().cpu().tolist()
    loss_mask = data.batch["loss_mask"][row].detach().cpu().tolist()
    prompt = data.batch["prompts"][row].detach().cpu().tolist()
    response = data.batch["responses"][row].detach().cpu().tolist()
    prompt_mask = data.batch["attention_mask"][row, : len(prompt)].detach().cpu().tolist()
    response_mask = data.batch["attention_mask"][row, -len(response) :].detach().cpu().tolist() if len(response) else []
    response_loss = data.batch["loss_mask"][row, -len(response) :].detach().cpu().tolist() if len(response) else []
    prompt_positions = position_ids[: len(prompt)]
    response_positions = position_ids[-len(response) :] if len(response) else []
    return {
        "prompt_ids": prompt,
        "prompt_attention_mask": prompt_mask,
        "prompt_position_ids": prompt_positions,
        "prompt_loss_mask": [0] * len(prompt),
        "response_ids": response,
        "response_attention_mask": response_mask,
        "response_position_ids": response_positions,
        "response_loss_mask": response_loss,
        "turn_boundaries": [],
        "global_turn_indices": [],
        "conversation_history": [],
        "actual_input_ids": input_ids,
        "actual_input_tokens": int(sum(attention)),
        "_legacy_position_ids": position_ids,
        "_legacy_loss_mask": loss_mask,
    }


def _repeat_tensor_rows(data: DataProto, source_rows: list[int]) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for key in data.batch.keys():
        if key in _ROLLOUT_TENSOR_KEYS:
            continue
        value = data.batch[key]
        pieces = []
        for row in source_rows:
            pieces.append(value[row].clone())
        if pieces:
            tensors[key] = torch.stack(pieces, dim=0)
    return tensors


def _expanded_reward_tensor(
    data: DataProto,
    reward_tensor: torch.Tensor,
    layouts: list[dict[str, Any]],
    source_rows: list[int],
    record_indices: list[int],
    record_counts: list[int],
    max_model_len: int,
) -> torch.Tensor:
    output = torch.zeros(
        (len(layouts), max_model_len),
        device=reward_tensor.device,
        dtype=reward_tensor.dtype,
    )
    original_mask = data.batch.get("response_mask")
    for expanded_row, (source_row, record_index, record_count) in enumerate(
        zip(source_rows, record_indices, record_counts)
    ):
        source_values = reward_tensor[source_row].reshape(-1)
        # TravelGym emits one terminal scalar at the final valid response
        # position.  Put the complete scalar on the final fragment only, so
        # splitting cannot duplicate terminal reward mass.
        if record_index == record_count - 1:
            target_positions = [
                index
                for index, value in enumerate(layouts[expanded_row]["segment_response_mask"])
                if int(value)
            ]
            if target_positions:
                output[expanded_row, target_positions[-1]] = source_values.sum()
        # Preserve a genuinely token-level reward when its active width is
        # exactly the source response width.  This path is not used by the
        # terminal TravelGym manager but keeps the adapter honest for generic
        # reward managers.
        if (
            record_count == 1
            and source_values.numel() == max_model_len
            and original_mask is not None
        ):
            output[expanded_row] = source_values.to(output.device, output.dtype)
    return output


def expand_segmented_batch(
    data: DataProto,
    reward_tensor: torch.Tensor,
    reward_extra_infos_dict: Mapping[str, list] | None,
    *,
    max_model_len: int,
    pad_token_id: int,
) -> tuple[DataProto, torch.Tensor, dict[str, list]]:
    """Expand one rollout row into its exact native-context fragments.

    The first axis is expanded only after reward calculation.  Every output
    fragment carries the original ``uid`` and reward metadata, while the
    terminal reward is written to the final fragment once.  No response
    target is copied into another fragment because each record was finalized
    at a cleanup boundary by the rollout ledger.
    """
    if not has_segmented_rollouts(data):
        return data, reward_tensor, dict(reward_extra_infos_dict or {})

    batch_size = len(data)
    raw_records = data.non_tensor_batch["segment_records"]
    source_rows: list[int] = []
    record_indices: list[int] = []
    records_per_row: list[list[dict[str, Any]]] = []
    record_counts: list[int] = []
    layouts: list[dict[str, Any]] = []
    chosen_records: list[dict[str, Any]] = []
    for row in range(batch_size):
        records = _records_for_row(raw_records[row])
        if not records:
            records = [_fallback_record(data, row)]
        records_per_row.append(records)
        for record_index, record in enumerate(records):
            layout = _record_layout(
                record,
                max_model_len=max_model_len,
                pad_token_id=pad_token_id,
            )
            source_rows.append(row)
            record_indices.append(record_index)
            record_counts.append(len(records))
            chosen_records.append(record)
            layouts.append(layout)

    source_input = data.batch["input_ids"]
    source_attention = data.batch["attention_mask"]
    source_position = data.batch["position_ids"]
    source_loss = data.batch["loss_mask"]
    device = source_input.device
    id_dtype = source_input.dtype
    mask_dtype = source_attention.dtype
    position_dtype = source_position.dtype
    loss_dtype = source_loss.dtype

    def tensor_rows(name: str, dtype: torch.dtype) -> torch.Tensor:
        return torch.tensor(
            [layout[name] for layout in layouts],
            device=device,
            dtype=dtype,
        )

    tensors = _repeat_tensor_rows(data, source_rows)
    tensors.update(
        {
            "prompts": tensor_rows("prompts", id_dtype),
            "responses": tensor_rows("responses", id_dtype),
            "input_ids": tensor_rows("input_ids", id_dtype),
            "attention_mask": tensor_rows("attention_mask", mask_dtype),
            "position_ids": tensor_rows("position_ids", position_dtype),
            "loss_mask": tensor_rows("loss_mask", loss_dtype),
            "segment_response_mask": tensor_rows("segment_response_mask", mask_dtype),
            "turn_boundaries": tensor_rows("turn_boundaries", mask_dtype),
        }
    )
    expanded_reward = _expanded_reward_tensor(
        data,
        reward_tensor,
        layouts,
        source_rows,
        record_indices,
        record_counts,
        max_model_len,
    )

    non_tensor: dict[str, np.ndarray] = {}
    for key, value in data.non_tensor_batch.items():
        if key in {"segment_records", "segment_record", "segment_conversation_histories"}:
            continue
        values = []
        for row in source_rows:
            values.append(deepcopy(value[row]))
        non_tensor[key] = make_1d_object_array(values)

    segment_records = []
    segment_histories = []
    trajectory_ids = []
    segment_ids = []
    segment_counts = []
    segment_is_last = []
    for expanded_row, (source_row, record_index, record) in enumerate(
        zip(source_rows, record_indices, chosen_records)
    ):
        segment_records.append([deepcopy(record)])
        segment_histories.append(deepcopy(record.get("conversation_history", [])))
        original_uid = (
            data.non_tensor_batch.get("uid", np.arange(batch_size, dtype=object))[source_row]
        )
        trajectory_ids.append(f"{original_uid}::rollout_row_{source_row}")
        segment_ids.append(int(record.get("segment_id", record_index)))
        segment_counts.append(len(records_per_row[source_row]))
        segment_is_last.append(record_index == len(records_per_row[source_row]) - 1)
    non_tensor["segment_records"] = make_1d_object_array(segment_records)
    non_tensor["segment_conversation_histories"] = make_1d_object_array(segment_histories)
    non_tensor["segment_trajectory_uid"] = np.asarray(trajectory_ids, dtype=object)
    non_tensor["segment_id"] = np.asarray(segment_ids, dtype=np.int64)
    non_tensor["segment_count"] = np.asarray(segment_counts, dtype=np.int64)
    non_tensor["segment_is_last"] = np.asarray(segment_is_last, dtype=bool)
    non_tensor["segment_source_row"] = np.asarray(source_rows, dtype=np.int64)

    meta_info = deepcopy(data.meta_info)
    meta_info["travel_segmented_rollouts"] = True
    meta_info["travel_segment_count"] = int(len(layouts))
    meta_info["travel_segment_source_rollout_count"] = int(batch_size)
    meta_info["travel_segment_actual_input_max_tokens"] = int(
        max(layout["actual_input_tokens"] for layout in layouts)
    )
    expanded = DataProto(
        batch=TensorDict(source=tensors, batch_size=len(layouts)),
        non_tensor_batch=non_tensor,
        meta_info=meta_info,
    )

    expanded_extras: dict[str, list] = {}
    for key, values in (reward_extra_infos_dict or {}).items():
        expanded_extras[key] = [
            deepcopy(values[row]) for row in source_rows
        ]
    return expanded, expanded_reward, expanded_extras


__all__ = ["expand_segmented_batch", "has_segmented_rollouts"]
