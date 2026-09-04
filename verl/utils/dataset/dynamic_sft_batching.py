"""Length-aware batching helpers for variable-length trajectory SFT."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch.utils.data import Sampler


SEQUENCE_FIELDS = (
    "input_ids",
    "attention_mask",
    "position_ids",
    "loss_mask",
    "reasoning_loss_mask",
    "tool_call_loss_mask",
)


class DynamicPaddingCollator:
    """Right-pad a batch only to its longest sequence.

    Rounding to a small kernel-friendly multiple preserves the original token
    stream and masks while avoiding the previous unconditional 32K padding.
    """

    def __init__(self, *, pad_token_id: int, pad_to_multiple_of: int = 128, max_length: int = 32768):
        if pad_to_multiple_of < 1:
            raise ValueError("pad_to_multiple_of must be positive")
        self.pad_token_id = int(pad_token_id)
        self.pad_to_multiple_of = int(pad_to_multiple_of)
        self.max_length = int(max_length)

    def __call__(self, features: Sequence[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
        if not features:
            raise ValueError("cannot collate an empty SFT batch")
        lengths = [int(feature["input_ids"].shape[-1]) for feature in features]
        longest = max(lengths)
        if longest > self.max_length:
            raise ValueError(f"batch contains sequence length {longest} > max_length {self.max_length}")
        padded_length = min(
            self.max_length,
            int(math.ceil(longest / self.pad_to_multiple_of) * self.pad_to_multiple_of),
        )

        batch: dict[str, torch.Tensor] = {}
        for field in SEQUENCE_FIELDS:
            values = [feature[field] for feature in features]
            pad_value = self.pad_token_id if field == "input_ids" else 0
            output = torch.full(
                (len(values), padded_length),
                pad_value,
                dtype=values[0].dtype,
            )
            for row, value in enumerate(values):
                output[row, : value.shape[-1]] = value
            batch[field] = output

        batch["sample_weight"] = torch.stack(
            [torch.as_tensor(feature["sample_weight"], dtype=torch.float32).reshape(()) for feature in features]
        )
        return batch


class LengthGroupedDistributedSampler(Sampler[int]):
    """Shuffle trajectories while keeping each global batch length-homogeneous.

    The same global batches are constructed on every rank and then sharded into
    contiguous per-rank slices.  A multi-rank final partial batch is padded with
    repeated indices so all ranks execute the same number of collectives; a
    single-rank run keeps the final batch partial and never repeats an example.
    """

    def __init__(
        self,
        lengths: Sequence[int],
        *,
        local_batch_size: int,
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 0,
        shuffle: bool = True,
        mega_batch_mult: int = 50,
    ) -> None:
        if local_batch_size < 1:
            raise ValueError("local_batch_size must be positive")
        if num_replicas < 1 or not 0 <= rank < num_replicas:
            raise ValueError("invalid distributed sampler rank/world size")
        if mega_batch_mult < 1:
            raise ValueError("mega_batch_mult must be positive")
        self.lengths = [int(value) for value in lengths]
        self.local_batch_size = int(local_batch_size)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.mega_batch_mult = int(mega_batch_mult)
        self.epoch = 0

    @property
    def global_batch_size(self) -> int:
        return self.local_batch_size * self.num_replicas

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _ordered_indices(self) -> list[int]:
        count = len(self.lengths)
        if not self.shuffle:
            return list(range(count))
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        return torch.randperm(count, generator=generator).tolist()

    def __iter__(self):
        indices = self._ordered_indices()
        if not indices:
            return iter(())

        global_batch_size = self.global_batch_size
        window_size = global_batch_size * self.mega_batch_mult
        full_batches: list[list[int]] = []
        partial_batch: list[int] | None = None
        for start in range(0, len(indices), window_size):
            window = indices[start : start + window_size]
            window.sort(key=lambda index: self.lengths[index], reverse=True)
            for offset in range(0, len(window), global_batch_size):
                batch = window[offset : offset + global_batch_size]
                if len(batch) == global_batch_size:
                    full_batches.append(batch)
                elif partial_batch is not None:
                    partial_batch.extend(batch)
                else:
                    partial_batch = batch

        # Keep batch order random so length grouping does not become a
        # short-to-long curriculum.  Every rank uses the same permutation.
        if self.shuffle and len(full_batches) > 1:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch + 1_000_003)
            order = torch.randperm(len(full_batches), generator=generator).tolist()
            full_batches = [full_batches[index] for index in order]

        if partial_batch:
            if self.num_replicas > 1:
                source = indices
                cursor = 0
                while len(partial_batch) < global_batch_size:
                    partial_batch.append(source[cursor % len(source)])
                    cursor += 1
                full_batches.append(partial_batch)
                partial_batch = None

        rank_indices: list[int] = []
        rank_start = self.rank * self.local_batch_size
        rank_end = rank_start + self.local_batch_size
        for batch in full_batches:
            rank_indices.extend(batch[rank_start:rank_end])
        if partial_batch and self.num_replicas == 1:
            rank_indices.extend(partial_batch)
        return iter(rank_indices)

    def __len__(self) -> int:
        count = len(self.lengths)
        if self.num_replicas == 1:
            return count
        global_batches = math.ceil(count / self.global_batch_size)
        return global_batches * self.local_batch_size


__all__ = ["DynamicPaddingCollator", "LengthGroupedDistributedSampler"]
