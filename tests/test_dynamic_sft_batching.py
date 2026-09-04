from __future__ import annotations

import unittest

import torch

from verl.utils.dataset.dynamic_sft_batching import (
    DynamicPaddingCollator,
    LengthGroupedDistributedSampler,
)


def _feature(length: int, *, weight: float = 1.0) -> dict[str, torch.Tensor]:
    input_ids = torch.arange(1, length + 1, dtype=torch.long)
    attention_mask = torch.ones(length, dtype=torch.long)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": torch.arange(length, dtype=torch.long),
        "loss_mask": attention_mask.clone(),
        "reasoning_loss_mask": torch.zeros(length, dtype=torch.long),
        "tool_call_loss_mask": attention_mask.clone(),
        "sample_weight": torch.tensor(weight, dtype=torch.float32),
    }


class DynamicPaddingTests(unittest.TestCase):
    def test_right_pads_only_to_rounded_batch_maximum(self):
        collator = DynamicPaddingCollator(
            pad_token_id=99,
            pad_to_multiple_of=4,
            max_length=32,
        )
        batch = collator([_feature(3, weight=0.5), _feature(5)])
        self.assertEqual(tuple(batch["input_ids"].shape), (2, 8))
        self.assertEqual(batch["input_ids"][0].tolist(), [1, 2, 3, 99, 99, 99, 99, 99])
        self.assertEqual(batch["attention_mask"][0].tolist(), [1, 1, 1, 0, 0, 0, 0, 0])
        self.assertEqual(batch["position_ids"][1].tolist(), [0, 1, 2, 3, 4, 0, 0, 0])
        self.assertEqual(batch["loss_mask"][1].tolist(), [1, 1, 1, 1, 1, 0, 0, 0])
        self.assertEqual(batch["sample_weight"].tolist(), [0.5, 1.0])

    def test_rejects_overlength_sequence(self):
        collator = DynamicPaddingCollator(pad_token_id=0, max_length=4)
        with self.assertRaises(ValueError):
            collator([_feature(5)])


class LengthGroupedSamplerTests(unittest.TestCase):
    def test_single_rank_covers_each_sample_once_and_groups_lengths(self):
        lengths = [1, 100, 2, 99, 3, 98, 4, 97]
        sampler = LengthGroupedDistributedSampler(
            lengths,
            local_batch_size=2,
            seed=7,
            mega_batch_mult=4,
        )
        indices = list(sampler)
        self.assertEqual(sorted(indices), list(range(len(lengths))))
        batches = [indices[start : start + 2] for start in range(0, len(indices), 2)]
        self.assertTrue(all(abs(lengths[left] - lengths[right]) <= 1 for left, right in batches))

    def test_multi_rank_tail_is_even_and_deterministic(self):
        lengths = [3, 7, 11, 15, 19]
        samplers = [
            LengthGroupedDistributedSampler(
                lengths,
                local_batch_size=1,
                num_replicas=2,
                rank=rank,
                seed=13,
            )
            for rank in range(2)
        ]
        rank_indices = [list(sampler) for sampler in samplers]
        self.assertEqual([len(values) for values in rank_indices], [3, 3])
        self.assertTrue(set(range(len(lengths))).issubset(set(rank_indices[0] + rank_indices[1])))
        self.assertEqual(rank_indices, [list(sampler) for sampler in samplers])
        for sampler in samplers:
            sampler.set_epoch(1)
        self.assertNotEqual(rank_indices, [list(sampler) for sampler in samplers])


if __name__ == "__main__":
    unittest.main()
