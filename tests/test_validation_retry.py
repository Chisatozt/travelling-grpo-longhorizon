from __future__ import annotations

import numpy as np
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from verl import DataProto
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    _extract_terminal_reward_metadata,
)


def _input_batch(size: int) -> DataProto:
    return DataProto(
        batch=TensorDict(
            {"input_ids": torch.arange(size).reshape(size, 1)},
            batch_size=[size],
        ),
        non_tensor_batch={
            "task_id": np.array([f"task-{index}" for index in range(size)])
        },
    )


def _output_batch(values: list[int], valid: list[bool]) -> DataProto:
    size = len(values)
    return DataProto(
        batch=TensorDict(
            {"responses": torch.tensor(values, dtype=torch.long).reshape(size, 1)},
            batch_size=[size],
        ),
        non_tensor_batch={
            "reward_scores": np.array(
                [
                    {
                        "interact_with_env": 0.5 if row_valid else 0.0,
                        "interact_with_env_reward_valid": float(row_valid),
                        "interact_with_env_terminal_only": 1.0,
                    }
                    for row_valid in valid
                ],
                dtype=object,
            )
        },
    )


class _RolloutWorker:
    world_size = 1

    def __init__(self):
        self.calls: list[int] = []

    def generate_sequences(self, batch: DataProto) -> DataProto:
        self.calls.append(len(batch))
        if len(self.calls) == 1:
            return _output_batch([10, 20, 30], [True, False, False])
        if len(self.calls) == 2:
            return _output_batch([21, 31], [True, False])
        return _output_batch([32], [True])


def _trainer(retries: int) -> RayPPOTrainer:
    trainer = object.__new__(RayPPOTrainer)
    trainer.config = OmegaConf.create(
        {"trainer": {"validation_retry_attempts": retries}}
    )
    trainer.actor_rollout_wg = _RolloutWorker()
    trainer.async_rollout_mode = False
    return trainer


def test_validation_retries_only_invalid_rows_and_restores_input_order():
    trainer = _trainer(retries=2)

    output, stats = trainer._generate_validation_batch_with_retries(
        _input_batch(3)
    )

    assert trainer.actor_rollout_wg.calls == [3, 2, 1]
    assert output.batch["responses"].reshape(-1).tolist() == [10, 21, 32]
    assert stats == {
        "retried_rows": 3,
        "recovered_rows": 2,
        "final_invalid_rows": 0,
    }


def test_validation_keeps_invalid_row_after_retry_budget_is_exhausted():
    trainer = _trainer(retries=1)

    output, stats = trainer._generate_validation_batch_with_retries(
        _input_batch(3)
    )

    _, valid = _extract_terminal_reward_metadata(output)
    assert valid.tolist() == [True, True, False]
    assert stats == {
        "retried_rows": 2,
        "recovered_rows": 1,
        "final_invalid_rows": 1,
    }
