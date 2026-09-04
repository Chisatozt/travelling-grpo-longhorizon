from types import SimpleNamespace

import numpy as np
import pytest
import torch

from verl.trainer.ppo.metric_utils import compute_data_metrics, compute_reward_component_metrics


def test_compute_data_metrics_exports_user_simulator_usage():
    batch = SimpleNamespace(
        batch={
            "responses": torch.ones((2, 2), dtype=torch.long),
            "attention_mask": torch.ones((2, 4), dtype=torch.long),
            "token_level_scores": torch.zeros((2, 2)),
            "token_level_rewards": torch.zeros((2, 2)),
            "advantages": torch.zeros((2, 2)),
            "returns": torch.zeros((2, 2)),
        },
        non_tensor_batch={
            "user_api_calls": np.array([2.0, 3.0]),
            "user_api_errors": np.array([0.0, 1.0]),
            "user_retries": np.array([0.0, 1.0]),
            "user_cache_hits": np.array([1.0, 0.0]),
            "user_total_tokens": np.array([100.0, 200.0]),
            "user_wall_time_seconds": np.array([1.5, 2.5]),
        },
    )

    metrics = compute_data_metrics(batch, use_critic=False)

    assert metrics["user_simulator/api_calls/sum"] == 5.0
    assert metrics["user_simulator/api_calls/mean_per_trajectory"] == 2.5
    assert metrics["user_simulator/api_errors/sum"] == 1.0
    assert metrics["user_simulator/retries/sum"] == 1.0
    assert metrics["user_simulator/cache_hits/sum"] == 1.0
    assert metrics["user_simulator/total_tokens/sum"] == 300.0
    assert metrics["user_simulator/wall_time_seconds/sum"] == 4.0
    assert metrics["user_simulator/api_error_rate"] == pytest.approx(0.2)
    reward_metrics = compute_reward_component_metrics(
        {
            "reward": [0.5, 1.0],
            "correct_completion": [0.0, 1.0],
            "total_penalty": [-0.2, -0.1],
        },
        prefix="training/reward",
    )
    assert reward_metrics["training/reward/final"] == pytest.approx(0.75)
    assert reward_metrics["training/reward/components/correct_completion"] == pytest.approx(0.5)
    assert reward_metrics["training/reward/components/total_penalty"] == pytest.approx(-0.15)
