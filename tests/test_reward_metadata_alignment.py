from __future__ import annotations

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from verl import DataProto
from verl.tools.travel_reward_metrics import TRAVEL_REWARD_METRIC_NAMES
from verl.trainer.ppo.core_algos import AdvantageEstimator
from verl.trainer.ppo.ray_trainer import (
    _aligned_reward_extra_info_arrays,
    _apply_turn_credit_stage,
    _extract_terminal_reward_components,
    compute_advantage,
)
from verl.workers.reward_manager.naive import NaiveRewardManager


class _Tokenizer:
    def decode(self, token_ids, skip_special_tokens=True):
        del token_ids, skip_special_tokens
        return ""


def _reward_scores(*, valid: bool, api_calls: float | None = None) -> dict:
    scores = {
        "interact_with_env": 0.5 if valid else 0.0,
        "interact_with_env_reward_valid": float(valid),
        "interact_with_env_terminal_only": 1.0,
    }
    if api_calls is not None:
        scores["interact_with_env_user_api_calls"] = api_calls
    return scores


def test_mixed_valid_and_quarantined_rows_keep_every_metric_aligned():
    batch_size = 16
    reward_scores = [
        *[_reward_scores(valid=True, api_calls=2.0) for _ in range(11)],
        *[_reward_scores(valid=False) for _ in range(4)],
        None,
    ]
    data = DataProto(
        batch=TensorDict(
            {
                "prompts": torch.ones((batch_size, 2), dtype=torch.long),
                "responses": torch.ones((batch_size, 3), dtype=torch.long),
                "attention_mask": torch.ones((batch_size, 5), dtype=torch.long),
            },
            batch_size=[batch_size],
        ),
        non_tensor_batch={
            "reward_model": np.array(
                [{"ground_truth": ""} for _ in range(batch_size)], dtype=object
            ),
            "data_source": np.array(
                ["interact_travelgym"] * batch_size, dtype=object
            ),
            "reward_scores": np.array(reward_scores, dtype=object),
            "extra_info": np.array([{} for _ in range(batch_size)], dtype=object),
        },
    )

    result = NaiveRewardManager(_Tokenizer(), num_examine=0)(
        data, return_dict=True
    )
    extra = result["reward_extra_info"]

    for metric_name in TRAVEL_REWARD_METRIC_NAMES:
        assert len(extra[metric_name]) == batch_size
    assert extra["user_api_calls"] == [2.0] * 11 + [0.0] * 5
    assert extra["reward_valid"] == [1.0] * 11 + [0.0] * 5
    arrays = _aligned_reward_extra_info_arrays(extra, batch_size=batch_size)
    assert all(len(value) == batch_size for value in arrays.values())


def test_ragged_reward_metadata_fails_before_actor_update():
    with pytest.raises(ValueError, match="user_api_calls=11"):
        _aligned_reward_extra_info_arrays(
            {
                "score": [0.0] * 16,
                "user_api_calls": [0.0] * 11,
            },
            batch_size=16,
        )


def test_terminal_component_reconstruction_requires_v2_and_closes_to_score():
    class Batch:
        pass

    data = Batch()
    data.non_tensor_batch = {
        "reward_version": np.array(["travelgym-terminal-v2", "travelgym-terminal-v2"]),
        "correct_completion": np.array([1.0, 0.0]),
        "coverage_adjusted_answer_quality": np.array([1.0, 0.0]),
        "coverage_adjusted_legal_chain_rate": np.array([1.0, 0.0]),
        "hidden_preference_hit_rate": np.array([1.0, 0.0]),
        "efficiency": np.array([0.5, 0.0]),
        "policy_penalty": np.array([0.0, 0.1]),
        "redundant_action_penalty": np.array([0.0, 0.2]),
        "incomplete_penalty": np.array([0.0, 1.0]),
        "zero_answer_penalty": np.array([0.0, 0.5]),
        "max_steps_penalty": np.array([0.0, 0.75]),
    }
    terminal_scores = torch.tensor([0.95, -0.70])
    names, components = _extract_terminal_reward_components(data, terminal_scores)

    assert names[-1] == "clip_residual"
    assert torch.allclose(components.sum(dim=-1), terminal_scores)

    data.non_tensor_batch["reward_version"][1] = "unknown"
    assert _extract_terminal_reward_components(data, terminal_scores) is None


def test_turn_credit_stage_shadow_is_unchanged_and_train_is_applied():
    original = torch.tensor([[1.0, 1.0]])
    credited = torch.tensor([[1.5, 0.5]])
    mask = torch.ones_like(original)

    shadow_diagnostics = {}
    shadow = _apply_turn_credit_stage(
        original, credited, mask, shadow_diagnostics, stage="shadow"
    )
    assert torch.equal(shadow, original)
    assert shadow_diagnostics["applied"] == 0.0

    train_diagnostics = {}
    train = _apply_turn_credit_stage(
        original, credited, mask, train_diagnostics, stage="train"
    )
    assert torch.equal(train, credited)
    assert not torch.equal(train, original)
    assert train_diagnostics["applied"] == 1.0


def test_turn_credit_train_rejects_real_conservation_corruption():
    original = torch.tensor([[1.0, 1.0]])
    corrupted = torch.tensor([[1.5, 1.0]])

    with pytest.raises(RuntimeError, match="turn credit conservation failed"):
        _apply_turn_credit_stage(
            original,
            corrupted,
            torch.ones_like(original),
            {},
            stage="train",
        )


def test_turn_credit_train_rejects_missing_quality_events():
    original = torch.tensor([[1.0, 1.0]])

    with pytest.raises(ValueError, match="quality events are missing"):
        _apply_turn_credit_stage(
            original,
            original.clone(),
            torch.ones_like(original),
            {"fallback_row_count": 1.0},
            stage="train",
        )


def test_grpo_multiturn_train_stage_reaches_actor_advantages():
    batch_size = 4
    response_length = 4
    histories = np.empty(batch_size, dtype=object)
    histories[:] = [
        [
            {
                "choice": "search",
                "turn_events": [{"choice": "search", "accepted": True, "new_search": True}],
            },
            {
                "choice": "answer",
                "turn_events": [{"choice": "answer", "accepted": True, "wrong_answer": True}],
            },
        ]
        for _ in range(batch_size)
    ]
    terminal_scores = [-0.8, -0.2, 0.4, 0.9]
    non_tensor_batch = {
        "uid": np.array(["task"] * batch_size, dtype=object),
        "reward_scores": np.array(
            [
                {
                    "interact_with_env": score,
                    "interact_with_env_reward_valid": 1.0,
                    "interact_with_env_terminal_only": 1.0,
                }
                for score in terminal_scores
            ],
            dtype=object,
        ),
        "reward_version": np.array(["travelgym-terminal-v2"] * batch_size, dtype=object),
        "conversation_histories": histories,
    }
    for component_name in (
        "correct_completion",
        "coverage_adjusted_answer_quality",
        "coverage_adjusted_legal_chain_rate",
        "hidden_preference_hit_rate",
        "efficiency",
        "policy_penalty",
        "redundant_action_penalty",
        "incomplete_penalty",
        "zero_answer_penalty",
        "max_steps_penalty",
    ):
        non_tensor_batch[component_name] = np.zeros(batch_size, dtype=np.float64)

    data = DataProto(
        batch=TensorDict(
            {
                "responses": torch.ones((batch_size, response_length), dtype=torch.long),
                "response_mask": torch.ones((batch_size, response_length)),
                "loss_mask": torch.ones((batch_size, response_length)),
                "turn_boundaries": torch.tensor([[1, 0, 1, 0]] * batch_size),
                "token_level_rewards": torch.zeros((batch_size, response_length)),
            },
            batch_size=[batch_size],
        ),
        non_tensor_batch=non_tensor_batch,
    )

    result = compute_advantage(
        data,
        AdvantageEstimator.GRPO_MULTITURN,
        num_repeat=4,
        multi_turn=True,
        turn_level_method="component_attribution",
        config={
            "dynamic_sampling": {"min_reward_spread": 0.005},
            "turn_credit": {
                "method": "component_attribution",
                "stage": "train",
                "mix_ratio": 0.30,
                "conservation_atol": 1.0e-5,
                "conservation_rtol": 1.0e-6,
            },
        },
    )

    assert result.meta_info["turn_credit_applied"] == 1.0
    assert result.meta_info["turn_credit_diagnostics"]["applied"] == 1.0
    assert not torch.equal(result.batch["advantages"][:, :2], result.batch["advantages"][:, 2:])
    assert torch.equal(result.batch["returns"], result.batch["advantages"])
