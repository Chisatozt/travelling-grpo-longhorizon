from __future__ import annotations

import numpy as np
import pytest
import torch

from verl.trainer.ppo.core_algos import (
    compute_terminal_component_advantages,
    compute_terminal_group_advantage,
    redistribute_behavior_component_turn_credit,
    turn_credit_conservation_error,
    turn_credit_conservation_stats,
)


def _history(event: dict) -> dict:
    return {
        "choice": event["choice"],
        "content": "public content",
        "turn_events": [event],
    }


def test_component_advantages_sum_to_terminal_grpo_advantage():
    scores = torch.tensor([0.8, 0.4, 0.0, -0.4])
    components = torch.tensor(
        [
            [0.9, 0.2, -0.3],
            [0.5, 0.1, -0.2],
            [0.2, 0.0, -0.2],
            [-0.1, 0.0, -0.3],
        ]
    )
    mask = torch.ones((4, 3))
    terminal, _ = compute_terminal_group_advantage(
        torch.zeros_like(mask),
        mask,
        np.array(["task"] * 4, dtype=object),
        terminal_scores=scores,
    )
    component_advantages = compute_terminal_component_advantages(
        scores,
        components,
        np.array(["task"] * 4, dtype=object),
    )

    assert torch.allclose(
        component_advantages.sum(dim=-1),
        terminal[:, 0],
        atol=1e-6,
    )


def test_behavior_routing_penalizes_redundant_action_in_successful_trajectory():
    terminal = torch.full((1, 8), 0.8)
    mask = torch.ones_like(terminal)
    boundaries = torch.tensor([[1, 0, 1, 0, 1, 0, 1, 0]])
    histories = [[
        _history(
            {
                "choice": "search",
                "aspect": "flight",
                "accepted": True,
                "new_search": True,
            }
        ),
        _history(
            {
                "choice": "action",
                "aspect": "flight",
                "accepted": True,
                "useful_action": True,
                "new_preference_count": 1,
            }
        ),
        _history(
            {
                "choice": "action",
                "aspect": "flight",
                "accepted": True,
                "duplicate_action": True,
                "redundant_action": True,
            }
        ),
        _history(
            {
                "choice": "answer",
                "aspect": "flight",
                "accepted": True,
                "completed_aspect": True,
                "correct_answer": True,
            }
        ),
    ]]
    component_advantages = torch.tensor([[1.0, 0.3, -0.4, -0.1]])
    credited, diagnostics = redistribute_behavior_component_turn_credit(
        terminal,
        mask,
        boundaries,
        histories,
        component_advantages,
        (
            "correct_completion",
            "hidden_preference_hit_rate",
            "redundant_action_penalty",
            "efficiency",
        ),
        mix_ratio=1.0,
    )

    assert credited[0, 2:4].mean() > 0
    assert credited[0, 4:6].mean() < 0
    assert diagnostics["mean_useful_action_credit"] > 0
    assert diagnostics["mean_redundant_action_credit"] < 0
    assert turn_credit_conservation_error(terminal, credited, mask) < 1e-6


def test_turn_length_normalization_preserves_original_token_credit_sum():
    terminal = torch.full((1, 8), -0.5)
    mask = torch.ones_like(terminal)
    boundaries = torch.tensor([[1, 1, 0, 0, 1, 0, 1, 0]])
    histories = [[
        _history(
            {
                "choice": "search",
                "aspect": "flight",
                "accepted": True,
                "new_search": True,
            }
        ),
        _history(
            {
                "choice": "action",
                "aspect": "flight",
                "accepted": True,
                "useful_action": True,
            }
        ),
        _history(
            {
                "choice": "action",
                "aspect": "flight",
                "accepted": False,
                "invalid_call": True,
                "redundant_action": True,
            }
        ),
        _history(
            {
                "choice": "finish",
                "aspect": "flight",
                "accepted": True,
                "termination_reason": "finish",
            }
        ),
    ]]
    component_advantages = torch.tensor([[0.2, -0.3, -0.4]])
    # The explicit residual closes the component sum to terminal A=-0.5.
    component_advantages[0, -1] += -0.5 - component_advantages.sum()

    credited, diagnostics = redistribute_behavior_component_turn_credit(
        terminal,
        mask,
        boundaries,
        histories,
        component_advantages,
        ("hidden_preference_hit_rate", "policy_penalty", "clip_residual"),
        mix_ratio=0.3,
    )

    assert diagnostics["conservation_error"] < 1e-6
    assert credited[0].sum() == pytest.approx(terminal[0].sum().item(), abs=1e-6)


def test_missing_quality_events_fall_back_to_terminal_credit():
    terminal = torch.tensor([[1.0, 1.0, -2.0, -2.0]])
    mask = torch.ones_like(terminal)
    boundaries = torch.tensor([[1, 0, 1, 0]])
    credited, diagnostics = redistribute_behavior_component_turn_credit(
        terminal,
        mask,
        boundaries,
        [[{"choice": "search"}, {"choice": "answer"}]],
        torch.tensor([[0.0]]),
        ("clip_residual",),
        mix_ratio=1.0,
    )

    assert torch.equal(credited, terminal)
    assert diagnostics["mean_fallback_rows"] == 1.0


def test_component_credit_never_targets_a_fully_masked_turn():
    terminal = torch.full((1, 5), -0.4)
    mask = torch.tensor([[1, 1, 0, 1, 1]])
    boundaries = torch.tensor([[1, 0, 1, 1, 0]])
    histories = [[
        _history({"choice": "search", "accepted": True, "new_search": True}),
        _history({"choice": "action", "accepted": False, "invalid_call": True}),
        _history({"choice": "finish", "accepted": True}),
    ]]

    credited, diagnostics = redistribute_behavior_component_turn_credit(
        terminal,
        mask,
        boundaries,
        histories,
        torch.tensor([[-0.4]]),
        ("policy_penalty",),
        mix_ratio=1.0,
    )

    assert credited[0, 2] == 0
    assert diagnostics["conservation_error"] < 1e-6
    assert credited.sum() == pytest.approx((terminal * mask).sum().item(), abs=1e-6)


def test_long_float32_rollout_is_projected_to_fp64_conservation_contract():
    sequence_length = 17_001
    terminal = torch.full((1, sequence_length), 0.732, dtype=torch.float32)
    mask = torch.ones_like(terminal)
    boundaries = torch.zeros_like(terminal, dtype=torch.long)
    boundaries[0, [0, 4_001, 9_001, 13_001]] = 1
    histories = [[
        _history({"choice": "search", "accepted": True, "new_search": True}),
        _history({"choice": "action", "accepted": True, "useful_action": True}),
        _history(
            {
                "choice": "action",
                "accepted": True,
                "duplicate_action": True,
                "redundant_action": True,
            }
        ),
        _history(
            {
                "choice": "answer",
                "accepted": True,
                "wrong_answer": True,
            }
        ),
    ]]

    credited, diagnostics = redistribute_behavior_component_turn_credit(
        terminal,
        mask,
        boundaries,
        histories,
        torch.tensor([[0.8, 0.2, -0.2, -0.068]], dtype=torch.float32),
        (
            "correct_completion",
            "hidden_preference_hit_rate",
            "redundant_action_penalty",
            "policy_penalty",
        ),
        mix_ratio=0.30,
    )
    conservation = turn_credit_conservation_stats(terminal, credited, mask)

    assert not torch.equal(credited, terminal)
    assert conservation["absolute_error"] <= 1.0e-5
    assert conservation["relative_error"] <= 1.0e-6
    assert diagnostics["preprojection_relative_error"] <= 1.0e-6
    assert diagnostics["conservation_finite"] == 1.0
    assert diagnostics["effective_mean_useful_action_credit"] == pytest.approx(
        0.30 * diagnostics["mean_useful_action_credit"]
    )
