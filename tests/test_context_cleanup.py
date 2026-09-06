from __future__ import annotations

import json
import unittest

import numpy as np
import torch
from tensordict import TensorDict

from verl import DataProto
from verl.protocol import make_1d_object_array
from verl.trainer.ppo.ray_trainer import compute_advantage, compute_response_mask
from verl.trainer.ppo.segmented_rollout import expand_segmented_batch
from verl.trainer.ppo.core_algos import AdvantageEstimator
from verl.tools.schemas import OpenAIFunctionCallSchema, OpenAIFunctionToolCall
from verl.workers.rollout.schemas import (
    AsyncRolloutRequest,
    AsyncRolloutRequestStateEnum,
    FinishReasonTypeEnum,
)


class _CharacterTemplateTokenizer:
    """Tiny native-template stand-in used without a model checkpoint."""

    pad_token_id = 0
    eos_token = "<eos>"

    @staticmethod
    def _message_dict(message):
        if hasattr(message, "model_dump"):
            return message.model_dump(exclude_none=True)
        return message

    def _render(self, messages, add_generation_prompt=False):
        parts = []
        for raw in messages:
            message = self._message_dict(raw)
            parts.append(f"<{message['role']}>:{message.get('content', '')}")
            if message.get("reasoning_content"):
                parts.append(f"<reasoning>:{message['reasoning_content']}")
            if message.get("tool_calls"):
                parts.append(
                    "<calls>:" + json.dumps(message["tool_calls"], sort_keys=True)
                )
        if add_generation_prompt:
            parts.append("<assistant>:")
        return "".join(parts)

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize=False,
        add_generation_prompt=False,
        **kwargs,
    ):
        del kwargs
        rendered = self._render(messages, add_generation_prompt=add_generation_prompt)
        if tokenize:
            return [ord(char) for char in rendered]
        return rendered

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(char) for char in str(text)]

    def decode(self, token_ids, **kwargs):
        del kwargs
        return "".join(chr(int(token)) for token in token_ids)


def _tool_call(choice: str, content: str, call_id: str) -> OpenAIFunctionToolCall:
    return OpenAIFunctionToolCall(
        id=call_id,
        function=OpenAIFunctionCallSchema(
            name="interact_with_env",
            arguments={"choice": choice, "content": content},
        ),
    )


def _request(*, max_model_len=720):
    return AsyncRolloutRequest(
        request_id="cleanup-smoke",
        state=AsyncRolloutRequestStateEnum.RUNNING,
        messages=[
            {"role": "system", "content": "Travel assistant."},
            {"role": "user", "content": "Book flight, hotel, and restaurant."},
        ],
        tool_schemas=[],
        tools_kwargs={},
        input_ids=[],
        response_ids=[],
        attention_mask=[],
        response_attention_mask=[],
        response_position_ids=[],
        response_loss_mask=[],
        reward_scores={},
        max_prompt_len=500,
        max_response_len=2000,
        max_model_len=max_model_len,
        use_inference_chat_template=True,
        enable_tokenization_sanity_check=True,
        tokenizer=_CharacterTemplateTokenizer(),
    )


def _append_completed_turn(request, turn_idx: int, aspect: str, body_size=100):
    tokenizer = _CharacterTemplateTokenizer()
    request.begin_turn(turn_idx)
    prompt = request.get_generation_prompt_ids(tokenizer)
    request.ensure_active_segment(prompt)
    request.record_model_output("generation", f"raw-answer-{aspect}", finish_reason="stop")
    request.add_assistant_message(
        tokenizer,
        "assistant-body-" + ("x" * body_size),
        tool_calls=[_tool_call("answer", f"{aspect}-answer-id", str(turn_idx))],
    )
    request.add_tool_response_messages(
        tokenizer,
        [f"Public feedback for {aspect}: the user prefers an early departure."],
        tool_call_ids=[str(turn_idx)],
        names=["interact_with_env"],
    )
    event = {
        "choice": "answer",
        "accepted": True,
        "completed_aspect": True,
        "aspect": aspect,
    }
    request.complete_turn(
        [event],
        {
            "choice": "answer",
            "turn_idx": turn_idx,
            "turn_events": [event],
        },
    )


def _append_public_turn(request, turn_idx: int, aspect: str, choice: str, content: str):
    tokenizer = _CharacterTemplateTokenizer()
    request.begin_turn(turn_idx)
    prompt = request.get_generation_prompt_ids(tokenizer)
    request.ensure_active_segment(prompt)
    request.record_model_output("generation", f"raw-{choice}-{content}", finish_reason="stop")
    request.add_assistant_message(
        tokenizer,
        f"raw-{choice}-{content}",
        tool_calls=[_tool_call(choice, content, str(turn_idx))],
    )
    request.add_tool_response_messages(
        tokenizer,
        [f"Public feedback for {aspect} after {choice}: {content}"],
        tool_call_ids=[str(turn_idx)],
        names=["interact_with_env"],
    )
    event = {
        "choice": choice,
        "accepted": True,
        "completed_aspect": choice == "answer",
        "aspect": aspect,
    }
    request.complete_turn(
        [event],
        {"choice": choice, "turn_idx": turn_idx, "turn_events": [event]},
    )


def test_cleanup_removes_the_completed_aspect_group_but_keeps_public_action_memory():
    tokenizer = _CharacterTemplateTokenizer()
    request = _request(max_model_len=2000)
    _append_public_turn(request, 0, "flight", "search", "search-flight")
    _append_public_turn(request, 1, "flight", "action", "traveler prefers aisle")
    _append_completed_turn(request, 2, "flight", body_size=30)

    result = request.maybe_cleanup_context(
        tokenizer,
        enabled=True,
        target_context_tokens=1,
        next_turn_reserve=1,
        reason="completed_aspect_group",
        force=True,
    )

    assert result["success"] is True
    assert result["cleaned_aspects"] == ["flight"]
    assert result["global_turn_ranges"] == [[0, 2]]
    active_text = tokenizer.decode(request.input_ids)
    assert "traveler prefers aisle" in active_text
    assert "raw-search-search-flight" not in active_text
    assert "raw-action-traveler prefers aisle" not in active_text
    assert "raw-answer-flight-answer-id" not in active_text
    assert "Submitted answer ID(s): flight-answer-id" in active_text

    # Re-attempting cleanup records no duplicate memory and does not mutate
    # the active native context or execute any tool again.
    before = list(request.input_ids)
    second = request.maybe_cleanup_context(
        tokenizer,
        enabled=True,
        target_context_tokens=1,
        next_turn_reserve=1,
        reason="repeat_cleanup",
        force=True,
    )
    assert second["success"] is False
    assert second["failure"] == "no_completed_aspect_history"
    assert request.input_ids == before
    assert active_text.count("traveler prefers aisle") == 1
    assert len(request.archive_model_outputs) == 3


def test_archive_keeps_raw_model_outputs_separate_from_active_context():
    request = _request(max_model_len=2000)
    tokenizer = _CharacterTemplateTokenizer()
    request.begin_turn(0)
    prompt = request.get_generation_prompt_ids(tokenizer)
    request.ensure_active_segment(prompt)
    request.record_model_output("generation", "raw completion", finish_reason="stop")
    assert request.archive_model_outputs == [
        {
            "output_index": 0,
            "phase": "generation",
            "global_turn": 0,
            "text": "raw completion",
            "finish_reason": "stop",
        }
    ]


def test_tool_response_over_limit_cleans_before_quarantine_and_keeps_response_in_next_prompt():
    tokenizer = _CharacterTemplateTokenizer()
    request = _request(max_model_len=1150)
    _append_completed_turn(request, 0, "flight", body_size=10)

    request.begin_turn(1)
    prompt = request.get_generation_prompt_ids(tokenizer)
    request.ensure_active_segment(prompt)
    request.record_model_output("generation", "current actor output", finish_reason="stop")
    request.add_assistant_message(
        tokenizer,
        "current actor output",
        tool_calls=[_tool_call("action", "hotel details", "current")],
    )
    request.add_tool_response_messages(
        tokenizer,
        ["current public tool response " + ("z" * 600)],
        tool_call_ids=["current"],
        names=["interact_with_env"],
    )
    assert len(request.input_ids) > request.max_model_len

    result = request.maybe_cleanup_context(
        tokenizer,
        enabled=True,
        target_context_tokens=100,
        next_turn_reserve=1,
        reason="tool_response_over_limit",
        force=True,
        require_next_turn=True,
    )

    assert result["success"] is True
    assert len(request.input_ids) <= request.max_model_len
    assert "current public tool response" in tokenizer.decode(request.input_ids)
    assert len(request.segment_records) == 1
    segment_text = tokenizer.decode(request.segment_records[0]["actual_input_ids"])
    assert "current public tool response" not in segment_text
    assert request.archive_messages[-1].content.startswith("current public tool response")


def test_cleanup_smoke_forces_two_compactions_and_keeps_running():
    tokenizer = _CharacterTemplateTokenizer()
    request = _request(max_model_len=1800)
    cleanup_results = []
    for turn_idx, aspect in enumerate(("flight", "hotel", "restaurant")):
        _append_completed_turn(request, turn_idx, aspect, body_size=120)
        if turn_idx < 2:
            cleanup_results.append(
                request.maybe_cleanup_context(
                    tokenizer,
                    enabled=True,
                    target_context_tokens=350,
                    next_turn_reserve=100,
                    reason="test_forced_danger",
                    force=True,
                    require_next_turn=True,
                )
            )

    request.finalize(
        tokenizer,
        {"interact_with_env": 1.0},
        finish_reason_type=FinishReasonTypeEnum.STOP,
    )

    assert len(cleanup_results) == 2
    assert all(result["success"] for result in cleanup_results)
    assert [aspect for result in cleanup_results for aspect in result["cleaned_aspects"]] == [
        "flight",
        "hotel",
    ]
    assert len(request.segment_records) == 3
    assert all(record["actual_input_tokens"] <= request.max_model_len for record in request.segment_records)
    assert len(request.archive_turns) == 3
    assert len(request.archive_messages) > len(request.messages)
    assert "Book flight, hotel, and restaurant." in request.messages[1].content
    assert "Public feedback for flight" in request.messages[2].content
    assert "Public feedback for hotel" in request.messages[3].content
    assert tokenizer.decode(request.input_ids) == tokenizer.apply_chat_template(
        request._message_dicts(), add_generation_prompt=False
    )
    assert request.length_summary()["segment_count"] == 3.0


def test_cleanup_is_threshold_driven_and_only_accepted_completed_answers_are_candidates():
    tokenizer = _CharacterTemplateTokenizer()
    request = _request(max_model_len=2000)
    _append_completed_turn(request, 0, "flight", body_size=10)
    enough = request.maybe_cleanup_context(
        tokenizer,
        enabled=True,
        target_context_tokens=100,
        next_turn_reserve=100,
        reason="budget_safe",
    )
    assert enough["attempted"] is False

    request.begin_turn(1)
    prompt = request.get_generation_prompt_ids(tokenizer)
    request.ensure_active_segment(prompt)
    request.add_assistant_message(
        tokenizer,
        "not accepted",
        tool_calls=[_tool_call("answer", "unknown", "bad")],
    )
    request.add_tool_response_messages(
        tokenizer,
        ["not accepted"],
        tool_call_ids=["bad"],
        names=["interact_with_env"],
    )
    request.complete_turn(
        [
            {
                "choice": "answer",
                "accepted": False,
                "completed_aspect": True,
                "aspect": "hotel",
            }
        ],
        {"choice": "answer", "turn_idx": 1},
    )
    # Even an explicit force cannot make an unaccepted answer cleanable.
    result = request.maybe_cleanup_context(
        tokenizer,
        enabled=True,
        target_context_tokens=1,
        next_turn_reserve=1,
        reason="no_candidate",
        force=True,
    )
    assert result["attempted"] is True
    assert result["success"] is True
    assert result["cleaned_aspects"] == ["flight"]
    assert request.archive_turns[1]["cleanable"] is False
    before = list(request.input_ids)
    no_history = request.maybe_cleanup_context(
        tokenizer,
        enabled=True,
        target_context_tokens=1,
        next_turn_reserve=1,
        reason="no_cleanable_history",
        force=True,
    )
    assert no_history["success"] is False
    assert no_history["failure"] == "no_completed_aspect_history"
    assert request.input_ids == before


def test_cleanup_disabled_keeps_the_legacy_active_ledger_unchanged():
    tokenizer = _CharacterTemplateTokenizer()
    request = _request(max_model_len=2000)
    _append_completed_turn(request, 0, "flight", body_size=10)
    before_ids = list(request.input_ids)
    before_messages = list(request.messages)

    result = request.maybe_cleanup_context(
        tokenizer,
        enabled=False,
        target_context_tokens=1,
        next_turn_reserve=1,
        reason="disabled",
        force=True,
    )

    assert result["attempted"] is False
    assert result["success"] is False
    assert request.input_ids == before_ids
    assert request.messages == before_messages
    assert request.cleanup_events == []


def test_mixed_tool_events_are_conservatively_kept_out_of_cleanup():
    tokenizer = _CharacterTemplateTokenizer()
    request = _request(max_model_len=2000)
    request.begin_turn(0)
    prompt = request.get_generation_prompt_ids(tokenizer)
    request.ensure_active_segment(prompt)
    request.add_assistant_message(
        tokenizer,
        "mixed tool calls",
        tool_calls=[
            _tool_call("search", "flight", "search-0"),
            _tool_call("answer", "F1", "answer-0"),
        ],
    )
    request.add_tool_response_messages(
        tokenizer,
        ["public search response", "public answer response"],
        tool_call_ids=["search-0", "answer-0"],
        names=["interact_with_env", "interact_with_env"],
    )
    record = request.complete_turn(
        [
            {
                "choice": "search",
                "accepted": True,
                "completed_aspect": False,
                "aspect": "flight",
            },
            {
                "choice": "answer",
                "accepted": True,
                "completed_aspect": True,
                "aspect": "flight",
            },
        ],
        {"turn_events": []},
    )

    assert record["cleanable"] is False
    result = request.maybe_cleanup_context(
        tokenizer,
        enabled=True,
        target_context_tokens=1,
        next_turn_reserve=1,
        reason="mixed_tool_events",
        force=True,
    )
    assert result["success"] is False
    assert result["failure"] == "no_completed_aspect_history"


def _segment_record(turn_idx: int, marker: int):
    event = {
        "choice": "action",
        "accepted": True,
        "completed_aspect": False,
        "aspect": "flight",
    }
    return {
        "segment_id": turn_idx,
        "global_turn_indices": [turn_idx],
        "turn_boundaries": [0],
        "conversation_history": [
            {"choice": "action", "turn_idx": turn_idx, "turn_events": [event]}
        ],
        "prompt_ids": [10 + marker, 11 + marker, 12 + marker],
        "prompt_attention_mask": [1, 1, 1],
        "prompt_position_ids": [0, 1, 2],
        "prompt_loss_mask": [0, 0, 0],
        "response_ids": [20 + marker, 21 + marker],
        "response_attention_mask": [1, 1],
        "response_position_ids": [3, 4],
        "response_loss_mask": [1, 1],
    }


def _segmented_source_batch():
    batch_size = 4
    tensors = TensorDict(
        {
            "prompts": torch.tensor([[1, 2, 3]] * batch_size),
            "responses": torch.tensor([[4, 5, 6, 7, 8]] * batch_size),
            "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]] * batch_size),
            "attention_mask": torch.ones(batch_size, 8, dtype=torch.long),
            "position_ids": torch.arange(8).repeat(batch_size, 1),
            "loss_mask": torch.tensor([[0, 0, 0, 1, 1, 1, 1, 1]] * batch_size),
            "response_mask": torch.tensor([[1, 1, 1, 1, 1]] * batch_size),
        },
        batch_size=batch_size,
    )
    reward_scores = [
        {"interact_with_env": 0.0, "interact_with_env_reward_valid": 1.0, "interact_with_env_terminal_only": 1.0},
        {"interact_with_env": 1.0, "interact_with_env_reward_valid": 1.0, "interact_with_env_terminal_only": 1.0},
        {"interact_with_env": 0.0, "interact_with_env_reward_valid": 1.0, "interact_with_env_terminal_only": 1.0},
        {"interact_with_env": 1.0, "interact_with_env_reward_valid": 1.0, "interact_with_env_terminal_only": 1.0},
    ]
    records = [
        [_segment_record(0, 0), _segment_record(1, 1)],
        [_segment_record(0, 2), _segment_record(1, 3)],
        [_segment_record(0, 4), _segment_record(1, 5)],
        [_segment_record(0, 6), _segment_record(1, 7)],
    ]
    return DataProto(
        batch=tensors,
        non_tensor_batch={
            "uid": np.asarray(["g0", "g0", "g1", "g1"], dtype=object),
            "reward_scores": np.asarray(reward_scores, dtype=object),
            "segment_records": make_1d_object_array(records),
        },
        meta_info={"travel_context_cleanup_enabled": True},
    )


def test_segment_expansion_preserves_uids_and_terminal_reward_once():
    source = _segmented_source_batch()
    reward = torch.tensor(
        [[0.0, 0.0, 0.0, 2.0]] * 4,
        dtype=torch.float32,
    )
    expanded, expanded_reward, _ = expand_segmented_batch(
        source,
        reward,
        {},
        max_model_len=16,
        pad_token_id=0,
    )
    assert len(expanded) == 8
    assert expanded.batch["input_ids"].shape == (8, 16)
    assert torch.equal(
        expanded.batch["segment_response_mask"],
        compute_response_mask(expanded),
    )
    assert expanded.non_tensor_batch["uid"].tolist() == ["g0", "g0", "g0", "g0", "g1", "g1", "g1", "g1"]
    assert expanded.non_tensor_batch["segment_trajectory_uid"][0] == expanded.non_tensor_batch["segment_trajectory_uid"][1]
    assert torch.all(expanded.batch["input_ids"].ne(0).sum(dim=-1) <= 5)
    assert torch.allclose(expanded_reward.sum(dim=-1).reshape(4, 2).sum(dim=-1), torch.tensor([2.0] * 4))
    assert torch.all(expanded_reward.reshape(4, 2, 16)[:, 0].sum(dim=-1) == 0)


def test_segmented_grpo_uses_one_terminal_group_statistic_per_original_rollout():
    source = _segmented_source_batch()
    reward = torch.tensor([[0.0, 0.0, 0.0, 1.0]] * 4, dtype=torch.float32)
    expanded, expanded_reward, _ = expand_segmented_batch(
        source,
        reward,
        {},
        max_model_len=16,
        pad_token_id=0,
    )
    expanded.batch["response_mask"] = expanded.batch["segment_response_mask"]
    expanded.batch["token_level_rewards"] = expanded_reward
    result = compute_advantage(
        expanded,
        adv_estimator=AdvantageEstimator.GRPO_MULTITURN,
        num_repeat=2,
        multi_turn=True,
        turn_level_method="off",
        norm_adv_by_std_in_grpo=True,
        config={
            "dynamic_sampling": {
                "enable": True,
                "numerical_epsilon": 1.0e-6,
                "min_reward_spread": 0.0,
            },
            "turn_credit": {"method": "off", "stage": "off"},
        },
    )
    advantages = result.batch["advantages"]
    # Both fragments of one source rollout receive the same scalar GRPO
    # advantage, while both source rows in each UID group participate once.
    for start in range(0, len(result), 2):
        active = result.batch["loss_mask"][start : start + 2].bool()
        assert torch.allclose(
            advantages[start][active[0]],
            advantages[start + 1][active[1]],
        )
    stats = result.meta_info["terminal_sampling_stats"]
    assert stats["num_trajectories"] == 4
    assert stats["trainable_group_count"] == 2


if __name__ == "__main__":
    unittest.main()
