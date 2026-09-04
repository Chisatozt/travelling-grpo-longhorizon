from __future__ import annotations

import asyncio
import unittest
from dataclasses import dataclass, field
from types import SimpleNamespace

from verl.tools.travel_reward_metrics import TRAVEL_REWARD_METRIC_NAMES
from verl.workers.rollout.schemas import (
    AsyncRolloutRequestStateEnum,
    FinishReasonTypeEnum,
    RolloutLengthExceededError,
    RolloutProtocolError,
    RolloutTemplateAlignmentError,
)
from verl.workers.rollout.sglang_rollout.sglang_rollout_customized import SGLangRollout


@dataclass
class _FakeRequest:
    request_id: str = "length-test"
    tools_kwargs: dict = field(default_factory=dict)
    state: AsyncRolloutRequestStateEnum = AsyncRolloutRequestStateEnum.RUNNING
    prompt_ids: list[int] = field(default_factory=lambda: [10, 11, 12])
    tool_schemas: list[dict] = field(default_factory=lambda: [{"type": "function"}])
    max_response_len: int = 4096
    max_model_len: int = 8192
    input_ids: list[int] = field(default_factory=lambda: [10, 11, 12, 13])
    attention_mask: list[int] = field(default_factory=lambda: [1, 1, 1, 1])
    position_ids: list[int] = field(default_factory=lambda: [0, 1, 2, 3])
    loss_mask: list[int] = field(default_factory=lambda: [0, 0, 0, 1])
    response_ids: list[int] = field(default_factory=lambda: [13])
    response_attention_mask: list[int] = field(default_factory=lambda: [1])
    response_position_ids: list[int] = field(default_factory=lambda: [3])
    response_loss_mask: list[int] = field(default_factory=lambda: [1])
    turn_boundaries: list[int] = field(default_factory=lambda: [3])
    conversation_histories: list[dict] = field(default_factory=lambda: [{"reward": 0.0}])
    metrics: dict = field(default_factory=lambda: {"foo": 1})
    reward_scores: dict = field(default_factory=dict)
    finish_reason: str | None = None
    length_events: list[dict] = field(default_factory=list)

    def record_length_event(self, phase: str, **metadata) -> None:
        self.length_events.append({"phase": phase, **metadata})


class _FakeTelemetryTool:
    def __init__(self):
        self.released = False

    async def get_reward_metadata(self, request_id):
        self.request_id = request_id
        return {"user_api_calls": 3, "user_total_tokens": 120}

    async def release(self, request_id, **kwargs):
        del request_id, kwargs
        self.released = True


class SGLangLengthQuarantineTests(unittest.TestCase):
    def test_phase_budget_uses_continuation_prompt_and_requested_cap(self):
        rollout = object.__new__(SGLangRollout)
        rollout._max_new_tokens_per_turn = 4096
        rollout._tool_response_token_reserve = 128
        rollout._template_token_reserve = 32
        request = _FakeRequest()
        continuation_prompt = request.prompt_ids + list(range(600))

        budget = rollout._compute_generation_budget(
            request,
            continuation_prompt,
            max_new_tokens_per_turn=512,
            phase="tool_call_generation",
        )

        self.assertEqual(budget, 512)
        self.assertEqual(
            request.length_events[-1]["phase"],
            "tool_call_generation",
        )
        self.assertEqual(request.length_events[-1]["context_tokens"], 603)
        self.assertEqual(request.length_events[-1]["response_tokens"], 600)

    def test_overlong_request_is_quarantined_without_crashing_batch(self):
        rollout = object.__new__(SGLangRollout)
        rollout._tool_map = {}
        request = _FakeRequest()

        invalid = asyncio.run(
            rollout._quarantine_length_exceeded(
                request,
                RolloutLengthExceededError("per-turn cap reached"),
            )
        )

        self.assertEqual(invalid.state, AsyncRolloutRequestStateEnum.COMPLETED)
        self.assertEqual(invalid.input_ids, invalid.prompt_ids)
        self.assertEqual(invalid.attention_mask, [1, 1, 1])
        self.assertEqual(invalid.position_ids, [0, 1, 2])
        self.assertEqual(invalid.loss_mask, [0, 0, 0])
        self.assertEqual(invalid.response_ids, [])
        self.assertEqual(invalid.finish_reason, FinishReasonTypeEnum.LENGTH.value)
        self.assertEqual(invalid.reward_scores["interact_with_env"], 0.0)
        for metric in TRAVEL_REWARD_METRIC_NAMES:
            self.assertEqual(invalid.reward_scores[f"interact_with_env_{metric}"], 0.0)
        self.assertTrue(invalid.length_events[-1]["invalid"])

    def test_quarantine_preserves_usage_before_releasing_tool(self):
        rollout = object.__new__(SGLangRollout)
        tool = _FakeTelemetryTool()
        rollout._tool_map = {"interact_with_env": tool}
        request = _FakeRequest(
            tools_kwargs={"interact_with_env": {"release_kwargs": {}}}
        )

        invalid = asyncio.run(
            rollout._quarantine_length_exceeded(
                request,
                RolloutLengthExceededError("per-turn cap reached"),
            )
        )

        self.assertTrue(tool.released)
        self.assertEqual(
            invalid.reward_scores["interact_with_env_user_api_calls"], 3.0
        )
        self.assertEqual(
            invalid.reward_scores["interact_with_env_user_total_tokens"], 120.0
        )
        self.assertEqual(
            invalid.reward_scores["interact_with_env_user_api_errors"], 0.0
        )

    def test_template_error_isolated_without_failing_other_request(self):
        rollout = object.__new__(SGLangRollout)
        rollout._tool_map = {}
        bad = _FakeRequest(request_id="bad-template")
        good = _FakeRequest(request_id="good-template")

        async def fake_rollout(request, *_args, **_kwargs):
            if request.request_id == "bad-template":
                raise RolloutTemplateAlignmentError("synthetic mismatch")
            return request

        rollout._async_rollout_a_request = fake_rollout

        async def run_batch():
            semaphore = asyncio.Semaphore(2)
            return await asyncio.gather(
                rollout._semaphore_wrapped_rollout(semaphore, bad, True, False),
                rollout._semaphore_wrapped_rollout(semaphore, good, True, False),
            )

        invalid, valid = asyncio.run(run_batch())
        self.assertEqual(invalid.state, AsyncRolloutRequestStateEnum.COMPLETED)
        self.assertEqual(invalid.finish_reason, FinishReasonTypeEnum.STOP.value)
        self.assertEqual(invalid.reward_scores["interact_with_env_reward_valid"], 0.0)
        self.assertEqual(invalid.length_events[-1]["phase"], "template_alignment")
        self.assertIs(valid, good)

    def test_unknown_tool_name_is_quarantined_without_failing_sibling(self):
        rollout = object.__new__(SGLangRollout)
        rollout._tool_map = {"interact_with_env": object()}
        bad = _FakeRequest(
            request_id="bad-tool",
            tools_kwargs={"interact_with_env": {}},
        )
        good = _FakeRequest(request_id="good-tool")

        async def fake_rollout(request, *_args, **_kwargs):
            if request.request_id == "bad-tool":
                tool_call = SimpleNamespace(
                    function=SimpleNamespace(name="interaction_with_env")
                )
                rollout._validate_tool_calls(request, [tool_call])
            return request

        rollout._async_rollout_a_request = fake_rollout

        async def run_batch():
            semaphore = asyncio.Semaphore(2)
            return await asyncio.gather(
                rollout._semaphore_wrapped_rollout(semaphore, bad, True, False),
                rollout._semaphore_wrapped_rollout(semaphore, good, True, False),
            )

        invalid, valid = asyncio.run(run_batch())
        self.assertEqual(invalid.state, AsyncRolloutRequestStateEnum.COMPLETED)
        self.assertEqual(invalid.reward_scores["interact_with_env_reward_valid"], 0.0)
        self.assertEqual(invalid.length_events[-1]["phase"], "tool_protocol")
        self.assertIs(valid, good)

    def test_tool_validation_reports_generated_and_available_names(self):
        rollout = object.__new__(SGLangRollout)
        rollout._tool_map = {"interact_with_env": object()}
        request = _FakeRequest(tools_kwargs={"interact_with_env": {}})
        tool_call = SimpleNamespace(
            function=SimpleNamespace(name="interaction_with_env")
        )

        with self.assertRaisesRegex(
            RolloutProtocolError,
            "interaction_with_env.*interact_with_env",
        ):
            rollout._validate_tool_calls(request, [tool_call])

    def test_batch_error_waits_for_sibling_before_propagating(self):
        sibling_finished = False

        async def fail():
            raise RuntimeError("synthetic infrastructure failure")

        async def sibling():
            nonlocal sibling_finished
            await asyncio.sleep(0)
            sibling_finished = True
            return "ok"

        async def run_batch():
            with self.assertRaisesRegex(RuntimeError, "synthetic infrastructure"):
                await SGLangRollout._gather_rollout_requests([fail(), sibling()])

        asyncio.run(run_batch())
        self.assertTrue(sibling_finished)


if __name__ == "__main__":
    unittest.main()
