from __future__ import annotations

import asyncio
import json
import unittest
from uuid import uuid4

from verl.tools.env_manager import EnvironmentManager
from verl.tools.interact_tool import InteractTool
from verl.workers.rollout.schemas import (
    AsyncRolloutRequest,
    AsyncRolloutRequestStateEnum,
)


TASK_ID = "flight:2-23|apartment:3-63|restaurant:3-159"


class DeterministicTemplateTokenizer:
    @staticmethod
    def _render(messages, add_generation_prompt=False):
        payload = json.dumps(messages, ensure_ascii=False, sort_keys=True)
        return payload + ("<GEN>" if add_generation_prompt else "")

    def apply_chat_template(self, messages, **kwargs):
        text = self._render(messages, kwargs.get("add_generation_prompt", False))
        return list(text.encode("utf-8")) if kwargs.get("tokenize", True) else text

    @staticmethod
    def encode(text, add_special_tokens=False):
        del add_special_tokens
        return list(text.encode("utf-8"))

    @staticmethod
    def decode(token_ids, **kwargs):
        del kwargs
        return bytes(token_ids).decode("utf-8")


class TravelInitialContextTests(unittest.TestCase):
    def test_environment_manager_retains_public_reset_state(self):
        manager = EnvironmentManager()
        request_id = str(uuid4())
        try:
            manager.create_environment(request_id, id=TASK_ID, max_turns=25)
            context = manager.get_initial_context(request_id)
            self.assertIn("Travel planning is ready.", context)
            self.assertIn("Current aspect: apartment", context)
            self.assertNotIn("correct_ids", context)
            self.assertNotIn("best_id", context)
        finally:
            manager.release_environment(request_id)
        self.assertEqual(manager.get_initial_context(request_id), "")

    def test_request_injects_context_into_prompt_not_training_target(self):
        tokenizer = DeterministicTemplateTokenizer()
        request = AsyncRolloutRequest(
            request_id="request",
            state=AsyncRolloutRequestStateEnum.PENDING,
            messages=[{"role": "user", "content": "Book my trip."}],
            input_ids=[],
            response_ids=[],
            attention_mask=[],
            response_attention_mask=[],
            response_position_ids=[],
            response_loss_mask=[],
            reward_scores={},
            max_prompt_len=4096,
            max_response_len=128,
            max_model_len=4224,
            use_inference_chat_template=True,
            enable_tokenization_sanity_check=True,
            tokenizer=tokenizer,
        )
        old_prompt = list(request.prompt_ids)
        request.inject_initial_user_context(
            tokenizer,
            "Travel planning is ready.\nCurrent aspect: apartment",
        )

        self.assertNotEqual(request.prompt_ids, old_prompt)
        self.assertEqual(request.input_ids, request.prompt_ids)
        self.assertEqual(request.response_ids, [])
        self.assertEqual(set(request.prompt_loss_mask), {0})
        self.assertIn("Current aspect: apartment", request.messages[-1].content)
        self.assertEqual(
            request.get_generation_prompt_ids(tokenizer),
            request.prompt_ids,
        )


    def test_interact_tool_returns_quality_event_only_in_private_metrics(self):
        tool = object.__new__(InteractTool)
        tool._conversation_data = {}
        tool._env_manager = EnvironmentManager()
        request_id = str(uuid4())
        try:
            asyncio.run(tool.create(request_id, id=TASK_ID, max_turns=1))
            response = asyncio.run(
                tool.execute(
                    request_id,
                    {"choice": "search", "content": "Search the current aspect."},
                )
            )
            public_text, _, done, choice, _, metrics = response
            self.assertEqual(choice, "search")
            self.assertTrue(done)
            self.assertTrue(metrics["turn_event"]["new_search"])
            self.assertNotIn("turn_event", public_text)
            self.assertNotIn("correct_ids", json.dumps(metrics))
            self.assertNotIn("best_id", json.dumps(metrics))
        finally:
            asyncio.run(tool.release(request_id))


if __name__ == "__main__":
    unittest.main()
