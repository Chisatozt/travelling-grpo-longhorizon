from __future__ import annotations

import unittest
from pathlib import Path

from transformers import AutoTokenizer

from verl.tools.schemas import OpenAIFunctionCallSchema, OpenAIFunctionToolCall
from verl.workers.rollout.schemas import (
    AsyncRolloutRequest,
    AsyncRolloutRequestStateEnum,
    _native_template_ids,
    _native_template_text,
)


MODEL_PATH = Path(
    "checkpoints/TravelGym/qwen35_4b_canonical_sft/merged_step_186"
)


def _tool_schema() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "interact_with_env",
            "description": "TravelGym public interaction protocol.",
            "strict": False,
            "parameters": {
                "type": "object",
                "properties": {
                    "choice": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["choice", "content"],
                "additionalProperties": False,
            },
        },
    }


def _tool_call(choice: str, content: str, index: str = "0") -> OpenAIFunctionToolCall:
    return OpenAIFunctionToolCall(
        id=index,
        function=OpenAIFunctionCallSchema(
            name="interact_with_env",
            arguments={"choice": choice, "content": content},
        ),
    )


class QwenTemplateLedgerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            local_files_only=True,
        )

    def _request(self) -> AsyncRolloutRequest:
        return AsyncRolloutRequest(
            request_id="template-ledger",
            state=AsyncRolloutRequestStateEnum.RUNNING,
            messages=[
                {"role": "system", "content": "You are a travel assistant."},
                {"role": "user", "content": "Find a hotel."},
            ],
            tool_schemas=[_tool_schema()],
            tools_kwargs={},
            input_ids=[],
            response_ids=[],
            attention_mask=[],
            response_attention_mask=[],
            response_position_ids=[],
            response_loss_mask=[],
            reward_scores={},
            max_prompt_len=8192,
            max_response_len=24576,
            max_model_len=32768,
            use_inference_chat_template=True,
            enable_tokenization_sanity_check=True,
            tokenizer=self.tokenizer,
        )

    def _assert_native_text_and_ledger_lengths(self, request, *, generation=False):
        rendered = _native_template_text(
            self.tokenizer,
            request._message_dicts(),
            tools=request._tool_dicts(),
            add_generation_prompt=generation,
        )
        decoded = self.tokenizer.decode(
            request.input_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        self.assertEqual(decoded, rendered)
        self.assertEqual(
            len(request.input_ids),
            len(request.attention_mask),
        )
        self.assertEqual(len(request.input_ids), len(request.position_ids))
        self.assertEqual(len(request.input_ids), len(request.loss_mask))

    def test_empty_reasoning_pure_tool_call_uses_append_only_bpe_ledger(self):
        request = self._request()
        generation_prefix = list(request.input_ids)
        prompt_length = len(request.prompt_ids)

        request.add_assistant_message(
            self.tokenizer,
            "",
            tool_calls=[_tool_call("search", "hotels")],
        )

        # This is the exact production failure: full retokenization merges the
        # two newlines and is not token-prefixed by the generation prompt.
        canonical_ids = _native_template_ids(
            self.tokenizer,
            request._message_dicts(),
            tools=request._tool_dicts(),
            add_generation_prompt=False,
        )
        self.assertNotEqual(canonical_ids[: len(generation_prefix)], generation_prefix)
        self._assert_native_text_and_ledger_lengths(request)
        assistant_end = len(request.input_ids)
        self.assertTrue(all(request.loss_mask[prompt_length:assistant_end]))

        request.add_tool_response_messages(
            self.tokenizer,
            ["Search results: H1, H2"],
            tool_call_ids=["0"],
            names=["interact_with_env"],
        )
        self._assert_native_text_and_ledger_lengths(request)
        tool_end = len(request.input_ids)
        self.assertTrue(all(mask == 0 for mask in request.loss_mask[assistant_end:tool_end]))

        returned_ids = request.get_generation_prompt_ids(self.tokenizer)
        self.assertIs(returned_ids, request.input_ids)
        self._assert_native_text_and_ledger_lengths(request, generation=True)
        self.assertTrue(all(mask == 0 for mask in request.loss_mask[tool_end:]))

    def test_reasoning_tool_call_remains_aligned_after_tool_round_trip(self):
        request = self._request()
        request.add_assistant_message(
            self.tokenizer,
            "I should search before answering.",
            tool_calls=[_tool_call("search", "business hotels")],
        )
        request.add_tool_response_messages(
            self.tokenizer,
            ["Search results: H3"],
            tool_call_ids=["0"],
            names=["interact_with_env"],
        )
        request.get_generation_prompt_ids(self.tokenizer)
        request.add_assistant_message(
            self.tokenizer,
            "H3 best matches the request.",
            tool_calls=[_tool_call("answer", "H3", index="1")],
        )
        self._assert_native_text_and_ledger_lengths(request)

    def test_forced_reasoning_close_is_masked_before_tool_call(self):
        request = self._request()
        assistant_start = len(request.input_ids)
        reasoning = "I should search for the requested hotel."
        tool_prompt_ids = request.build_tool_call_prompt_ids(
            self.tokenizer,
            reasoning,
        )
        tool_prompt_text = self.tokenizer.decode(
            tool_prompt_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        self.assertTrue(tool_prompt_text.endswith("</think>\n\n"))

        request.add_assistant_message(
            self.tokenizer,
            "",
            tool_calls=[_tool_call("search", "hotels")],
            reasoning_content=reasoning,
            forced_reasoning_end=True,
        )
        self._assert_native_text_and_ledger_lengths(request)
        content_ids = request.input_ids[assistant_start:]
        closing_ids = self.tokenizer.encode("</think>", add_special_tokens=False)
        closing_start = next(
            index
            for index in range(len(content_ids) - len(closing_ids), -1, -1)
            if content_ids[index : index + len(closing_ids)] == closing_ids
        )
        self.assertTrue(
            all(
                request.loss_mask[assistant_start + index] == 0
                for index in range(closing_start, closing_start + len(closing_ids))
            )
        )
        self.assertTrue(any(request.loss_mask[assistant_start : assistant_start + closing_start]))
        self.assertTrue(any(request.loss_mask[assistant_start + closing_start + len(closing_ids) :]))


if __name__ == "__main__":
    unittest.main()
