from __future__ import annotations

import unittest

from verl.workers.rollout.schemas import (
    RolloutLengthExceededError,
    compute_generation_budget,
)


class RolloutBudgetTests(unittest.TestCase):
    def test_per_turn_cap_and_tool_reserve_are_applied(self):
        budget, clamped = compute_generation_budget(
            generation_prompt_tokens=8192,
            current_response_tokens=0,
            max_model_len=32768,
            max_response_len=24576,
        )
        self.assertEqual(budget, 2048)
        self.assertFalse(clamped)

    def test_cumulative_response_budget_clamps_later_turn(self):
        budget, clamped = compute_generation_budget(
            generation_prompt_tokens=22000,
            current_response_tokens=17000,
            max_model_len=32768,
            max_response_len=24576,
        )
        self.assertEqual(budget, 1432)
        self.assertTrue(clamped)

    def test_no_room_after_reserve_is_a_typed_length_error(self):
        with self.assertRaises(RolloutLengthExceededError):
            compute_generation_budget(
                generation_prompt_tokens=28000,
                current_response_tokens=18000,
                max_model_len=32768,
                max_response_len=24576,
            )

    def test_no_tool_requests_use_small_template_reserve(self):
        budget, clamped = compute_generation_budget(
            generation_prompt_tokens=32700,
            current_response_tokens=0,
            max_model_len=32768,
            max_response_len=24576,
            has_tools=False,
        )
        self.assertEqual(budget, 35)
        self.assertTrue(clamped)


if __name__ == "__main__":
    unittest.main()
