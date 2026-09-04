from __future__ import annotations

import unittest

from verl.trainer.ppo.rollout_health import (
    InitialRolloutHealthError,
    validate_initial_rollout_health,
)


class InitialRolloutHealthTests(unittest.TestCase):
    def test_rejects_repeated_punctuation_collapse(self):
        with self.assertRaisesRegex(InitialRolloutHealthError, "repeated characters"):
            validate_initial_rollout_health(["!" * 8192], response_token_limit=8192)

    def test_rejects_full_budget_without_tool_protocol(self):
        with self.assertRaisesRegex(InitialRolloutHealthError, "without a tool-call"):
            validate_initial_rollout_health(
                ["multilingual gibberish " * 500],
                response_token_limit=8192,
            )

    def test_accepts_short_final_answer(self):
        validate_initial_rollout_health(
            ["The requested itinerary is complete."],
            response_token_limit=8192,
        )

    def test_accepts_long_tool_trajectory(self):
        validate_initial_rollout_health(
            ['<tool_call>{"name":"search"}</tool_call>' + "x" * 9000],
            response_token_limit=8192,
        )


if __name__ == "__main__":
    unittest.main()
