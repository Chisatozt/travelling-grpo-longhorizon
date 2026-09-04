from __future__ import annotations

import unittest

from verl.trainer.ppo.validation_baseline import load_step0_validation_metrics
from verl.trainer.ppo.validation_passes import aggregate_validation_attempts


class ValidationPassAggregationTests(unittest.TestCase):
    def test_pass_at_k_uses_task_success_and_actual_early_stop_attempts(self):
        attempts = {
            "travel22::task-a": [
                {"completion_success": 0, "reward_valid": 1, "terminal_reward": 0.1},
                {"completion_success": 1, "reward_valid": 1, "terminal_reward": 0.9},
            ],
            "travel33::task-b": [
                {"completion_success": 1, "reward_valid": 1, "terminal_reward": 0.8},
            ],
            "travel44::task-c": [
                {"completion_success": 0, "reward_valid": 0, "terminal_reward": 0.0},
                {"completion_success": 0, "reward_valid": 1, "terminal_reward": 0.2},
                {"completion_success": 0, "reward_valid": 1, "terminal_reward": 0.3},
            ],
        }
        summary = aggregate_validation_attempts(attempts, pass_k=3)
        self.assertEqual(summary["task_count"], 3)
        self.assertEqual(summary["pass_count"], 2)
        self.assertAlmostEqual(summary["pass@3"], 2 / 3)
        self.assertAlmostEqual(summary["pass@1"], 1 / 3)
        self.assertEqual(summary["attempt_count"], 6)
        self.assertEqual(summary["early_stopped_tasks"], 2)
        self.assertEqual(summary["invalid_attempt_count"], 1)
        self.assertAlmostEqual(summary["valid_attempt_rate"], 5 / 6)
        self.assertAlmostEqual(summary["terminal_reward/mean@1"], 0.3)
        self.assertAlmostEqual(summary["terminal_reward/mean_best"], (0.9 + 0.8 + 0.3) / 3)

    def test_native_step0_baseline_loader_returns_public_scalars(self):
        import json
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "baseline.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": "travelgym-grpo-validation-baseline-v1",
                        "source": "native_validation",
                        "protocol": {
                            "split": "validation_smoke",
                            "native_two_stage": True,
                            "template_prefill": "<think>",
                            "reasoning_max_tokens": 2560,
                            "tool_call_max_tokens": 512,
                            "forced_reasoning_end_loss_mask": 0,
                            "pass_k": 3,
                            "task_level_early_stop": True,
                            "validation_retry_attempts": 0,
                        },
                        "step0_metrics": {
                            "grpo/val/smoke20/pass@3": 0.25,
                            "grpo/val/smoke20/attempt_count": 42,
                        },
                    }
                ),
                encoding="utf-8",
            )
            metrics = load_step0_validation_metrics(path)
            self.assertEqual(metrics["grpo/val/smoke20/pass@3"], 0.25)
            self.assertEqual(metrics["grpo/val/smoke20/attempt_count"], 42.0)

    def test_attempts_over_k_are_rejected(self):
        with self.assertRaises(ValueError):
            aggregate_validation_attempts(
                {"task": [{"completion_success": 0}] * 4}, pass_k=3
            )


if __name__ == "__main__":
    unittest.main()
