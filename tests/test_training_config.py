from __future__ import annotations

import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


class TrainingConfigTests(unittest.TestCase):
    def test_travel_sft_uses_three_epochs(self):
        config = yaml.safe_load(
            (ROOT / "verl" / "trainer" / "config" / "travel_qwen35_sft.yaml").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(config["trainer"]["total_epochs"], 3)

    def test_grpo_defaults(self):
        config = yaml.safe_load(
            (ROOT / "examples" / "sglang_multiturn" / "config" / "grpo_multiturn.yaml").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(config["data"]["train_batch_size"], 8)
        self.assertTrue(config["algorithm"]["dynamic_sampling"]["enable"])
        self.assertEqual(config["actor_rollout_ref"]["rollout"]["multi_turn"]["max_turns"], 25)
        self.assertEqual(config["trainer"]["save_freq"], 20)


if __name__ == "__main__":
    unittest.main()
