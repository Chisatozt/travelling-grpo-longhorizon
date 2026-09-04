from __future__ import annotations

import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


class TrainingConfigTests(unittest.TestCase):
    def test_travel_sft_uses_length_aware_batch_eight_for_three_epochs(self):
        config = yaml.safe_load(
            (ROOT / "verl" / "trainer" / "config" / "travel_qwen35_sft.yaml").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(config["data"]["train_batch_size"], 8)
        self.assertEqual(config["data"]["micro_batch_size_per_gpu"], 1)
        self.assertTrue(config["data"]["multiturn"]["dynamic_padding"])
        self.assertTrue(config["data"]["multiturn"]["length_bucketing"])
        self.assertEqual(config["data"]["multiturn"]["pad_to_multiple_of"], 128)
        self.assertEqual(config["trainer"]["total_epochs"], 3)

    def test_grpo_defaults(self):
        config = yaml.safe_load(
            (ROOT / "examples" / "sglang_multiturn" / "config" / "grpo_multiturn.yaml").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(config["data"]["train_batch_size"], 4)
        self.assertTrue(config["algorithm"]["dynamic_sampling"]["enable"])
        self.assertEqual(config["algorithm"]["turn_credit"]["method"], "component_attribution")
        self.assertEqual(config["algorithm"]["turn_credit"]["stage"], "train")
        self.assertEqual(config["algorithm"]["turn_credit"]["mix_ratio"], 0.30)
        self.assertEqual(config["algorithm"]["turn_credit"]["conservation_atol"], 1.0e-5)
        self.assertEqual(config["algorithm"]["turn_credit"]["conservation_rtol"], 1.0e-6)
        self.assertEqual(
            config["algorithm"]["turn_credit"]["routing"]["causal_routing"]["correct_completion"],
            [0.15, 0.25, 0.60],
        )
        self.assertEqual(
            config["actor_rollout_ref"]["rollout"]["multi_turn"]["turn_level_method"],
            "component_attribution",
        )
        self.assertEqual(config["algorithm"]["dynamic_sampling"]["numerical_epsilon"], 1.0e-6)
        self.assertEqual(config["algorithm"]["dynamic_sampling"]["min_reward_spread"], 5.0e-3)
        self.assertEqual(config["actor_rollout_ref"]["rollout"]["multi_turn"]["max_turns"], 25)
        self.assertEqual(config["actor_rollout_ref"]["actor"]["clip_ratio_low"], 0.15)
        self.assertEqual(config["actor_rollout_ref"]["actor"]["clip_ratio_high"], 0.25)
        self.assertEqual(config["actor_rollout_ref"]["actor"]["ppo_mini_batch_size"], 2)
        self.assertEqual(config["actor_rollout_ref"]["actor"]["optim"]["lr"], 1.0e-5)
        self.assertEqual(config["actor_rollout_ref"]["model"]["lora_rank"], 32)
        self.assertEqual(config["actor_rollout_ref"]["model"]["lora_alpha"], 64)
        self.assertEqual(config["trainer"]["experiment_profile"], "production")
        self.assertTrue(config["trainer"]["initial_rollout_health_gate"])
        self.assertEqual(config["trainer"]["validation_retry_attempts"], 2)
        self.assertEqual(config["trainer"]["validation_pass_k"], 1)
        self.assertTrue(config["trainer"]["validation_task_level_early_stop"])
        self.assertFalse(config["trainer"]["val_only"])
        self.assertEqual(config["actor_rollout_ref"]["rollout"]["multi_turn"]["response_token_buffer"], 0)
        self.assertEqual(config["actor_rollout_ref"]["rollout"]["multi_turn"]["max_new_tokens_per_turn"], 4096)
        self.assertEqual(config["actor_rollout_ref"]["rollout"]["multi_turn"]["max_reasoning_tokens_per_turn"], 2560)
        self.assertEqual(config["actor_rollout_ref"]["rollout"]["multi_turn"]["max_tool_call_tokens_per_turn"], 512)
        self.assertEqual(config["actor_rollout_ref"]["rollout"]["multi_turn"]["tool_response_token_reserve"], 6144)
        self.assertEqual(config["actor_rollout_ref"]["rollout"]["multi_turn"]["template_token_reserve"], 32)
        self.assertFalse(config["actor_rollout_ref"]["rollout"]["val_kwargs"]["do_sample"])
        self.assertEqual(config["actor_rollout_ref"]["rollout"]["val_kwargs"]["temperature"], 0)
        self.assertEqual(config["actor_rollout_ref"]["rollout"]["val_kwargs"]["n"], 1)
        self.assertEqual(config["trainer"]["total_training_steps"], 100)
        self.assertEqual(config["trainer"]["milestones"], [20, 40, 60, 80, 100])
        self.assertEqual(config["trainer"]["save_freq"], 20)

    def test_grpo_launchers_expose_runtime_defaults(self):
        stage = (ROOT / "examples" / "sglang_multiturn" / "run_grpo_stage.sh").read_text()
        train = (ROOT / "examples" / "sglang_multiturn" / "train.sh").read_text()
        native = (ROOT / "eval" / "native_validate.sh").read_text()

        self.assertIn('actor_rollout_ref.actor.optim.lr=1e-5', train)
        self.assertIn('actor_rollout_ref.model.lora_rank=32', train)
        self.assertIn('actor_rollout_ref.model.lora_alpha=64', train)
        self.assertIn('TURN_CREDIT_STAGE="${TURN_CREDIT_STAGE:-train}"', train)
        self.assertIn('export OMP_NUM_THREADS=8', train)
        self.assertIn('GRPO_AUTO_SHUTDOWN', train)
        self.assertIn('--task-kind "$GRPO_MONITOR_KIND"', train)

        self.assertIn('OVERFIT_ACTOR_LR:-1e-5', stage)
        self.assertIn('GRPO_MONITOR_KIND=training', stage)
        self.assertIn('OMP_NUM_THREADS=8', stage)

        self.assertIn('trainer.validation_pass_k="$PASS_K"', native)
        self.assertIn('max_reasoning_tokens_per_turn=2560', native)
        self.assertIn('max_tool_call_tokens_per_turn=512', native)
        self.assertIn('GRPO_MONITOR_KIND=validation', native)
        self.assertIn('OMP_NUM_THREADS=8', native)



if __name__ == "__main__":
    unittest.main()
