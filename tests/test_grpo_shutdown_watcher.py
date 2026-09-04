from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.grpo_shutdown_watcher import grpo_artifact_status, poweroff_commands, training_log_status


class GRPOShutdownWatcherTests(unittest.TestCase):
    def test_final_artifact_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoints, artifacts = root / "checkpoints", root / "artifacts"
            self.assertFalse(grpo_artifact_status(checkpoints, artifacts, 20)["complete"])
            actor = checkpoints / "global_step_20/actor"
            actor.mkdir(parents=True)
            (actor / "adapter_model.safetensors").write_text("weights", encoding="utf-8")
            (checkpoints / "global_step_20/data.pt").write_text("data", encoding="utf-8")
            (checkpoints / "global_step_20/rng_state.pt").write_text("rng", encoding="utf-8")
            (checkpoints / "latest_checkpointed_iteration.txt").write_text("20\n", encoding="utf-8")
            (artifacts / "validation").mkdir(parents=True)
            (artifacts / "validation/20.jsonl").write_text("{}\n", encoding="utf-8")
            (artifacts / "rollouts").mkdir(parents=True)
            (artifacts / "rollouts/20.jsonl").write_text("{}\n", encoding="utf-8")
            status = grpo_artifact_status(checkpoints, artifacts, 20)
            self.assertTrue(status["complete"], status["errors"])
            self.assertEqual(status["completed_checkpoint_steps"], [20])
            self.assertEqual(status["completed_validation_steps"], [20])
            self.assertEqual(status["completed_rollout_steps"], [20])

    def test_validation_artifact_contract_without_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoints, artifacts = root / "checkpoints", root / "artifacts"
            self.assertFalse(
                grpo_artifact_status(checkpoints, artifacts, 0, "validation")["complete"]
            )
            validation = artifacts / "validation"
            validation.mkdir(parents=True)
            (validation / "0_pass3.jsonl").write_text("{}\n", encoding="utf-8")
            (validation / "0_pass3_summary.json").write_text("{}\n", encoding="utf-8")
            status = grpo_artifact_status(checkpoints, artifacts, 0, "validation")
            self.assertTrue(status["complete"], status["errors"])
            self.assertEqual(status["completed_validation_steps"], [0])
            self.assertEqual(status["task_kind"], "validation")

    def test_training_log_status(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.log"
            path.write_text(
                "step:2 - grpo/train/reward:0.1\n"
                "RuntimeError: response length 25000 exceeds max_response_len=24576\n",
                encoding="utf-8",
            )
            status = training_log_status(path)
            self.assertEqual(status["latest_step"], 2)
            self.assertTrue(status["failure_lines"])

    def test_missing_log(self):
        with tempfile.TemporaryDirectory() as directory:
            status = training_log_status(Path(directory) / "missing.log")
            self.assertFalse(status["exists"])
            self.assertIsNone(status["latest_step"])

    def test_autodl_shutdown_wrapper_is_default(self):
        safe_commands = poweroff_commands(False)
        opted_in_commands = poweroff_commands(True)
        wrapper = ["/bin/bash", "/usr/bin/shutdown"]
        self.assertEqual(safe_commands[0], wrapper)
        self.assertIn(wrapper, opted_in_commands)


if __name__ == "__main__":
    unittest.main()
