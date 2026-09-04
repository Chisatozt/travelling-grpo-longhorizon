from __future__ import annotations
import tempfile
import unittest
from pathlib import Path
from scripts.grpo_one_shutdown_runner import merged_model_complete, verify_overfit_artifacts


class GRPOOneShutdownRunnerTests(unittest.TestCase):
    def test_merged_model_requires_weights_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertFalse(merged_model_complete(root)[0])
            for name in ("model-00001-of-00001.safetensors", "config.json", "tokenizer_config.json", "merge_metadata.json"):
                (root / name).write_text("x", encoding="utf-8")
            ok, errors = merged_model_complete(root)
            self.assertTrue(ok, errors)

    def test_final_artifacts_require_checkpoint_and_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, artifacts = root / "checkpoints", root / "artifacts"
            self.assertFalse(verify_overfit_artifacts(checkpoint, artifacts)["complete"])
            actor = checkpoint / "global_step_10/actor"
            actor.mkdir(parents=True)
            (actor / "adapter_model.safetensors").write_text("weights", encoding="utf-8")
            (checkpoint / "global_step_10/data.pt").write_text("data", encoding="utf-8")
            (checkpoint / "global_step_10/rng_state.pt").write_text("rng", encoding="utf-8")
            (checkpoint / "latest_checkpointed_iteration.txt").write_text("10", encoding="utf-8")
            (artifacts / "validation").mkdir(parents=True)
            (artifacts / "validation/10.jsonl").write_text("{}\n", encoding="utf-8")
            (artifacts / "rollouts").mkdir(parents=True)
            (artifacts / "rollouts/10.jsonl").write_text("{}\n", encoding="utf-8")
            status = verify_overfit_artifacts(checkpoint, artifacts)
            self.assertTrue(status["complete"], status["errors"])


if __name__ == "__main__":
    unittest.main()
