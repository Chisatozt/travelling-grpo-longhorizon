from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from sft.sft_split import SFTSplitError, build_sft_split


ENVS = ["travel22", "travel33", "travel44", "travel233", "travel333", "travel334", "travel444", "travel2222"]


def _record(task_key: str, category: str = "strict_gold", length: int = 10):
    return {
        "messages": [{"role": "user", "content": "request"}, {"role": "assistant", "content": "answer", "tool_calls": [], "reasoning_content": "think"}],
        "assistant_train_mask": [0, 1],
        "trainer_metadata": {"trajectory_class": category, "sample_weight": 1.0},
        "_test_length": length,
    }


class SFTSplitTests(unittest.TestCase):
    def test_task_grouped_validation_and_exact_audit(self):
        records, audits = [], []
        for index, env in enumerate(ENVS + ["travel22", "travel33", "travel44", "travel233"]):
            key = f"{env}::task-{index}"
            records.append(_record(key, length=5 + index * 10))
            audits.append({"task_key": key})
        # A second trajectory for the first task must follow its task group.
        records.append(_record("travel22::task-0", category="recoverable_correct", length=20))
        audits.append({"task_key": "travel22::task-0"})
        with tempfile.TemporaryDirectory() as directory:
            manifest = build_sft_split(
                records, audits, output_dir=Path(directory),
                token_length_fn=lambda row: row["_test_length"],
                require_exact_token_audit=True,
            )
            self.assertEqual(manifest["validation_task_count"], 10)
            self.assertTrue(manifest["token_audit_exact"])
            train = (Path(directory) / "train.jsonl").read_text(encoding="utf-8")
            validation = (Path(directory) / "val_gold10.jsonl").read_text(encoding="utf-8")
            self.assertNotIn('"trajectory_class": "totally_wrong"', train)
            self.assertGreater(len(validation.splitlines()), 0)

    def test_formal_split_requires_token_audit(self):
        with self.assertRaises(SFTSplitError):
            build_sft_split([_record("travel22::a")], [{"task_key": "travel22::a"}], output_dir=".")

    def test_default_split_tolerates_missing_environment_strict_gold(self):
        records, audits = [], []
        for index, env in enumerate(ENVS[:6]):
            key = f"{env}::task-{index}"
            records.append(_record(key, length=10 + index))
            audits.append({"task_key": key})
        with tempfile.TemporaryDirectory() as directory:
            manifest = build_sft_split(
                records,
                audits,
                output_dir=Path(directory),
                token_length_fn=lambda row: row["_test_length"],
                validation_count=6,
            )
            self.assertEqual(manifest["validation_task_count"], 6)

    def test_formal_split_rejects_task_outside_sft_pool(self):
        records, audits = [], []
        for index, env in enumerate(ENVS + ["travel22", "travel33"]):
            key = f"{env}::task-{index}"
            records.append(_record(key, length=10 + index))
            audits.append({"task_key": key})
        pool = {
            "manifest_version": "travelgym-task-pools-v2",
            "pools": {
                "sft": {"records": [{"task_key": f"{env}::task-{i}"} for i, env in enumerate(ENVS + ["travel22", "travel33"])]},
                "grpo": {"records": []},
                "validation": {"records": []},
                "validation_smoke": {"records": []},
            },
        }
        # The first record is deliberately outside the formal SFT pool.
        pool["pools"]["sft"]["records"] = pool["pools"]["sft"]["records"][1:]
        with self.assertRaises(SFTSplitError):
            build_sft_split(
                records,
                audits,
                output_dir=tempfile.mkdtemp(),
                task_pool_manifest=pool,
                token_length_fn=lambda row: row["_test_length"],
            )


if __name__ == "__main__":
    unittest.main()
