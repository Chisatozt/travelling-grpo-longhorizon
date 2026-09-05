from __future__ import annotations

import json
import unittest
from pathlib import Path

from travel_grpo.collection.task_pools import assert_task_pools_disjoint
from travel_grpo.collection.travel_canonical import canonical_hash


ROOT = Path(__file__).resolve().parents[1]
FOUR_TASK_ENV_ORDER = {
    "travel22",
    "travel33",
    "travel44",
    "travel233",
}


class GRPOPreflightPoolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest_path = ROOT / "data/task_pools/travel_grpo_overfit_pools.json"
        cls.manifest = json.loads(cls.manifest_path.read_text(encoding="utf-8"))

    def test_generated_manifest_is_current(self):
        import subprocess

        subprocess.run(
            ["python", str(ROOT / "scripts/prepare_grpo_overfit_pools.py"), "--check"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

    def test_overfit_pools_are_sft_seen_and_formal_pools_remain_disjoint(self):
        assert_task_pools_disjoint(self.manifest, require_strict=True)
        pools = self.manifest["pools"]
        one_keys = {record["task_key"] for record in pools["grpo_overfit_one"]["records"]}
        four_keys = {record["task_key"] for record in pools["grpo_overfit_four"]["records"]}
        sft_keys = {record["task_key"] for record in pools["sft"]["records"]}
        grpo_keys = {record["task_key"] for record in pools["grpo"]["records"]}
        validation_keys = {record["task_key"] for record in pools["validation"]["records"]}

        self.assertEqual(len(one_keys), 1)
        self.assertEqual(len(four_keys), 4)
        self.assertLessEqual(one_keys, four_keys)
        self.assertLessEqual(four_keys, sft_keys)
        self.assertFalse(four_keys & grpo_keys)
        self.assertFalse(four_keys & validation_keys)
        self.assertEqual({key.split("::", 1)[0] for key in four_keys}, FOUR_TASK_ENV_ORDER)

    def test_selected_tasks_are_in_actual_sft_train_split(self):
        audits = json.loads(
            (ROOT / "data/sft/travel_sft_qwen35_merged.audit.json").read_text(encoding="utf-8")
        )["records"]
        audit_by_hash = {
            record["canonical_hash_after"]: record
            for record in audits
            if record.get("canonical_hash_after")
        }
        train_task_keys = set()
        for line in (ROOT / "data/sft/travel_sft_qwen35_split/train.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            audit = audit_by_hash[canonical_hash(record)]
            train_task_keys.add(audit["task_key"])

        selected = self.manifest["grpo_overfit_selection"]
        self.assertIn(selected["one_task_key"], train_task_keys)
        self.assertLessEqual(set(selected["four_task_keys"]), train_task_keys)
        for record in selected["records"]:
            self.assertEqual(record["trajectory_class"], "partial_correct")
            self.assertEqual(record["sample_weight"], 0.5)
            self.assertGreater(record["correct_completion"], 0.0)
            self.assertLess(record["correct_completion"], 1.0)
            self.assertEqual(record["completion_success"], 0)


if __name__ == "__main__":
    unittest.main()
