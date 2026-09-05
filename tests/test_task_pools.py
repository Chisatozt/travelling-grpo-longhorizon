"""Acceptance tests for the disjoint TravelGym task partition."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from travel_grpo.collection.task_pools import (
    OPAQUE_TASK_KEY_PREFIX,
    TaskPoolError,
    assert_task_pools_disjoint,
    build_sft_candidate_audit,
    is_permanently_quarantined_opaque,
    load_pool_manifest,
    pool_task_keys,
)
from travel_grpo.collection.merge_travel_sft import prepare_canonical_inputs
from travel_grpo.collection.travel_canonical import canonical_hash, canonicalize_record
from travel_grpo.collection.travel_task_resolver import TravelTaskResolver


ROOT = Path(__file__).resolve().parents[1]


class TaskPoolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = ROOT / "data" / "task_pools" / "travel_task_pools.json"
        cls.manifest = load_pool_manifest(cls.path)

    def test_active_pools_are_pairwise_disjoint(self):
        assert_task_pools_disjoint(self.manifest)
        sets = {name: pool_task_keys(self.manifest, name) for name in ("sft", "grpo", "validation")}
        self.assertEqual(len(sets["sft"] & sets["grpo"]), 0)
        self.assertEqual(len(sets["sft"] & sets["validation"]), 0)
        self.assertEqual(len(sets["grpo"] & sets["validation"]), 0)

    def test_expected_pool_sizes_and_validation_views(self):
        self.assertEqual(len(self.manifest["pools"]["sft"]["records"]), 600)
        self.assertEqual(len(self.manifest["pools"]["grpo"]["records"]), 2051)
        self.assertEqual(len(self.manifest["pools"]["validation"]["records"]), 200)
        self.assertEqual(len(self.manifest["pools"]["validation_smoke"]["records"]), 20)
        self.assertEqual(self.manifest["sft_target_count"], 600)
        self.assertEqual(self.manifest["sft_historical_count"], 238)
        self.assertEqual(self.manifest["sft_expansion_count"], 362)
        self.assertEqual(self.manifest["quarantined_sft_count"], 6)
        self.assertTrue(self.manifest["strict_task_identity"])
        self.assertEqual(
            self.manifest["task_alignment"]["quarantine_policy"],
            "isolate_discard",
        )
        self.assertEqual(
            [item["record_index"] for item in self.manifest["quarantined_sft"]],
            [97, 144, 159, 180, 206, 208],
        )
        self.assertTrue(
            all(item.get("env_name") != "opaque_sft" for item in self.manifest["pools"]["sft"]["records"])
        )
        self.assertEqual(
            sum(item.get("role") == "teacher_expansion" for item in self.manifest["pools"]["sft"]["records"]),
            362,
        )
        smoke = pool_task_keys(self.manifest, "validation_smoke")
        validation = pool_task_keys(self.manifest, "validation")
        self.assertTrue(smoke <= validation)

        final200 = json.loads(
            (ROOT / "data" / "evaluation" / "test_manifests" / "final200.json").read_text(encoding="utf-8")
        )
        final_ids = {(item["env_name"] + "::" + item["task_id"]) for item in final200["records"]}
        self.assertEqual(validation, final_ids)

    def test_strict_mode_accepts_discarded_opaque_rows(self):
        # The six opaque source rows are discarded rather than mapped, so the
        # active pools are strict without a reviewed alignment sidecar.
        self.assertTrue(self.manifest["strict_task_identity"])
        assert_task_pools_disjoint(self.manifest, require_strict=True)

        # A hand-edited string value must not bypass the strict gate.
        string_flag = json.loads(json.dumps(self.manifest))
        string_flag["strict_task_identity"] = "false"
        with self.assertRaises(TaskPoolError):
            assert_task_pools_disjoint(string_flag, require_strict=True)

    def test_candidate_audit_never_guesses_opaque_rows(self):
        report = json.loads(
            (ROOT / "data" / "task_pools" / "sft_task_alignment_candidates.json").read_text(encoding="utf-8")
        )
        self.assertFalse(report["requires_review"])
        self.assertEqual(report["quarantine_policy"], "isolate_discard")
        self.assertEqual([row["record_index"] for row in report["records"]], [97, 144, 159, 180, 206, 208])
        self.assertTrue(
            all(
                not row["requires_review"]
                and not row["resolved_without_review"]
                and row["quarantine_status"] == "discarded"
                for row in report["records"]
            )
        )

    def test_opaque_quarantine_is_permanent_and_active_pools_reject_it(self):
        first = self.manifest["quarantined_sft"][0]
        self.assertTrue(first["opaque_task_key"].startswith(OPAQUE_TASK_KEY_PREFIX))
        source = json.loads(
            (ROOT / "data" / "sft" / "travel_sft_public.json").read_text(encoding="utf-8")
        )
        self.assertTrue(
            is_permanently_quarantined_opaque(
                source[first["record_index"]],
                source_index=first["record_index"],
                source_path=ROOT / "data" / "sft" / "travel_sft_public.json",
            )
        )
        broken = json.loads(json.dumps(self.manifest))
        broken["pools"]["sft"]["records"].append(
            {"env_name": "opaque_sft", "task_id": first["opaque_task_key"], "split": "quarantine"}
        )
        with self.assertRaises(TaskPoolError):
            assert_task_pools_disjoint(broken)

    def test_canonical_merge_discards_all_six_opaque_rows(self):
        records, audits, stats = prepare_canonical_inputs(
            [ROOT / "data" / "sft" / "travel_sft_public.json"],
            resolver=TravelTaskResolver(project_root=ROOT),
            require_think=False,
        )
        self.assertEqual(stats["source_records"], 244)
        self.assertEqual(stats["opaque_quarantined"], 6)
        self.assertEqual(len(records), 238)
        self.assertEqual(len(audits), 238)
        self.assertTrue(all("opaque_task_key" not in audit for audit in audits))

    def test_reviewed_opaque_key_from_audit_is_resolvable(self):
        # The private audit exposes canonical_hash-based opaque keys.  Ensure
        # the resolver consumes that exact key rather than a second hash
        # spelling that would make a reviewed map ineffective.
        with tempfile.TemporaryDirectory() as directory:
            task_path = Path(directory) / "travelgym_data_22.json"
            task_path.write_text(json.dumps({
                "task-1": {
                    "initial_description": "Plan a flight to Kyoto.",
                    "dimensions": ["flight"],
                    "preferences": {"flight": {"correct_ids": ["F1"], "best_id": "F1"}},
                }
            }), encoding="utf-8")
            record = {"messages": [{"role": "user", "content": "Plan a flight to Kyoto."}]}
            opaque = "opaque_sft::" + canonical_hash(canonicalize_record(record))
            resolver = TravelTaskResolver(
                project_root=directory,
                task_paths=[task_path],
                explicit_map={opaque: "travel22::task-1"},
            )
            task_id, _ = resolver.resolve(record, {"source_index": 0})
            self.assertEqual(task_id, "task-1")

    def test_reviewed_path_map_accepts_repository_relative_spelling(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task_path = root / "travelgym_data_22.json"
            task_path.write_text(json.dumps({
                "task-1": {
                    "initial_description": "Plan a flight to Kyoto.",
                    "dimensions": ["flight"],
                    "preferences": {"flight": {"correct_ids": ["F1"], "best_id": "F1"}},
                }
            }), encoding="utf-8")
            record = {"messages": [{"role": "user", "content": "opaque request"}]}
            resolver = TravelTaskResolver(
                project_root=root,
                task_paths=[task_path],
                explicit_map={"data/sft/travel_sft_public.json#97": "travel22::task-1"},
            )
            task_id, _ = resolver.resolve(
                record,
                {"source_index": 97, "source_path": str(root / "data" / "sft" / "travel_sft_public.json")},
            )
            # The map key is intentionally portable and should resolve only
            # because the reviewed sidecar explicitly names the task.
            self.assertEqual(task_id, "task-1")

    def test_overlap_is_rejected(self):
        broken = json.loads(json.dumps(self.manifest))
        broken["pools"]["grpo"]["records"].append(
            dict(broken["pools"]["sft"]["records"][0])
        )
        with self.assertRaises(TaskPoolError):
            assert_task_pools_disjoint(broken)


if __name__ == "__main__":
    unittest.main()
