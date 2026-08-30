from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path

from eval.teacher_collection import TeacherCache, TeacherCacheError, make_provenance, sanitize_tracking_payload
from sft.clean_travel_trajectories import clean_trajectory
from sft.merge_travel_sft import merge_canonical_records
from sft.travel_canonical import iter_source_records


class TeacherCollectionTests(unittest.TestCase):
    def test_pass_level_resume_and_deduplication(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "teacher.json"
            cache = TeacherCache(path, collection_run_id="run-1")
            tasks = [{"env_name": "travel22", "task_id": "t1"}]
            self.assertEqual(len(cache.pending(tasks)), 2)
            provenance = make_provenance(
                env_name="travel22", task_id="t1", pass_index=0, model="deepseek-v4-flash",
                collection_run_id="run-1",
            )
            trajectory = {"messages": [{"role": "assistant", "reasoning_content": "plan", "tool_calls": []}]}
            self.assertTrue(cache.record_success(env_name="travel22", task_id="t1", pass_index=0, trajectory=trajectory, provenance=provenance))
            self.assertFalse(cache.record_success(env_name="travel22", task_id="t1", pass_index=0, trajectory=trajectory, provenance=provenance))
            self.assertEqual(cache.pending(tasks), [{"env_name": "travel22", "task_id": "t1", "pass_index": 1, "task_key": "travel22::t1::1"}])
            resumed = TeacherCache(path, collection_run_id="run-1")
            self.assertEqual(len(resumed.pending(tasks)), 1)
            self.assertIsNotNone(resumed.claim(env_name="travel22", task_id="t1", pass_index=1))
            self.assertEqual(
                resumed.records["travel22::t1::1"]["provenance"]["task_key"],
                "travel22::t1",
            )
            self.assertIn("code_revision", resumed.records["travel22::t1::1"]["provenance"])
            self.assertEqual(resumed.pending(tasks), [])

    def test_tracking_sanitizes_private_fields_but_cache_does_not(self):
        sanitized = sanitize_tracking_payload({"reasoning_content": "secret", "correct_ids": ["F1"], "collection/completed_tasks": 2})
        self.assertNotIn("reasoning_content", sanitized)
        self.assertNotIn("correct_ids", sanitized)
        self.assertEqual(sanitized["collection/completed_tasks"], 2)

    def test_invalid_completed_pass_is_not_rebilled_on_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "teacher.json"
            cache = TeacherCache(path, collection_run_id="run-1")
            provenance = make_provenance(
                env_name="travel22", task_id="t1", pass_index=0, model="deepseek-v4-flash",
                collection_run_id="run-1",
            )
            self.assertTrue(cache.record_success(
                env_name="travel22", task_id="t1", pass_index=0,
                trajectory={"messages": []}, provenance=provenance, valid=False,
            ))
            pending = cache.pending([{"env_name": "travel22", "task_id": "t1"}])
            self.assertEqual(pending, [{"env_name": "travel22", "task_id": "t1", "pass_index": 1, "task_key": "travel22::t1::1"}])
            self.assertIsNone(cache.claim(env_name="travel22", task_id="t1", pass_index=0))

    def test_collection_stats_include_invalid_and_missing_reasoning(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = TeacherCache(Path(directory) / "teacher.json", collection_run_id="run-1")
            provenance = make_provenance(
                env_name="travel22", task_id="t1", pass_index=0, model="deepseek-v4-flash",
                collection_run_id="run-1",
            )
            cache.record_success(
                env_name="travel22", task_id="t1", pass_index=0,
                trajectory={"messages": []}, provenance=provenance,
                valid=False, missing_reasoning_count=2,
            )
            self.assertEqual(cache.stats["invalid_trajectories"], 1)
            self.assertEqual(cache.stats["missing_reasoning"], 2)

    def test_provenance_must_match_task_and_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = TeacherCache(Path(directory) / "teacher.json", collection_run_id="run-1")
            trajectory = {"messages": []}
            provenance = make_provenance(
                env_name="travel22", task_id="t1", pass_index=0, model="deepseek-v4-flash",
                collection_run_id="run-1",
            )
            provenance["pass_index"] = 1
            with self.assertRaises(TeacherCacheError):
                cache.record_success(
                    env_name="travel22", task_id="t1", pass_index=0,
                    trajectory=trajectory, provenance=provenance,
                )

    def test_offline_fixture_replays_reasoning_without_network(self):
        fixture = Path(__file__).parent / "fixtures" / "offline_teacher_cache.json"
        payload = json.loads(fixture.read_text(encoding="utf-8"))
        rows = list(iter_source_records(payload))
        self.assertEqual(len(rows), 2)
        self.assertIn("<think>", rows[0][0]["messages"][1]["content"])
        self.assertEqual(rows[0][2]["pass_index"], 0)

    def test_offline_fixture_cleans_and_merges_complete_teacher_passes(self):
        """Exercise the cache -> canonical cleaner -> dedupe path without API/GPU."""
        fixture = Path(__file__).parent / "fixtures" / "offline_teacher_cache.json"
        payload = json.loads(fixture.read_text(encoding="utf-8"))
        task = {
            "id": "fixture-task",
            "dimensions": ["flight"],
            "flight": {"all_ids": ["F1", "F2"], "correct_ids": ["F1"], "best_id": "F1"},
        }
        cleaned = []
        for raw, _, meta in iter_source_records(payload):
            record, audit = clean_trajectory(
                raw,
                task=task,
                task_id="fixture-task",
                require_think=True,
                source=f"offline-pass-{meta['pass_index']}",
            )
            self.assertEqual(audit["trajectory_class"], "strict_gold")
            self.assertEqual(record["trainer_metadata"]["trajectory_class"], "strict_gold")
            self.assertEqual(record["assistant_train_mask"], [0, 1, 0, 1, 0, 1, 0])
            cleaned.append(record)
        merged, duplicate_count = merge_canonical_records(cleaned[:1], cleaned[1:])
        self.assertEqual(len(merged), 1)
        self.assertEqual(duplicate_count, 1)


if __name__ == "__main__":
    unittest.main()
