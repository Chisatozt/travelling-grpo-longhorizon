from __future__ import annotations

import unittest

from travel_grpo.evaluation.final200 import Final200Error, build_final200_plan, evaluate_final200


class Final200PlanTests(unittest.TestCase):
    def setUp(self):
        records = [{"env_name": "travel22" if i < 25 else "travel33", "task_id": f"task-{i}"} for i in range(200)]
        self.pool = {"strict_task_identity": True, "pools": {"validation": {"records": records}}}
        self.smoke = {"records": records[:20]}
        self.models = {"base": "base", "sft": "sft", "grpo": "grpo", "deepseek": "deepseek"}

    def test_plan_has_all200_seen_unseen_and_manual_grpo(self):
        plan = build_final200_plan(task_pool_manifest=self.pool, smoke_manifest=self.smoke, models=self.models, seed=1, max_turns=25, selected_grpo_checkpoint="ckpt")
        self.assertEqual(len(plan["splits"]["all200"]), 200)
        self.assertEqual(len(plan["splits"]["smoke20_seen"]), 20)
        self.assertEqual(len(plan["splits"]["unseen180"]), 180)
        self.assertEqual(plan["models"]["grpo"], "ckpt")
        self.assertEqual(plan["protocol"]["pass_k"], 3)
        self.assertTrue(plan["protocol"]["task_level_early_stop"])
        self.assertTrue(plan["protocol"]["native_two_stage"])
        self.assertEqual(plan["protocol"]["template_prefill"], "<think>")
        self.assertEqual(plan["protocol"]["reasoning_max_tokens_per_turn"], 2560)
        self.assertEqual(plan["protocol"]["tool_call_max_tokens_per_turn"], 512)

    def test_missing_selection_is_blocked(self):
        with self.assertRaises(Final200Error):
            build_final200_plan(task_pool_manifest=self.pool, smoke_manifest=self.smoke, models=self.models, seed=1, max_turns=25)

    def test_reward_version_cannot_drift(self):
        with self.assertRaises(Final200Error):
            build_final200_plan(
                task_pool_manifest=self.pool,
                smoke_manifest=self.smoke,
                models=self.models,
                seed=1,
                max_turns=25,
                reward_version="different-reward-v0",
                selected_grpo_checkpoint="ckpt",
            )

    def test_offline_evaluator_runs_four_models_and_comparison_without_raw_reports(self):
        plan = build_final200_plan(
            task_pool_manifest=self.pool,
            smoke_manifest=self.smoke,
            models=self.models,
            seed=1,
            max_turns=25,
            selected_grpo_checkpoint="ckpt",
        )
        calls = []
        trackers = []

        class Tracker:
            def __init__(self, name):
                self.name = name
                self.logs = []

            def log(self, *, data, step):
                self.logs.append((data, step))

            def finish(self):
                pass

        def runner(*, model_name, task_key, protocol):
            calls.append((model_name, task_key, protocol["reward_version"]))
            # Deliberately include a private-looking field to prove the
            # orchestration result keeps aggregate scalars only.
            return {
                "terminal_reward": 0.5,
                "completion_success": 1.0,
                "correct_completion": 1.0,
                "best_answer_rate": 1.0,
                "answer_coverage": 1.0,
                "legal_chain_rate": 1.0,
                "efficiency": 0.8,
                "correct_ids": ["F1"],
            }

        def tracking_factory(*, name, run_spec, protocol):
            tracker = Tracker(name)
            trackers.append((name, run_spec, tracker))
            return tracker

        result = evaluate_final200(plan, runner=runner, tracking_factory=tracking_factory)
        self.assertEqual(len(calls), 4 * 200)
        self.assertEqual([item[0] for item in trackers], ["base", "sft", "grpo", "deepseek", "comparison"])
        self.assertEqual(result["models"]["base"]["all"]["episodes"], 200.0)
        self.assertEqual(result["models"]["base"]["smoke20_seen"]["episodes"], 20.0)
        self.assertEqual(result["models"]["base"]["unseen180"]["episodes"], 180.0)
        serialized = str(result)
        self.assertNotIn("correct_ids", serialized)
        self.assertIn("all/base_terminal_reward", [
            key for key, _value in result["comparison"].items()
        ])
        self.assertEqual(result["models"]["base"]["all"]["pass_at_k"], 1.0)
        self.assertEqual(result["models"]["base"]["all"]["attempt_count"], 200.0)

    def test_final200_stops_each_task_after_second_attempt(self):
        plan = build_final200_plan(
            task_pool_manifest=self.pool,
            smoke_manifest=self.smoke,
            models=self.models,
            seed=1,
            max_turns=25,
            selected_grpo_checkpoint="ckpt",
        )
        calls = []

        def runner(*, model_name, task_key, protocol):
            calls.append((model_name, task_key, protocol["attempt"]))
            success = 1.0 if protocol["attempt"] == 2 else 0.0
            return {
                "terminal_reward": success,
                "completion_success": success,
                "correct_completion": success,
                "best_answer_rate": success,
                "answer_coverage": success,
                "legal_chain_rate": success,
                "efficiency": 0.5,
            }

        result = evaluate_final200(plan, runner=runner)
        self.assertEqual(len(calls), 4 * 200 * 2)
        self.assertEqual(result["models"]["base"]["all"]["pass_at_k"], 1.0)
        self.assertEqual(result["models"]["base"]["all"]["attempt_count"], 400.0)
        self.assertEqual(result["models"]["base"]["all"]["early_stopped_tasks"], 200.0)
        self.assertEqual(result["models"]["base"]["all"]["completion_success_count"], 200.0)


if __name__ == "__main__":
    unittest.main()
