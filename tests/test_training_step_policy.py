from __future__ import annotations

import unittest

from verl.trainer.ppo.experiment_integrity import (
    ExperimentIntegrityError,
    resolve_training_step_policy,
    validate_process_run_until_step,
    validate_run_until_step,
    validate_total_training_steps,
)


class TrainingStepPolicyTests(unittest.TestCase):
    def test_named_training_profiles(self):
        self.assertEqual(
            resolve_training_step_policy("production"),
            ("production", 100, (20, 40, 60, 80, 100)),
        )
        self.assertEqual(
            resolve_training_step_policy("overfit_one"),
            ("overfit_one", 10, (10,)),
        )
        self.assertEqual(
            resolve_training_step_policy("overfit_four"),
            ("overfit_four", 20, (20,)),
        )
        with self.assertRaises(ExperimentIntegrityError):
            resolve_training_step_policy("custom")

    def test_supported_training_milestones(self):
        self.assertEqual(validate_run_until_step(None, 100), 100)
        self.assertEqual(validate_run_until_step(20, 100), 20)
        with self.assertRaises(ExperimentIntegrityError):
            validate_run_until_step(101, 100)
        self.assertEqual(validate_process_run_until_step(None), 100)
        for step in (20, 40, 60, 80, 100):
            self.assertEqual(validate_process_run_until_step(step), step)
        with self.assertRaises(ExperimentIntegrityError):
            validate_process_run_until_step(50)
        self.assertEqual(validate_total_training_steps(None), 100)
        with self.assertRaises(ExperimentIntegrityError):
            validate_total_training_steps(50)


if __name__ == "__main__":
    unittest.main()
