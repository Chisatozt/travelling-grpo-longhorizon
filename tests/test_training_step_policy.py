from __future__ import annotations

import unittest

from verl.trainer.ppo.experiment_integrity import (
    ExperimentIntegrityError,
    validate_process_run_until_step,
    validate_run_until_step,
    validate_total_training_steps,
)


class TrainingStepPolicyTests(unittest.TestCase):
    def test_supported_training_milestones(self):
        self.assertEqual(validate_run_until_step(None, 200), 200)
        self.assertEqual(validate_run_until_step(5, 200), 5)
        with self.assertRaises(ExperimentIntegrityError):
            validate_run_until_step(201, 200)
        self.assertEqual(validate_process_run_until_step(None), 200)
        self.assertEqual(validate_process_run_until_step(5), 5)
        with self.assertRaises(ExperimentIntegrityError):
            validate_process_run_until_step(100)
        self.assertEqual(validate_total_training_steps(None), 200)
        with self.assertRaises(ExperimentIntegrityError):
            validate_total_training_steps(50)


if __name__ == "__main__":
    unittest.main()
