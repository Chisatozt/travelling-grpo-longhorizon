from __future__ import annotations

import unittest

from verl.trainer.ppo.dynamic_sampling import (
    DEFAULT_MIN_REWARD_SPREAD,
    DEFAULT_NUMERICAL_EPSILON,
    resolve_reward_spread_thresholds,
    select_reward_varying_groups,
)


class DynamicSamplingThresholdTests(unittest.TestCase):
    def test_numerical_equality_and_semantic_spread_are_distinct(self):
        kept, stats = select_reward_varying_groups(
            ["constant", "constant", "small", "small", "useful", "useful"],
            [0.5, 0.5000005, 0.5, 0.503, 0.5, 0.505],
            expected_group_size=2,
        )

        self.assertEqual(kept, [4, 5])
        self.assertEqual(stats["skip_reason_counts"]["constant_reward"], 1)
        self.assertEqual(stats["skip_reason_counts"]["insufficient_reward_spread"], 1)
        self.assertAlmostEqual(stats["groups"][2]["reward_spread"], 0.005)
        self.assertEqual(stats["numerical_epsilon"], DEFAULT_NUMERICAL_EPSILON)
        self.assertEqual(stats["min_reward_spread"], DEFAULT_MIN_REWARD_SPREAD)

    def test_minimum_semantic_spread_is_inclusive(self):
        kept, stats = select_reward_varying_groups(
            ["task", "task"],
            [0.25, 0.255],
            expected_group_size=2,
            numerical_epsilon=1.0e-6,
            min_reward_spread=0.005,
        )

        self.assertEqual(kept, [0, 1])
        self.assertEqual(stats["trainable_group_count"], 1)

    def test_spread_clearly_below_semantic_minimum_is_skipped(self):
        kept, stats = select_reward_varying_groups(
            ["task", "task"],
            [0.1, 0.1049],
            expected_group_size=2,
        )

        self.assertEqual(kept, [])
        self.assertEqual(
            stats["skip_reason_counts"],
            {"insufficient_reward_spread": 1},
        )

    def test_legacy_reward_tolerance_is_a_numerical_alias(self):
        self.assertEqual(
            resolve_reward_spread_thresholds(
                {"reward_tolerance": 2.0e-6, "min_reward_spread": 0.01}
            ),
            (2.0e-6, 0.01),
        )

    def test_invalid_thresholds_fail_fast(self):
        for config in (
            {"numerical_epsilon": -1.0},
            {"numerical_epsilon": float("nan")},
            {"min_reward_spread": -1.0},
            {"min_reward_spread": float("inf")},
        ):
            with self.subTest(config=config), self.assertRaises(ValueError):
                resolve_reward_spread_thresholds(config)


if __name__ == "__main__":
    unittest.main()
