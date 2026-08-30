from __future__ import annotations

import unittest

from verl.utils.dataset.rl_dataset import _infer_travel_variant


class RLDatasetPoolTests(unittest.TestCase):
    def test_variant_identity_is_path_derived(self):
        self.assertEqual(
            _infer_travel_variant(
                r"D:\data\travel22_multiturn_onechoice\train.parquet"
            ),
            "travel22",
        )
        # The aggregate compatibility parquet has no collision-resistant
        # composition identity and must therefore be rejected when a formal
        # env_name::task_id pool is requested.
        self.assertIsNone(_infer_travel_variant(r"D:\data\alltrain_multiturn\train.parquet"))


if __name__ == "__main__":
    unittest.main()
