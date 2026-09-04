from __future__ import annotations

import unittest

import torch.nn as nn

from verl.utils.fsdp_utils import get_fsdp_wrap_policy


class PresentLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)


class MixedLayerModel(nn.Module):
    _no_split_modules = ["PresentLayer", "AbsentVisionLayer"]

    def __init__(self):
        super().__init__()
        self.layer = PresentLayer()


class MissingLayerModel(nn.Module):
    _no_split_modules = ["AbsentTextLayer", "AbsentVisionLayer"]


class FSDPWrapPolicyTests(unittest.TestCase):
    def test_uses_present_layer_when_optional_declared_layer_is_absent(self):
        policy = get_fsdp_wrap_policy(MixedLayerModel())
        self.assertIsNotNone(policy)

    def test_rejects_model_when_no_declared_layer_exists(self):
        with self.assertRaisesRegex(Exception, "Could not find any transformer layer"):
            get_fsdp_wrap_policy(MissingLayerModel())


if __name__ == "__main__":
    unittest.main()
