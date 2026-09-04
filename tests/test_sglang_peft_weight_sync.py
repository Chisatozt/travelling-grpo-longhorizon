from __future__ import annotations

import unittest

import torch

from verl.utils.model import merge_peft_state_dict_for_inference


class SGLangPeftWeightSyncTests(unittest.TestCase):
    def test_merges_lora_and_emits_hf_names(self):
        state = {
            "base_model.model.model.embed_tokens.weight": torch.tensor([[1.0, 2.0]]),
            "base_model.model.model.layers.0.self_attn.q_proj.base_layer.weight": torch.tensor(
                [[1.0, 0.0], [0.0, 1.0]]
            ),
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight": torch.tensor(
                [[1.0, 2.0]]
            ),
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight": torch.tensor(
                [[3.0], [4.0]]
            ),
            "base_model.model.lm_head.weight": torch.tensor([[5.0, 6.0]]),
        }
        module_name = "base_model.model.model.layers.0.self_attn.q_proj"
        merged = merge_peft_state_dict_for_inference(
            state, lora_scalings={module_name: 2.0}
        )
        self.assertEqual(
            list(merged),
            [
                "model.embed_tokens.weight",
                "model.layers.0.self_attn.q_proj.weight",
                "lm_head.weight",
            ],
        )
        torch.testing.assert_close(
            merged["model.layers.0.self_attn.q_proj.weight"],
            torch.tensor([[7.0, 12.0], [8.0, 17.0]]),
        )

    def test_zero_initialized_lora_preserves_merged_sft_weight(self):
        state = {
            "_fsdp_wrapped_module.base_model.model.model.layers.0.mlp.up_proj.base_layer.weight": torch.tensor(
                [[2.0, 3.0]]
            ),
            "_fsdp_wrapped_module.base_model.model.model.layers.0.mlp.up_proj.lora_A.default.weight": torch.tensor(
                [[7.0, 8.0]]
            ),
            "_fsdp_wrapped_module.base_model.model.model.layers.0.mlp.up_proj.lora_B.default.weight": torch.zeros(
                1, 1
            ),
        }
        module_name = "base_model.model.model.layers.0.mlp.up_proj"
        merged = merge_peft_state_dict_for_inference(
            state, lora_scalings={module_name: 2.0}
        )
        torch.testing.assert_close(
            merged["model.layers.0.mlp.up_proj.weight"], torch.tensor([[2.0, 3.0]])
        )

    def test_missing_scaling_fails_instead_of_silently_loading_base(self):
        state = {
            "base_model.model.model.layers.0.mlp.up_proj.base_layer.weight": torch.ones(1, 1),
            "base_model.model.model.layers.0.mlp.up_proj.lora_A.default.weight": torch.ones(1, 1),
            "base_model.model.model.layers.0.mlp.up_proj.lora_B.default.weight": torch.ones(1, 1),
        }
        with self.assertRaisesRegex(ValueError, "Missing LoRA scaling"):
            merge_peft_state_dict_for_inference(state, lora_scalings={})


if __name__ == "__main__":
    unittest.main()
