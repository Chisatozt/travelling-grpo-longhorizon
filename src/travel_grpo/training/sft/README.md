# training/sft

这里是 canonical SFT 的项目侧预处理层：负责 Qwen token mask、task-level split 和输入校验；实际 FSDP trainer 仍在 verl/trainer/fsdp_sft_trainer.py。

- qwen35_mask.py：将监督限定在正确的 assistant 行为 token；
- sft_split.py：生成固定的 train/validation task split；
- validate_travel_canonical.py：校验消息、工具、mask 和长度。

训练入口、数据格式和 LoRA 合并见 [SFT 文档](../../../../docs/sft.md)。
