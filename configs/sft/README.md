# configs/sft

本目录保存 SFT 的兼容性配置示例。当前 canonical SFT 的权威配置不是这里的旧配置，而是 verl/trainer/config/travel_qwen35_sft.yaml。

configs/sft/qwen3_customized.yaml 只用于旧 ShareGPT renderer 回归检查；canonical 数据、Qwen token mask、训练命令和模型合并见 [SFT 文档](../../docs/sft.md)。

configs/sft/task_alignment.example.json 是任务对齐字段示例，不是当前活动任务池。活动任务身份以 data/task_pools/travel_task_pools.json 为准。
