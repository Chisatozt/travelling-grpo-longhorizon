# configs

这里放项目侧的配置入口，不放训练器源码。配置的作用是把项目的任务、工具协议和训练参数接到 veRL/SGLang 上。

| 目录/文件 | 用途 |
| --- | --- |
| configs/grpo/ | GRPO 的项目配置和默认 recipe |
| configs/sft/ | 旧 renderer 兼容配置与任务对齐示例 |
| configs/tools/ | interact_with_env 的 schema 和 AgentLoop 配置 |

GRPO 入口是 configs/grpo/grpo_multiturn.yaml 与 scripts/train_grpo.sh 的组合；SFT 的权威配置在 verl/trainer/config/travel_qwen35_sft.yaml。配置读取和运行顺序见 [GRPO 文档](../docs/grpo.md) 与 [SFT 文档](../docs/sft.md)。
