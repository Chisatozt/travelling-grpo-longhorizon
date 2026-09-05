# training/grpo

当前 GRPO 核心实现仍位于 verl/：trainer、AgentLoop、动态采样、TurnCredit 和 Hard Case Pool 都由 veRL 侧运行。本目录只记录项目适配边界，避免产生第二份 trainer。

项目入口是 scripts/train_grpo.sh，项目配置是 configs/grpo/grpo_multiturn.yaml，环境反馈来自 environments/TravelGym。运行顺序见 [GRPO 文档](../../../../docs/grpo.md)。
