# src/travel_grpo

src/travel_grpo 是项目适配层的 canonical home：把 TravelGym、veRL、SGLang 和数据/评测流程连接起来。它保存项目自有逻辑，不复制 veRL trainer，也不保存第二份 TravelGym 实现。

| 模块 | 在整体流程中的位置 |
| --- | --- |
| collection | 任务池、Teacher 轨迹、canonical SFT 数据 |
| environment | 环境与工具之间的项目边界说明 |
| training/sft | Qwen mask、split 和 canonical 校验 |
| training/grpo | GRPO 适配边界说明，核心 trainer 仍在 verl/ |
| evaluation | public contract、HTTP/native 评测、Final-200 和分析 |

常用模块入口：

~~~bash
python -m travel_grpo.collection.task_pools --help
python -m travel_grpo.evaluation.eval --help
python -m travel_grpo.evaluation.final200 --help
~~~

先读 [项目文档总览](../../docs/README.md)，再按阶段进入 [数据](../../docs/data_collection.md)、[SFT](../../docs/sft.md)、[GRPO](../../docs/grpo.md) 或 [评测](../../docs/evaluation.md)。
