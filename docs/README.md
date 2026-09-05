# 项目文档总览

这里先说明项目是什么、各阶段如何连接，再进入数据、训练、评测和运维细节。根目录 [README.md](../README.md) 适合第一次了解项目；本页是按工作阶段组织的导航页。

## 一句话理解项目

TravelGym 把多 aspect 旅行规划变成一个长程交互任务：模型必须通过工具搜索和询问用户，基于环境返回的证据完成多个选择。项目先用 Teacher 轨迹和 LoRA SFT 学习基本行为，再用 veRL + SGLang 在线 GRPO 在真实环境反馈中优化，最后在固定的 Smoke20/Final-200 上比较模型。

## 总体流程

~~~text
任务池
  -> Teacher 真实环境轨迹
  -> canonical 清洗与 SFT mask
  -> LoRA SFT
  -> 合并后的 Actor
  -> veRL GRPO + SGLang rollout
  -> TravelGym terminal Reward
  -> 固定留出集评测与报告
~~~

| 阶段 | 你要回答的问题 | 入口/产物 | 深入文档 |
| --- | --- | --- | --- |
| TravelGym | Agent 能看到什么、能调用什么？ | environments/TravelGym、public contract | [公开协议](travel_public_protocol.md) |
| 数据 | 训练任务从哪里来，如何避免泄漏？ | data/task_pools、data/sft、Teacher cache | [数据采集](data_collection.md) |
| SFT | 如何把完整轨迹变成可监督样本？ | canonical JSONL、Qwen token mask、LoRA | [SFT](sft.md) |
| GRPO | 如何让模型在环境中在线学习？ | veRL trainer、SGLang rollout、checkpoint | [GRPO](grpo.md) |
| Evaluation | 如何公平比较 Base、SFT 和 GRPO？ | Smoke20、Final-200、HTTP/native evaluator | [评测](evaluation.md) |
| Operations | 如何安装、预检、恢复和排错？ | scripts、watcher、日志 | [运维](operations.md) |

## 先看这三件事

1. [TravelGym 公开协议](travel_public_protocol.md)：确认模型实际看到的 observation 和工具动作。
2. [训练链路与数据隔离](travel_training_chain.md)：理解 SFT、GRPO、validation 的任务关系。
3. [GRPO 文档](grpo.md)：理解 veRL trainer、SGLang rollout 和环境 Reward 如何闭环。

## 按需求阅读

- 第一次运行： [根 README](../README.md) → [运行文档](operations.md) → [GRPO 阶段预检](grpo_overfit_preflight.md)。
- 重新构造数据： [任务池与 Teacher 数据](data_collection.md) → [SFT 管线](sft_pipeline.md) → [SFT 操作指南](sft.md)。
- 修改环境或工具： [公开协议](travel_public_protocol.md) → [Reward](reward.md) → [TravelGym README](../environments/TravelGym/README.md)。
- 修改评测： [评测数据集](evaluation_dataset.md) → [评测流水线](evaluation.md) → [Final-200 plan](/root/autodl-tmp/travelling-grpo-longhorizon/src/travel_grpo/evaluation/final200.py)。
- 整理仓库： [仓库结构](repository_layout.md) → [项目源码说明](../src/travel_grpo/README.md)。

## 文档分层原则

根 README 和本页只保留项目目标、阶段关系、核心边界和最短入口；具体公式、字段、阈值、清洗规则和恢复条件放在专项文档中。目录级 README 只说明目录归属和下一步阅读位置，不复制源码实现。

文档中的路径均以仓库根目录为基准。checkpoints/、outputs/ 和本地 API cache 是运行时产物；data/task_pools/travel_task_pools.json 是任务身份分区的权威来源；data/evaluation/travel_manifest.json 是 public protocol 与 Reward 版本的权威来源。

本项目的文档层次参考了 [shopping-grpo-longhorizon 的公开 README](https://github.com/YYHDBL/shopping-grpo-longhorizon/tree/main) 以及其 [数据采集](https://github.com/YYHDBL/shopping-grpo-longhorizon/blob/main/docs/data-collection.md)、[SFT](https://github.com/YYHDBL/shopping-grpo-longhorizon/blob/main/docs/sft.md)、[GRPO](https://github.com/YYHDBL/shopping-grpo-longhorizon/blob/main/docs/grpo.md) 和 [评测](https://github.com/YYHDBL/shopping-grpo-longhorizon/blob/main/docs/evaluation.md) 文档；项目实际行为以本仓库代码、配置和 manifest 为准。
