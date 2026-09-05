# data/evaluation

本目录保存评测所需的固定清单和协议版本。固定清单位于 data/evaluation/test_manifests/，评测源数据来自 data/grpo 的八个 variant test split。

| 文件 | 用途 |
| --- | --- |
| travel_manifest.json | public protocol 与 terminal Reward 版本 |
| test_manifests/smoke20.json | 20-task 快速检查，属于 Final-200 子集 |
| test_manifests/final200.json | 200-task 固定留出评测 |

Final-200 的任务选择、数据泄漏约束和使用入口见 [评测数据集](../../docs/evaluation_dataset.md)；native/HTTP 评测流程见 [评测文档](../../docs/evaluation.md)。
