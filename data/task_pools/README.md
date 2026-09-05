# data/task_pools

这里是项目的任务身份总表，负责把同一批 TravelGym task 分配给 SFT、GRPO、validation 和诊断阶段。它解决的是 task-level 数据泄漏问题，不只是给 Parquet 添加标签。

| 文件 | 用途 |
| --- | --- |
| travel_task_pools.json | 当前正式 SFT、GRPO、validation 和 smoke 分区 |
| travel_grpo_overfit_pools.json | overfit-one/overfit-four 诊断子池 |
| sft_task_alignment_candidates.json | 历史任务对齐和 quarantine 审计 |

活动身份统一为 env_name::task_id。当前 manifest 的选择 seed 为 20260801；重新构建后必须重新检查活动池互斥、Final-200 一致性和 opaque task quarantine。

~~~bash
python -m travel_grpo.collection.task_pools --help
python scripts/prepare_grpo_overfit_pools.py --help
~~~

流程说明见 [数据采集](../../docs/data_collection.md)、[训练链路](../../docs/travel_training_chain.md) 和 [评测数据集](../../docs/evaluation_dataset.md)。
