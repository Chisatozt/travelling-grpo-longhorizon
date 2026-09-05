# collection

collection 负责把“固定任务 → Teacher 轨迹 → canonical SFT 数据”串起来。

主要模块：

- task_pools.py：生成和检查任务身份分区；
- teacher_collection.py：采集 cache、断点续跑和 provenance；
- travel_canonical.py：统一多轮消息和工具调用格式；
- prepare_travel_sft.py / merge_travel_sft.py：清洗、分类、合并和审计；
- travel_task_resolver.py：恢复 env/task identity。

标准顺序和数据隐私见 [数据采集文档](../../../docs/data_collection.md) 与 [SFT 管线](../../../docs/sft_pipeline.md)。
