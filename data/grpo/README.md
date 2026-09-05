# data/grpo

本目录保存 TravelGym 多轮 GRPO 的 Parquet。实际 variant 数据位于 data/grpo/travelXX_multiturn_onechoice/；当前训练入口读取八个 variant 的 train/test split。

八个 variant 合计 2,651 个 train task 和 471 个 test task。alltrain_multiturn、alltest_multiturn 与 travel_multiturn_onechoice 是聚合或兼容数据，当前 scripts/train_grpo.sh 不直接读取它们。

任务身份由 env_name::task_id 表示，任务池会在 tokenization 前完成 SFT、GRPO 和 validation 分区。不要直接删 Parquet 行改变实验分区，应修改并重新生成 task-pool manifest。

进一步阅读：[数据采集](../../docs/data_collection.md)、[GRPO](../../docs/grpo.md)、[评测数据集](../../docs/evaluation_dataset.md)。
