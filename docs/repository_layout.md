# 仓库结构与源码边界

## Canonical tree

~~~text
configs/
  grpo/                         GRPO 配置
  sft/                          legacy renderer 和对齐示例
  tools/                        interact_with_env schema/config
data/
  grpo/                         八个 TravelGym variant 的 Parquet
  sft/                          canonical SFT 语料和 task-level split
  evaluation/                   protocol、smoke20、final200
  task_pools/                   SFT/GRPO/Validation 互斥分区
docs/                           项目文档
environments/TravelGym/         唯一 TravelGym package 和内置任务数据
scripts/                        用户入口和运行检查
src/travel_grpo/
  collection/                   Teacher、清洗、canonical、task pool
  evaluation/                   HTTP/native 评测和 Final-200 plan
  training/sft/                 token mask、split、canonical 校验
  environment/                  环境/工具边界说明
  training/grpo/                veRL 边界说明
verl/                           veRL trainer、worker、tool 和 SGLang runtime
tests/                          单元、入口、布局和安装检查
~~~

checkpoints/、outputs/、日志、.env 和 virtualenv 是运行时目录，由 .gitignore 排除，
不能作为源码或公共数据的 import root。

## 稳定边界

1. Python 包名保持 verl 和 travelgym，不能因为目录整理改名。
2. GRPO 入口保持 verl.trainer.main_ppo；Hydra 基础配置在 verl/trainer/config/，
   项目 override 在 configs/grpo/。
3. 权威的项目源码只在 src/travel_grpo/；通过 python -m travel_grpo... 或 scripts/
   入口调用。
4. TravelGym 的唯一物理实现是 environments/TravelGym/travelgym/；这里保持 one physical source。
5. veRL 的训练器、worker、tool 和 SGLang 集成仍位于 verl/；src/travel_grpo 中的
   training/grpo README 只说明边界，不复制 veRL 实现。
6. task-pool 的 source path 只应指向 data/grpo/、data/sft/ 或指定 evaluation
   manifest；不能重新引入已经删除的旧数据路径。
7. 不创建旧目录的 symlink 或 README-only path mapping 作为运行入口。

## 已移除的旧入口

以下路径不再存在，也不应被文档或新代码引用：

~~~text
eval/
sft/
examples/data_preprocess/
examples/sglang_multiturn/
gyms/TravelGym/
data/alltrain_multiturn/
data/alltest_multiturn/
data/travel*_multiturn_onechoice/
~~~

如果外部脚本仍引用这些路径，应迁移到对应的 src/travel_grpo 模块、data/ canonical
路径或 scripts/入口，而不是恢复兼容 symlink。

## 修改后的检查

~~~bash
python -m pytest -q tests/test_repository_layout.py
python -m pytest -q
~~~

布局测试同时检查 canonical 文件存在、旧路径不存在、入口不是 symlink，以及组织视图
README 指向唯一源码。
