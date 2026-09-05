# Final-200 与 Smoke20 评测数据集

## 固定来源

仓库的八个 multiturn_onechoice test split 一共 471 个 task：

| Variant | Test task |
| --- | ---: |
| travel22 | 101 |
| travel33 | 87 |
| travel44 | 67 |
| travel233 | 61 |
| travel333 | 53 |
| travel334 | 47 |
| travel444 | 38 |
| travel2222 | 17 |
| 合计 | 471 |

data/evaluation/test_manifests/final200.json 从这 471 个 task 中按固定 seed 20260801
分层选择 200 个，smoke20.json 是 final200 内的固定 20-task 子集：

| Manifest | 数量 | 用途 |
| --- | ---: | --- |
| smoke20.json | 20 | 快速检查、native pass@3、GRPO step-0 baseline |
| final200.json | 200 | recipe 固定后的正式横向比较 |

final200 的 variant 配额是 43/37/28/26/23/20/16/7；smoke20 的配额是
3/3/3/3/2/2/2/2。清单保存 source_index，评测器会再次检查 parquet 中的
reward_model.id 与 manifest 的 task ID 一致。

## 防止泄漏

任务身份统一使用 env_name::task_id。travel_task_pools.json 同时管理：

- SFT 历史/Teacher task；
- GRPO train task；
- validation/final200 task。

加载 task pool 时会检查 active pool 两两不交叉；不能只按 parquet 行号或聚合文件名
判断是否泄漏。修改 SFT 或 GRPO 数据后，必须重新检查 task pool，而不是手工认为
test 仍然独立。

Final-200 不能用于：

- Prompt 调优；
- Teacher 采集；
- Reward 调参；
- checkpoint 选择；
- 训练过程中的反复试验。

## 重新生成清单

只有在明确要产生新的评测集 revision 时才运行：

~~~bash
python -m travel_grpo.evaluation.build_test_manifests \
  --project-root . \
  --output-dir data/evaluation/test_manifests
~~~

修改 selection seed、配额或源数据会改变清单。变更后应同步更新 task-pool manifest、
测试和文档，不能把新清单的结果与旧清单的结果直接并列。

## 协议版本

data/evaluation/travel_manifest.json 固定：

- contract_version=travelgym-public-v1；
- reward_version=travelgym-terminal-v2；
- protocol 为 Search、Action、Answer；
- Search 返回完整候选；
- Action 不改变候选列表；
- Answer 只能提交当前 Search 可见 ID；
- actor forbidden fields deny-list。

评测器在启动前校验这些字段；协议或 Reward 版本不匹配会 fail closed。
