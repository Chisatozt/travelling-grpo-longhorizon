# TravelGym 训练与评测分析报告

> 报告日期：2026-09-05
\n> 分析范围：2026-08-30 至 2026-09-04 的本地 SFT、4-task overfit、100-step GRPO、recovery 和 Final-200 评测记录。

## 结论摘要

当前实验已经验证了完整的“Teacher 数据 → SFT → GRPO → 固定任务评测”流程，并取得了清晰的阶段性收益：

- SFT 将模型从“基本不会稳定完成任务”提升到能够较稳定地执行合法工具链；
- GRPO-step100 在 Final-200 上将完整任务成功率从 SFT 的 11.5% 提升到 30.5%，正确 aspect 完成率从 41.4% 提升到 52.3%；
- 训练侧的 Turn Credit 在 overfit、主运行和 recovery 中均实际生效，轨迹优势守恒误差为 0，未出现 fallback row；
- 主要问题不是完全不会做，而是 GRPO 后的任务覆盖和协议稳定性下降：Final-200 中 GRPO-step100 的平均答案覆盖率只有 73.2%，并且 200 次评测中有 80 次被标记为无效尝试；
- recovery 确实生成了 `global_step_100`，但恢复运行的最终 validation 阶段因 shape mismatch 失败，因此 step100 checkpoint 可用，但 recovery 运行记录不能视为完全成功；
- 当前结果足以支持“训练链路有效”的结论，但还不足以证明 Turn Credit 或动态采样分别带来了多少增益，需要补充消融实验。

## 1. 指标口径与数据来源

### 1.1 Final-200 指标

Final-200 使用固定的 200 个任务、`pass_k=1`、temperature 0 和 native protocol。这里的 `pass@1` 表示一个任务的所有 aspect 都正确完成；`correct_completion` 则是每个任务中正确完成的 aspect 数占该任务 aspect 总数的比例，再对 200 个任务取平均。

因此，`pass@1` 衡量完整任务成功，`correct_completion` 衡量 aspect 级进展，二者不能互相替代。报告中将 `correct_completion/mean@1` 称为“正确 aspect 完成率”。

其他关键指标：

- `answer_coverage`：被回答的 aspect 数占任务 aspect 总数的比例；
- `legal_chain_rate`：满足 Search → Action → Answer 合法链路的回答比例；
- `terminal_reward`：`travelgym-terminal-v2` 在 episode 结束时计算的终局 Reward；
- `valid_attempt_rate`：评测过程中没有基础设施或协议级无效的尝试比例。它不是主结果指标，但对解释 GRPO 的稳定性很重要。

### 1.2 主要数据源

- SFT checkpoint 指标：`checkpoints/TravelGym/qwen35_4b_canonical_sft/best_checkpoint.json`；
- GRPO 训练日志：
  - `outputs/travelgym_grpo_overfit_4tasks_sft186_rewardv2_noval20_turncreditfix/training.log`；
  - `outputs/travelgym_grpo_production_deepseek_user_sft186_100_liveval/training.log`；
  - `outputs/travelgym_grpo_production_deepseek_user_sft186_100_recovery80to100/training.log`；
- Final-200 评测摘要：三个 `outputs/travelgym_native_*_final200_pass1/validation/*summary.json` 文件；
- 任务与评测清单：`data/task_pools/travel_task_pools.json`、`data/evaluation/test_manifests/final200.json`。

`outputs/`、`checkpoints/` 和日志被 `.gitignore` 忽略，报告中的结果来自当前机器上的运行产物，不会随 Git 提交自动同步。

## 2. SFT 阶段分析

### 2.1 数据与训练配置

当前 canonical SFT 语料共有 795 条记录，由 238 条历史记录和 557 条 DeepSeek Teacher 记录组成。正式 split 为 490 条训练轨迹和 10 条 `strict_gold` 验证轨迹；无法恢复任务身份的历史行被隔离，不参与训练。

SFT 运行配置如下：

| 项目 | 配置/结果 |
| --- | --- |
| 基础模型 | Qwen3.5-4B |
| 训练精度 | BF16 |
| 参数高效微调 | LoRA rank 16、alpha 32 |
| 最大长度 | 32768 tokens |
| 训练轮数 | 3 epochs |
| 最终 checkpoint | `global_step_186` |
| SFT 验证 masked token NLL | 0.3677 |
| SFT 验证 perplexity | 1.4444 |
| tool parse rate | 100% |
| structured choice/content rate | 100% |
| protocol non-degraded | 100% |

### 2.2 SFT 的效果

与基础模型相比，SFT 在 Final-200 上的变化如下：

| 指标 | Qwen3.5-4B base | SFT-step186 | 变化 |
| --- | ---: | ---: | ---: |
| 完整任务 `pass@1` | 4.5% | 11.5% | +7.0 个百分点 |
| 正确 aspect 完成率 | 23.4% | 41.4% | +18.0 个百分点 |
| 答案覆盖率 | 99.0% | 92.8% | -6.2 个百分点 |
| 合法链路率 | 32.0% | 91.2% | +59.1 个百分点 |
| terminal Reward 均值 | 0.1547 | 0.2946 | +0.1399 |
| 隐藏偏好命中率 | 7.0% | 21.4% | +14.4 个百分点 |
| efficiency | 69.7% | 50.6% | -19.1 个百分点 |

SFT 最明显的收益是协议能力：合法工具链率从 32.0% 提升到 91.2%，说明模型开始稳定遵守“先搜索、再询问或分析、最后回答”的交互顺序。正确 aspect 完成率和终局 Reward 也同步提升。

同时，SFT 使轨迹变得更长、更谨慎，效率下降，答案覆盖率略有下降，冗余 Action 和用户模拟器调用增加。这说明 SFT 已经学会了“如何交互”，但还没有完全学会在有限预算内快速、完整地结束多 aspect 任务。

需要区分两类验证结果：SFT checkpoint 中的 100% tool parse/structured rate 是训练验证阶段的格式指标，不代表模型在 200 个新任务上的完整成功率；Final-200 才反映实际环境泛化能力。

## 3. 4-task Overfit 诊断

### 3.1 运行情况

4-task overfit 对固定的 4 个任务运行 20 steps，实际训练日志从约 2026-09-03 08:23 开始，到 step20 完成，耗时约 1 小时 29 分。对应路径为：

```text
outputs/2026-09-03/08-22-45/.hydra/
outputs/travelgym_grpo_overfit_4tasks_sft186_rewardv2_noval20_turncreditfix/
outputs/travelgym_grpo_overfit_4tasks_sft186_rewardv2_noval20_turncreditfix_launcher.log
```

### 3.2 训练信号

| 训练 step | terminal Reward | 正确 aspect 完成率 | 完整任务成功率 | 答案覆盖率 | 合法链路率 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.582 | 63.5% | 31.2% | 100.0% | 93.8% |
| 10 | 0.517 | 58.3% | 31.2% | 100.0% | 94.8% |
| 20 | 0.676 | 75.0% | 43.8% | 100.0% | 100.0% |

20 个 step 内的平均正确 aspect 完成率为 66.9%，平均完整任务成功率为 36.3%。动态采样始终只需要 1 个 generation batch，没有 invalid group；Turn Credit 的 `mix_ratio` 为 0.30，conservation error 为 0，fallback row 为 0。

这说明训练入口、工具协议、Reward metadata 和 Turn Credit 路由至少能够在小规模实验中闭环运行。但由于任务池只有 4 个任务，step20 的提升不能被解释为对新任务的泛化提升，只能作为工程和 Reward 的 sanity check。

### 3.3 checkpoint 问题

日志曾记录写入 `global_step_20` LoRA adapter，但当前工作区中对应的模型目录已经不存在。当前只剩：

```text
checkpoints/TravelGym/travelgym_grpo_overfit_4tasks_sft186_rewardv2_noval20_turncreditfix/hard_case_pool.json
checkpoints/TravelGym/travelgym_grpo_overfit_4tasks_sft186_rewardv2_noval20_turncreditfix/latest_checkpointed_iteration.txt
```

因此，overfit 的日志、rollout 和审计信息可以同步，但不能从当前文件恢复 overfit 模型权重。

## 4. 100-step GRPO 分析

### 4.1 主运行与 recovery

主运行从约 2026-09-03 11:10 开始，目标是 100 steps；当前主日志包含 step0–99 的训练记录和 `rollouts/1.jsonl` 至 `rollouts/99.jsonl`。主运行实际保留的 checkpoint 是 `global_step_40` 和 `global_step_80`。

主运行路径：

```text
outputs/2026-09-03/11-10-01/.hydra/
outputs/travelgym_grpo_production_deepseek_user_sft186_100_liveval/
checkpoints/TravelGym/travelgym_grpo_production_deepseek_user_sft186_100_liveval/
```

recovery 从主运行的 `global_step_80` 继续，生成 step81–100 的 rollout，并写出 `global_step_100`：

```text
outputs/2026-09-04/06-46-44/.hydra/
outputs/travelgym_grpo_production_deepseek_user_sft186_100_recovery80to100/
checkpoints/TravelGym/travelgym_grpo_production_deepseek_user_sft186_100_recovery80to100/global_step_100/
```

### 4.2 主运行训练趋势

下表是训练日志中若干关键 step 的 batch 平均值。step40 的 0 值不是模型在所有任务上的真实能力，而是该 step 经过 3 个 generation batch 后仍有 12 个 invalid group、没有形成有效更新的结果。

| step | terminal Reward | 正确 aspect 完成率 | 完整任务成功率 | 答案覆盖率 | generation batches | invalid groups | 平均 response tokens |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.270 | 36.5% | 0.0% | 97.9% | 1 | 0 | 9182 |
| 20 | 0.310 | 35.4% | 0.0% | 100.0% | 1 | 0 | 6446 |
| 40 | 0.000 | — | — | — | 3 | 12 | 1988 |
| 60 | 0.500 | 58.3% | 50.0% | 100.0% | 2 | 1 | 9258 |
| 80 | 0.602 | 67.7% | 18.8% | 100.0% | 2 | 2 | 13489 |
| 99 | 0.670 | 71.9% | 43.8% | 100.0% | 2 | 3 | 12292 |

整体趋势是后半段的正确 aspect 完成率和 Reward 高于前半段，但并不单调。step60–99 仍存在 invalid group 和较大的 response length 波动，说明动态采样在帮助训练获得可用组的同时，也带来了额外计算成本。

平均 response length 从早期约 6k–9k tokens 增长到后期约 12k–15k tokens，单 step 耗时也从约 434 秒上升到超过 900 秒，step80 曾达到约 1304 秒。这是 100-step 运行耗时较长的主要原因之一。

### 4.3 Recovery 的表现和异常

recovery 的 step81–99 日志仍保持 100% 答案覆盖率和约 99.9% 的合法链路率，但正确 aspect 完成率在 41.7%–71.9% 之间波动，step99 为 57.3%，terminal Reward 为 0.541。它没有表现出相对于主运行 step99 的明确单调提升。

recovery 最终生成了完整的 `global_step_100` checkpoint，且 checkpoint 的 `latest_checkpointed_iteration.txt` 为 100。但是，最终验证生成文件：

```text
outputs/travelgym_grpo_production_deepseek_user_sft186_100_recovery80to100/validation/100.jsonl
```

并不存在；`monitor/shutdown_report.json` 将本次运行标记为 `incomplete_or_failed`，日志中的错误是：

```text
ValueError: shape mismatch: value array of shape (1,1,14)
could not be broadcast to indexing result of shape (1,1)
```

因此，step100 checkpoint 可以用于后续 native evaluation，但不能把 recovery 的最终 validation 视为成功完成。

### 4.4 Turn Credit 与动态采样

主运行和 recovery 的日志都显示：

- `turn_credit_applied=1`；
- `turn_credit_mix_ratio=0.30`；
- `turn_credit_conservation_abs_error=0`；
- `turn_credit_fallback_row_count=0`。

这证明 Turn Credit 路由在这些训练记录中实际参与了优化，并且没有改变每条响应的优势总量。它将终局 Reward 的不同组件分配到产生相应行为的 turn，再以 30% 的比例与原始终局优势混合。

不过，上述指标只能证明机制运行正确，不能证明 Turn Credit 本身带来了多少性能提升。当前没有同 seed、同 task pool 的 `turn_credit.stage=off` 对照组，因此不能把 GRPO 的全部收益归因于 Turn Credit。

动态采样方面，早期 step 通常一次 generation batch 即得到有效组，后期经常需要 2–3 个 batch；step40 的 12 个 invalid group 是最明显的异常点。它反映出 rollout 的有效性和 Reward spread 在训练过程中并不稳定，也解释了为什么训练耗时和单 step token 数波动明显。

## 5. Final-200 最终结果

三组结果使用相同的 200 个任务、native protocol、temperature 0 和 `pass_k=1`：

| 模型/阶段 | 完整任务 `pass@1` | 正确 aspect 完成率 | 答案覆盖率 | 合法链路率 | terminal Reward 均值 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-4B base | 4.5%（9/200） | 23.4% | 99.0% | 32.0% | 0.1547 |
| SFT-step186 | 11.5%（23/200） | 41.4% | 92.8% | 91.2% | 0.2946 |
| GRPO-step100 | 30.5%（61/200） | 52.3% | 73.2% | 85.0% | 0.3987 |

### 5.1 结果解读

1. **SFT 首先解决协议问题。** 合法链路率从 32.0% 提升到 91.2%，说明模型更能按照环境要求完成 Search、Action 和 Answer；aspect 级正确性也明显提升。

2. **GRPO 进一步提高决策质量。** 相比 SFT，GRPO-step100 的完整任务成功率提高 19.0 个百分点，正确 aspect 完成率提高 10.9 个百分点，terminal Reward 提高 0.1041。

3. **GRPO 的收益伴随覆盖率下降。** GRPO 的答案覆盖率从 SFT 的 92.8% 降到 73.2%，平均 unanswered aspect 从 0.185 增加到 0.730。也就是说，模型在已经回答的部分更可能做出正确选择，但更容易漏答部分 aspect 或提前结束。

4. **协议能力没有完全丢失，但稳定性变差。** GRPO 合法链路率 85.0% 仍明显高于 base 的 32.0%，但低于 SFT 的 91.2%。Final-200 中 base、SFT、GRPO 的有效尝试数分别为 200/200、198/200、120/200；GRPO 有 80 次无效尝试。这是当前最需要优先解决的工程问题。

5. **结果不是单纯的“Reward 越高越好”。** GRPO 的 Reward 和 pass@1 都更高，但同时出现 coverage 和有效率下降，说明当前目标在“正确完成部分 aspect”和“完整、稳定地完成整个任务”之间存在张力。

## 6. 资源消耗与可复现性

### 6.1 硬件和耗时

运行配置是单节点单卡 NVIDIA GPU。当前日志没有可靠记录显卡型号。

| 阶段 | 运行信息 |
| --- | --- |
| SFT | BF16、LoRA 16/32、3 epochs，最终 step186；完整总耗时未保留 |
| 4-task overfit | 20 steps，约 1 小时 29 分；峰值约 49.6 GiB allocated、87.4 GiB reserved |
| GRPO 主运行 | 0–99 step 日志在 step99 显示累计约 15 小时 45 分；后期约 66.3 GiB allocated、90.6 GiB reserved |
| GRPO recovery | 约 4 小时 50 分；生成 step100 checkpoint，但最终 validation 失败 |

模型和 checkpoint 容量较大：合并后的 SFT 模型约 7.9 GiB；GRPO 的 `global_step_40`、`global_step_80` 和 `global_step_100` 各约 20 GiB。若只做推理，可以只保留 SFT merged model 和 GRPO actor 的 LoRA adapter；若要恢复训练，则需要完整 checkpoint。

### 6.2 历史配置路径问题

Hydra 运行产物保存的是当时运行机器上的绝对路径。部分 `.hydra/config.yaml` 和 `.hydra/overrides.yaml` 仍引用旧的：

```text
data/travel22_multiturn_onechoice/...
examples/sglang_multiturn/...
```

当前仓库的数据实际位于 `data/grpo/travel22_multiturn_onechoice/...`，因此这些历史配置不能直接复制到另一台机器运行，需要根据当前目录结构替换路径并重新做预检。

此外，当前工作区没有独立的 `fsdp_sft_trainer.log` 或 `main_ppo.log`；SFT 只有 Hydra 配置和 checkpoint，GRPO 的主要过程记录在各实验目录的 `training.log` 中。

## 7. 主要问题与建议

### P0：修复 recovery 的 validation shape mismatch

先定位 `validation/100.jsonl` 生成阶段的 batch 拼接问题，重点检查带 retry 的 pending index、嵌套数组维度以及不同 rollout 的 metadata 对齐。修复后应先运行 Smoke20，再重新执行 step100 validation，确认不会影响已有 checkpoint。

### P0：把“完整任务”与“aspect 级正确”分开监控

后续 checkpoint 选择不应只看 terminal Reward 或 pass@1，应同时记录：

- `correct_completion`；
- `answer_coverage`；
- `completion_success`；
- `unanswered_count`；
- invalid rollout/attempt 数；
- legal chain rate。

当前 GRPO 的主要退化正是 coverage 和有效率，不能被较高的 Reward 掩盖。

### P1：补充 Turn Credit 消融

在完全相同的 task pool、随机种子、初始化 SFT checkpoint 和训练步数下，至少比较：

- `turn_credit.stage=off`；
- `turn_credit.stage=shadow`；
- `turn_credit.stage=train`。

同时保留 conservation error、fallback row 和各 Reward component 的日志，才能判断 Turn Credit 是真正提升了学习效果，还是只改善了 credit 的可解释性。

### P1：补充动态采样消融和稳定性统计

比较 dynamic sampling 开启/关闭时的：generation batches、invalid groups、skip 次数、单 step 时间和最终有效率。step40 的异常应单独作为回归样例保留。

### P1：针对 coverage 退化做目标检查

当前 Reward 已有 incomplete penalty，但 GRPO 后仍出现大量 unanswered aspect。应先确认模型是因为预算不足、错误停止、invalid tool call，还是偏向只完成容易的 aspect；再决定是否调整终止行为、任务覆盖约束或 Reward 组件权重。建议先在 Smoke20 和固定小池上验证，避免直接启动大规模训练。

### P2：建立运行产物归档清单

建议每次实验结束时同时保存：

- resolved Hydra config；
- training/validation log；
- task pool 和评测 manifest 的 hash；
- checkpoint 及其来源 checkpoint；
- Final-200 summary；
- 运行状态和异常报告。

这些内容应放在 Git LFS、对象存储或本地归档目录中，而不是依赖普通 `git push`。

## 8. 实验记录索引

### SFT

```text
outputs/2026-08-30/23-07-42/.hydra/
checkpoints/TravelGym/qwen35_4b_canonical_sft/global_step_186/
checkpoints/TravelGym/qwen35_4b_canonical_sft/merged_step_186/
```

### 4-task overfit

```text
outputs/2026-09-03/08-22-45/.hydra/
outputs/travelgym_grpo_overfit_4tasks_sft186_rewardv2_noval20_turncreditfix/
checkpoints/TravelGym/travelgym_grpo_overfit_4tasks_sft186_rewardv2_noval20_turncreditfix/
```

### GRPO 主运行与 recovery

```text
outputs/2026-09-03/11-10-01/.hydra/
outputs/travelgym_grpo_production_deepseek_user_sft186_100_liveval/
checkpoints/TravelGym/travelgym_grpo_production_deepseek_user_sft186_100_liveval/

outputs/2026-09-04/06-46-44/.hydra/
outputs/travelgym_grpo_production_deepseek_user_sft186_100_recovery80to100/
checkpoints/TravelGym/travelgym_grpo_production_deepseek_user_sft186_100_recovery80to100/
```

### Final-200

```text
outputs/travelgym_native_qwen35_4b_final200_pass1/
outputs/travelgym_native_sft186_final200_pass1/
outputs/travelgym_native_grpo100_final200_pass1/
outputs/final200_sequence.log
```

对应的摘要文件分别是：

```text
outputs/travelgym_native_qwen35_4b_final200_pass1/validation/0_pass1_summary.json
outputs/travelgym_native_sft186_final200_pass1/validation/0_pass1_summary.json
outputs/travelgym_native_grpo100_final200_pass1/validation/100_pass1_summary.json
```
