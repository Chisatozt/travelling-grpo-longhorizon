# 使用 veRL + SGLang 进行 GRPO

## 训练边界

GRPO 的一次更新由以下组件协作完成：

1. verl.trainer.main_ppo 启动 Ray/veRL 训练进程并读取 Hydra 配置；
2. veRL 的 actor/ref worker 负责 FSDP 参数、log probability、GRPO advantage 和优化器更新；
3. rollout 配置的 name=sglang 使用 SGLang 根据当前 Actor 生成多轮响应；
4. verl.tools.interact_tool.InteractTool 将每个工具调用路由到持久化 TravelGym；
5. TravelGym 返回公开 observation，episode 结束后由私有 ledger 生成 terminal reward；
6. veRL 汇总 rollout group、计算更新并保存 checkpoint。

换句话说，veRL 负责训练流程和参数更新，SGLang 负责 rollout 推理，TravelGym 负责
环境状态和 Reward。GRPO 训练不需要另起一个 vLLM 模型服务；scripts/serve_vllm.sh
只用于手动服务或 HTTP 评测。

## 输入和权威配置

- 初始策略：完整合并后的 SFT 模型，例如
  checkpoints/TravelGym/qwen35_4b_canonical_sft/merged_step_186；
- train/validation Parquet：data/grpo/travel*_multiturn_onechoice/；
- task pool：data/task_pools/travel_task_pools.json；
- Reward：travelgym-terminal-v2；
- 项目配置：configs/grpo/grpo_multiturn.yaml；
- 基础 veRL/Hydra 配置：verl/trainer/config/。

configs/grpo/grpo_multiturn.yaml 通过 Hydra search path 继承 veRL 的 ppo_trainer，
因此修改字段时要确认最终 resolved config，而不是只看单个 YAML。

## 默认训练 recipe

| 设置 | 默认值 |
| --- | --- |
| algorithm | grpo_multiturn |
| rollout backend | sglang |
| rollouts / prompt | 4 |
| global train batch | 4 |
| PPO mini-batch | 2 |
| LoRA rank / alpha | 32 / 64 |
| actor learning rate | 1e-5 |
| prompt / model max length | 1280 / 32768 |
| max turns | 25 |
| total training steps | 100 |
| save milestones | 20、40、60、80、100 |
| terminal reward | step reward 为 0，episode 结束时一次计算 |
| user simulator | launcher 默认 DeepSeek API |
| OMP threads | 8 |

正式 GRPO 的 validation 是每个 task 一次普通 greedy attempt；独立的
evaluate_native.sh 才使用 pass@3 sampled evaluation。

## 启动前检查

先确认模型是完整权重而不是 adapter-only：

~~~bash
MODEL_PATH=checkpoints/TravelGym/qwen35_4b_canonical_sft/merged_step_186

python -m travel_grpo.evaluation.check_qwen35_runtime \
  --backend sglang --tokenizer "$MODEL_PATH"

python scripts/check_grpo_runtime.py "$MODEL_PATH"

python scripts/check_grpo_prompt_budget.py "$MODEL_PATH" \
  --pools grpo validation --max-prompt-length 1280

ACTOR_MODEL_PATH="$MODEL_PATH" bash scripts/run_grpo_stage.sh check
~~~

预检会验证 tokenizer、SGLang/FlashAttention 导入、FSDP 模型映射、工具 schema 和
reset 后 prompt 长度；它不会启动 Ray 或 CUDA 训练。

## 运行正式 GRPO

~~~bash
ACTOR_MODEL_PATH=checkpoints/TravelGym/qwen35_4b_canonical_sft/merged_step_186 \
GRPO_AUTO_SHUTDOWN=0 \
  bash scripts/train_grpo.sh
~~~

追加 Hydra override 时直接放在脚本后面：

~~~bash
ACTOR_MODEL_PATH=/path/to/merged-sft \
GRPO_AUTO_SHUTDOWN=0 \
  bash scripts/train_grpo.sh \
  trainer.total_training_steps=20 \
  trainer.save_freq=10
~~~

训练脚本会从自身位置推导仓库根目录，并自动加入 src/、TravelGym 和 veRL 到
PYTHONPATH。默认输出为：

~~~text
checkpoints/TravelGym/travelgym_qwen35_4b_terminal_bs4/
outputs/travelgym_qwen35_4b_terminal_bs4/
~~~

如果修改 experiment name、checkpoint root 或 output root，应让 watcher 的环境变量
和 Hydra 路径保持一致。

## 动态采样、TurnCredit 和 Hard Case Pool

### Bounded dynamic sampling

配置默认：

- 最多生成 3 个 generation batch；
- 最多连续跳过 10 个无效更新；
- 数值相等 epsilon 为 1e-6；
- 有效 Reward spread 下限为 0.005。

所有 rollout 的 terminal reward 相同或 spread 太小时，当前 update 会被记录为
skipped，而不是无限重采样。Hard Case Pool 不会触发重采样或 batch 注入。

### TurnCredit

algorithm.turn_credit.stage=train 默认开启 component attribution。它在终局 GRPO
advantage 确定后，把行为组件按 causal routing 归因到对应 turn，并执行 float64
conservation check。mix_ratio=0.30 控制与 terminal signal 的混合比例；关闭方式为：

~~~bash
GRPO_AUTO_SHUTDOWN=0 bash scripts/train_grpo.sh \
  algorithm.turn_credit.stage=off
~~~

### Hard Case Pool

它是 trainer-side、append-only 的审计器。当同一 task 连续 3 个完整 group 的 rollout
全部 reward_valid=true 且 correct_completion=0 时，记录 task、step、Reward 版本
和 streak。它不会改变 Reward、advantage、loss 或采样分布。

## 诊断阶段

run_grpo_stage.sh 将诊断和生产隔离，并拒绝复用已有目录：

~~~bash
ACTOR_MODEL_PATH=/path/to/merged-sft \
  bash scripts/run_grpo_stage.sh overfit-one

CONFIRM_OVERFIT_FOUR=YES ACTOR_MODEL_PATH=/path/to/merged-sft \
  bash scripts/run_grpo_stage.sh overfit-four

CONFIRM_PRODUCTION_GRPO=YES ACTOR_MODEL_PATH=/path/to/merged-sft \
  bash scripts/run_grpo_stage.sh production
~~~

one-task/four-task 会使用 SFT-seen task 做诊断；production 使用正式 grpo pool。详见
grpo_overfit_preflight.md。

## 恢复和 checkpoint

veRL checkpoint 位于 checkpoints/TravelGym/<experiment>/global_step_<n>/。恢复前
必须确认：

- 初始模型、task pool、Reward 版本和 tool schema 相同；
- trainer.resume_mode 与 trainer.resume_from_path 配置一致；
- 目标 output/checkpoint 目录没有混入另一个实验；
- 运行步数和 watcher 的 GRPO_EXPECTED_STEP 与实际目标一致。

生产阶段的 checkpoint 选择应看 validation 的 Reward、完成率、有效 rollout 率、
重复动作和 KL/clip 指标，不要只按最后 step 选择。
