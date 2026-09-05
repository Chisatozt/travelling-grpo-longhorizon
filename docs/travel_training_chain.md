# TravelGym 训练链路与数据隔离

## 端到端数据血缘

~~~text
data/task_pools/travel_task_pools.json
  -> Teacher/API collection
  -> public transcript replay + canonicalization
  -> SFT classification/audit
  -> Qwen native chat template + assistant_train_mask
  -> FSDP LoRA SFT
  -> merged SFT model
  -> veRL GRPO trainer
  -> SGLang rollout + InteractTool
  -> TravelGym terminal Reward
  -> GRPO advantage/update
  -> native smoke20/final200 evaluation
~~~

每条链路只使用一个物理源码位置；数据和 checkpoint 是显式输入，不通过旧目录或隐式
symlink 寻址。

## 任务隔离

身份键是 env_name::task_id：

~~~text
SFT        = 238 historical + 362 reserved Teacher expansion task
GRPO       = train task - SFT identities
Validation = fixed 200 test task
Smoke20    = fixed 20-task subset of Validation
~~~

任务池构建器会验证 active pool 互斥、source row 与 task ID 一致、selection seed
固定。6 条无法恢复的历史记录写入 quarantine，永不自动猜测。Parquet loader 在
tokenization/rollout 之前按 pool 过滤，因此聚合 Parquet 不能绕过任务隔离。

## Canonical 和清洗

canonical schema 是 travelgym-canonical-v1：

- 标准 system/user/assistant/tool messages；
- 独立 reasoning_content；
- 结构化 tool_calls；
- 内部 dict arguments；
- assistant_train_mask 与 messages 对齐；
- sample_weight 记录 strict/recoverable/partial 的监督权重。

cleaner 从公开拒绝和 tool/observation 对齐信息回放状态机：

- 可明确修复的 action-before-search、answer-before-search、cross-aspect、repeated
  search、invisible ID、duplicate answer、vague action 会保留为 mask=0 上下文；
- malformed JSON、缺少 tool call、think/tool 截断、tool_call_id mismatch、observation
  错位、环境失败和无法修复的 terminal error 会截断后缀；
- 清洗后重新计算 completion/coverage/合法链指标；
- wrong/infrastructure/overlength 不进入 SFT。

## SFT 与 GRPO 边界

SFT 训练只优化 assistant 目标 token，user/tool observation 是上下文。训练使用
verl.trainer.fsdp_sft_trainer，不依赖 SGLang。

GRPO 从合并后的 SFT 模型开始。veRL 负责 optimizer、FSDP、Ray、group advantage 和
checkpoint；SGLang 负责当前策略的多轮 rollout；InteractTool 保持 TravelGym
实例；terminal Reward 只通过 trainer metadata 回传。SGLang 反馈中的 observation
不会包含 private Reward。

## Validation 边界

GRPO 内置 validation 是训练健康检查，默认一题一次；native evaluate_native.sh
的 pass@3 是独立的策略评测。Final-200 不能反向影响 task pool、prompt、checkpoint
或 Reward。所有正式比较都保留 invalid/not-judged 任务在分母和审计中。
