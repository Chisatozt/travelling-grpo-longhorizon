# 任务池与 Teacher 数据

## 目标

SFT 需要完整的多轮 TravelGym 轨迹，而不是只有最终答案的单轮文本。数据管线保留：

- Agent 的 assistant 文本、thinking 和结构化工具调用；
- TravelGym 返回的公开 observation；
- 每个动作对应的公开拒绝或成功反馈；
- 终局分类、任务身份和清洗审计。

偏好 ID、正确/最佳选项、Reward ledger 和其他私有标签只进入本地 cache/audit，
不会进入模型可见的 canonical 消息。

## 1. 固定任务池

先生成或检查任务池：

~~~bash
python -m travel_grpo.collection.task_pools \
  --sft-target-count 600 \
  --output data/task_pools/travel_task_pools.json
~~~

当前 checked-in manifest 的身份键是 env_name::task_id，选择 seed 是 20260801：

| Pool | 数量/来源 | 用途 |
| --- | --- | --- |
| sft | 238 个可解析历史 task + 362 个 Teacher expansion reservation | Teacher 采集和 SFT |
| sft_smoke | sft 中固定分层的 20 个 task | 低成本采集验证 |
| grpo | train split 中排除 SFT 身份后的 task | 在线 GRPO rollout |
| validation | test split 中固定的 200 个 task | 正式验证/Final-200 |
| validation_smoke | validation 中固定的 20 个 task | 快速验证 |

历史语料有 6 条记录无法唯一恢复 task ID。它们被明确写入
quarantined_sft 并弃用，既不进入活动池，也不会由代码猜测或重新引入。任务池
加载时要求 strict_task_identity=true，任何 pool 交叉都会失败。

## 2. Teacher 采集

Teacher 入口复用 travel_grpo.evaluation.eval，它会实际驱动 TravelGym 并保存可断点
续跑的 JSON cache。SFT 采集必须使用 DeepSeek、开启 Thinking，并要求每个工具回合
存在非空 reasoning：

~~~bash
python -m travel_grpo.evaluation.eval \
  --model_name deepseek-v4-flash \
  --sft-collection \
  --thinking enabled \
  --include-think \
  --require-think \
  --task-pool-manifest data/task_pools/travel_task_pools.json \
  --task-pool sft_smoke \
  --pass_k 2 \
  --max_turns 25 \
  --collection-cache outputs/evaluation/deepseek_teacher_sft_teacher_cache.json \
  --collection-run-id deepseek_teacher_sft \
  --save_name deepseek_teacher_sft
~~~

smoke 通过后，可以把 sft_smoke 改为 sft，继续使用相同 cache 和 run id。已完成的
env/task/pass 不会重复请求。正式采集可以用 collection-task-limit 限制新增 task
数量。

Teacher cache 的每个记录键是：

~~~text
env_name::task_id::pass_index
~~~

cache provenance 至少记录 model、thinking、reasoning effort、max turns、代码版本、
任务池 hash 和 collection run id。cache 是私有可恢复输入，不应提交到公开数据目录。

## 3. 合并、清洗和分类

历史公开语料 data/sft/travel_sft_public.json 是不可覆盖的输入。将它和 Teacher
cache 合并为新的 canonical 文件：

~~~bash
python -m travel_grpo.collection.merge_travel_sft \
  --base data/sft/travel_sft_public.json \
  --input outputs/evaluation/deepseek_teacher_sft_teacher_cache.json \
  --output data/sft/travel_sft_qwen35_merged.jsonl \
  --audit-output data/sft/travel_sft_qwen35_merged.audit.json \
  --manifest-output data/sft/travel_sft_qwen35_merged.manifest.json \
  --tokenizer Qwen/Qwen3.5-4B \
  --max-length 32768 \
  --split-output-dir data/sft/travel_sft_qwen35_split \
  --task-pool-manifest data/task_pools/travel_task_pools.json
~~~

tokenizer 使长度和监督 token 审计使用 Qwen 原生 chat template；max-length 默认是
32768，超长样本按 error/quarantine 处理，不做字符级截断。脚本只去除 canonical
内容完全相同的重复记录，不会把同一 task 的不同合法轨迹错误合并。

分类优先级为：

1. infrastructure_invalid：环境/API、消息对齐、tool-call 或任务身份不可用；
2. recoverable_correct：存在公开可修复错误，但修复后完整成功；
3. strict_gold：没有保留的协议错误且完整成功；
4. partial_correct：至少完成一个正确 aspect，但未完整成功；
5. totally_wrong：没有正确完成的 aspect。

当前 checked-in manifest 的 795 行由 238 条历史记录和 557 条 Teacher 记录组成：

| 类别 | 行数 | SFT 处理 |
| --- | ---: | --- |
| strict_gold | 13 | 权重 1.0 |
| recoverable_correct | 50 | 权重 1.0 |
| partial_correct | 438 | 权重 0.5 |
| totally_wrong | 294 | 排除 |
| 合计 | 795 | — |

另外 36 条 canonical duplicate 已被记录在 manifest 中；6 条 opaque historical row
属于任务池 quarantine，不进入这 795 行。

## 4. 产物和隐私

典型输出包括：

~~~text
data/sft/
  travel_sft_qwen35_merged.jsonl
  travel_sft_qwen35_merged.train.jsonl
  travel_sft_qwen35_merged.<class>.jsonl
  travel_sft_qwen35_merged.audit.json
  travel_sft_qwen35_merged.manifest.json
  travel_sft_qwen35_split/
    train.jsonl
    val_gold10.jsonl
    split_manifest.json
outputs/evaluation/
  <run>_teacher_cache.json
  <run>_collection_tracking.json
~~~

当前正式 split 是 490 条 train 轨迹、10 条 strict-gold validation 轨迹，按 task
group 分开。原始 assistant reasoning、用户模拟器请求和完整轨迹可能包含敏感信息，
只允许出现在本地 outputs/ 或受控 audit 中。SwanLab/W&B 等 tracking 只发送脱敏
scalar，不发送 API key、完整 transcript 或私有答案。
