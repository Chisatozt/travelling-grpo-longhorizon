# TravelGym canonical SFT 数据管线

SFT 的公开协议固定为：

```text
Search 获取当前 aspect 的完整候选
  -> Action 获取用户自然语言证据（不改变候选列表）
  -> Actor 在上下文中隐式比较
  -> Answer 提交 Search 中可见的一个 option ID
```

候选不会由环境过滤或缩小；`preference_id`、`correct_ids`、`best_id`、Reward
和清洗诊断只在 User Simulator、Reward ledger 或 audit sidecar 中存在，不会
进入 canonical `messages`、tool observation 或 Actor observation。

## 1. Teacher 采集

先生成一次互斥任务清单。预览阶段将历史语料中可解析的 238 个 Task 加上 362
个确定性扩展 Task，组成 600-task 的 `sft` pool；GRPO 使用其余 train task，
Validation 使用固定 test task：

```powershell
python .\sft\task_pools.py `
  --sft-target-count 600 `
  --output .\data\task_pools\travel_task_pools.json
```

当前历史语料中有 6 条无法仅凭公开内容唯一对齐的记录。它们被列入
`quarantined_sft`，永久隔离/弃用，不属于任何活动池，也不会被 Teacher
重新采集或用于 SFT。正式清单对活动池直接标记
`strict_task_identity=true`；`sft_task_alignment_candidates.json` 仅保留为
审计证据，不需要 reviewed map，代码也不会猜测或重新引入这些记录。

DeepSeek-V4-Flash 的 SFT 采集必须启用 Thinking，并让导出的每一个工具回合
包含非空 `<think>...</think>`：

```powershell
python .\eval\eval.py `
  --model_name deepseek-v4-flash `
  --sft-collection --thinking enabled --include-think --require-think `
  --task-pool-manifest .\data\task_pools\travel_task_pools.json `
  --task-pool sft `
  --max_turns 25 --save_name outputs/deepseek_teacher_sft
```

首次真实采集先将 `--task-pool sft_smoke`（固定分层 20-task）并使用
`pass_k=2` 的 Teacher cache；smoke 通过后改为 `--task-pool sft`，沿用同一
`--save_name`/cache 即可从已完成的 task/pass 继续，不会重复调用。

这一步只生成 Teacher cache，不覆盖仓库中的 244 条原始文件；6 条 quarantine
记录不会进入采集清单。API 的
`reasoning_content` 在工作 transcript 中保持独立字段；为兼容旧数据，cache
中的 SFT 导出仍可包含 `<think>` 与 `<tool_call>` 文本。

## 2. 合并、清洗和分类

基础语料始终是 `sft/travel_sft_public.json`。新 cache 通过 `--input` 显式传入，
脚本只删除 canonical 公共内容完全相同的重复记录；同一 task 的不同合法轨迹
会保留。默认 `require-think`，且默认 32K、超长报错：

```powershell
python .\sft\merge_travel_sft.py `
  --base .\sft\travel_sft_public.json `
  --input .\eval\outputs\deepseek_teacher_sft_teacher_cache.json `
  --output .\sft\travel_sft_canonical.jsonl `
  --audit-output .\sft\travel_sft_canonical.audit.json `
  --manifest-output .\sft\travel_sft_canonical.manifest.json `
  --tokenizer Qwen/Qwen3.5-4B --max-length 32768 `
  --split-output-dir .\sft `
  --task-pool-manifest .\data\task_pools\travel_task_pools.json
```

`--tokenizer` 用于按 Qwen 原生 template 做精确 token 长度审计，首次运行需要
本地缓存该 tokenizer；不希望联网时可传本地路径。没有安全 task ID 对齐时，
记录会进入 `infrastructure_invalid`，不会猜测；那 6 条历史记录已经被
`quarantined_sft` 明确弃用，并在 canonical 合并前直接丢弃。不要用初始文本做多对一去重。

输出包括：

- `*.jsonl`：canonical 完整轨迹（每行一个完整 episode）；
- `*.train.jsonl`：SFT 可用行；
- `*.strict_gold.jsonl`、`*.recoverable_correct.jsonl`、`*.partial_correct.jsonl`、
  `*.totally_wrong.jsonl` 和 quarantine 文件；
- `*.audit.json`：私有清洗事件、任务对齐和终局诊断；
- `*.manifest.json`：schema/template hash、长度分位数、分类和有效监督 token。

正式训练使用 `travel_sft_qwen35_split/train.jsonl` 和
`travel_sft_qwen35_split/val_gold10.jsonl`；后者只保留固定 10 个 task group
中的 `strict_gold` 记录，`partial_correct` 与非 SFT 分类仅保留在 canonical
和分类审计文件中。

清洗器从错误 Assistant Turn 开始删除不可恢复错误的整个后缀。公开拒绝且有
后续修复的 action-before-search、answer-before-search、跨 aspect、重复 Search、
无效参数、不可见 ID、重复 Answer 和模糊 Action 会保留为 mask=0 上下文，修复
Turn 为 mask=1。错误终局 Answer 会截断；非终局错误 Answer 保留但不监督。
原始 fatal 后缀被截断而清洗后成功的轨迹仍分类为 `strict_gold`。

检查 canonical 文件：

```powershell
python .\sft\validate_travel_canonical.py `
  --input .\sft\travel_sft_canonical.jsonl --max-length 32768
```

## 3. canonical 格式和 loss mask

每条样本使用标准 `system/user/assistant/tool` 消息：

```json
{
  "schema_version": "travelgym-canonical-v1",
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "", "reasoning_content": "...", "tool_calls": [{"id": "call_0001", "type": "function", "function": {"name": "interact_with_env", "arguments": {"choice": "search", "content": "..."}}}]},
    {"role": "tool", "tool_call_id": "call_0001", "name": "interact_with_env", "content": "..."}
  ],
  "tools": [{"type": "function", "function": {"name": "interact_with_env", "parameters": {"type": "object", "properties": {"choice": {"type": "string"}, "content": {"type": "string"}}, "required": ["choice", "content"]}}}],
  "enable_thinking": true,
  "assistant_train_mask": [0, 1, 0]
}
```

`function.arguments` 在 canonical 内部是 dict；发送 OpenAI-compatible 请求时由
`verl.tools.travel_tool_adapter` 转成 JSON string。`assistant_train_mask` 与
`messages` 等长：system/user/tool 为 0，正确 Search/具体 Action/正确 Answer
及合法修复为 1；错误 Assistant（包括 CoT 和 tool call）整体为 0。完整轨迹
只对应一个 Dataset 样本，绝不按正确 Turn 展开前缀样本。Qwen3.5 原生
`apply_chat_template(..., enable_thinking=True)` 直接生成 input IDs，token-level
mask 通过 `sft/qwen35_mask.py` 对齐，不使用字符截断。

分类权重：strict/recoverable=1.0，partial=0.5（必须存在监督 Turn），
totally_wrong、infrastructure_invalid、overlength 不进入 SFT。

## 4. VERL LoRA SFT

`verl/trainer/config/travel_qwen35_sft.yaml` 是权威配置，默认
`Qwen/Qwen3.5-4B`、LoRA rank 16、全局 trajectory batch 8、micro batch 1、
3 epochs、32K 最大长度、`truncation=error` 和原生 Qwen template。490 条训练
轨迹对应每 epoch 62 个 optimizer steps；每个 epoch 都保存 checkpoint，最终模型
应结合 held-out TravelGym 成功率在 epoch 1/2/3 中选择，而不是固定使用最后一个。

训练样本保持原始完整 token stream，不再逐条补到 32K。训练集先按原生 Qwen
template 的精确 token 长度分桶并打乱 batch 顺序，再在每个本地 batch 内右侧补齐
到最长轨迹（向上取 128 的倍数）；attention、position 和所有 loss mask 的 padding
区域均为 0。32K 仍是硬上限，合法轨迹不会被截断。不要把不同 trajectory 直接
拼成一条可互相 attention 的序列；若未来启用 sequence packing，必须使用隔离的
attention block 和 position IDs。

启动命令：

```bash
torchrun --standalone --nproc_per_node=1 -m verl.trainer.fsdp_sft_trainer \
  --config-name=travel_qwen35_sft \
  trainer.n_gpus_per_node=1 \
  model.partial_pretrain=Qwen/Qwen3.5-4B
```

该 TravelGym SFT 配置默认使用 SwanLab 记录训练指标和验证样本。训练服务器
上可在项目根目录 `.env` 设置 `SWANLAB_API_KEY`；无网络时设置
`SWANLAB_MODE=local` 或 `offline`。

不同 VERL 版本的入口参数可能略有差异；必须确认 dataloader 实际读取
`assistant_train_mask` 和 `sample_weight`。`sft/qwen3_customized.yaml` 仅保留
旧 ShareGPT renderer 的回归用途，不支持 canonical 逐 Turn mask，不应作为
本管线的权威训练配置。
