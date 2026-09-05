# LoRA SFT 操作指南

## 目标和输入

SFT 的目标是让基础模型先学会 TravelGym 的合法动作协议、证据使用和多 aspect 完成，
再把合并后的模型交给 GRPO。权威输入和产物是：

- 基础模型：默认 Qwen/Qwen3.5-4B；
- 训练集：data/sft/travel_sft_qwen35_split/train.jsonl，当前 490 条；
- 验证集：data/sft/travel_sft_qwen35_split/val_gold10.jsonl，当前 10 条；
- 配置：verl/trainer/config/travel_qwen35_sft.yaml；
- 输出：checkpoints/TravelGym/qwen35_4b_canonical_sft/。

当前 split 的 10 个验证 task group 与训练 task 不重叠。最终 Final-200 不能用于
checkpoint 选择；它只用于 recipe 固定后的独立比较。

## Canonical 样本

每行是一个完整 episode，而不是按正确回合展开的前缀样本：

~~~json
{
  "schema_version": "travelgym-canonical-v1",
  "messages": [
    {"role": "user", "content": "Please plan the trip."},
    {
      "role": "assistant",
      "content": "",
      "reasoning_content": "I should inspect the current aspect.",
      "tool_calls": [
        {
          "id": "call_0001",
          "type": "function",
          "function": {
            "name": "interact_with_env",
            "arguments": {"choice": "search", "content": "Search the current aspect."}
          }
        }
      ]
    },
    {
      "role": "tool",
      "tool_call_id": "call_0001",
      "name": "interact_with_env",
      "content": "Search results ..."
    }
  ],
  "tools": ["the canonical interact_with_env schema"],
  "enable_thinking": true,
  "assistant_train_mask": [0, 1, 0],
  "sample_weight": 1.0
}
~~~

canonical 内部的 function.arguments 是 dict；到 OpenAI-compatible API 边界时，
verl.tools.travel_tool_adapter 才会将它序列化为 JSON string。assistant_train_mask
与 messages 等长：

- user、tool 和 system 为 0；
- 合法 Search、Action、Answer 及可修复后的正确 Assistant turn 为 1；
- 错误 Assistant turn 的 thinking 和 tool call 整体为 0；
- totally wrong、基础设施无效、超长记录不进入 SFT。

token mask 由 travel_grpo.training.sft.qwen35_mask 对 Qwen 原生 template 做精确
对齐，不能使用字符截断代替 token-level mask。

## 运行训练

安装完成并确认数据后运行：

~~~bash
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=1 \
  -m verl.trainer.fsdp_sft_trainer \
  --config-name=travel_qwen35_sft \
  trainer.n_gpus_per_node=1 \
  model.partial_pretrain=Qwen/Qwen3.5-4B
~~~

当前配置的关键默认值：

| 设置 | 值 |
| --- | --- |
| 最大序列长度 | 32768 tokens |
| batch | 全局 trajectory batch 8 |
| micro batch | 每 GPU 1 |
| LoRA rank / alpha | 16 / 32 |
| epoch | 3 |
| padding | 动态 padding，长度分桶，补齐到 128 的倍数 |
| truncation | error |
| 验证 | 每 epoch |
| logger | console + SwanLab |

490 条 train trajectory 在单 GPU 当前配置下每 epoch 约 62 个 optimizer step；实际
步数仍以 trainer 日志和 checkpoint metadata 为准。训练保存每个 epoch 的 checkpoint，
应依据 held-out 验证指标和协议指标选择，而不是默认最后一个。

训练前可只检查 canonical 结构：

~~~bash
python -m travel_grpo.training.sft.validate_travel_canonical \
  --input data/sft/travel_sft_qwen35_merged.jsonl \
  --max-length 32768
~~~

## 合并 LoRA

GRPO 和 native SGLang 评测需要完整 Hugging Face 权重，而不是 adapter-only 目录：

~~~bash
python scripts/merge_sft_lora.py \
  --adapter checkpoints/TravelGym/qwen35_4b_canonical_sft/global_step_186 \
  --base-model Qwen/Qwen3.5-4B \
  --output checkpoints/TravelGym/qwen35_4b_canonical_sft/merged_step_186
~~~

脚本会拒绝覆盖非空输出目录，并保存 tokenizer 和 merge_metadata.json。如果选择其他
epoch/step，必须同时调整 adapter 和 output 路径：

~~~bash
python scripts/merge_sft_lora.py \
  --adapter checkpoints/TravelGym/qwen35_4b_canonical_sft/global_step_<selected> \
  --output checkpoints/TravelGym/qwen35_4b_canonical_sft/merged_step_<selected>
~~~

## Legacy 配置边界

configs/sft/qwen3_customized.yaml 只用于旧 ShareGPT renderer 的回归检查。它不能
读取 assistant_train_mask、完整 trajectory 权重或 canonical 的逐回合 loss mask，
不是当前权威 SFT 配置。不要用它产出的旧格式模型直接宣称复现 canonical SFT。
