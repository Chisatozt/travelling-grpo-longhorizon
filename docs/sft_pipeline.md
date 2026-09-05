# Canonical SFT 管线（实现约定）

本文是 [SFT 操作指南](sft.md) 的实现补充，说明清洗器、数据格式和 token mask 的
不可变约定。

## 输入和输出

入口模块：

~~~text
travel_grpo.collection.prepare_travel_sft
travel_grpo.collection.merge_travel_sft
travel_grpo.collection.clean_travel_trajectories
travel_grpo.collection.travel_canonical
travel_grpo.training.sft.sft_split
travel_grpo.training.sft.qwen35_mask
travel_grpo.training.sft.validate_travel_canonical
~~~

推荐用 module CLI：

~~~bash
python -m travel_grpo.collection.prepare_travel_sft --help
python -m travel_grpo.collection.merge_travel_sft --help
python -m travel_grpo.training.sft.sft_split --help
~~~

基础语料 data/sft/travel_sft_public.json 不可覆盖。合并输出必须是派生路径；脚本
会拒绝相对/绝对路径等价的 in-place overwrite。

## Replay cleaner

cleaner 只根据公共 transcript、工具参数、tool feedback 和可选私有 task sidecar
回放协议。它不生成 candidate filter，也不把私有 label 添加回 messages。

可修复错误（如 action-before-search、answer-before-search、cross-aspect、repeated
search、invalid visible ID、duplicate answer、vague action）会保留在完整上下文中，
错误 Assistant turn 的 assistant_train_mask 为 0；之后的合法修复 turn 可以为 1。

fatal 情况（malformed tool JSON、missing tool call、reasoning/tool 截断、tool ID
错位、observation 错位、环境/API failure、无法修复的终局 Answer）从 offending turn
开始删除不可恢复后缀。清洗后的终局重新计算分类，不能因为原始轨迹曾失败就自动把
修复后的成功样本判为 wrong。

## Canonical schema

~~~json
{
  "schema_version": "travelgym-canonical-v1",
  "messages": [
    {"role": "user", "content": "..."},
    {
      "role": "assistant",
      "content": "",
      "reasoning_content": "...",
      "tool_calls": [
        {
          "id": "call_0001",
          "type": "function",
          "function": {
            "name": "interact_with_env",
            "arguments": {"choice": "search", "content": "Search flight."}
          }
        }
      ]
    },
    {
      "role": "tool",
      "tool_call_id": "call_0001",
      "name": "interact_with_env",
      "content": "..."
    }
  ],
  "tools": [{"type": "function", "function": {"name": "interact_with_env"}}],
  "enable_thinking": true,
  "assistant_train_mask": [0, 1, 0],
  "sample_weight": 1.0
}
~~~

校验要求包括 schema version、message role、tool_call_id 配对、工具名/choice/content、
mask 长度、thinking 字段和无私有字段泄漏。canonical hash 排除 audit-only annotation，
只对公开内容去重。

## Token-level mask

Qwen3.5 原生 template 先将完整 message sequence 渲染为 input IDs，再由
travel_grpo.training.sft.qwen35_mask 将 message-level mask 对齐到 token spans：

- user/tool/system 和 padding token 的 loss mask 为 0；
- 被选中的 assistant turn 的 reasoning 与 tool-call token 均为 1；
- 不监督 assistant 的可见前缀、错误 turn 或模板控制 token；
- 动态 padding 只补当前 batch，attention/position/loss mask 的 padding 区域为 0；
- 32K 是硬上限，不能用字符截断掩盖超长。

训练样本保持一个完整 episode 一个 dataset row；不要把每个正确 turn 展开为独立
prefix，因为那会改变长程 credit assignment 和 task 分布。

## Split 与审计

travel_grpo.training.sft.sft_split 先按 task group 选择固定 10 个 strict-gold
validation group，再将剩余可监督轨迹写入 train。split manifest 保存 tokenizer、
seed、任务键、数量和 token audit 状态。当前 checked-in split 为 490 train / 10 val。

*.audit.json 保留私有 replay event、任务对齐和终局诊断；*.manifest.json 保留公开可
复现所需的 schema/template hash、长度分位数、类别和监督 token 统计。二者不能被误
当作模型输入。
