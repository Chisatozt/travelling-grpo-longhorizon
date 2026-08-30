# TravelGym public protocol

协议只有一条主链路：

```text
search 获取完整候选 -> action 获取自然语言筛选证据
-> Actor 在上下文中隐式判断 -> answer 提交可见候选 ID
```

Search 将 `task["all_options"][aspect]` 的完整候选原样作为公开反馈；Action
只增加用户的自然语言证据和 action 计数，不生成过滤列表、不改变
`visible_option_ids`。最终由 Actor 自己理解候选属性并提交一个 Search 结果中的
ID。环境中不存在 `CandidateFilter`、`filtered_candidates`、per-preference
matching IDs 或 filter F1。

## Actor 可见状态

`PublicControl.snapshot()` 只包含：

- `current_aspect`
- `searched_aspects`
- `visible_option_ids`
- `action_count_by_aspect`
- `answered_aspects`
- `public_conversation_history`

控制器只拒绝 action/answer-before-search、跨 aspect、重复/无效调用、重复 Answer
以及不在当前 Search 结果中的 ID。拒绝 observation 是公开自然语言；没有 step
Reward。`preference_id`、命中状态、correct/best ID、任务 ID 和所有 Reward/诊断
均不得进入 Actor observation 或 tool feedback。现有 scrubber 继续作为防御检查，
但“文本里出现隐藏字段”本身不是新增的 Turn 删除规则。

## Private terminal reward

Reward ledger 完成整个 episode 后一次计算（step reward 恒为 0）：

```text
raw = 3.00 * correct_completion
    + 0.30 * answer_quality
    + 0.20 * legal_chain_rate
    + 0.15 * hidden_preference_hit_rate
    + 0.05 * efficiency
    - policy_penalty
terminal = clip(raw / 3.70, -1, 1)
```

其中 `correct_completion` 是正确 Answer aspect 数除以总 aspect 数，
`completion_success` 要求所有 aspect 都回答且正确；`answer_quality` 区分 best 和
其他 correct，`legal_chain_rate` 检查 Search→Action→Answer，
`hidden_preference_hit_rate` 与效率来自私有 ledger。非法调用、重复/跨 aspect、
不可见 ID、错误答案、无输出和超步数只在终局 penalty 中体现。无效基础设施
Reward 的 rollout 标记 `reward_valid=false`，训练器隔离。

## State machine

每个 aspect 的公开状态按 `unsearched -> searched -> evidence -> answered` 前进，
当前 aspect 完成后切换到下一个；环境不保存候选缩小状态。相同的 reducer 用在
在线 TravelGym、offline SFT 清洗和评测，确保拒绝原因与轨迹分类一致。

