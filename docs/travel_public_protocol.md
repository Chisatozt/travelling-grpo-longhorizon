# TravelGym 公开交互协议

## 协议目标

TravelGym 的模型可见主链路是：

~~~text
search 完整候选
  -> action 获取自然语言证据
  -> actor 自己比较候选
  -> answer 提交可见 option ID
~~~

环境不会因为用户偏好替 Agent 过滤或排序候选。Search 返回当前 aspect 的完整
候选集合；Action 只产生自然语言用户反馈和计数；Answer 必须使用当前 Search 结果中
出现的一个 ID。

## 工具 schema

模型通过一个工具调用：

~~~json
{
  "name": "interact_with_env",
  "arguments": {
    "choice": "search | action | answer",
    "content": "query, question, or one visible option ID"
  }
}
~~~

内部 canonical record 使用 dict arguments；OpenAI-compatible API 使用 JSON string。
verl.tools.travel_tool_adapter 是两种表示之间的唯一转换边界。

环境原始 wire string 仍兼容：

~~~text
[search] Search the current flight.
[action] Which flight preferences matter most?
[answer] F5
[finish]
~~~

finish 是环境动作格式，但不属于模型工具 schema 的三个 choice；它由 evaluator/
collector 的兼容路径使用。

## 动作规则

| 动作 | 前置条件 | 公共结果 |
| --- | --- | --- |
| Search | 当前 aspect 尚未 Search | 返回该 aspect 的全部候选 ID 和公开属性 |
| Action | 当前 aspect 已 Search，且仍有足够步数 | 返回自然语言用户证据；候选列表不变 |
| Answer | 当前 aspect 已 Search，ID 在可见列表中 | 记录一个答案并切换到下一个 aspect |
| Finish | 任意尚未结束的 episode | 立即结束，未答 aspect 由 Reward 扣分 |

控制器还会拒绝：

- 空参数、未知 choice 或 malformed tool call；
- action/answer-before-search；
- cross-aspect 操作；
- repeated Search、重复 Answer；
- 不在当前 Search 结果中的 option ID；
- episode 结束后的调用；
- 在剩余步数不足以完成未答 aspect 时继续无效 Action。

当前默认 max_steps=25。回答可在 Search 后直接进行；不要把 Action 误认为环境会
强制过滤候选。require_action_before_answer 只是配置兼容字段，当前公开 reducer 的
实际约束仍以 Search-before-Answer 为准。

## Actor 可见状态

公开 observation 只允许包含：

- feedback；
- current_aspect；
- searched_aspects；
- visible_option_ids；
- action_count_by_aspect；
- answered_aspects；
- public_conversation_history。

以下内容永远是私有的：task ID、完整 task、偏好 ID、correct/best ID、候选全集的
内部 gold 字段、Reward report、step/terminal reward、成功标签、diagnostics、
用户模拟器内部 judgment 和 API 细节。

sanitize_public_feedback() 和 TravelGym 的 public projection 是防御性检查；tool
adapter 不负责重算候选或 Reward。

## 状态机

每个 aspect 公开地从：

~~~text
unsearched -> searched -> evidence (可选) -> answered
~~~

向前推进。回答当前 aspect 后切换到下一个；全部 aspect 完成时终止。SFT cleaner
通过回放同一类公共反馈识别协议错误，保证离线分类与在线环境的拒绝原因一致。

## 开发约束

修改 Search feedback、工具 schema、状态字段或拒绝文本时，必须同步：

1. data/evaluation/travel_manifest.json；
2. verl/tools/travel_tool_adapter.py 和相关 config；
3. src/travel_grpo/collection/travel_canonical.py；
4. tests/test_canonical_travel_pipeline.py、环境和 Reward 测试；
5. 本文档和运行 preflight。

不要向 actor observation 添加任何“方便训练”的 gold 字段；若 trainer 需要指标，应从
环境私有 report 通过独立 metadata 通道传递。
