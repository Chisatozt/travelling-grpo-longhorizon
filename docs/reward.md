# TravelGym terminal Reward v2

## 设计原则

当前唯一有效的 Reward 版本是 travelgym-terminal-v2。它是确定性的 episode-end
Reward，不依赖另一个 LLM 对最终答案打分。DeepSeek User Simulator 只负责 Action
时的自然语言用户回复，不是 Reward Judge。

交互 step 的返回值恒为 0.0；trainer/evaluator 在 episode 结束后调用
get_terminal_reward() 或读取 get_reward_report()。

## 指标定义

设任务有 T 个 aspect：

| 指标 | 定义 |
| --- | --- |
| correct_completion | 正确回答的 aspect 数 / T |
| completion_success | 所有 aspect 都已回答且全部正确时为 1，否则为 0 |
| answer_coverage | 已回答 aspect 数 / T |
| answer_quality / best_answer_rate | 最佳答案数 / 已回答 aspect 数 |
| legal_chain_rate | 满足 Search 且有 Action 后再 Answer 的回答数 / 已回答 aspect 数 |
| coverage_adjusted_answer_quality | 最佳答案数 / T |
| coverage_adjusted_legal_chain_rate | 合法回答链数 / T |
| hidden_preference_hit_rate | Agent 实际 elicitation 的偏好数 / 任务偏好总数；无偏好任务为 1 |
| efficiency | max(0, 1 - steps / max_steps) |

Reward 使用 coverage-adjusted 的两项质量指标，因此漏答一个 aspect 不能维持满分。

## 公式

~~~text
policy_penalty = min(1.0, 0.05 * invalid_call_count
                          + 0.10 * wrong_answer_count)

redundant_action_penalty =
    min(0.60, 0.05 * min(3, redundant_action_count)
             + 0.10 * max(0, redundant_action_count - 3))

incomplete_penalty = 1.00 * (1.0 - answer_coverage)
zero_answer_penalty = 0.50 if no aspect is answered else 0
max_steps_penalty = 0.75 if termination_reason == "max_steps" else 0

total_penalty = policy_penalty
              + redundant_action_penalty
              + incomplete_penalty
              + zero_answer_penalty
              + max_steps_penalty

raw = 3.00 * correct_completion
    + 0.30 * coverage_adjusted_answer_quality
    + 0.20 * coverage_adjusted_legal_chain_rate
    + 0.15 * hidden_preference_hit_rate
    + 0.05 * efficiency
    - total_penalty

terminal_reward = clip(raw / 3.70, -1, 1)
~~~

## 终止和惩罚

- 非法 tool call 计入 invalid_call_count；
- 对错误 option ID 的 Answer 计入 wrong_answer_count；
- Action 无新证据会进入 redundant/no-gain 统计；前三个冗余 Action 每个扣 0.05，
  后续每个扣 0.10，冗余惩罚最多 0.60；
- 未回答 aspect 按 coverage 扣分；完全没有 Answer 时额外扣 0.50；
- 达到默认 25 步而没有自然结束时额外扣 0.75；
- finish 可以提前结束，但如果覆盖不足，仍会受到 incomplete penalty；
- 用户模拟器或环境外层调用失败会令 reward_valid=false，最终标量返回 0.0，
  训练器应隔离该 rollout。

max_steps、系数和 actor forbidden fields 以 data/evaluation/travel_manifest.json 为准；
实现位于 environments/TravelGym/travelgym/env/travel_env.py。

## 私有信息边界

Reward 计算可以读取任务的 correct/best IDs 和偏好，但这些字段不得进入：

- TravelEnv 返回给 Actor 的 observation；
- InteractTool.execute() 的 feedback；
- SGLang/Qwen 的下一轮 prompt；
- 外部 tracking 的 transcript。

允许发送给 tracking 的是脱敏后的 scalar，例如 Reward、完成率、步骤数、工具计数和
有效性标记。
