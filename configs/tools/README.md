# configs/tools

本目录定义 Actor 与 TravelGym 交互时使用的 interact_with_env 工具：

- interact_tool_schema.yaml：公开 function schema；
- interact_tool_config.yaml：AgentLoop/SGLang 多轮调用配置。

公开动作是 search、action、answer 和 finish。环境实际状态机位于 environments/TravelGym/travelgym/env/travel_env.py，协议检查位于 src/travel_grpo/evaluation/travel_contract.py。

修改工具字段时，请同步检查 [公开协议](../../docs/travel_public_protocol.md) 和 [Reward 说明](../../docs/reward.md)，不要把隐藏答案、偏好或 Reward ledger 放进 Actor 上下文。
