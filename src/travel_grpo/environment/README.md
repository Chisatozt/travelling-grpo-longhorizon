# environment

这里描述项目适配层与 TravelGym 的边界，不保存第二份环境实现。实际状态机位于 environments/TravelGym/travelgym/，工具 schema 位于 configs/tools/，公开字段检查位于 src/travel_grpo/evaluation/travel_contract.py。

如果要修改动作、observation 或隐私边界，请从 [公开交互协议](../../../docs/travel_public_protocol.md) 开始，并同步更新环境、schema、contract 和测试。
