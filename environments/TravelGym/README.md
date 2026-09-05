# TravelGym

TravelGym 是本项目的核心交互环境。它把旅行规划任务组织成多个 aspect，例如航班、酒店、公寓、租车和餐厅；模型需要在有限步数内获取信息、理解用户偏好、选择候选并完成任务。

它回答的是“模型如何与世界交互、环境如何反馈和评分”，而不是“如何训练模型”。训练、评测和 Teacher 采集都复用这一份环境实现。

## 项目中的位置

~~~text
Teacher / GRPO / Evaluation
          |
          v
  interact_with_env
          |
          v
environments/TravelGym/travelgym/
          |
          v
  public observation + terminal Reward
~~~

- 状态机和动作解析：travelgym/env/travel_env.py；
- 本地/DeepSeek User Simulator：travelgym/env/user_simulator.py；
- 内置 TravelGym 数据：travelgym/data/；
- 配置：travelgym/config.py；
- 项目侧工具 schema：configs/tools/；
- public contract：src/travel_grpo/evaluation/travel_contract.py。

## 交互方式

模型使用四类动作：Search 获取当前 aspect 的完整候选，Action 询问用户或获取证据，Answer 提交当前可见的 option ID，Finish 结束 episode。Action 不会替环境自动过滤候选；正确答案和用户偏好仍然属于隐藏环境状态。

环境默认最大 25 步，Reward 只在 episode 结束时计算。正式 GRPO/native evaluation 的 Action 使用 question-aware DeepSeek User Simulator，local simulator 只适合离线测试和协议调试。

## 安装和最小检查

~~~bash
python -m pip install -e environments/TravelGym
python -m pytest -q environments/TravelGym/test_human.py
~~~

直接使用时：

~~~python
from travelgym import TravelEnv
from travelgym.config import TravelGymConfig

env = TravelEnv(TravelGymConfig(user_simulator_mode="local", max_steps=25))
observation, info = env.reset(seed=7)
observation, reward, terminated, truncated, info = env.step(
    "[search] Search the current travel aspect."
)
~~~

环境的完整动作规则、隐私边界和终局评分分别见 [公开协议](../../docs/travel_public_protocol.md)、[Reward](../../docs/reward.md) 和 [评测文档](../../docs/evaluation.md)。
