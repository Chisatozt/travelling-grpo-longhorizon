# TravelGym

本项目只保留并训练 `TravelGym`。它是一个 Gymnasium 兼容的多轮旅行
规划环境，训练链路为：

```text
Search 获取完整候选 → Action 获取自然语言证据 → Actor 隐式比较 → Answer 提交可见 ID
```

## 目录

```text
gyms/TravelGym/
├── travelgym/
│   ├── config.py
│   ├── data/
│   └── env/
├── README.md
├── requirements.txt
└── setup.py
```

## 安装

```bash
pip install -e gyms/TravelGym
```

API 配置从项目根目录 `.env` 读取；进程环境变量优先于 `.env`。可参考
`.env.example`。

## 公开交互协议

- `search`：为当前 travel aspect 获取完整候选列表。
- `action`：向 User Simulator 获取自然语言偏好证据；不会修改候选列表。
- `answer`：提交一个出现在当前 Search 结果中的候选 ID。

环境只维护公开控制状态：当前 aspect、已 Search 的 aspect、可见候选
ID、Action 次数、已回答 aspect 和公开对话历史。偏好 ID、正确/最佳 ID、
命中率和终局 Reward 只留在环境内部。

## 数据与训练

编号数据集位于 `data/`，例如 `travel22_multiturn_onechoice` 和
`travel2222_multiturn_onechoice`。使用
`examples/data_preprocess/travel_multiturn_w_tool.py` 生成 parquet，使用
`examples/data_preprocess/merge_customize.py` 合并 TravelGym-only 的训练
和验证集。

SGLang/VERL 训练入口是 `examples/sglang_multiturn/train.sh`，评测入口是
`eval/eval.py` 或 `eval/eval.sh`。终局 Reward 版本固定为
`travelgym-terminal-v2`。

更多状态机、奖励和隐私边界见 `docs/travel_public_protocol.md`。
