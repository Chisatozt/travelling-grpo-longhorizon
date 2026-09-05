# configs/grpo

configs/grpo/grpo_multiturn.yaml 是 TravelGym GRPO 的项目侧基础配置。它把 veRL trainer、SGLang rollout、多轮工具、任务池和 terminal Reward 连接成一个可运行的 recipe。

使用 scripts/train_grpo.sh 启动。脚本会补充当前仓库的模型路径、数据路径、任务池和运行保护：

~~~bash
ACTOR_MODEL_PATH=/path/to/merged-sft \
GRPO_AUTO_SHUTDOWN=0 \
  bash scripts/train_grpo.sh
~~~

需要单次调整时，把 Hydra override 追加到脚本后面；需要 overfit 或 production 闸门时使用 scripts/run_grpo_stage.sh。完整流程见 [GRPO 文档](../../docs/grpo.md)。
