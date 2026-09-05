# GRPO 阶段预检与诊断闸门

run_grpo_stage.sh 将短诊断和 production GRPO 分成独立目录、独立 checkpoint 和
独立 task pool。任何阶段都从同一个完整合并 SFT 模型重新开始，不把 overfit checkpoint
串接到下一阶段。

## 预检

~~~bash
ACTOR_MODEL_PATH=/path/to/merged-sft \
  bash scripts/run_grpo_stage.sh check
~~~

预检会检查：

- data/task_pools/travel_grpo_overfit_pools.json 是否由当前 SFT split 生成；
- merged model 是否有顶层权重；
- Qwen/FlashAttention/SGLang/veRL runtime；
- GRPO/validation reset prompt 是否在 1280 token 上限内；
- GPU 是否可用。

## 诊断阶段

one-task：

~~~bash
ACTOR_MODEL_PATH=/path/to/merged-sft \
GRPO_AUTO_SHUTDOWN=0 \
  bash scripts/run_grpo_stage.sh overfit-one
~~~

它使用 grpo_overfit_one，默认 1 个 SFT-seen task、batch 1、mini-batch 1、10 steps。

four-task：

~~~bash
CONFIRM_OVERFIT_FOUR=YES \
ACTOR_MODEL_PATH=/path/to/merged-sft \
GRPO_AUTO_SHUTDOWN=0 \
  bash scripts/run_grpo_stage.sh overfit-four
~~~

它使用 grpo_overfit_four，默认 4 个 SFT-seen task、batch 4、mini-batch 2、20 steps，
10/20 保存 checkpoint。该阶段需要显式确认，避免误花费 GPU/API 预算。

## Production

production 默认暂停，必须显式解锁：

~~~bash
CONFIRM_PRODUCTION_GRPO=YES \
ACTOR_MODEL_PATH=/path/to/merged-sft \
GRPO_AUTO_SHUTDOWN=0 \
  bash scripts/run_grpo_stage.sh production
~~~

production 使用正式 grpo pool、validation pool 和 DeepSeek User Simulator。默认从
step 20 开始跑到 RUN_UNTIL_STEP（脚本可通过环境变量或 Hydra override 调整），
运行前会检查 merged model 和 prompt budget。恢复已有生产目录时只能显式设置
PRODUCTION_RESUME_MODE=auto，且 checkpoint/artifact 必须成对存在。

## 观察指标

不要把 PPO loss 当作 SFT loss 期待单调下降。诊断至少检查：

- terminal reward、completion_success、answer coverage；
- finite/non-zero gradient norm、PPO KL、clip fraction、response length；
- dynamic sampling admitted/skipped counters；
- invalid、wrong、redundant、repeat 和 User Simulator error；
- checkpoint metadata 与实际 optimizer step。

同一 task 的 rollout group 全部 reward 相同时，skip 可能是有效保护；连续全零则说明
当前策略没有产生可学习差异。

## 关机 watcher

train_grpo.sh 默认会 armed watcher。第一次人工调试必须使用：

~~~bash
GRPO_AUTO_SHUTDOWN=0 bash scripts/run_grpo_stage.sh overfit-one
~~~

无人值守运行只有在确认 checkpoint、日志和关机行为均已验证后才使用默认 watcher。
