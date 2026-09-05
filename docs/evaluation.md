# TravelGym 评测流水线

评测必须使用固定的任务 manifest 和 travelgym-public-v1 协议。当前项目提供两条
评测路径：

| 路径 | Actor | 适用场景 |
| --- | --- | --- |
| HTTP evaluator | OpenAI-compatible endpoint | Teacher/API 采集、外部模型或快速诊断 |
| Native evaluator | 本地完整模型 + Ray/SGLang | SFT/GRPO 的 smoke20、Final-200 正式策略评测 |

两条路径都使用 TravelGym terminal Reward；它们不使用 LLM-as-a-Judge 给 Reward
重新打分。DeepSeek 在 Action 中扮演用户模拟器，不能与最终评分 Judge 混淆。

## 1. HTTP 评测

travel_grpo.evaluation.eval 的 CLI 支持 model_name、port、max_turns、pass_k、
temperature、envs、test-manifest、task-pool-manifest、task-pool、max_tasks、
save_name 和 dry-run。

仓库入口会自动加载 .env，默认评估 data/evaluation/test_manifests/final200.json：

~~~bash
ACTOR_MODEL_NAME=your-served-model \
  bash scripts/evaluate_http.sh
~~~

如果使用自建 Actor endpoint：

~~~bash
CUSTOMIZED_SERVED_MODEL_NAME=local-actor \
CUSTOMIZED_SERVED_MODEL_PORT=8500 \
ACTOR_MODEL_NAME=local-actor \
  bash scripts/evaluate_http.sh
~~~

策略模型服务可以由 scripts/serve_vllm.sh 启动，但该脚本是模板：运行前需要编辑
模型路径、served name、端口和 GPU。用户模拟器仍由 USER_MODEL_NAME 和
OPENAI_BASE_URL 决定。

直接运行一个 20-task manifest：

~~~bash
python -m travel_grpo.evaluation.eval \
  --model_name local-actor \
  --port 8500 \
  --test-manifest data/evaluation/test_manifests/smoke20.json \
  --envs travel22 travel33 travel44 travel233 travel333 travel334 travel444 travel2222 \
  --max_turns 25 \
  --pass_k 1 \
  --temperature 0 \
  --save_name http_smoke20
~~~

dry-run 只验证 manifest、任务池和待处理工作，不创建 API client 或进行 rollout。

## 2. Native smoke20 / final200

native 入口会调用同一套 veRL/SGLang 多轮路径，但通过 trainer.val_only=true 在验证
阶段结束。它会：

- 读取本地完整模型和 tokenizer；
- 运行 Qwen3.5 原生 template、thinking 和 qwen3_coder tool parser；
- 对每个 task 最多执行 VALIDATION_PASS_K 次独立 sampled attempt；
- 某 task 的一次 attempt completion_success==1 后提前停止该 task；
- 默认禁用 validation retry，保留真实 attempt_count 和 valid rate；
- 拒绝覆盖已有 artifact/checkpoint/baseline 路径。

冒烟评测：

~~~bash
VALIDATION_MODEL_PATH=/path/to/merged-model \
VALIDATION_EXPERIMENT=my_smoke20 \
  bash scripts/evaluate_native.sh smoke20
~~~

正式评测：

~~~bash
VALIDATION_MODEL_PATH=/path/to/merged-model \
VALIDATION_EXPERIMENT=my_final200 \
VALIDATION_PASS_K=3 \
  bash scripts/evaluate_native.sh final200
~~~

当前脚本默认 pass_k=3、temperature/top-p 为 1.0/1.0、do_sample=true，因此
native smoke/final 是 pass@3 sampled evaluation；这与 GRPO 训练内部的单次 greedy
validation 不同。native evaluator 自动运行 tokenizer/parser、GPU、prompt budget
和 output preflight。

## 3. 评测产物

HTTP evaluator 的默认根目录是 outputs/evaluation/，native evaluator 的默认根目录
是 outputs/<VALIDATION_EXPERIMENT>/：

~~~text
<save_name>_manifest.json
  TravelGym protocol、Reward version、task manifest、采样设置和 hash

<save_name>_results.json
  每个 environment/pass_k 的 terminal reward、pass@k、valid rate 等 scalar

<save_name>_reward_cache.json
  rollout history、tool feedback、terminal report 和 telemetry；属于私有审计数据
~~~

交互 step Reward 不会被写入模型 feedback；只在结束时写入 trainer/evaluator 的
private metadata。reward_valid=false 的 rollout 不应当作为正常零分样本解释。

## 4. Final-200 plan

travel_grpo.evaluation.final200 负责离线校验并生成 execution plan，不会自动启动
模型、Ray 或 API 执行器。它要求显式提供 base、SFT、DeepSeek 和待选 GRPO checkpoint：

~~~bash
python -m travel_grpo.evaluation.final200 \
  --task-pool-manifest data/task_pools/travel_task_pools.json \
  --smoke-manifest data/evaluation/test_manifests/smoke20.json \
  --base /path/to/base-model \
  --sft /path/to/merged-sft \
  --deepseek deepseek-v4-flash \
  --selected-grpo-checkpoint /path/to/grpo-checkpoint \
  --output outputs/evaluation/final200_plan.json \
  --dry-run
~~~

plan 固定 all200、smoke20_seen 和 unseen180 三个 split，以及 reward version、seed、
max turns、tool parser、thinking budget 和 pass@3 约定。CLI 的真实模型执行 adapter
当前保持关闭；不要把生成 plan 误认为已经完成 Final-200 rollout。

## 5. 评测口径

汇总至少保留：

- terminal reward、Reward validity 和 termination reason；
- completion_success、correct_completion、answer coverage；
- best-answer 和 legal-chain 指标；
- steps、tool calls、invalid/wrong/redundant actions；
- User Simulator API calls、errors、cache hits 和 token telemetry；
- pass@k、attempt_count、early-stopped tasks 和 valid attempt rate。

Reward、协议和确定性诊断分开报告，不合成为一个不可解释的总分。比较不同模型时，
必须使用相同的 task ID、环境 variant、Reward 版本、tool schema 和采样设置。
