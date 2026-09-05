# 运行、监控与故障排查

## 1. 安装顺序

从仓库根目录执行：

~~~bash
python -m pip install -e ".[sglang]"
python -m pip install flash-attn --no-build-isolation
python -m pip install -e environments/TravelGym
~~~

根 setup.py 安装 veRL 和项目 Python 包；.[sglang] 提供 SGLang extra；TravelGym
是单独的 editable package。项目没有额外的 setup.sh 或环境启动服务，TravelGym
在进程内由 EnvironmentManager 创建。

检查 import：

~~~bash
python -c "import travelgym, travel_grpo, verl; print('imports: OK')"
python -m pytest -q
~~~

## 2. 环境文件

~~~bash
cp .env.example .env
~~~

.env 已被忽略。训练/native validation 使用 DeepSeek API User Simulator 时，需要
OPENAI_API_KEY 或 DEEPSEEK_API_KEY、对应的 OPENAI_BASE_URL 和 DeepSeek
USER_MODEL_NAME。离线单元测试可以把 TRAVELGYM_USER_SIMULATOR 设为 local，但它
不能代表正式 DeepSeek 用户模拟结果。

常用本地设置：

~~~dotenv
OPENAI_API_KEY=<key>
OPENAI_BASE_URL=https://api.deepseek.com
USER_MODEL_NAME=deepseek-v4-flash
CUDA_VISIBLE_DEVICES=0
TRAINER_LOGGER=swanlab
SWANLAB_MODE=local
~~~

## 3. 预检顺序

给定完整的本地模型目录 MODEL_PATH，推荐按以下顺序：

~~~bash
python -m travel_grpo.evaluation.check_qwen35_runtime \
  --backend sglang --tokenizer "$MODEL_PATH"

python scripts/check_grpo_runtime.py "$MODEL_PATH"

python scripts/check_grpo_prompt_budget.py "$MODEL_PATH" \
  --pools grpo validation --max-prompt-length 1280

ACTOR_MODEL_PATH="$MODEL_PATH" bash scripts/run_grpo_stage.sh check
~~~

最后一个阶段还会检查 overfit manifest、GPU 和模型完整性。native eval 和 GRPO 会拒绝
没有顶层模型权重的 adapter-only 路径。

## 4. 日志和 watcher

train_grpo.sh 默认设置 GRPO_AUTO_SHUTDOWN=1，并启动
scripts/grpo_shutdown_watcher.py。watcher 会等待被监控的训练进程，根据 checkpoint
和输出 marker 写入成功/失败报告，并可能调用机器关机命令。

手动运行或调试时必须显式关闭：

~~~bash
GRPO_AUTO_SHUTDOWN=0 bash scripts/train_grpo.sh
~~~

典型产物：

~~~text
outputs/<experiment>/
  training.log
  monitor/watcher.log
  monitor/shutdown_report.json
  monitor/shutdown_state.json
  rollouts/
  validation/
checkpoints/TravelGym/<experiment>/
  global_step_<n>/
  last_checkpoint.json
  best_checkpoint.json
~~~

outputs/ 和 checkpoints/ 都被 Git 忽略；日志、用户 API 事件和完整 rollout 不应上传
到公开仓库。

## 5. 评测输出

HTTP/native evaluator 会把结果写入 outputs/evaluation/ 或指定的 experiment 目录：

~~~text
<save_name>_manifest.json       协议、数据集和采样配置
<save_name>_results.json        按环境/pass_k 的聚合 scalar
<save_name>_reward_cache.json   完整 rollout、Reward report 和 telemetry
<save_name>_collection_tracking.json  Teacher 采集的脱敏 tracking（如适用）
~~~

reward_cache.json 可能含原始轨迹和用户 API 相关信息，只能作为本地审计输入。评测
入口拒绝在协议或 task manifest 改变后静默复用旧 cache；换 save_name 可创建全新 run。

## 6. 常见故障

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| No usable NVIDIA GPU | native/GRPO 需要 CUDA | 先确认 GPU、driver 和 nvidia-smi -L |
| SGLang/FlashAttention runtime import failed | extra 或 CUDA 扩展未安装 | 在当前 Python 环境重装兼容版本，再运行 runtime check |
| parser/tokenizer mismatch | SGLang parser 与 Qwen template 不匹配 | 不要绕过 preflight；确认 qwen3_coder 和 tokenizer 版本 |
| prompt 超过 1280 | reset 后初始控制上下文过长 | 运行 prompt budget checker，检查数据/工具 schema，而不是直接截断 |
| Merged model is missing | 使用了不存在或 adapter-only 路径 | 先运行 scripts/merge_sft_lora.py |
| output already exists | 入口防止覆盖和混跑 | 选择新的 VALIDATION_EXPERIMENT/experiment name |
| DeepSeek User Simulator error | key、base URL、模型名或限流问题 | 检查 .env，降低并发和 timeout；无网络时只做 local 单测 |
| 训练结束后机器关机 | watcher 默认 armed | 手动运行设置 GRPO_AUTO_SHUTDOWN=0 |
| 任务池 identity error | manifest 与 parquet/source 不一致 | 重新生成并检查 task pool，不要手工修改 task ID |

## 7. 代码变更后的最小验证

~~~bash
python -m pytest -q tests/test_repository_layout.py
python -m pytest -q tests/test_canonical_travel_pipeline.py \
  tests/test_travel_reward_v2.py \
  tests/test_grpo_preflight.py
~~~

如果修改了工具 schema、TravelGym observation、Reward 或 Qwen mask，应额外运行完整
python -m pytest -q，并重新执行对应的 runtime/prompt preflight。
