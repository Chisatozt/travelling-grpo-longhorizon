# TravelGym 评测与 Teacher 采集

`eval.py` 只评测 TravelGym，并复用同一公开协议：

```text
Search 完整候选 -> Action 自然语言证据 -> Actor 隐式比较 -> Answer 可见 ID
```

Action 不会缩小候选列表；环境的 task ID、偏好和终局 Reward 仅用于 evaluator
和 trainer 私有 metadata。

## 固定评测集

支持 `travel22`、`travel33`、`travel44`、`travel233`、`travel333`、`travel334`、
`travel444`、`travel2222`。固定测试清单为：

- `test_manifests/smoke20.json`：20 条冒烟任务；
- `test_manifests/final200.json`：200 条最终评测任务；

三类任务由 `data/task_pools/travel_task_pools.json` 统一管理：历史/Teacher
轨迹和确定性扩展任务属于 `sft`（默认 600 个 train Task），GRPO rollout 属于
`grpo`，上述 smoke/final200 属于 `validation`。清单按 `(env_name, task_id)` 做互斥校验；不要直接把聚合
parquet 当作独立任务池。当前历史 244 条中有 6 条无法唯一恢复 task ID，已被
列入 `quarantined_sft`，永久隔离/弃用并从所有活动池剔除。checked-in 清单
对活动池直接满足 `strict_task_identity=true`；`data/task_pools/sft_task_alignment_candidates.json`
仅作审计证据，不需要 reviewed map，Teacher/SFT/GRPO 可直接使用正式活动池。

评测器按清单中的 `(env_name, task_id)` 选择固定任务。示例：

```powershell
python .\eval\eval.py `
  --test-manifest .\eval\test_manifests\smoke20.json `
  --envs travel22 travel33 travel44 travel233 travel333 travel334 travel444 travel2222 `
  --max_turns 25 --pass_k 1 --save_name outputs/smoke20
```

## DeepSeek-V4-Flash Teacher

SFT 采集必须启用 Thinking；`--sft-collection` 会强制 DeepSeek、Thinking、
`<think>` 导出和非空 reasoning 检查：

```powershell
python .\eval\eval.py `
  --model_name deepseek-v4-flash `
  --sft-collection --thinking enabled --include-think --require-think `
  --task-pool-manifest .\data\task_pools\travel_task_pools.json `
  --task-pool sft `
  --max_turns 25 --save_name outputs/deepseek_teacher_sft
```

API 请求使用 `extra_body.thinking={"type":"enabled"}` 和
`reasoning_effort=high`；每个工作 transcript 的
`reasoning_content` 独立保存，不把 `<think>` 手工重复拼到 Qwen 输入。为了和
历史 244 条 ShareGPT 数据对齐，cache 的 SFT 导出仍提供兼容的
`<think>...</think><tool_call>...</tool_call>` 文本；随后由
`sft/merge_travel_sft.py` 转回 canonical。Thinking disabled 只能作为诊断运行，
不能当作 SFT Teacher 数据源。

## Qwen3.5 serving

启动 vLLM 前先运行无下载 preflight：

```powershell
python .\eval\check_qwen35_runtime.py --backend both --tokenizer Qwen/Qwen3.5-4B
```

服务必须使用和 SFT 完全相同的 tokenizer/template、工具 schema 和
`enable_thinking=true`。vLLM 示例（具体版本以 preflight 为准）：

```bash
vllm serve "$MERGED_MODEL_PATH" \
  --max-model-len 32768 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --served-model-name "$CUSTOMIZED_SERVED_MODEL_NAME"
```

`eval/host.sh` 提供同样参数。当前仓库不无验证地升级 vLLM/SGLang；若版本不
包含所需 parser，先升级并重新运行 preflight。SGLang 配置中的
`multi_turn.tool_call_parser` 也会在初始化时检查，避免静默退回 Hermes 等旧
parser。

## final200 编排

`eval/final200.py` 先校验 strict task-pool、Reward 版本和
`SELECTED_GRPO_CHECKPOINT`，再生成固定的 all200/seen20/unseen180 计划。步骤
200 后不会自动挑选 GRPO checkpoint，也不会自动启动评测。真实服务器可将
SGLang/TravelGym 执行器注入 `evaluate_final200(plan, runner=..., tracking_factory=...)`；
该接口会以同一 protocol 调用 base、SFT、手选 GRPO、DeepSeek 四个独立 run，
并以仅含标量的 `final200/comparison/*` run 汇总比较。CPU fixture 可使用同一
接口测试编排，而不会创建 API client、Ray 或 GPU 进程。

## 输出和隐私

每个 `save_name` 会产生：

- `*_results.json`：终局指标汇总；
- `*_reward_cache.json`：完整轨迹、tool feedback、终局报告和 telemetry（私有）；
- `*_manifest.json`：协议、Reward 版本和采样配置 hash。

交互 step reward 恒为 0，只有 episode 结束时计算
`travelgym-terminal-v1`。含 API/环境故障的 rollout 标记
`reward_valid_for_training=false`，由 SFT cleaner/GRPO trainer 隔离。
