# UserRL：TravelGym 长程 Agent 后训练

面向长程、用户中心 Agent 的可复现后训练与评测项目。

Teacher 轨迹与 LoRA SFT → 在线 GRPO → 固定留出集评测

[论文](https://arxiv.org/pdf/2509.19736) · [原始 UserRL 数据](https://github.com/SalesforceAIResearch/UserRL/tree/main/data) · [项目文档](docs/README.md)

## TravelGym 是什么？

TravelGym 是一个多轮旅行规划环境，用来训练和评估需要长期交互的 Agent。一个任务可能同时要求规划航班、酒店、公寓、租车或餐厅；模型需要在环境中逐步搜索信息、询问用户偏好、比较候选，并为每个 aspect 提交选择。

因此，TravelGym 关注的不是一次性生成最终答案，而是模型能否在有限步数内完成一条可执行的交互轨迹：

- 理解多 aspect 的用户需求；
- 正确使用 Search、Action、Answer、Finish 等工具；
- 在长对话中保存证据并完成约束满足；
- 处理错误、重复动作和用户澄清；
- 在信息足够时完成任务并结束 episode。

环境、训练和评测使用同一套公开交互协议。正确答案、用户偏好和终局 Reward 等隐藏信息只保留在环境或评测侧，不直接提供给 Actor。

## 项目做了什么？

仓库围绕一条连续的后训练流水线组织：先用 Teacher 收集真实的多轮环境轨迹，再通过 SFT 建立工具使用和交互协议的基础能力，最后用 GRPO 让模型根据环境终局反馈继续优化长程决策。

~~~mermaid
flowchart LR
    A[TravelGym 任务池] --> B[Teacher 轨迹]
    B --> C[协议回放与数据清洗]
    C --> D[LoRA SFT]
    D --> E[合并后的 SFT 模型]
    E --> F[在线 GRPO]
    F --> G[SGLang rollout]
    G --> H[TravelGym 环境]
    H --> I[终局 Reward]
    I --> F
    E --> J[固定留出集评测]
    F --> J
    J --> K[可审计结果]
~~~

| 阶段 | 目标 | 主要产物 | 入口 |
| --- | --- | --- | --- |
| 任务池与 Teacher | 固定任务身份，收集真实多轮轨迹 | task pool、Teacher cache | `travel_grpo.collection`、`travel_grpo.evaluation.eval` |
| SFT | 学会合法工具行为和多轮交互格式 | canonical JSONL、LoRA adapter | `verl/trainer/fsdp_sft_trainer.py` |
| GRPO | 利用环境终局反馈优化长程决策 | rollout、Reward、checkpoint | `scripts/train_grpo.sh` |
| Evaluation | 用相同任务和协议比较不同模型 | Smoke20、Final-200、报告 | `scripts/evaluate_native.sh` |

## SFT 数据是怎么收集的？

SFT 数据不是把用户问题和最终答案简单拼接起来，而是记录 Agent 在 TravelGym 中完成任务的完整过程。数据流程如下：

1. 从固定 task pool 中选择任务，并用 `env_name::task_id` 保持任务身份稳定；
2. 用 DeepSeek Teacher 实际驱动 TravelGym，生成包含 thinking、工具调用、公开 observation 和工具反馈的多轮轨迹；
3. 将历史公开轨迹与 Teacher cache 合并，做消息对齐、工具调用、任务身份和长度检查；
4. 按 `strict_gold`、`recoverable_correct`、`partial_correct` 等类别审计轨迹，排除完全错误或基础设施无效的样本；
5. 使用 Qwen3.5-4B 的 chat template 做 token 审计，并按 task group 生成固定的训练/验证 split。

当前 canonical 合并语料有 795 条记录：238 条历史记录和 557 条 DeepSeek Teacher 记录。当前正式 SFT split 是 490 条训练轨迹和 10 条 `strict_gold` 验证轨迹；另有无法恢复任务身份的历史行被隔离，不会进入活动数据集。

SFT 的监督重点是 Assistant 的行为：环境 observation 和工具反馈作为上下文，只有符合训练策略的 Assistant 文本/动作 token 参与 loss。这样模型学习的是“在当前环境状态下如何行动”，而不是背诵隐藏答案。

Teacher 采集通过真实环境回放完成，支持 smoke 验证、断点续跑和 provenance 记录。一个低成本的采集示例是：

~~~bash
python -m travel_grpo.evaluation.eval \
  --model_name deepseek-v4-flash \
  --sft-collection --thinking enabled --include-think --require-think \
  --task-pool-manifest data/task_pools/travel_task_pools.json \
  --task-pool sft_smoke --pass_k 2 --max_turns 25 \
  --collection-cache outputs/evaluation/deepseek_teacher_sft_teacher_cache.json \
  --collection-run-id deepseek_teacher_sft \
  --save_name deepseek_teacher_sft
~~~

完整的数据合并、清洗和隐私边界见[任务池与 Teacher 数据文档](docs/data_collection.md)。

## GRPO 是怎么训练的？

GRPO 从合并后的 SFT 模型开始。对同一个 prompt，当前策略会生成多个候选交互轨迹；当前实验默认每个 prompt 生成 4 条轨迹。每条轨迹都要真实执行工具调用、接收 TravelGym observation，并在 episode 结束时得到一个终局 Reward。

一次训练更新可以概括为：

1. 训练器从 GRPO 任务池取出一批 prompt；
2. 当前策略为每个 prompt 生成多条候选轨迹，AgentLoop 将工具调用交给 TravelGym；Action 阶段的用户回复由 DeepSeek User Simulator 提供；
3. TravelGym 根据轨迹中的搜索、用户询问、选择和回答，计算 `travelgym-terminal-v2` 终局 Reward；
4. 对同一 prompt 的候选轨迹进行组内比较，得到相对优势，并用它更新 Actor；
5. 保存训练状态和 checkpoint，继续处理下一批任务。

当前配置使用 SGLang 作为 rollout 推理后端，但训练入口、环境交互、Reward 计算和参数更新由同一套 GRPO 配置统一串联；运行 GRPO 不需要额外启动一个 vLLM HTTP 服务。`scripts/serve_vllm.sh` 仅用于手动服务或 HTTP 评测。

当前正式 recipe 的关键设置是：单节点单卡、global train batch 4、每个 prompt 4 个 rollout、prompt 最大长度 1280、模型最大长度 32768、LoRA rank/alpha 为 32/64、最多 25 个交互回合、总训练步数 100。动态采样和 hard-case 审计属于训练侧机制，具体开关见 [GRPO 文档](docs/grpo.md)。

### Turn Credit 设计

TravelGym 的主 Reward 是 episode 结束时才计算的轨迹级信号。Turn Credit 不改变这个 Reward，而是在组内优势已经计算完成后，把一部分优势重新路由到更可能造成结果的交互 turn，使搜索、询问、选择和回答得到更有区分度的训练信号。

当前默认方法是 `component_attribution`，主要按以下因果关系分配：

- `correct_completion`：搜索 15%、有用 Action 25%、最终 Answer 60%；
- `coverage_adjusted_answer_quality`：搜索 10%、有用 Action 35%、最终 Answer 55%；
- `coverage_adjusted_legal_chain_rate`：搜索 25%、有用 Action 35%、最终 Answer 40%；
- 偏好命中主要归给产生有效偏好信息的 Action；非法调用、错误回答、重复/无收益 Action 则归给对应的问题 turn；未完成、零回答或超步数惩罚会分配给浪费预算的 turn 和最后一个 turn。

行为分量路由得到的优势与原始终局优势按 `mix_ratio=0.30` 混合：

~~~text
final_advantage = 0.70 * terminal_advantage + 0.30 * behavior_turn_credit
~~~

每个 turn 的 credit 会再按该 turn 的可训练 token 数均分，避免较长的 thinking 仅因 token 更多而获得更多 credit；同时对每条响应执行 float64 轨迹和守恒检查，防止路由过程改变原始优势总量。Turn Credit 只使用训练侧的 Reward metadata，不会把隐藏答案或偏好写入 Actor 的 observation。

默认配置是 `algorithm.turn_credit.stage=train`；设置为 `shadow` 时只记录诊断、不影响优化，设置为 `off` 可关闭。相关配置位于 [`configs/grpo/grpo_multiturn.yaml`](configs/grpo/grpo_multiturn.yaml)，实现细节见 [GRPO 文档](docs/grpo.md)。

当前 100-step 实验由两段运行记录组成：主运行覆盖 0–99 step，随后用 80–100 recovery 补齐并产生 `global_step_100` checkpoint。4-task overfit 是诊断实验，用来检查 Reward、工具协议和训练入口，不应当当作正式 benchmark 结果。

~~~bash
ACTOR_MODEL_PATH=checkpoints/TravelGym/qwen35_4b_canonical_sft/merged_step_186 \
GRPO_AUTO_SHUTDOWN=0 \
  bash scripts/train_grpo.sh
~~~

训练前建议先运行 [GRPO 运行时预检](docs/grpo.md)。它会检查模型权重、tokenizer、rollout 依赖、工具 schema 和 reset 后 prompt 长度。

## 评估流水线是怎么设计的？

评估使用固定任务清单和统一的 TravelGym native protocol，避免不同模型因为任务抽样不同而产生不可比结果。当前主要有两种规模：

- `Smoke20`：固定 20 个任务，用于快速检查模型、环境和 User Simulator；
- `Final-200`：从 471 个 test task 中固定选择 200 个任务，用于最终比较。

native evaluation 使用 SGLang 直接在评测进程中运行模型；HTTP 评测则用于已经启动的服务。结果同时记录 `pass@k`、正确 aspect 完成率、答案覆盖率、terminal Reward 和工具行为指标。当前 README 的最终结果采用 `pass_k=1`、temperature 0 的 greedy 评测，因此表中的 `pass@1` 不是 sampled `pass@3`。

评测任务清单位于 [`data/evaluation/test_manifests/`](data/evaluation/test_manifests/)，详细流程见[评估文档](docs/evaluation.md)和[Final-200 说明](docs/evaluation_dataset.md)。

## 实验结果

下表汇总当前本地运行产物中保留的 Final-200 native evaluation 结果。三组评测使用相同的 200 个任务、`pass_k=1` 和 temperature 0；表中的 aspect 指标来自 `correct_completion/mean@1`。

| 模型/阶段 | 评测任务 | pass@1 | terminal Reward 均值 | 正确 aspect 完成率 |
| --- | --- | ---: | ---: | ---: |
| Qwen3.5-4B base（step 0） | Final-200 | 4.5%（9/200） | 0.1547 | 23.4% |
| SFT-step186 | Final-200 | 11.5%（23/200） | 0.2946 | 41.4% |
| GRPO-step100 | Final-200 | 30.5%（61/200） | 0.3987 | 52.3% |

“正确 aspect 完成率”是每个任务中正确完成的 aspect 数除以该任务 aspect 总数，再在 200 个任务上取平均；它不同于要求所有 aspect 都完成的 `pass@1`。这组结果显示，SFT 先提升了基础的 aspect 级正确性，GRPO 进一步提高了 aspect 级完成率、完整任务成功率和平均终局 Reward。原始评测摘要保存在被忽略的本地 `outputs/` 运行目录中，表格是 README 中的固定汇总。

## 训练硬件与耗时

当前保留的训练配置和运行日志表明，实验使用单节点、单张 NVIDIA GPU：

| 阶段 | 主要配置 | 运行记录中的耗时/显存信息 |
| --- | --- | --- |
| SFT | Qwen3.5-4B、BF16、LoRA rank/alpha 16/32、3 epochs | 最终 checkpoint 为 `global_step_186`；当前仓库未保留可核验的完整 SFT 总耗时 |
| 4-task overfit | 单卡、20 steps | 日志累计约 1 小时 29 分；用于诊断 |
| GRPO 主运行 | batch 4、每 prompt 4 rollout、LoRA 32/64、SGLang `gpu_memory_utilization=0.4` | 0–99 step 的日志在 step 99 显示累计约 15 小时 45 分；日志曾记录约 47.7 GiB allocated、89.5 GiB reserved |
| GRPO recovery | 从 80 step 恢复到 100 step | 日志进度约 4 小时 50 分，并生成 `global_step_100` |

当前日志没有可靠记录具体 GPU 型号，因此这里不对显卡型号作推断。实际复现实验时，还需要为基础模型、SFT/GRPO checkpoint、SGLang cache 和 outputs 预留足够磁盘空间。

## 环境要求

- Linux、Python 环境，以及可用的 NVIDIA GPU/CUDA；仅运行部分协议测试时可以不启动 GPU；
- 可导入的 PyTorch、veRL、SGLang、FlashAttention 和 TravelGym editable package；
- Teacher 采集、GRPO 的 User Simulator 和部分 native evaluation 需要 OpenAI-compatible 的 DeepSeek API；
- Qwen3.5-4B 基础模型和合并后的 SFT 权重必须可从本地路径加载。

## 快速开始

以下命令从仓库根目录执行。完整安装和故障排查见[运行文档](docs/operations.md)。

### 1. 安装

~~~bash
python -m pip install -e ".[sglang]"
python -m pip install flash-attn --no-build-isolation
python -m pip install -e environments/TravelGym
~~~

### 2. 配置 User Simulator

~~~bash
cp .env.example .env
~~~

至少配置 `OPENAI_API_KEY` 或 `DEEPSEEK_API_KEY`、`OPENAI_BASE_URL` 和 `USER_MODEL_NAME`。密钥不要提交到仓库。

### 3. 预检、训练和合并

~~~bash
MODEL_PATH=/path/to/complete-qwen35-model
python scripts/check_grpo_runtime.py "$MODEL_PATH"
python scripts/check_grpo_prompt_budget.py "$MODEL_PATH" --pools grpo validation --max-prompt-length 1280

OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=1 \
  -m verl.trainer.fsdp_sft_trainer \
  --config-name=travel_qwen35_sft \
  trainer.n_gpus_per_node=1 \
  model.partial_pretrain=Qwen/Qwen3.5-4B

python scripts/merge_sft_lora.py \
  --adapter checkpoints/TravelGym/qwen35_4b_canonical_sft/global_step_186 \
  --base-model Qwen/Qwen3.5-4B \
  --output checkpoints/TravelGym/qwen35_4b_canonical_sft/merged_step_186
~~~

然后运行 [GRPO 训练](docs/grpo.md) 或固定评测：

~~~bash
VALIDATION_MODEL_PATH=/path/to/complete-model \
VALIDATION_EXPERIMENT=my_final200 \
  bash scripts/evaluate_native.sh final200
~~~

## Reward 简介

当前项目使用确定性的 `travelgym-terminal-v2`，Reward 只在 episode 结束时计算，交互 step 的即时 Reward 为 0。它不依赖另一个 LLM 对最终答案打分；DeepSeek 只在 Action 阶段模拟用户回复。

Reward 的正向部分主要衡量：

- 完整正确完成所有 aspect：权重 3.00；
- 覆盖率调整后的答案质量：权重 0.30；
- 覆盖率调整后的合法工具链：权重 0.20；
- 隐藏用户偏好命中率：权重 0.15；
- 步数效率：权重 0.05。

非法调用、错误答案、冗余 Action、未覆盖 aspect、零回答和达到最大步数会受到惩罚。总分归一化后裁剪到 `[-1, 1]`。隐藏答案和偏好只用于环境侧评分，不会进入 Actor 的 observation、工具反馈或下一轮 prompt。

完整公式、终止条件和私有信息边界见 [Reward 文档](docs/reward.md)。

## 仓库结构

~~~text
configs/                        Hydra、工具 schema 和项目配置
data/                           canonical SFT、GRPO Parquet、评测清单、任务池
docs/                           项目总览与分阶段深入文档
environments/TravelGym/         唯一 TravelGym 环境实现和内置任务数据
scripts/                        面向用户的训练、评测、合并和预检入口
src/travel_grpo/                项目自有的数据、协议和评测适配层
verl/                            veRL trainer、worker、tool 和 SGLang 集成
tests/                           协议、数据、训练入口和布局检查
~~~

`checkpoints/`、`outputs/`、本地日志和 `.env` 属于运行时内容，不是源码或正式数据集。

## 常用配置

| 变量/配置 | 用途 |
| --- | --- |
| `ACTOR_MODEL_PATH` | GRPO 使用的完整 Actor/SFT 模型路径 |
| `VALIDATION_MODEL_PATH` | native evaluation 使用的完整模型路径 |
| `USER_MODEL_NAME` | User Simulator 使用的 OpenAI-compatible 模型名 |
| `CUDA_VISIBLE_DEVICES` | 限制可见 GPU |
| `GRPO_AUTO_SHUTDOWN` | 控制 GRPO 结束后的服务清理行为 |
| `configs/grpo/grpo_multiturn.yaml` | GRPO 的主配置 |
| `data/task_pools/travel_task_pools.json` | 任务身份、split 和选择 seed 的 manifest |

## 文档导航

先读[项目文档总览](docs/README.md)，再按需求进入专项文档：

- 环境和工具： [公开交互协议](docs/travel_public_protocol.md)、[TravelGym 环境](environments/TravelGym/README.md)；
- 数据和 SFT： [任务池与 Teacher 数据](docs/data_collection.md)、[SFT 操作指南](docs/sft.md)；
- GRPO 训练： [GRPO 训练文档](docs/grpo.md)、[阶段预检](docs/grpo_overfit_preflight.md)；
- 模型比较： [评测流水线](docs/evaluation.md)、[Final-200 数据集](docs/evaluation_dataset.md)；
- 实验复盘： [训练与评测分析报告](docs/experiment_report.md)；
- 目录和开发： [训练链路](docs/travel_training_chain.md)、[仓库结构](docs/repository_layout.md)。
