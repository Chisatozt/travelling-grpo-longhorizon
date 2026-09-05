# scripts

scripts 是用户操作层的薄入口：负责把配置、数据和项目适配层接到 veRL、SGLang、TravelGym 上。训练器和环境实现不在这里重复维护。

| 入口 | 用途 |
| --- | --- |
| scripts/train_grpo.sh | GRPO 主入口 |
| scripts/run_grpo_stage.sh | check、overfit-one、overfit-four、production 闸门 |
| scripts/evaluate_native.sh | SGLang native smoke20/final200 |
| scripts/evaluate_http.sh | OpenAI-compatible HTTP Actor 评测 |
| scripts/serve_vllm.sh | 手工 vLLM 服务模板 |
| scripts/merge_sft_lora.py | 合并 SFT LoRA adapter |
| scripts/merge_checkpoint.sh | 合并 veRL Actor checkpoint |
| scripts/check_*.py | 运行时、prompt budget 和 User Simulator 检查 |

最短入口：

~~~bash
bash scripts/run_grpo_stage.sh check
bash scripts/evaluate_native.sh smoke20
~~~

脚本默认从仓库根目录解析路径。部分入口默认启用 shutdown watcher，交互式运行时设置 GRPO_AUTO_SHUTDOWN=0。旧的 setup.sh、start_environment.sh、baseline.sh、sft.sh、evaluate.sh 和 grpo.sh 不属于当前项目入口。
