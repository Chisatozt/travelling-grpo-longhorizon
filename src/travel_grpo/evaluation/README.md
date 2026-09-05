# evaluation

evaluation 负责把同一套 TravelGym 协议用于 Teacher 采集、native/HTTP 评测和 Final-200 对比。

| 模块 | 作用 |
| --- | --- |
| eval.py | HTTP Actor 评测和 Teacher collection |
| check_qwen35_runtime.py | 模型/tokenizer/SGLang 预检 |
| build_test_manifests.py | 生成 smoke20/final200 |
| final200.py | 构造和校验 Final-200 plan |
| travel_contract.py | public observation 与隐私契约 |
| merge.py / analyze.py | checkpoint 合并和结果分析 |

使用 scripts/evaluate_native.sh 或 scripts/evaluate_http.sh；详细口径见 [评测文档](../../../docs/evaluation.md) 和 [评测数据集](../../../docs/evaluation_dataset.md)。
