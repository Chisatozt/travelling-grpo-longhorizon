# Evaluation

This package is the canonical implementation of TravelGym evaluation. It
keeps evaluator outputs under the canonical `outputs/evaluation/` runtime
directory so historical result caches remain separate from source code.

Key module entry points are:

- `travel_grpo.evaluation.eval`: HTTP/API evaluator and Teacher collection;
- `travel_grpo.evaluation.check_qwen35_runtime`: tokenizer/parser preflight;
- `travel_grpo.evaluation.build_test_manifests`: fixed smoke20/final200 sets;
- `travel_grpo.evaluation.final200`: offline final200 planning;
- `travel_grpo.evaluation.merge`: checkpoint merge helper;
- `travel_grpo.evaluation.analyze`: evaluation-cache summaries;
- `travel_grpo.evaluation.travel_contract`: shared public observation contract.

Use the module paths above for all evaluation and Teacher-collection commands.
