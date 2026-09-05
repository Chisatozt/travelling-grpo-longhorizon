# User-facing entry points

These are the canonical user-facing entry points:

- scripts/train_grpo.sh
- scripts/run_grpo_stage.sh
- scripts/evaluate_native.sh
- scripts/evaluate_http.sh

There are no compatibility shell wrappers in the old example or evaluation
directories. These scripts add the checkout's `src/` and TravelGym roots to
`PYTHONPATH` themselves.
