# SFT training

The veRL trainer remains `verl/trainer/fsdp_sft_trainer.py`; the project-owned
SFT helpers now live in this package:

- `qwen35_mask.py`: exact Qwen3.5 assistant/reasoning/tool masks;
- `sft_split.py`: task-level train/validation splitting and token audits;
- `validate_travel_canonical.py`: strict canonical-corpus checks.

Data construction and task-pool preparation are in
`travel_grpo.collection`.
