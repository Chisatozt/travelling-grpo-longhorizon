# Repository layout

The checkout uses one physical source of truth for every project-owned
implementation. Runtime artifacts and user-facing entry points are separated
from the veRL and TravelGym packages.

## Canonical tree

```text
configs/                    project-level GRPO, SFT, and tool configuration
data/
  grpo/                     veRL Parquet datasets
  sft/                      canonical SFT corpora and split artifacts
  evaluation/               smoke20/final200 and protocol manifests
  task_pools/               task identity and disjoint-pool manifests
docs/                       data, training, evaluation, and layout docs
environments/TravelGym/     canonical TravelGym package and task data
outputs/
  evaluation/               historical and HTTP evaluation caches
  <experiment>/             GRPO/SGLang rollouts and validation artifacts
scripts/                    canonical launchers and operational tools
src/travel_grpo/
  collection/               Teacher, cleaning, canonicalization, task pools
  evaluation/               HTTP/native evaluation and result analysis
  training/sft/              masks, splits, and canonical validation
verl/                       veRL runtime, trainers, tools, and SGLang rollout
tests/                      unit, entry-point, layout, and packaging checks
```

## Stable package boundaries

1. The `verl` and `travelgym` Python package names remain unchanged.
2. The GRPO trainer entry remains `verl.trainer.main_ppo`; Hydra still reads
   `verl/trainer/config/`.
3. Project-owned collection, evaluation, and SFT helpers are imported through
   `travel_grpo.collection`, `travel_grpo.evaluation`, and
   `travel_grpo.training.sft`.
4. Task-pool `source_path` values point only to `data/grpo/` and `data/sft/`.
5. Checkpoints and experiment outputs retain their existing roots; historical
   evaluation caches are now stored under `outputs/evaluation/`.

There are no top-level `eval/` or `sft/` implementation directories, no
`examples/data_preprocess/` or `examples/sglang_multiturn/` compatibility
wrappers, and no project-level path-mapping symlinks. Use the canonical module
commands and `scripts/` launchers shown in the component documentation.
