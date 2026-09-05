# TravelGym task pools

`travel_task_pools.json` is the checked-in `travelgym-task-pools-v2`
deterministic partition used by the training/evaluation code.  A task key is
`env_name::task_id`.

| Pool | Source | Use |
| --- | --- | --- |
| `sft` | 238 resolvable historical tasks + 362 deterministic train-task reservations | historical SFT and DeepSeek Teacher collection |
| `sft_smoke` | stratified 20-task subset of `sft` | paid DeepSeek Teacher smoke collection |
| `grpo` | train tasks not in `sft` | Actor rollout during GRPO |
| `validation` | fixed 200-task test selection | final validation |
| `validation_smoke` | fixed 20-task subset of `validation` | quick validation |

The checked-in formal manifest reserves 600 resolved train tasks for SFT.  The expansion
reservations are marked `role=teacher_expansion`; they are task reservations,
not trajectories, so a task may later receive multiple Teacher trajectories.

Regenerate the deterministic partition with:

```powershell
python -m travel_grpo.collection.task_pools `
  --sft-target-count 600 `
  --output .\data\task_pools\travel_task_pools.json
```

Six historical rows whose task IDs cannot be recovered unambiguously are
explicitly listed under `quarantined_sft` and permanently excluded from every
active pool, including canonical SFT merge and Teacher collection.  The checked-in
manifest is strict for active task identities; `sft_task_alignment_candidates.json`
is retained as audit evidence only.  No ID is inferred or reintroduced by the code.
