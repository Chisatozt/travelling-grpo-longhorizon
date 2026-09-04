# GRPO overfit preflight

The two overfit diagnostics and production are deliberately separate runs. All
three must initialize a fresh rank-32 GRPO LoRA (alpha 64) from the same complete model:
Qwen3.5-4B with the canonical SFT `global_step_186` adapter merged into it.
Never feed an overfit checkpoint into another stage.

## Selected SFT-seen tasks

The generated manifest is `data/task_pools/travel_grpo_overfit_pools.json`.
It is reproducibly generated from the actual canonical SFT train split and its
audit, without new rollout sampling:

```bash
python scripts/prepare_grpo_overfit_pools.py --check
```

The one-task pool uses the selected `travel233` task. The four-task pool keeps
the three shortest two-tool compositions (`travel22`, `travel33`, `travel44`)
plus that `travel233` three-tool task. Every selected trajectory is an actual
partial-correct SFT-train sample. These diagnostic pools intentionally overlap
SFT, but remain disjoint from formal GRPO and Validation pools.

## Model preparation

Low-memory download (safe in no-GPU mode):

```bash
HF_HUB_DISABLE_XET=1 hf download Qwen/Qwen3.5-4B \
  --local-dir /root/autodl-tmp/models/Qwen3.5-4B \
  --max-workers 1
```

Merge when the process has more than 12 GiB of host memory available:

```bash
python scripts/merge_sft_lora.py \
  --base-model /root/autodl-tmp/models/Qwen3.5-4B \
  --device cuda
```

The output is
`checkpoints/TravelGym/qwen35_4b_canonical_sft/merged_step_186`. The stage
launcher refuses an adapter-only directory.

## Stages

Check readiness without starting training:

```bash
examples/sglang_multiturn/run_grpo_stage.sh check
```

With a GPU attached, run the one-task diagnostic:

```bash
examples/sglang_multiturn/run_grpo_stage.sh overfit-one
```

Review SwanLab and the rollout dumps, then start the independent four-task,
20-step run from merged SFT step-186:

```bash
CONFIRM_OVERFIT_FOUR=YES \
  examples/sglang_multiturn/run_grpo_stage.sh overfit-four
```

The four-task run saves and validates at steps 10 and 20. The one-task run
remains a 10-step diagnostic.

Production is guarded and remains paused by default. Only after both diagnostic
runs have been reviewed should it be explicitly unlocked:

```bash
CONFIRM_PRODUCTION_GRPO=YES \
  examples/sglang_multiturn/run_grpo_stage.sh production
```

The overfit runs use their training tasks for baseline/final validation on
purpose. Production restores the authoritative GRPO/Validation split while
retaining the validated four-task optimizer settings: global batch size 4,
PPO mini-batch size 2, LoRA rank/alpha `32/64`, and actor learning rate `1e-5`. OMP uses 8 threads and TurnCredit is enabled by default.

## Review gates

Do not treat PPO loss as an SFT-style monotonic curve. Review terminal reward,
completion/protocol metrics, finite nonzero gradient norm, PPO KL, clip
fractions, response length, and the dynamic-sampling counters. A constant group
near reward ceiling can mean successful saturation; a constant all-zero group
means no learnable update. Global steps may contain fewer optimizer updates when dynamic sampling
rejects a group, so always check `dynamic_sampling_skipped_update`.

## One-task unattended run and shutdown

The first GPU session must stop after the one-task diagnostic. The guarded
workflow merges step 186 when needed, runs the complete preflight, runs only
`overfit-one`, verifies the step-10 checkpoint and final validation artifacts,
records success or failure, and then powers off:

```bash
mkdir -p outputs
nohup setsid python scripts/grpo_one_shutdown_runner.py --arm \
  > outputs/grpo_one_shutdown_launcher.log 2>&1 < /dev/null &
```

The durable report is `outputs/grpo_one_shutdown/report.json`; detailed stage
output is `outputs/grpo_one_shutdown/workflow.log`. The workflow never launches
`overfit-four` or `production`. The 4-task stage additionally requires
`CONFIRM_OVERFIT_FOUR=YES`; production retains its independent
`CONFIRM_PRODUCTION_GRPO=YES` guard.
