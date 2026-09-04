#!/usr/bin/env python3
"""Validate the GRPO worker, Qwen3.5 mapping and FSDP wrap policy without weights."""

from __future__ import annotations

import argparse
from importlib.metadata import version

from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM

from verl.utils.fsdp_utils import get_fsdp_wrap_policy
from verl.workers.fsdp_workers import ActorRolloutRefWorker  # noqa: F401


def check_rollout_runtime() -> dict[str, str]:
    try:
        import flash_attn
        import flash_attn_2_cuda  # noqa: F401
        import sglang
        from verl.workers.rollout.sglang_rollout.sglang_rollout_customized import SGLangRollout  # noqa: F401
        return {
            "sglang": version("sglang"),
            "flash_attn": flash_attn.__version__,
        }
    except Exception as exc:
        raise SystemExit(f"SGLang/FlashAttention runtime import failed: {exc!r}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    args = parser.parse_args()

    runtime = check_rollout_runtime()
    config = AutoConfig.from_pretrained(args.model)
    model_class = AutoModelForCausalLM._model_mapping.get(type(config), None)
    if model_class is None:
        raise SystemExit(f"AutoModelForCausalLM does not support {type(config).__name__}")
    with init_empty_weights():
        model = model_class(config)
    policy = get_fsdp_wrap_policy(model, config=None, is_lora=True)
    if policy is None:
        raise SystemExit(f"no FSDP wrap policy for {model_class.__name__}")
    print(
        "GRPO runtime: OK "
        f"(config={type(config).__name__}, model={model_class.__name__}, "
        f"no_split={model._no_split_modules}, "
        f"sglang={runtime['sglang']}, flash_attn={runtime['flash_attn']})"
    )


if __name__ == "__main__":
    main()
