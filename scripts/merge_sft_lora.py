#!/usr/bin/env python3
"""Merge the canonical SFT LoRA adapter into a complete Hugging Face model."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adapter",
        type=Path,
        default=Path("checkpoints/TravelGym/qwen35_4b_canonical_sft/global_step_186"),
    )
    parser.add_argument("--base-model", default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("checkpoints/TravelGym/qwen35_4b_canonical_sft/merged_step_186"),
    )
    parser.add_argument("--max-shard-size", default="4GB")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    adapter = args.adapter.resolve()
    output = args.output.resolve()
    adapter_config_path = adapter / "adapter_config.json"
    adapter_weights = adapter / "adapter_model.safetensors"
    if not adapter_config_path.is_file() or not adapter_weights.is_file():
        raise SystemExit(f"incomplete PEFT adapter: {adapter}")
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty merge output: {output}")

    adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    base_model = args.base_model or adapter_config.get("base_model_name_or_path")
    if not base_model:
        raise SystemExit("base model is missing from both CLI and adapter_config.json")

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output.mkdir(parents=True, exist_ok=True)
    print(f"Loading base model on {args.device}: {base_model}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map={"": args.device},
    )
    print(f"Loading SFT adapter: {adapter}", flush=True)
    model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
    print("Merging LoRA into base weights", flush=True)
    merged = model.merge_and_unload(progressbar=True, safe_merge=True)
    print(f"Saving complete model: {output}", flush=True)
    merged.save_pretrained(
        output,
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )
    tokenizer = AutoTokenizer.from_pretrained(str(adapter), trust_remote_code=True)
    tokenizer.save_pretrained(output)
    metadata = {
        "schema_version": "travelgym-sft-merged-model-v1",
        "base_model": str(base_model),
        "adapter": str(adapter),
        "output": str(output),
        "merged_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "fresh initialization for GRPO overfit diagnostics and production",
    }
    (output / "merge_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    weight_files = sorted(output.glob("model*.safetensors"))
    if not weight_files:
        raise SystemExit(f"merge produced no complete model weights in {output}")
    print(f"Merge complete: {len(weight_files)} weight shard(s)", flush=True)


if __name__ == "__main__":
    main()
