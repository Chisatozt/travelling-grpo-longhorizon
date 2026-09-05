"""Build the TravelGym-only train and validation parquet files.

The former version merged several unrelated environments.  This project trains
only the TravelGym public ``search -> action -> answer`` contract, so the
source list below is deliberately explicit and no cross-environment sampling
remains.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import yaml
from datasets import Dataset

from .._paths import REPOSITORY_ROOT
from .task_pools import TaskPoolError, load_pool_manifest, pool_task_keys


PROJECT_ROOT = REPOSITORY_ROOT
TRAVEL_DATASETS = (
    "travel22_multiturn_onechoice",
    "travel33_multiturn_onechoice",
    "travel44_multiturn_onechoice",
    "travel233_multiturn_onechoice",
    "travel333_multiturn_onechoice",
    "travel334_multiturn_onechoice",
    "travel444_multiturn_onechoice",
    "travel2222_multiturn_onechoice",
)
TOOL_CONFIG_PATH = PROJECT_ROOT / "configs" / "tools" / "interact_tool_config.yaml"


def load_dataset_split(
    dataset_name: str,
    split: str = "train",
    *,
    allowed_task_keys: set[str] | None = None,
) -> list[dict]:
    """Load one TravelGym parquet split as Python records."""
    parquet_path = PROJECT_ROOT / "data" / "grpo" / dataset_name / f"{split}.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
    records = Dataset.from_parquet(str(parquet_path)).to_list()
    invalid = [
        index
        for index, record in enumerate(records)
        if record.get("data_source") != "interact_travelgym"
        or not isinstance(record.get("reward_model"), dict)
        or record["reward_model"].get("env_name") != "TravelGym"
    ]
    if invalid:
        preview = ", ".join(str(index) for index in invalid[:5])
        raise ValueError(
            f"{parquet_path} contains non-TravelGym records at row(s) {preview}; "
            "regenerate or sanitize the source split before merging"
        )
    if allowed_task_keys is None:
        return records
    env_name = dataset_name.removesuffix("_multiturn_onechoice")
    kept: list[dict] = []
    for record in records:
        reward_model = record.get("reward_model", {})
        task_id = str(reward_model.get("id", ""))
        key = f"{env_name}::{task_id}"
        if key in allowed_task_keys:
            kept.append(record)
    return kept


def merge_datasets(
    dataset_names: tuple[str, ...],
    output_dir: Path,
    tokenizer,
    tools: list[dict],
    split: str = "train",
    max_prompt_tokens: int = 1152,
    seed: int = 42,
    task_pool_manifest: dict | None = None,
    task_pool_name: str | None = None,
) -> Dataset:
    """Merge and length-filter only TravelGym records."""
    rng = np.random.default_rng(seed)
    samples: list[dict] = []
    discarded = 0

    for dataset_name in dataset_names:
        allowed = None
        if task_pool_manifest is not None:
            if not task_pool_name:
                raise TaskPoolError("task_pool_name is required with task_pool_manifest")
            allowed = pool_task_keys(task_pool_manifest, task_pool_name)
        dataset = load_dataset_split(dataset_name, split, allowed_task_keys=allowed)
        rng.shuffle(dataset)
        # Retain the historical sampling policy while applying it uniformly
        # to TravelGym variants only.
        # A task-pool constrained Validation build must retain all selected
        # final200 tasks; the legacy 50% test sampling would silently shrink
        # the promised Validation-Task-Pool.
        fraction = 0.35 if split == "train" else (
            1.0 if task_pool_manifest is not None and task_pool_name == "validation" else 0.5
        )
        dataset = dataset[: max(1, int(len(dataset) * fraction))]
        dataset_kept = 0
        for sample in dataset:
            messages = sample["prompt"]
            try:
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    tools=tools,
                    add_generation_prompt=True,
                    enable_thinking=True,
                )
            except TypeError:
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    tools=tools,
                    add_generation_prompt=True,
                )
            if len(prompt) >= max_prompt_tokens:
                discarded += 1
                continue
            samples.append(sample)
            dataset_kept += 1
        print(
            f"Dataset {dataset_name} - {split}: selected {len(dataset)}, "
            f"kept {dataset_kept}"
        )

    rng.shuffle(samples)
    merged_dataset = Dataset.from_list(samples)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{split}.parquet"
    merged_dataset.to_parquet(str(output_path))
    print(
        f"Saved TravelGym {split} dataset: {len(merged_dataset)} samples to "
        f"{output_path} (discarded overlength: {discarded})"
    )
    return merged_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokenizer",
        default=os.environ.get("TOKENIZER_PATH"),
        help="Tokenizer path/name used to apply the chat template (or TOKENIZER_PATH).",
    )
    parser.add_argument("--max_prompt_tokens", type=int, default=1152)
    parser.add_argument("--train_output", type=Path, default=PROJECT_ROOT / "data" / "grpo" / "alltrain_multiturn")
    parser.add_argument("--test_output", type=Path, default=PROJECT_ROOT / "data" / "grpo" / "alltest_multiturn")
    parser.add_argument(
        "--task-pool-manifest",
        type=Path,
        default=(Path(os.environ["TRAVEL_TASK_POOL_MANIFEST"]) if os.environ.get("TRAVEL_TASK_POOL_MANIFEST") else None),
        help="Disjoint task-pool manifest. Train uses grpo; test uses validation.",
    )
    parser.add_argument("--train-pool", default="grpo", choices=("sft", "grpo", "validation"))
    parser.add_argument("--test-pool", default="validation", choices=("sft", "grpo", "validation"))
    args = parser.parse_args()
    if not args.tokenizer:
        parser.error("--tokenizer or TOKENIZER_PATH is required")
    if args.max_prompt_tokens <= 0:
        parser.error("--max_prompt_tokens must be positive")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    tool_config = yaml.safe_load(TOOL_CONFIG_PATH.read_text(encoding="utf-8"))
    tools = [tool_config["tools"][0]["tool_schema"]]
    task_pool_manifest = None
    if args.task_pool_manifest:
        task_pool_manifest = load_pool_manifest(args.task_pool_manifest)
    merge_datasets(
        TRAVEL_DATASETS,
        args.train_output,
        tokenizer,
        tools,
        split="train",
        max_prompt_tokens=args.max_prompt_tokens,
        seed=42,
        task_pool_manifest=task_pool_manifest,
        task_pool_name=args.train_pool if task_pool_manifest is not None else None,
    )
    merge_datasets(
        TRAVEL_DATASETS,
        args.test_output,
        tokenizer,
        tools,
        split="test",
        max_prompt_tokens=args.max_prompt_tokens,
        seed=42,
        task_pool_manifest=task_pool_manifest,
        task_pool_name=args.test_pool if task_pool_manifest is not None else None,
    )


if __name__ == "__main__":
    main()
