from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_physical_layout_has_canonical_runtime_paths() -> None:
    required_paths = (
        "verl/__init__.py",
        "environments/TravelGym/setup.py",
        "configs/grpo/grpo_multiturn.yaml",
        "configs/tools/interact_tool_config.yaml",
        "data/grpo/travel22_multiturn_onechoice/train.parquet",
        "data/sft/travel_sft_qwen35_split/train.jsonl",
        "data/evaluation/test_manifests/final200.json",
        "data/task_pools/travel_task_pools.json",
        "scripts/train_grpo.sh",
        "scripts/evaluate_native.sh",
        "scripts/evaluate_http.sh",
        "scripts/serve_vllm.sh",
        "scripts/merge_checkpoint.sh",
        "docs/evaluation.md",
        "docs/sft_pipeline.md",
        "outputs/evaluation/travelgym_sft_step186_smoke20_pass3_manifest.json",
        "src/travel_grpo/collection/task_pools.py",
        "src/travel_grpo/evaluation/eval.py",
        "src/travel_grpo/evaluation/final200.py",
        "src/travel_grpo/training/sft/qwen35_mask.py",
        "src/travel_grpo/training/sft/sft_split.py",
        "src/travel_grpo/collection/merge_customize.py",
        "src/travel_grpo/collection/travel_multiturn_w_tool.py",
    )
    for relative_path in required_paths:
        assert (ROOT / relative_path).exists(), relative_path
    legacy_paths = (
        "eval",
        "sft",
        "examples/data_preprocess",
        "examples/sglang_multiturn",
        "gyms/TravelGym",
        "data/alltest_multiturn",
        "data/alltrain_multiturn",
        "data/travel22_multiturn_onechoice",
        "data/travel33_multiturn_onechoice",
        "data/travel44_multiturn_onechoice",
        "data/travel233_multiturn_onechoice",
        "data/travel333_multiturn_onechoice",
        "data/travel334_multiturn_onechoice",
        "data/travel444_multiturn_onechoice",
        "data/travel2222_multiturn_onechoice",
        "data/travel_multiturn_onechoice",
    )
    for relative_path in legacy_paths:
        path = ROOT / relative_path
        assert not path.exists(), relative_path
        assert not path.is_symlink(), relative_path


def test_organization_views_point_to_single_sources_of_truth() -> None:
    expected_text = {
        "configs/grpo/README.md": "configs/grpo/grpo_multiturn.yaml",
        "data/grpo/README.md": "data/grpo/travel",
        "data/evaluation/README.md": "data/evaluation/test_manifests",
        "scripts/README.md": "scripts/train_grpo.sh",
        "docs/repository_layout.md": "one physical source",
        "src/travel_grpo/README.md": "canonical home",
    }
    for relative_path, marker in expected_text.items():
        assert marker in (ROOT / relative_path).read_text(encoding="utf-8")


def test_canonical_entrypoints_are_self_contained() -> None:
    for relative_path in (
        "scripts/train_grpo.sh",
        "scripts/run_grpo_stage.sh",
        "scripts/evaluate_native.sh",
        "scripts/evaluate_http.sh",
        "scripts/serve_vllm.sh",
        "scripts/merge_checkpoint.sh",
    ):
        path = ROOT / relative_path
        assert path.is_file(), relative_path
        assert not path.is_symlink(), relative_path
