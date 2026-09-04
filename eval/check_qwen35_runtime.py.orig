"""Preflight Qwen3.5 template and parser compatibility without launching a model.

This command is intentionally conservative: it never downloads a checkpoint
or changes the pinned training stack.  Run it on the serving host before a
Qwen3.5 SFT/GRPO job.  A non-zero result means the installed runtime must be
upgraded or the configuration must be reviewed before training.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


def _result(name: str, ok: bool, detail: str) -> dict[str, Any]:
    return {"check": name, "ok": bool(ok), "detail": detail}


def check_transformers(tokenizer_path: str | None) -> list[dict[str, Any]]:
    try:
        transformers = importlib.import_module("transformers")
    except ImportError as exc:
        return [_result("transformers", False, f"not installed: {exc}")]
    checks = [_result("transformers", True, str(getattr(transformers, "__version__", "unknown")))]
    if not tokenizer_path:
        checks.append(_result("qwen_template", True, "not loaded (pass --tokenizer for a local-template check)"))
        return checks
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=True, local_files_only=True
        )
        template = getattr(tokenizer, "chat_template", None)
        if not template:
            return checks + [_result("qwen_template", False, "tokenizer has no chat_template")]
        signature = inspect.signature(tokenizer.apply_chat_template)
        has_thinking = "enable_thinking" in signature.parameters
        # Exercise the exact public tool schema and both completed/generation
        # renders.  Merely finding a ``chat_template`` is insufficient: a
        # tokenizer can silently ignore Qwen's thinking flag or render tool
        # calls with a different special-token stream.
        try:
            from sft.travel_canonical import canonical_tools_schema

            tools = canonical_tools_schema()
        except ImportError:
            tools = None
        messages = [
            {"role": "system", "content": "TravelGym"},
            {"role": "user", "content": "Find a flight."},
        ]
        completed = tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=True,
        )
        generation = tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        if hasattr(completed, "tolist"):
            completed = completed.tolist()
        if hasattr(generation, "tolist"):
            generation = generation.tolist()
        if completed and isinstance(completed[0], list):
            completed = completed[0]
        if generation and isinstance(generation[0], list):
            generation = generation[0]
        if not completed or not generation or generation[: len(completed)] != completed:
            raise ValueError("completed and generation Qwen template streams are not prefix-aligned")
        checks.append(
            _result(
                "qwen_template",
                bool(has_thinking or "enable_thinking" in signature.parameters),
                f"loaded locally; native render OK; enable_thinking argument={has_thinking}; template_chars={len(str(template))}",
            )
        )
    except Exception as exc:  # noqa: BLE001 - report preflight failure, do not hide it
        checks.append(_result("qwen_template", False, str(exc)))
    return checks


def check_vllm() -> list[dict[str, Any]]:
    executable = shutil.which("vllm")
    if not executable:
        return [_result("vllm", False, "vllm executable not found")]
    try:
        completed = subprocess.run(
            [executable, "serve", "--help"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        help_text = (completed.stdout or "") + (completed.stderr or "")
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [_result("vllm", False, str(exc))]
    required = ("--tool-call-parser", "qwen3_coder", "--reasoning-parser", "qwen3")
    missing = [value for value in required if value not in help_text]
    return [_result("vllm", not missing, "supports qwen3_coder/qwen3" if not missing else f"missing CLI entries: {missing}")]


def check_sglang() -> list[dict[str, Any]]:
    try:
        module = importlib.import_module("sglang.srt.function_call.function_call_parser")
    except ImportError:
        try:
            module = importlib.import_module("sglang.srt.function_call_parser")
        except ImportError as exc:
            return [_result("sglang", False, f"FunctionCallParser unavailable: {exc}")]
    parser = getattr(module, "FunctionCallParser", None)
    enum = getattr(parser, "ToolCallParserEnum", {}) if parser else {}
    names = set(enum.keys()) if hasattr(enum, "keys") else set()
    ok = "qwen3_coder" in names
    return [_result("sglang", ok, "qwen3_coder parser found" if ok else f"available parsers: {sorted(names)}")]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("vllm", "sglang", "both"), default="both")
    parser.add_argument("--tokenizer", default=None, help="Local Qwen3.5 tokenizer/checkpoint path (no download).")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    checks = check_transformers(args.tokenizer)
    if args.backend in {"vllm", "both"}:
        checks.extend(check_vllm())
    if args.backend in {"sglang", "both"}:
        checks.extend(check_sglang())
    if args.as_json:
        print(json.dumps(checks, ensure_ascii=False, indent=2))
    else:
        for item in checks:
            print(f"{'OK' if item['ok'] else 'FAIL'} {item['check']}: {item['detail']}")
    return 0 if all(item["ok"] for item in checks) else 2


if __name__ == "__main__":
    raise SystemExit(main())
