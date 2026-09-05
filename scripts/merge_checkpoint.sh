#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$REPOSITORY_ROOT/src:$REPOSITORY_ROOT/environments/TravelGym:$REPOSITORY_ROOT${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-path/to/your/model/global_step_XXX}"

exec "$PYTHON_BIN" -m travel_grpo.evaluation.merge merge \
    --backend fsdp \
    --local_dir "${MODEL_PATH}/actor" \
    --target_dir "${MODEL_PATH}_hf" "$@"
