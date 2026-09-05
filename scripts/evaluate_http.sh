#!/usr/bin/env bash

# TravelGym-only evaluation launcher.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$REPOSITORY_ROOT/src:$REPOSITORY_ROOT/environments/TravelGym:$REPOSITORY_ROOT${PYTHONPATH:+:$PYTHONPATH}"
if [[ -f "$REPOSITORY_ROOT/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$REPOSITORY_ROOT/.env"
    set +a
fi
export PROJECT_ROOT="$REPOSITORY_ROOT/outputs/evaluation"
mkdir -p "$PROJECT_ROOT"
cd "$REPOSITORY_ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"

ACTOR_MODEL_NAME="${ACTOR_MODEL_NAME:-gpt-4o}"
TRAVEL_ENVS=(travel22 travel33 travel44 travel233 travel333 travel334 travel444 travel2222)
TRAVEL_TEST_MANIFEST="${TRAVEL_TEST_MANIFEST:-$REPOSITORY_ROOT/data/evaluation/test_manifests/final200.json}"

"$PYTHON_BIN" -m travel_grpo.evaluation.eval \
    --model_name "$ACTOR_MODEL_NAME" \
    --max_turns 25 \
    --pass_k 1 \
    --temperature 0 \
    --envs "${TRAVEL_ENVS[@]}" \
    --test-manifest "$TRAVEL_TEST_MANIFEST" \
    --save_name "results_${ACTOR_MODEL_NAME}"

# For a self-hosted Actor, set CUSTOMIZED_SERVED_MODEL_NAME and
# CUSTOMIZED_SERVED_MODEL_PORT. User simulation still uses USER_MODEL_NAME.
if [[ -n "${CUSTOMIZED_SERVED_MODEL_NAME:-}" ]]; then
    "$PYTHON_BIN" -m travel_grpo.evaluation.eval \
        --model_name "$CUSTOMIZED_SERVED_MODEL_NAME" \
        --port "${CUSTOMIZED_SERVED_MODEL_PORT:-8500}" \
        --max_turns 25 \
        --pass_k 1 \
        --temperature 0 \
        --envs "${TRAVEL_ENVS[@]}" \
        --test-manifest "$TRAVEL_TEST_MANIFEST" \
        --save_name "results_${CUSTOMIZED_SERVED_MODEL_NAME}"
fi
