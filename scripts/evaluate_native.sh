#!/usr/bin/env bash
set -euo pipefail

# Canonical native TravelGym evaluator.  It starts the same Ray/SGLang stack
# as GRPO, but trainer.val_only=true makes the process exit after validation.
# Every task gets up to pass_k independent attempts; successful tasks are
# removed before the next attempt.  Invalid-row retries are disabled here.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$REPOSITORY_ROOT/src:$REPOSITORY_ROOT/environments/TravelGym:$REPOSITORY_ROOT${PYTHONPATH:+:$PYTHONPATH}"
if [[ -z "${PYTHON_BIN:-}" ]]; then
    if [[ -x "$REPOSITORY_ROOT/.venv-grpo/bin/python" ]]; then
        PYTHON_BIN="$REPOSITORY_ROOT/.venv-grpo/bin/python"
    else
        PYTHON_BIN="$(command -v python3)"
    fi
fi

MODE="${1:-}"
if [[ $# -gt 0 ]]; then
    shift
fi
case "$MODE" in
    smoke20)
        VALIDATION_POOL="validation_smoke"
        INITIAL_SMOKE=true
        DEFAULT_EXPERIMENT="travelgym_native_sft186_smoke20_pass3"
        DEFAULT_BASELINE_OUTPUT="$REPOSITORY_ROOT/outputs/travelgym_sft_step186_validation_baseline.json"
        ;;
    final200)
        VALIDATION_POOL="validation"
        INITIAL_SMOKE=false
        DEFAULT_EXPERIMENT="travelgym_native_sft186_final200_pass3"
        DEFAULT_BASELINE_OUTPUT=""
        ;;
    *)
        echo "Usage: $0 {smoke20|final200} [Hydra overrides...]" >&2
        exit 2
        ;;
esac

MODEL_PATH="${VALIDATION_MODEL_PATH:-${ACTOR_MODEL_PATH:-$REPOSITORY_ROOT/checkpoints/TravelGym/qwen35_4b_canonical_sft/merged_step_186}}"
PASS_K="${VALIDATION_PASS_K:-3}"
# pass@k requires independent sampled attempts; callers can pin the exact
# sampling policy, but the default must not silently fall back to greedy.
VALIDATION_TEMPERATURE="${VALIDATION_TEMPERATURE:-1.0}"
VALIDATION_TOP_P="${VALIDATION_TOP_P:-1.0}"
VALIDATION_DO_SAMPLE="${VALIDATION_DO_SAMPLE:-true}"
VALIDATION_RESUME_PATH="${VALIDATION_RESUME_PATH:-}"
if [[ -n "$VALIDATION_RESUME_PATH" ]]; then
    if [[ ! -d "$VALIDATION_RESUME_PATH" || "$VALIDATION_RESUME_PATH" != *global_step_* ]]; then
        echo "Validation resume checkpoint is invalid: $VALIDATION_RESUME_PATH" >&2
        exit 3
    fi
    VALIDATION_RESUME_MODE="resume_path"
    VALIDATION_RESUME_FROM_OVERRIDE="$VALIDATION_RESUME_PATH"
    VALIDATION_LORA_RANK="${VALIDATION_LORA_RANK:-32}"
    VALIDATION_LORA_ALPHA="${VALIDATION_LORA_ALPHA:-64}"
else
    VALIDATION_RESUME_MODE="disable"
    VALIDATION_RESUME_FROM_OVERRIDE="null"
    VALIDATION_LORA_RANK="${VALIDATION_LORA_RANK:-0}"
    VALIDATION_LORA_ALPHA="${VALIDATION_LORA_ALPHA:-64}"
fi
case "$VALIDATION_DO_SAMPLE" in
    true|false)
        ;;
    *)
        echo "VALIDATION_DO_SAMPLE must be true or false, got: $VALIDATION_DO_SAMPLE" >&2
        exit 2
        ;;
esac
EXPERIMENT="${VALIDATION_EXPERIMENT:-$DEFAULT_EXPERIMENT}"
ARTIFACT_DIR="${VALIDATION_ARTIFACT_DIR:-$REPOSITORY_ROOT/outputs/$EXPERIMENT}"
CHECKPOINT_DIR="${VALIDATION_CHECKPOINT_DIR:-$REPOSITORY_ROOT/checkpoints/TravelGym/$EXPERIMENT}"
BASELINE_OUTPUT_PATH="${VALIDATION_BASELINE_OUTPUT_PATH:-$DEFAULT_BASELINE_OUTPUT}"
VALIDATION_EXPECTED_STEP="${VALIDATION_EXPECTED_STEP:-0}"

if [[ ! -d "$MODEL_PATH" ]]; then
    echo "Validation model is missing: $MODEL_PATH" >&2
    exit 3
fi
if ! find "$MODEL_PATH" -maxdepth 1 -type f \( -name 'model*.safetensors' -o -name 'pytorch_model*.bin' \) -print -quit | grep -q .; then
    echo "Validation model has no top-level model weights: $MODEL_PATH" >&2
    exit 3
fi
if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L >/dev/null 2>&1; then
    echo "No usable NVIDIA GPU is attached; refusing to start native validation." >&2
    exit 4
fi
if [[ ! "$PASS_K" =~ ^[1-9][0-9]*$ ]]; then
    echo "VALIDATION_PASS_K must be a positive integer, got: $PASS_K" >&2
    exit 2
fi
if [[ -e "$ARTIFACT_DIR" || -e "$CHECKPOINT_DIR" || ( -n "$BASELINE_OUTPUT_PATH" && -e "$BASELINE_OUTPUT_PATH" ) ]]; then
    echo "Refusing to overwrite an existing validation run:" >&2
    echo "  artifacts: $ARTIFACT_DIR" >&2
    echo "  checkpoint: $CHECKPOINT_DIR" >&2
    if [[ -n "$BASELINE_OUTPUT_PATH" ]]; then
        echo "  baseline: $BASELINE_OUTPUT_PATH" >&2
    fi
    exit 5
fi

# Validate the installed native tokenizer/parser before allocating the
# validation run.  This is intentionally the same preflight used by GRPO.
OMP_NUM_THREADS=8 "$PYTHON_BIN" -m travel_grpo.evaluation.check_qwen35_runtime \
    --backend sglang --tokenizer "$MODEL_PATH"
OMP_NUM_THREADS=8 "$PYTHON_BIN" "$REPOSITORY_ROOT/scripts/check_grpo_prompt_budget.py" \
    "$MODEL_PATH" --pools validation --max-prompt-length 1280

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TRAINER_LOGGER="${TRAINER_LOGGER:-swanlab}"
export MULTITURN_MODEL_NAME="${VALIDATION_USER_MODEL_NAME:-${MULTITURN_MODEL_NAME:-deepseek-chat}}"
export TRAVELGYM_USER_SIMULATOR=deepseek_api
export TRAVELGYM_USER_TIMEOUT="${TRAVELGYM_USER_TIMEOUT:-45}"
export TRAVELGYM_STEP_TIMEOUT="${TRAVELGYM_STEP_TIMEOUT:-300}"
export TRAVELGYM_USER_JUDGE_MAX_TOKENS="${TRAVELGYM_USER_JUDGE_MAX_TOKENS:-128}"
export TRAVELGYM_USER_RESPONSE_MAX_TOKENS="${TRAVELGYM_USER_RESPONSE_MAX_TOKENS:-2048}"
export TRAVELGYM_USER_REQUEST_CONCURRENCY="${TRAVELGYM_USER_REQUEST_CONCURRENCY:-8}"
export TRAVELGYM_USER_CACHE_PATH="$ARTIFACT_DIR/user_simulator/responses.sqlite3"
export TRAVELGYM_USER_API_LOG_PATH="$ARTIFACT_DIR/user_simulator/api.events.jsonl"

# User-supplied overrides are accepted first; the protocol-critical values
# below are last so a run cannot silently fall back to the generic HTTP path.
ACTOR_MODEL_PATH="$MODEL_PATH" \
PYTHON_BIN="$PYTHON_BIN" \
RUN_UNTIL_STEP=20 \
GRPO_AUTO_SHUTDOWN="${GRPO_AUTO_SHUTDOWN:-1}" \
GRPO_MONITOR_KIND=validation \
GRPO_EXPECTED_STEP="$VALIDATION_EXPECTED_STEP" \
GRPO_CHECKPOINT_ROOT="$CHECKPOINT_DIR" \
GRPO_ARTIFACT_ROOT="$ARTIFACT_DIR" \
GRPO_TRAINING_LOG="$ARTIFACT_DIR/validation.log" \
USE_TRAIN_FILES_FOR_VAL=false \
bash "$REPOSITORY_ROOT/scripts/train_grpo.sh" "$@" \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    trainer.experiment_profile=production \
    trainer.experiment_name="$EXPERIMENT" \
    trainer.default_local_dir="$CHECKPOINT_DIR" \
    trainer.resume_mode="$VALIDATION_RESUME_MODE" \
    trainer.resume_from_path="$VALIDATION_RESUME_FROM_OVERRIDE" \
    trainer.total_training_steps=100 \
    trainer.run_until_step=20 \
    trainer.total_epochs=15 \
    trainer.val_before_train=true \
    trainer.val_only=true \
    trainer.initial_validation_smoke="$INITIAL_SMOKE" \
    trainer.validation_pass_k="$PASS_K" \
    trainer.validation_task_level_early_stop=true \
    trainer.validation_retry_attempts=0 \
    trainer.initial_rollout_health_gate=true \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.log_val_generations=0 \
    trainer.rollout_data_dir="$ARTIFACT_DIR/rollouts" \
    trainer.validation_data_dir="$ARTIFACT_DIR/validation" \
    trainer.validation_baseline_output_path="$BASELINE_OUTPUT_PATH" \
    trainer.wait_for_selected_grpo_checkpoint=false \
    algorithm.dynamic_sampling.enable=false \
    algorithm.hard_case_pool.enable=false \
    actor_rollout_ref.model.lora_rank="$VALIDATION_LORA_RANK" \
    actor_rollout_ref.model.lora_alpha="$VALIDATION_LORA_ALPHA" \
    data.shuffle=false \
    data.validation_shuffle=false \
    data.task_pool_name=grpo \
    data.task_pool_train_name=grpo \
    data.task_pool_val_name="$VALIDATION_POOL" \
    data.task_pool_smoke_name=validation_smoke \
    actor_rollout_ref.rollout.temperature="$VALIDATION_TEMPERATURE" \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.top_p="$VALIDATION_TOP_P" \
    actor_rollout_ref.rollout.val_kwargs.temperature="$VALIDATION_TEMPERATURE" \
    actor_rollout_ref.rollout.val_kwargs.do_sample="$VALIDATION_DO_SAMPLE" \
    actor_rollout_ref.rollout.multi_turn.enable=true \
    actor_rollout_ref.rollout.multi_turn.enable_thinking=true \
    actor_rollout_ref.rollout.multi_turn.use_inference_chat_template=true \
    actor_rollout_ref.rollout.multi_turn.enable_tokenization_sanity_check=true \
    actor_rollout_ref.rollout.multi_turn.max_new_tokens_per_turn=4096 \
    actor_rollout_ref.rollout.multi_turn.max_reasoning_tokens_per_turn=2560 \
    actor_rollout_ref.rollout.multi_turn.max_tool_call_tokens_per_turn=512 \
    actor_rollout_ref.rollout.multi_turn.tool_call_parser=qwen3_coder \
    actor_rollout_ref.rollout.multi_turn.response_token_buffer=0
