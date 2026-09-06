#!/bin/bash

# TravelGym terminal-reward GRPO training.
# Run this script from any working directory on the available GPU set.

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

# Load API/runtime configuration from the repository-root .env.  Values can be
# overridden by exporting them after this block or by passing Hydra overrides.
if [[ -f "$REPOSITORY_ROOT/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$REPOSITORY_ROOT/.env"
    set +a
fi
export OMP_NUM_THREADS=8

PROJECT_DIR="${PROJECT_DIR:-$REPOSITORY_ROOT}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-${DEEPSEEK_BASE_URL:-https://api.openai.com/v1}}"
export MULTITURN_MODEL_NAME="${MULTITURN_MODEL_NAME:-${USER_MODEL_NAME:-gpt-4o}}"
export TRAVELGYM_USER_SIMULATOR="${TRAVELGYM_USER_SIMULATOR:-deepseek_api}"
export TRAVELGYM_USER_TIMEOUT="${TRAVELGYM_USER_TIMEOUT:-45}"
export TRAVELGYM_STEP_TIMEOUT="${TRAVELGYM_STEP_TIMEOUT:-300}"
export MODEL_MAX_ATTEMPTS="${MODEL_MAX_ATTEMPTS:-3}"
export TRAVELGYM_USER_JUDGE_MAX_TOKENS="${TRAVELGYM_USER_JUDGE_MAX_TOKENS:-128}"
export TRAVELGYM_USER_RESPONSE_MAX_TOKENS="${TRAVELGYM_USER_RESPONSE_MAX_TOKENS:-2048}"
export TRAVELGYM_USER_REQUEST_CONCURRENCY="${TRAVELGYM_USER_REQUEST_CONCURRENCY:-8}"
TRAINER_LOGGER="${TRAINER_LOGGER:-swanlab}"
RUN_UNTIL_STEP="${RUN_UNTIL_STEP:-100}"
USE_TRAIN_FILES_FOR_VAL="${USE_TRAIN_FILES_FOR_VAL:-false}"
TURN_CREDIT_STAGE="${TURN_CREDIT_STAGE:-train}"
TURN_CREDIT_MIX_RATIO="${TURN_CREDIT_MIX_RATIO:-0.30}"
CONTEXT_CLEANUP_ENABLED="${CONTEXT_CLEANUP_ENABLED:-false}"
CONTEXT_CLEANUP_TARGET_TOKENS="${CONTEXT_CLEANUP_TARGET_TOKENS:-20000}"
CONTEXT_CLEANUP_TEMPLATE_MARGIN_TOKENS="${CONTEXT_CLEANUP_TEMPLATE_MARGIN_TOKENS:-32}"

case "$CONTEXT_CLEANUP_ENABLED" in
    true|false)
        ;;
    *)
        echo "CONTEXT_CLEANUP_ENABLED must be true or false" >&2
        exit 2
        ;;
esac
if [[ ! "$CONTEXT_CLEANUP_TARGET_TOKENS" =~ ^[1-9][0-9]*$ || ! "$CONTEXT_CLEANUP_TEMPLATE_MARGIN_TOKENS" =~ ^[0-9]+$ ]]; then
    echo "CONTEXT_CLEANUP_TARGET_TOKENS must be positive and CONTEXT_CLEANUP_TEMPLATE_MARGIN_TOKENS must be non-negative" >&2
    exit 2
fi

case "$TURN_CREDIT_STAGE" in
    off|shadow|train)
        ;;
    *)
        echo "TURN_CREDIT_STAGE must be one of off, shadow, train" >&2
        exit 2
        ;;
esac

case "$TRAINER_LOGGER" in
    swanlab|wandb|mlflow|tensorboard|clearml)
        ;;
    *)
        echo "TRAINER_LOGGER must be one of swanlab, wandb, mlflow, tensorboard, clearml" >&2
        exit 2
        ;;
esac
TRACKING_LOGGERS="['console','${TRAINER_LOGGER}']"

if [[ "$TRAVELGYM_USER_SIMULATOR" == "deepseek_api" ]]; then
    if [[ -z "${OPENAI_API_KEY:-${DEEPSEEK_API_KEY:-}}" ]]; then
        echo "DeepSeek User Simulator requires OPENAI_API_KEY or DEEPSEEK_API_KEY." >&2
        exit 2
    fi
    if [[ "${MULTITURN_MODEL_NAME,,}" != *deepseek* ]]; then
        echo "DeepSeek User Simulator requires a DeepSeek USER_MODEL_NAME." >&2
        exit 2
    fi
    if [[ -z "$OPENAI_BASE_URL" ]]; then
        echo "DeepSeek User Simulator requires OPENAI_BASE_URL or DEEPSEEK_BASE_URL." >&2
        exit 2
    fi
fi

MODEL_PATH="${ACTOR_MODEL_PATH:-${MODEL_PATH:-Qwen/Qwen3.5-4B}}"
if [[ -z "$MODEL_PATH" ]]; then
    echo "ACTOR_MODEL_PATH (or MODEL_PATH) must point to the Actor checkpoint." >&2
    exit 2
fi

IFS=',' read -ra GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-${#GPU_IDS[@]}}"
CONFIG_PATH="$PROJECT_DIR/configs/grpo"

# Read all eight authoritative variant splits.  The RLHFDataset applies the
# task-pool filter before tokenisation, so validation receives the complete
# final200 view instead of the legacy 50%-sampled aggregate parquet.
TRAIN_FILES="[$PROJECT_DIR/data/grpo/travel22_multiturn_onechoice/train.parquet,$PROJECT_DIR/data/grpo/travel33_multiturn_onechoice/train.parquet,$PROJECT_DIR/data/grpo/travel44_multiturn_onechoice/train.parquet,$PROJECT_DIR/data/grpo/travel233_multiturn_onechoice/train.parquet,$PROJECT_DIR/data/grpo/travel333_multiturn_onechoice/train.parquet,$PROJECT_DIR/data/grpo/travel334_multiturn_onechoice/train.parquet,$PROJECT_DIR/data/grpo/travel444_multiturn_onechoice/train.parquet,$PROJECT_DIR/data/grpo/travel2222_multiturn_onechoice/train.parquet]"
VAL_FILES="[$PROJECT_DIR/data/grpo/travel22_multiturn_onechoice/test.parquet,$PROJECT_DIR/data/grpo/travel33_multiturn_onechoice/test.parquet,$PROJECT_DIR/data/grpo/travel44_multiturn_onechoice/test.parquet,$PROJECT_DIR/data/grpo/travel233_multiturn_onechoice/test.parquet,$PROJECT_DIR/data/grpo/travel333_multiturn_onechoice/test.parquet,$PROJECT_DIR/data/grpo/travel334_multiturn_onechoice/test.parquet,$PROJECT_DIR/data/grpo/travel444_multiturn_onechoice/test.parquet,$PROJECT_DIR/data/grpo/travel2222_multiturn_onechoice/test.parquet]"

if [[ "$USE_TRAIN_FILES_FOR_VAL" == "true" ]]; then
    VAL_FILES="$TRAIN_FILES"
fi

set -x

ulimit -n 65535

GRPO_AUTO_SHUTDOWN="${GRPO_AUTO_SHUTDOWN:-1}"
GRPO_MONITOR_KIND="${GRPO_MONITOR_KIND:-training}"
GRPO_EXPECTED_STEP="${GRPO_EXPECTED_STEP:-$RUN_UNTIL_STEP}"
GRPO_MONITOR_EXPERIMENT="${GRPO_MONITOR_EXPERIMENT:-travelgym_qwen35_4b_terminal_bs4}"
GRPO_CHECKPOINT_ROOT="${GRPO_CHECKPOINT_ROOT:-$PROJECT_DIR/checkpoints/TravelGym/$GRPO_MONITOR_EXPERIMENT}"
GRPO_ARTIFACT_ROOT="${GRPO_ARTIFACT_ROOT:-$PROJECT_DIR/outputs/$GRPO_MONITOR_EXPERIMENT}"
GRPO_TRAINING_LOG="${GRPO_TRAINING_LOG:-$GRPO_ARTIFACT_ROOT/training.log}"
GRPO_MONITOR_DIR="${GRPO_MONITOR_DIR:-$GRPO_ARTIFACT_ROOT/monitor}"

if [[ "$GRPO_MONITOR_KIND" != "training" && "$GRPO_MONITOR_KIND" != "training_no_validation" && "$GRPO_MONITOR_KIND" != "validation" ]]; then
    echo "GRPO_MONITOR_KIND must be training, training_no_validation, or validation" >&2
    exit 2
fi

# Keep the complete prompt+response sequence within the fixed 32768-token
# Qwen3.5 context (1280 prompt + 31488 generated response).
GRPO_COMMAND=(
    "$PYTHON_BIN" -m verl.trainer.main_ppo
    --config-path="$CONFIG_PATH"
    --config-name='grpo_multiturn'
    algorithm.adv_estimator=grpo_multiturn
    algorithm.dynamic_sampling.enable=true
    algorithm.turn_credit.stage="$TURN_CREDIT_STAGE"
    algorithm.turn_credit.mix_ratio="$TURN_CREDIT_MIX_RATIO"
    algorithm.gamma=0.8
    data.train_batch_size=4
    data.max_prompt_length=1280
    data.max_response_length=31488
    data.filter_overlong_prompts=True
    data.truncation='error'
    data.return_raw_chat=True
    actor_rollout_ref.model.path="$MODEL_PATH"
    actor_rollout_ref.model.lora_rank=32
    actor_rollout_ref.model.lora_alpha=64
    actor_rollout_ref.actor.optim.lr=1e-5
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.actor.ppo_mini_batch_size=2
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.clip_ratio=0.2
    actor_rollout_ref.actor.clip_ratio_low=0.15
    actor_rollout_ref.actor.clip_ratio_high=0.25
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.kl_loss_coef=0.001
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    actor_rollout_ref.model.enable_activation_offload=True
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.name=sglang
    actor_rollout_ref.rollout.mode=sync
    actor_rollout_ref.rollout.gpu_memory_utilization=0.40
    actor_rollout_ref.rollout.max_model_len=32768
    actor_rollout_ref.rollout.n=4
    actor_rollout_ref.rollout.multi_turn.max_turns=25
    actor_rollout_ref.rollout.multi_turn.model_name="$MULTITURN_MODEL_NAME"
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$PROJECT_DIR/configs/tools/interact_tool_config.yaml"
    actor_rollout_ref.rollout.multi_turn.turn_level_method="component_attribution"
    actor_rollout_ref.rollout.multi_turn.trajectory_score_method="Terminal"
    actor_rollout_ref.rollout.multi_turn.context_cleanup.enabled="$CONTEXT_CLEANUP_ENABLED"
    actor_rollout_ref.rollout.multi_turn.context_cleanup.target_context_tokens="$CONTEXT_CLEANUP_TARGET_TOKENS"
    actor_rollout_ref.rollout.multi_turn.context_cleanup.template_margin_tokens="$CONTEXT_CLEANUP_TEMPLATE_MARGIN_TOKENS"
    actor_rollout_ref.hybrid_engine=True
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    algorithm.use_kl_in_reward=False
    trainer.critic_warmup=0
    trainer.experiment_profile='production'
    trainer.logger="$TRACKING_LOGGERS"
    trainer.project_name='TravelGym'
    trainer.experiment_name='travelgym_qwen35_4b_terminal_bs4'
    trainer.n_gpus_per_node="$N_GPUS_PER_NODE"
    trainer.nnodes=1
    trainer.save_freq=20
    trainer.test_freq=5
    trainer.val_before_train=False
    data.train_files="$TRAIN_FILES"
    data.val_files="$VAL_FILES"
    data.task_pool_manifest=$PROJECT_DIR/data/task_pools/travel_task_pools.json
    data.task_pool_name=grpo
    data.task_pool_train_name=grpo
    data.task_pool_val_name=validation
    data.task_pool_smoke_name=validation_smoke
    trainer.total_training_steps=100
    trainer.run_until_step="$RUN_UNTIL_STEP"
    trainer.milestones='[20,40,60,80,100]'
    trainer.total_epochs=15
    "$@"
)

if [[ "$GRPO_AUTO_SHUTDOWN" == "1" && "${GRPO_MONITOR_SUPERVISED:-0}" != "1" ]]; then
    mkdir -p "$GRPO_ARTIFACT_ROOT" "$GRPO_MONITOR_DIR" "$(dirname "$GRPO_TRAINING_LOG")"
    # The watcher owns the AutoDL shutdown decision. The training process is
    # kept as the watched PID so exit status and PID-reuse checks stay exact.
    "${GRPO_COMMAND[@]}" > >(tee -a "$GRPO_TRAINING_LOG") 2>&1 &
    target_pid=$!
    "$PYTHON_BIN" "$REPOSITORY_ROOT/scripts/grpo_shutdown_watcher.py" \
        --pid "$target_pid" \
        --task-kind "$GRPO_MONITOR_KIND" \
        --checkpoint-root "$GRPO_CHECKPOINT_ROOT" \
        --artifact-root "$GRPO_ARTIFACT_ROOT" \
        --training-log "$GRPO_TRAINING_LOG" \
        --expected-step "$GRPO_EXPECTED_STEP" \
        --log-file "$GRPO_MONITOR_DIR/watcher.log" \
        --report-file "$GRPO_MONITOR_DIR/shutdown_report.json" \
        --state-file "$GRPO_MONITOR_DIR/shutdown_state.json" \
        --lock-file "$GRPO_MONITOR_DIR/shutdown.lock" \
        --arm &
    watcher_pid=$!
    set +e
    wait "$target_pid"
    target_status=$?
    wait "$watcher_pid"
    watcher_status=$?
    set -e
    if [[ "$target_status" -eq 0 && "$watcher_status" -ne 0 ]]; then
        exit "$watcher_status"
    fi
    exit "$target_status"
fi

"${GRPO_COMMAND[@]}"
