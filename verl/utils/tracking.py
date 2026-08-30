# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A unified tracking interface that supports logging data to different backend
"""

import dataclasses
import os
import re
import warnings
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Union
from collections.abc import Mapping


_REASONING_BLOCK_RE = re.compile(
    r"<(?:think|analysis|reasoning)>.*?</(?:think|analysis|reasoning)>",
    re.IGNORECASE | re.DOTALL,
)
_REASONING_FIELD_RE = re.compile(
    r"(?:reasoning_content|chain_of_thought|scratchpad)\s*[:=]\s*.*?(?=\n\s*[A-Za-z_][\w -]*\s*[:=]|$)",
    re.IGNORECASE | re.DOTALL,
)


def sanitize_tracking_payload(payload):
    """Drop credentials, hidden labels, private reward and raw CoT."""
    forbidden = {
        "api_key", "apikey", "authorization", "token", "preference_id", "preference_ids",
        "correct_id", "correct_ids", "best_id", "best_ids", "hidden_ids", "hidden_labels",
        "reward_ledger", "raw_reward", "reasoning_content", "cot", "messages", "transcript",
    }
    if isinstance(payload, Mapping):
        result = {}
        for key, value in payload.items():
            lowered = str(key).casefold()
            if lowered in forbidden or any(part in lowered for part in ("api_key", "preference_id", "correct_id", "best_id", "reward_ledger", "reasoning_content")):
                continue
            result[str(key)] = sanitize_tracking_payload(value)
        return result
    if isinstance(payload, (list, tuple)):
        return [sanitize_tracking_payload(item) for item in payload]
    if isinstance(payload, str):
        # Redact sensitive content even when a caller uses a generic key such
        # as ``output``/``text``.  Scalar tracking is allowed to carry public
        # aggregate metrics, never a raw reasoning block or simulator label.
        text = _REASONING_BLOCK_RE.sub("[REASONING_REDACTED]", payload)
        text = _REASONING_FIELD_RE.sub("[REASONING_REDACTED]", text)
        text = re.sub(
            r"^\s*(?:preference[_ ]?ids?|correct[_ ]?ids?|best[_ ]?ids?|"
            r"ground[_ ]?truth|reward[_ ]?report|diagnostics?)\s*[:=].*$",
            "[PRIVATE_REDACTED]",
            text,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        text = re.sub(r"(?<![A-Za-z0-9])P\d+(?![A-Za-z0-9])", "[PRIVATE_ID]", text, flags=re.IGNORECASE)
        if "sk-" in text or "api_key" in text.casefold():
            return "[REDACTED]"
        return text
    return payload


def sanitize_validation_text(value: Any) -> str:
    """Remove reasoning/scratchpad blocks before optional sample logging.

    Validation tables are useful for a small number of cleaned examples, but
    decoded Qwen/DeepSeek output can still contain a provider's hidden
    reasoning markers.  Keep the public action/answer text while replacing
    recognizable reasoning fields with a fixed marker.  Scalar metrics are
    logged separately and are unaffected.
    """

    text = str(value or "")
    text = _REASONING_BLOCK_RE.sub("[REASONING_REDACTED]", text)
    text = _REASONING_FIELD_RE.sub("[REASONING_REDACTED]", text)
    return text


def sanitize_validation_samples(samples):
    """Return a backend-safe copy of ``(input, output, score)`` samples."""

    cleaned = []
    for sample in samples or []:
        if isinstance(sample, (list, tuple)) and len(sample) >= 3:
            cleaned.append((sanitize_validation_text(sample[0]), sanitize_validation_text(sample[1]), sample[2]))
        else:
            cleaned.append((sanitize_validation_text(sample), "", 0.0))
    return cleaned


class Tracking:
    """A unified tracking interface for logging experiment data to multiple backends.

    This class provides a centralized way to log experiment metrics, parameters, and artifacts
    to various tracking backends including WandB, MLflow, SwanLab, TensorBoard, and console.

    Attributes:
        supported_backend: List of supported tracking backends.
        logger: Dictionary of initialized logger instances for each backend.
    """

    supported_backend = ["wandb", "mlflow", "swanlab", "vemlp_wandb", "tensorboard", "console", "clearml"]

    def __init__(self, project_name, experiment_name, default_backend: Union[str, List[str]] = "console", config=None):
        if isinstance(default_backend, str):
            default_backend = [default_backend]
        for backend in default_backend:
            if backend == "tracking":
                import warnings

                warnings.warn("`tracking` logger is deprecated. use `wandb` instead.", DeprecationWarning, stacklevel=2)
            else:
                assert backend in self.supported_backend, f"{backend} is not supported"

        self.logger = {}
        self._finished = False
        # Tracking backends receive the resolved experiment config at init
        # time, before ``log`` has a chance to sanitize metrics.  Sanitize a
        # detached copy here as well so API credentials, hidden IDs, private
        # reward ledgers, messages and raw reasoning cannot leak through
        # backend metadata/configuration.
        safe_config = sanitize_tracking_payload(config or {})

        if "tracking" in default_backend or "wandb" in default_backend:
            import wandb

            settings = None
            trainer_config = safe_config.get("trainer") if isinstance(safe_config, Mapping) else None
            if isinstance(trainer_config, Mapping) and trainer_config.get("wandb_proxy", None):
                settings = wandb.Settings(https_proxy=trainer_config["wandb_proxy"])
            wandb.init(project=project_name, name=experiment_name, config=safe_config, settings=settings)
            self.logger["wandb"] = wandb

        if "mlflow" in default_backend:
            import mlflow

            MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", None)
            if MLFLOW_TRACKING_URI:
                mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

            # Project_name is actually experiment_name in MLFlow
            # If experiment does not exist, will create a new experiment
            experiment = mlflow.set_experiment(project_name)
            mlflow.start_run(experiment_id=experiment.experiment_id, run_name=experiment_name)
            mlflow.log_params(_compute_mlflow_params_from_objects(safe_config))
            self.logger["mlflow"] = _MlflowLoggingAdapter()

        if "swanlab" in default_backend:
            import swanlab

            SWANLAB_API_KEY = os.environ.get("SWANLAB_API_KEY", None)
            SWANLAB_LOG_DIR = os.environ.get("SWANLAB_LOG_DIR", "swanlog")
            # SwanLab 0.9.x names the cloud mode ``online``.  Keep accepting
            # the historical ``cloud`` spelling so existing .env files do
            # not silently change behavior.
            SWANLAB_MODE = os.environ.get("SWANLAB_MODE", "online").strip().lower()
            if SWANLAB_MODE == "cloud":
                SWANLAB_MODE = "online"
            if SWANLAB_MODE not in {"online", "local", "offline", "disabled"}:
                raise ValueError(
                    "SWANLAB_MODE must be one of online, local, offline, disabled"
                )
            if SWANLAB_API_KEY:
                swanlab.login(SWANLAB_API_KEY)  # NOTE: previous login information will be overwritten

            if safe_config is None:
                safe_config = {}  # defensive; sanitize_tracking_payload always returns an object here
            # New SwanLab versions support run resumption/group/tags. Keep a
            # compatibility fallback for older cluster images.
            swan_kwargs = {
                "project": project_name,
                "name": experiment_name,
                "config": {"FRAMEWORK": "verl", **safe_config},
                "log_dir": SWANLAB_LOG_DIR,
                "mode": SWANLAB_MODE,
            }
            run_id = os.environ.get("SWANLAB_RUN_ID")
            resume = os.environ.get("SWANLAB_RESUME")
            group = os.environ.get("SWANLAB_GROUP")
            tags_raw = os.environ.get("SWANLAB_TAGS", "")
            tags = [tag.strip() for tag in tags_raw.split(",") if tag.strip()]
            if run_id:
                swan_kwargs["id"] = run_id
            if resume:
                swan_kwargs["resume"] = resume
            if group:
                swan_kwargs["group"] = group
            if tags:
                swan_kwargs["tags"] = tags
            try:
                swanlab.init(**swan_kwargs)
            except TypeError:
                for key in ("id", "resume", "group", "tags"):
                    swan_kwargs.pop(key, None)
                swanlab.init(**swan_kwargs)
            self.logger["swanlab"] = swanlab

        if "vemlp_wandb" in default_backend:
            import volcengine_ml_platform
            from volcengine_ml_platform import wandb as vemlp_wandb

            volcengine_ml_platform.init(
                ak=os.environ["VOLC_ACCESS_KEY_ID"],
                sk=os.environ["VOLC_SECRET_ACCESS_KEY"],
                region=os.environ["MLP_TRACKING_REGION"],
            )

            vemlp_wandb.init(
                project=project_name,
                name=experiment_name,
                config=safe_config,
                sync_tensorboard=True,
            )
            self.logger["vemlp_wandb"] = vemlp_wandb

        if "tensorboard" in default_backend:
            self.logger["tensorboard"] = _TensorboardAdapter()

        if "console" in default_backend:
            from verl.utils.logger.aggregate_logger import LocalLogger

            self.console_logger = LocalLogger(print_to_console=True)
            self.logger["console"] = self.console_logger

        if "clearml" in default_backend:
            self.logger["clearml"] = ClearMLLogger(project_name, experiment_name, config)

    def log(self, data, step, backend=None):
        data = sanitize_tracking_payload(data)
        for default_backend, logger_instance in self.logger.items():
            if backend is None or default_backend in backend:
                logger_instance.log(data=data, step=step)

    def finish(self):
        """Finish all active tracking runs once, tolerating interpreter shutdown."""
        if self._finished:
            return
        self._finished = True
        for backend, logger_instance in list(self.logger.items()):
            try:
                if backend == "wandb":
                    logger_instance.finish(exit_code=0)
                elif backend == "swanlab":
                    logger_instance.finish()
                elif backend == "vemlp_wandb":
                    logger_instance.finish(exit_code=0)
                elif backend == "tensorboard":
                    logger_instance.finish()
                elif backend == "mlflow" and hasattr(logger_instance, "finish"):
                    logger_instance.finish()
                elif backend == "clearml" and hasattr(logger_instance, "finish"):
                    logger_instance.finish()
            except Exception as exc:  # pragma: no cover - shutdown-dependent
                warnings.warn(f"failed to finish {backend} tracking run: {exc}", RuntimeWarning)

    def __del__(self):
        # During Python interpreter teardown imported modules can already be
        # cleared.  Never turn a completed training run into an exception.
        try:
            self.finish()
        except Exception:
            pass


class ClearMLLogger:
    def __init__(self, project_name: str, experiment_name: str, config):
        self.project_name = project_name
        self.experiment_name = experiment_name

        import clearml

        self._task: clearml.Task = clearml.Task.init(
            task_name=experiment_name,
            project_name=project_name,
            continue_last_task=True,
            output_uri=False,
        )

        self._task.connect_configuration(sanitize_tracking_payload(config or {}), name="Hyperparameters")

    def _get_logger(self):
        return self._task.get_logger()

    def log(self, data, step):
        import numpy as np
        import pandas as pd

        # logs = self._rewrite_logs(data)
        logger = self._get_logger()
        for k, v in data.items():
            title, series = k.split("/", 1)

            if isinstance(v, (int, float, np.floating, np.integer)):
                logger.report_scalar(
                    title=title,
                    series=series,
                    value=v,
                    iteration=step,
                )
            elif isinstance(v, pd.DataFrame):
                logger.report_table(
                    title=title,
                    series=series,
                    table_plot=v,
                    iteration=step,
                )
            else:
                logger.warning(f'Trainer is attempting to log a value of "{v}" of type {type(v)} for key "{k}". This invocation of ClearML logger\'s function is incorrect so this attribute was dropped. ')

    def finish(self):
        self._task.mark_completed()


class _TensorboardAdapter:
    def __init__(self):
        import os

        from torch.utils.tensorboard import SummaryWriter

        tensorboard_dir = os.environ.get("TENSORBOARD_DIR", "tensorboard_log")
        os.makedirs(tensorboard_dir, exist_ok=True)
        print(f"Saving tensorboard log to {tensorboard_dir}.")
        self.writer = SummaryWriter(tensorboard_dir)

    def log(self, data, step):
        for key in data:
            self.writer.add_scalar(key, data[key], step)

    def finish(self):
        self.writer.close()


class _MlflowLoggingAdapter:
    def log(self, data, step):
        import mlflow

        results = {k.replace("@", "_at_"): v for k, v in data.items()}
        mlflow.log_metrics(metrics=results, step=step)


def _compute_mlflow_params_from_objects(params) -> Dict[str, Any]:
    if params is None:
        return {}

    return _flatten_dict(_transform_params_to_json_serializable(params, convert_list_to_dict=True), sep="/")


def _transform_params_to_json_serializable(x, convert_list_to_dict: bool):
    _transform = partial(_transform_params_to_json_serializable, convert_list_to_dict=convert_list_to_dict)

    if dataclasses.is_dataclass(x):
        return _transform(dataclasses.asdict(x))
    if isinstance(x, dict):
        return {k: _transform(v) for k, v in x.items()}
    if isinstance(x, list):
        if convert_list_to_dict:
            return {"list_len": len(x)} | {f"{i}": _transform(v) for i, v in enumerate(x)}
        else:
            return [_transform(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, Enum):
        return x.value

    return x


def _flatten_dict(raw: Dict[str, Any], *, sep: str) -> Dict[str, Any]:
    import pandas as pd

    ans = pd.json_normalize(raw, sep=sep).to_dict(orient="records")[0]
    assert isinstance(ans, dict)
    return ans


@dataclasses.dataclass
class ValidationGenerationsLogger:
    def log(self, loggers, samples, step):
        # Sample logging is optional; sanitize once before any backend sees a
        # decoded prompt/output so raw CoT cannot be uploaded accidentally.
        samples = sanitize_validation_samples(samples)
        if "wandb" in loggers:
            self.log_generations_to_wandb(samples, step)
        if "swanlab" in loggers:
            self.log_generations_to_swanlab(samples, step)
        if "mlflow" in loggers:
            self.log_generations_to_mlflow(samples, step)

        if "clearml" in loggers:
            self.log_generations_to_clearml(samples, step)
        if "tensorboard" in loggers:
            self.log_generations_to_tensorboard(samples, step)

    def log_generations_to_wandb(self, samples, step):
        """Log samples to wandb as a table"""
        import wandb

        # Create column names for all samples
        columns = ["step"] + sum([[f"input_{i + 1}", f"output_{i + 1}", f"score_{i + 1}"] for i in range(len(samples))], [])

        if not hasattr(self, "validation_table"):
            # Initialize the table on first call
            self.validation_table = wandb.Table(columns=columns)

        # Create a new table with same columns and existing data
        # Workaround for https://github.com/wandb/wandb/issues/2981#issuecomment-1997445737
        new_table = wandb.Table(columns=columns, data=self.validation_table.data)

        # Add new row with all data
        row_data = []
        row_data.append(step)
        for sample in samples:
            row_data.extend(sample)

        new_table.add_data(*row_data)

        # Update reference and log
        wandb.log({"val/generations": new_table}, step=step)
        self.validation_table = new_table

    def log_generations_to_swanlab(self, samples, step):
        """Log samples to swanlab as text"""
        import swanlab

        swanlab_text_list = []
        for i, sample in enumerate(samples):
            row_text = f"""
            input: {sample[0]}
            
            ---
            
            output: {sample[1]}
            
            ---
            
            score: {sample[2]}
            """
            swanlab_text_list.append(swanlab.Text(row_text, caption=f"sample {i + 1}"))

        # Log to swanlab
        swanlab.log({"val/generations": swanlab_text_list}, step=step)

    def log_generations_to_mlflow(self, samples, step):
        """Log validation generation to mlflow as artifacts"""
        # https://mlflow.org/docs/latest/api_reference/python_api/mlflow.html?highlight=log_artifact#mlflow.log_artifact

        import json
        import tempfile

        import mlflow

        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                validation_gen_step_file = Path(tmp_dir, f"val_step{step}.json")
                row_data = []
                for sample in samples:
                    data = {"input": sample[0], "output": sample[1], "score": sample[2]}
                    row_data.append(data)
                with open(validation_gen_step_file, "w") as file:
                    json.dump(row_data, file)
                mlflow.log_artifact(validation_gen_step_file)
        except Exception as e:
            print(f"WARNING: save validation generation file to mlflow failed with error {e}")

    def log_generations_to_clearml(self, samples, step):
        """Log validation generation to clearml as table"""

        import clearml
        import pandas as pd

        task: clearml.Task | None = clearml.Task.current_task()
        if task is None:
            return

        table = [
            {
                "step": step,
                "input": sample[0],
                "output": sample[1],
                "score": sample[2],
            }
            for sample in samples
        ]

        logger = task.get_logger()
        logger.report_table(
            series="Validation generations",
            title="Validation",
            table_plot=pd.DataFrame.from_records(table),
            iteration=step,
        )

    def log_generations_to_tensorboard(self, samples, step):
        """Log samples to tensorboard as text"""
        # Initialize tensorboard writer if not exists
        if not hasattr(self, "writer"):
            from torch.utils.tensorboard import SummaryWriter

            tensorboard_dir = os.environ.get("TENSORBOARD_DIR", "tensorboard_log")
            os.makedirs(tensorboard_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=tensorboard_dir)

        # Format the samples data into readable text
        text_content = f"**Generation Results - Step {step}**\n\n"

        for i, sample in enumerate(samples):
            text_content += f"### Sample {i + 1}\n"

            # Assuming sample contains [input, output, score]
            if len(sample) >= 3:
                input_text, output_text, score = sample[0], sample[1], sample[2]

                text_content += f"**Input:** {input_text}\n\n"
                text_content += f"**Output:** {output_text}\n\n"
                text_content += f"**Score:** {score}\n\n"
            else:
                # Handle cases where sample format might be different
                text_content += f"**Data:** {sample}\n\n"

            text_content += "---\n\n"

        # Log to tensorboard as text
        self.writer.add_text("val/generations", text_content, step)
        # Flush to ensure data is written
        self.writer.flush()
