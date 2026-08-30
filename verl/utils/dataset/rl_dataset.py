# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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

import copy
import logging
import os
import re
from collections import defaultdict
from typing import List, Optional, Union

try:  # Keep lightweight helpers importable in offline/unit-test environments.
    import datasets
except ModuleNotFoundError:  # pragma: no cover - exercised by dependency-free tests
    datasets = None
try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - training dependency
    np = None
try:
    import torch
    from torch.utils.data import Dataset
except ModuleNotFoundError:  # pragma: no cover - training dependency
    torch = None

    class Dataset:  # type: ignore[no-redef]
        """Import-time fallback so path-only helpers remain usable offline."""

        pass

try:
    from omegaconf import DictConfig, ListConfig
except ModuleNotFoundError:  # pragma: no cover - training dependency
    DictConfig = ListConfig = ()  # type: ignore[assignment]
try:
    from transformers import PreTrainedTokenizer, ProcessorMixin
except ModuleNotFoundError:  # pragma: no cover - training dependency
    PreTrainedTokenizer = ProcessorMixin = object  # type: ignore[assignment]

try:
    import verl.utils.torch_functional as verl_F
    from verl.utils.model import compute_position_id_with_mask
except ModuleNotFoundError:  # pragma: no cover - training dependency
    verl_F = None
    compute_position_id_with_mask = None

logger = logging.getLogger(__name__)


# The authoritative TravelGym parquet inputs are the eight composition
# variants (``travel22_multiturn_onechoice``, ...).  Their reward-model
# records intentionally keep ``env_name=TravelGym`` for backwards
# compatibility, so the variant identity must come from the source path.  A
# pool key is ``env_name::task_id``; silently treating an aggregate parquet as
# one of the variants would make that key ambiguous and could violate the
# train/validation isolation contract.
_TRAVEL_VARIANT_RE = re.compile(
    r"^(travel22|travel33|travel44|travel233|travel333|travel334|travel444|travel2222)_multiturn_onechoice$",
    re.IGNORECASE,
)


def _as_bool(value) -> bool:
    """Parse config booleans fail-closed (``"false"`` must stay false)."""

    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value == 1
    return str(value or "").strip().casefold() in {"true", "1", "yes", "y"}


def _infer_travel_variant(path: str | os.PathLike[str]) -> str | None:
    """Infer the authoritative TravelGym composition from a parquet path."""

    parent_name = os.path.basename(os.path.dirname(os.fspath(path)))
    match = _TRAVEL_VARIANT_RE.fullmatch(parent_name)
    return match.group(1).casefold() if match else None


__all__ = ["RLHFDataset", "collate_fn", "_infer_travel_variant"]


def _native_prompt_ids(tokenizer, messages):
    """Render a prompt with Qwen thinking enabled when supported.

    Qwen3.5 exposes ``enable_thinking`` in the native template.  The fallback
    keeps the generic RL dataset usable with older tokenizers while the Travel
    preflight/rollout configuration remains fail-fast for a required Qwen
    revision.
    """
    try:
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, enable_thinking=True
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True)


def collate_fn(data_list: list[dict]) -> dict:
    """
    Collate a batch of sample dicts into batched tensors and arrays.

    Args:
        data_list: List of dicts mapping feature names to torch.Tensor or other values.

    Returns:
        Dict where tensor entries are stacked into a torch.Tensor of shape
        (batch_size, *dims) and non-tensor entries are converted to
        np.ndarray of dtype object with shape (batch_size,).
    """
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    return {**tensors, **non_tensors}


class RLHFDataset(Dataset):
    """
    Load and preprocess RLHF data from Parquet files.

    - Caches files locally.
    - Reads into a HuggingFace Dataset and tokenizes prompts.
    - Optionally handles images/videos via a ProcessorMixin.
    - Filters prompts over a max length.
    - Supports resuming from checkpoints.

    Args:
        data_files (str or list): Path(s) to Parquet file(s).
        tokenizer (PreTrainedTokenizer): For the tokenization of text to token IDs.
        config (DictConfig): Options like cache_dir, prompt_key, max_prompt_length, truncation, etc.
        processor (ProcessorMixin, optional): Multimodal preprocessor for images/videos.
    """

    def __init__(
        self,
        data_files: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        if datasets is None or torch is None or np is None or verl_F is None:
            raise ImportError(
                "RLHFDataset requires the training dependencies (torch, "
                "datasets, numpy, and verl tensor utilities); install them "
                "before constructing it."
            )
        if not isinstance(data_files, (List, ListConfig)):
            data_files = [data_files]

        self.data_files = copy.deepcopy(data_files)
        self.original_data_files = copy.deepcopy(data_files)  # use for resume
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/rlhf"))
        self.prompt_key = config.get("prompt_key", "prompt")
        self.image_key = config.get("image_key", "images")
        self.video_key = config.get("video_key", "videos")
        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.return_raw_chat = config.get("return_raw_chat", False)
        self.return_full_prompt = config.get("return_full_prompt", False)
        self.truncation = config.get("truncation", "error")
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)

        self.num_workers = config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4))
        self.num_workers = min(self.num_workers, os.cpu_count())
        self.use_shm = config.get('use_shm', False)
        self.chat_template_func = config.get("chat_template_func", None)
        self.need_tools_kwargs = config.get("need_tools_kwargs", False)
        self.filter_prompts = config.get("filter_prompts", True)
        # TravelGym task-pool filtering is performed before tokenisation.  The
        # manifest contains evaluator-side IDs only; none of these fields are
        # copied into Actor messages.  Keeping the filter here also protects
        # callers that reuse an older aggregate parquet file.
        self.task_pool_manifest_path = config.get("task_pool_manifest", None)
        self.task_pool_name = config.get("task_pool_name", None)
        self.task_pool_require_strict = _as_bool(config.get("task_pool_require_strict", False))
        if bool(self.task_pool_manifest_path) != bool(self.task_pool_name):
            raise ValueError(
                "task_pool_manifest and task_pool_name must be provided together"
            )
        self._task_pool_allowed: dict[str, set[tuple[str, str]]] | None = None
        if self.task_pool_manifest_path:
            try:
                from sft.task_pools import load_pool_manifest
            except ImportError as exc:  # pragma: no cover - package install path
                raise RuntimeError("TravelGym task-pool filtering requires sft.task_pools") from exc
            pool_manifest = load_pool_manifest(
                self.task_pool_manifest_path,
                require_strict=self.task_pool_require_strict,
            )
            records = pool_manifest.get("pools", {}).get(self.task_pool_name, {}).get("records", [])
            allowed: dict[str, set[tuple[str, str]]] = defaultdict(set)
            for item in records:
                if not isinstance(item, dict) or not item.get("task_id"):
                    # Opaque historical SFT reservations cannot occur in an
                    # RL parquet row and are intentionally ignored here.
                    continue
                env_name = str(item.get("env_name", ""))
                if not env_name or env_name == "opaque_sft":
                    continue
                task_id = str(item["task_id"])
                split = str(item.get("split", ""))
                allowed[task_id].add((env_name, split))
            self._task_pool_allowed = dict(allowed)
            if not self._task_pool_allowed:
                raise ValueError(
                    f"task pool {self.task_pool_name!r} has no resolvable task IDs"
                )
        self.serialize_dataset = False
        self._download()
        self._read_files_and_tokenize()

    def _download(self, use_origin_parquet=False):
        from verl.utils.fs import copy_to_local

        data_files = self.data_files if not use_origin_parquet else self.original_data_files
        for i, parquet_file in enumerate(data_files):
            self.data_files[i] = copy_to_local(src=parquet_file, cache_dir=self.cache_dir, use_shm=self.use_shm)

    def _read_files_and_tokenize(self):
        dataframes = []
        for index, parquet_file in enumerate(self.data_files):
            # read parquet files and cache
            dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            # ``copy_to_local`` may replace a remote path with a cache path
            # that no longer contains the composition directory.  Prefer the
            # original user-supplied path for variant inference and keep the
            # result as a private row field used only by the pool filter and
            # Hard Case audit.
            original_path = (
                self.original_data_files[index]
                if index < len(self.original_data_files)
                else parquet_file
            )
            source_variant = _infer_travel_variant(original_path)
            if self._task_pool_allowed is not None:
                if source_variant is None:
                    raise ValueError(
                        "TravelGym task-pool filtering requires an authoritative "
                        "<travel_variant>_multiturn_onechoice parquet path; "
                        f"cannot prove env_name::task_id identity for {original_path!r}"
                    )
                if "_travel_source_env_name" in dataframe.column_names:
                    # A cached/fixture parquet may already carry the private
                    # provenance column. Replace it with the authoritative
                    # source-path variant rather than creating a duplicate
                    # column or trusting a user-provided value.
                    dataframe = dataframe.remove_columns(["_travel_source_env_name"])
                dataframe = dataframe.add_column(
                    "_travel_source_env_name",
                    [source_variant] * len(dataframe),
                )
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        if self._task_pool_allowed is not None:
            # The historical parquet rows store ``TravelGym`` as the reward
            # model environment, not the composition variant.  Task IDs are
            # globally unique in the authoritative eight variant files; if a
            # future dataset violates that assumption, ambiguous IDs are
            # rejected instead of silently selecting a wrong task.
            original_len = len(self.dataframe)
            keep_indices: list[int] = []
            ambiguous: list[str] = []
            for index in range(len(self.dataframe)):
                row = self.dataframe[index]
                reward_model = row.get("reward_model", {})
                task_id = str(reward_model.get("id", "")) if isinstance(reward_model, dict) else ""
                candidates = self._task_pool_allowed.get(task_id, set())
                if not candidates:
                    continue
                source_variant = str(row.get("_travel_source_env_name", "")).casefold()
                # Never infer a composition variant from the task ID alone:
                # the public parquet schema deliberately uses the aggregate
                # ``TravelGym`` label and task-pool isolation is defined on
                # the pair env_name::task_id.
                candidates = {
                    candidate for candidate in candidates if candidate[0].casefold() == source_variant
                }
                if not candidates:
                    continue
                extra_info = row.get("extra_info", {})
                row_split = str(extra_info.get("split", "")) if isinstance(extra_info, dict) else ""
                split_candidates = {
                    candidate for candidate in candidates if not row_split or candidate[1] == row_split
                }
                if len(split_candidates) == 1:
                    keep_indices.append(index)
                elif len(split_candidates) > 1:
                    ambiguous.append(task_id)
            if ambiguous:
                raise ValueError(
                    "task-pool filtering found ambiguous task IDs without a composition "
                    f"label: {sorted(set(ambiguous))[:5]}"
                )
            self.dataframe = self.dataframe.select(keep_indices)
            logger.info(
                "TravelGym task pool %s kept %d/%d rows",
                self.task_pool_name,
                len(self.dataframe),
                original_len,
            )
            if len(self.dataframe) == 0:
                raise ValueError(
                    f"TravelGym task pool {self.task_pool_name!r} selected no rows from "
                    f"{self.data_files}"
                )

        print(f"dataset len: {len(self.dataframe)}")

        # filter out too long prompts
        if self.filter_overlong_prompts:
            tokenizer = self.tokenizer
            prompt_key = self.prompt_key
            self.dataframe = self.dataframe.filter(
                lambda doc: len(_native_prompt_ids(tokenizer, doc[prompt_key])) <= self.max_prompt_length,
                num_proc=self.num_workers,
                desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
            )

            print(f"filter dataset len: {len(self.dataframe)}")

    def resume_dataset_state(self):
        self.serialize_dataset = not hasattr(self, "original_data_files")
        # resume dataframe if not it's serialized in data.pt
        if not self.serialize_dataset:
            self._download(use_origin_parquet=True)  # download and resume from original parquet files
            self._read_files_and_tokenize()
        else:
            print(r"old dataloader ckpt file is used, please train from scratch for better ckpt performance")

    def __len__(self):
        return len(self.dataframe)

    def _build_messages(self, example: dict):
        messages: list = example.pop(self.prompt_key)

        if self.image_key in example or self.video_key in example:
            for message in messages:
                content = message["content"]
                content_list = []
                for segment in re.split("(<image>|<video>)", content):
                    if segment == "<image>":
                        content_list.append({"type": "image"})
                    elif segment == "<video>":
                        content_list.append({"type": "video"})
                    else:
                        content_list.append({"type": "text", "text": segment})

                message["content"] = content_list

        return messages

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe[item]
        messages = self._build_messages(row_dict)
        model_inputs = {}

        if self.processor is not None:
            from verl.utils.dataset.vision_utils import process_image, process_video

            try:
                raw_prompt = self.processor.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False, enable_thinking=True
                )
            except TypeError:
                raw_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            multi_modal_data = {}

            images = None
            if self.image_key in row_dict:
                images = [process_image(image) for image in row_dict.pop(self.image_key)]
                multi_modal_data["image"] = images

            videos = None
            if self.video_key in row_dict:
                videos = [process_video(video) for video in row_dict.pop(self.video_key)]
                multi_modal_data["video"] = [video.numpy() for video in videos]

            model_inputs = self.processor(text=[raw_prompt], images=images, videos=videos, return_tensors="pt")

            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

            if "second_per_grid_ts" in model_inputs:
                model_inputs.pop("second_per_grid_ts")

            # There's a trap here, multi_modal_inputs has to be a dict, not BatchFeature
            row_dict["multi_modal_data"] = multi_modal_data
            row_dict["multi_modal_inputs"] = dict(model_inputs)

            # second_per_grid_ts isn't used for training, just for mrope
            row_dict["multi_modal_inputs"].pop("second_per_grid_ts", None)

        else:
            try:
                raw_prompt = self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False, enable_thinking=True
                )
            except TypeError:
                raw_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        if self.processor is not None and self.processor.image_processor.__class__.__name__ == "Qwen2VLImageProcessor":
            from verl.models.transformers.qwen2_vl import get_rope_index

            position_ids = [
                get_rope_index(
                    self.processor,
                    input_ids=input_ids[0],
                    image_grid_thw=model_inputs.get("image_grid_thw"),
                    video_grid_thw=model_inputs.get("video_grid_thw"),
                    second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                    attention_mask=attention_mask[0],
                )
            ]  # (1, 3, seq_len)

        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        row_dict["raw_prompt_ids"] = raw_prompt_ids
        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages

        # get prompts with chat template
        if self.return_full_prompt:
            row_dict["full_prompts"] = raw_prompt  # array of strings

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        need_tools_kwargs = row_dict.get("extra_info", {}).get("need_tools_kwargs", self.need_tools_kwargs)
        if need_tools_kwargs and not tools_kwargs:
            logger.warning("tools_kwargs is empty for index {}, data source: {}", index, row_dict["data_source"])
        row_dict["index"] = index
        row_dict["tools_kwargs"] = tools_kwargs
        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if "dataframe" in state:
                del state["dataframe"]
            return state

        return self.__dict__.copy()
