# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
from __future__ import annotations

import asyncio
import logging
import multiprocessing as mp
import os
import time
from contextlib import contextmanager
from copy import deepcopy
from json import JSONDecodeError
from importlib.metadata import PackageNotFoundError, version
from typing import List, Optional, Tuple
from uuid import uuid4

import numpy as np
import sglang.srt.entrypoints.engine
import torch
import torch.distributed as dist
from omegaconf import DictConfig
try:
    from sglang.srt.managers.tokenizer_manager import (
        ReleaseMemoryOccupationReqInput,
        ResumeMemoryOccupationReqInput,
        UpdateWeightsFromTensorReqInput,
    )
except ImportError:
    from sglang.srt.managers.io_struct import (
        ReleaseMemoryOccupationReqInput,
        ResumeMemoryOccupationReqInput,
        UpdateWeightsFromTensorReqInput,
    )
try:
    from sglang.srt.openai_api.protocol import Tool
except ImportError:
    from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
try:
    from sglang.srt.utils import (
        MultiprocessingSerializer,
        assert_pkg_version,
        get_ip,
        get_open_port,
        is_cuda,
        maybe_set_triton_cache_manager,
        set_prometheus_multiproc_dir,
        set_ulimit,
    )
except ImportError:
    import socket

    from sglang.srt.utils.common import (
        MultiprocessingSerializer,
        assert_pkg_version,
        is_cuda,
        set_prometheus_multiproc_dir,
        set_ulimit,
    )
    from sglang.srt.utils.network import get_open_port

    def get_ip():
        return socket.gethostbyname(socket.gethostname())

    def maybe_set_triton_cache_manager():
        return None
from tensordict import TensorDict
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizer

from verl import DataProto
from verl.protocol import make_1d_object_array
from verl.third_party.sglang import parallel_state as sglang_ps
from verl.tools.base_tool import BaseTool
from verl.tools.travel_reward_metrics import (
    TRAVEL_ACTOR_ASPECT_METRIC_NAMES,
    TRAVEL_QUALITY_METRIC_NAMES,
    TRAVEL_REWARD_METRIC_NAMES,
    TRAVEL_USER_METRIC_NAMES,
)
from verl.tools.schemas import (
    OpenAIFunctionCallSchema,
    OpenAIFunctionParsedSchema,
    OpenAIFunctionToolCall,
)
from verl.utils.debug import GPUMemoryLogger
from verl.utils.model import compute_position_id_with_mask
from verl.utils.net_utils import is_ipv6
from verl.utils.torch_functional import (
    get_response_mask,
    pad_sequence_to_length,
)
from verl.workers.rollout.base import BaseRollout
from verl.workers.rollout.schemas import (
    AsyncRolloutRequest,
    AsyncRolloutRequestStateEnum,
    FinishReasonTypeEnum,
    Message,
    RolloutLengthExceededError,
    RolloutProtocolError,
    RolloutTemplateAlignmentError,
    compute_generation_budget,
)
from verl.workers.rollout.sglang_rollout.utils import broadcast_pyobj

try:
    from sglang.srt.function_call.function_call_parser import FunctionCallParser
except ImportError:
    from sglang.srt.function_call_parser import FunctionCallParser


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# patch to avoid issue https://github.com/sgl-project/sglang/issues/6723
def _set_envs_and_config(server_args: ServerArgs):
    # Set global environments
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["NCCL_CUMEM_ENABLE"] = "0"
    os.environ["NCCL_NVLS_ENABLE"] = str(int(server_args.enable_nccl_nvls))
    os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "4"
    os.environ["CUDA_MODULE_LOADING"] = "AUTO"

    # Set prometheus env vars
    if server_args.enable_metrics:
        set_prometheus_multiproc_dir()

    # Set ulimit
    set_ulimit()

    # Fix triton bugs
    if server_args.tp_size * server_args.dp_size > 1:
        # FIXME: remove this after https://github.com/triton-lang/triton/pull/4295 is used as a dependency.
        maybe_set_triton_cache_manager()

    # Check flashinfer version
    if server_args.attention_backend == "flashinfer":
        assert_pkg_version(
            "flashinfer_python",
            "0.2.5",
            "Please uninstall the old version and reinstall the latest version by following the instructions at https://docs.flashinfer.ai/installation.html.",
        )
    if is_cuda():
        try:
            version("sglang-kernel")
            kernel_package = "sglang-kernel"
            minimum_kernel_version = "0.4.6"
        except PackageNotFoundError:
            # SGLang <= 0.4 published the CUDA extension as ``sgl-kernel``.
            kernel_package = "sgl-kernel"
            minimum_kernel_version = "0.1.1"
        assert_pkg_version(
            kernel_package,
            minimum_kernel_version,
            f"Please reinstall it with `pip install {kernel_package} --force-reinstall`",
        )

    # Set mp start method
    mp.set_start_method("spawn", force=True)


sglang.srt.entrypoints.engine._set_envs_and_config = _set_envs_and_config


# because chatCompletion is an async method, it makes the whole ray actor be an async actor
# which can not call loop.run_until_complete. So we need to make the engine to be an async class
class AsyncEngine(sglang.srt.entrypoints.engine.Engine):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # default to use dummy load format, which need to reload weights in first time
        self._need_reload = True

    async def release_memory_occupation(self):
        """Release GPU occupation temporarily."""
        obj = ReleaseMemoryOccupationReqInput()
        return await self.tokenizer_manager.release_memory_occupation(obj, None)

    async def resume_memory_occupation(self):
        """Resume GPU occupation."""

        # because __init__ is a sync method, it can not call the async release_memory_occupation
        # have to move release_memory_occupation from __init__ to here
        if self._need_reload:
            await self.release_memory_occupation()
            self._need_reload = False

        obj = ResumeMemoryOccupationReqInput()
        return await self.tokenizer_manager.resume_memory_occupation(obj, None)

    async def update_weights_from_tensor(
        self,
        named_tensors: List[Tuple[str, torch.Tensor]],  # noqa: UP006
        load_format: Optional[str] = None,
        flush_cache: bool = True,
    ):
        """Update weights from distributed source. If there are going to be more updates, set `flush_cache` to be false
        to avoid duplicated cache cleaning operation."""
        obj = UpdateWeightsFromTensorReqInput(
            serialized_named_tensors=[MultiprocessingSerializer.serialize(named_tensors) for _ in range(self.server_args.tp_size)],
            load_format=load_format,
            flush_cache=flush_cache,
        )
        return await self.tokenizer_manager.update_weights_from_tensor(obj, None)

    async def flush_cache(self):
        return await self.tokenizer_manager.flush_cache()


# NOTE(sgm): add for verl. We can optimize it by making
#  the dataloader yield List[int] without padding.
def _pre_process_inputs(
    pad_token_id,
    prompt_token_ids: torch.Tensor,
) -> list[int]:
    # remove the left padding in the prompt token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


# NOTE(linjunrong): adhoc
def _post_process_outputs(tokenizer, output):
    def _map_each_response(resp):
        output_token_logprobs = resp["meta_info"]["output_token_logprobs"]
        log_probs, output_token_ids = zip(*[(log_prob, token_ids) for log_prob, token_ids, _ in output_token_logprobs])
        return torch.tensor(output_token_ids), torch.tensor(log_probs)

    out_map = map(lambda x: _map_each_response(x), output)
    batched_output_token_ids = []
    batched_logprobs = []
    for output_token_ids, log_probs in out_map:
        batched_output_token_ids.append(output_token_ids)
        batched_logprobs.append(log_probs)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    batched_output_token_ids = pad_sequence(batched_output_token_ids, batch_first=True, padding_value=pad_token_id)
    if len(batched_logprobs) > 0:
        batched_logprobs = pad_sequence(batched_logprobs, batch_first=True, padding_value=pad_token_id)
    return batched_output_token_ids, batched_logprobs


def get_tool_call_parser_type(tokenizer: PreTrainedTokenizer, config=None) -> str:
    """Select a configured parser and fail loudly when it is unavailable.

    A Qwen3.5 deployment should set ``multi_turn.tool_call_parser`` to the
    parser verified for the installed SGLang build (normally
    ``qwen3_coder``). Automatic token detection is retained only as a
    compatibility fallback for non-Travel rollouts.
    """
    items = list(FunctionCallParser.ToolCallParserEnum.items())
    requested = None
    if config is not None:
        try:
            requested = config.multi_turn.get("tool_call_parser")
        except (AttributeError, TypeError):
            requested = None
    if requested:
        requested = str(requested)
        parser_cls = dict(items).get(requested)
        if parser_cls is None:
            available = ", ".join(sorted(dict(items)))
            raise ValueError(
                f"Configured SGLang tool_call_parser={requested!r} is unavailable; "
                f"upgrade/verify SGLang before running Qwen3.5 (available: {available})"
            )
        parser = parser_cls()
        vocab = tokenizer.get_vocab()
        # Newer structural-tag parsers expose start/end fields instead of
        # bot_token/eot_token; accept either representation.
        bot_token = (getattr(parser, "bot_token", "") or getattr(parser, "tool_call_start_token", "")).strip()
        eot_token = (getattr(parser, "eot_token", "") or getattr(parser, "tool_call_end_token", "")).strip()
        if not bot_token or bot_token not in vocab or (eot_token and eot_token not in vocab):
            raise ValueError(f"Configured parser {requested!r} is incompatible with the tokenizer")
        return requested
    for parser_type, parser_cls in items:
        parser = parser_cls()
        bot_token = (getattr(parser, "bot_token", "") or getattr(parser, "tool_call_start_token", "")).strip()
        eot_token = (getattr(parser, "eot_token", "") or getattr(parser, "tool_call_end_token", "")).strip()
        if bot_token in tokenizer.get_vocab() and (not eot_token or eot_token in tokenizer.get_vocab()):
            return parser_type
    else:
        raise ValueError(f"No tool call parser found for tokenizer {tokenizer}")


class SGLangRollout(BaseRollout):
    def __init__(
        self,
        actor_module: str,
        config: DictConfig,
        tokenizer,
        model_hf_config,
        port=None,
        trust_remote_code: bool = False,
        device_mesh: DeviceMesh | None = None,
        **kwargs,
    ):
        """Synchronized SGLang rollout engine.

        Args:
            actor_module: Huggingface model name or path to the model. The
                model should be supported by SGLang.
            config: A DictConfig object containing SGLang-specific operational
                parameters and rollout settings.
                Refer to https://docs.sglang.ai/backend/server_arguments.html
            tokenizer: The tokenizer instance compatible with the actor_module.
            model_hf_config: The Hugging Face model's configuration (e.g.,
                `transformers.PretrainedConfig`). It provides architectural
                details and hyperparameters like `max_position_embeddings`,
                used by SGLang for correct model initialization. This is
                the model's inherent design, not SGLang's runtime behavior.
            port: Optional port for multi-node initialization when nnodes > 1.
            trust_remote_code: Whether or not to allow for custom models
                defined on the Hub in their own modeling files.
            device_mesh: Optional `DeviceMesh` object for distributed setup.
            **kwargs: Additional keyword arguments, primarily `train_tp` for
                Megatron Backend integration to initialize hybrid engine
                process groups.
        """
        super().__init__()
        self.config = config
        multi_turn_config = config.get("multi_turn", {})
        from travelgym.env.actor_aspects import actor_aspect_extraction_enabled

        extraction_flag = config.get("enable_actor_aspect_extraction")
        if extraction_flag is None:
            extraction_flag = multi_turn_config.get("enable_actor_aspect_extraction")
        self._enable_actor_aspect_extraction = actor_aspect_extraction_enabled(
            extraction_flag
        )
        self._actor_aspect_extraction_max_tokens = max(
            1,
            int(multi_turn_config.get("actor_aspect_extraction_max_tokens", 128)),
        )
        self._max_new_tokens_per_turn = int(
            multi_turn_config.get("max_new_tokens_per_turn", 2048)
        )
        self._max_reasoning_tokens_per_turn = int(
            multi_turn_config.get("max_reasoning_tokens_per_turn", 2560)
        )
        self._max_tool_call_tokens_per_turn = int(
            multi_turn_config.get("max_tool_call_tokens_per_turn", 512)
        )
        self._tool_response_token_reserve = int(
            multi_turn_config.get("tool_response_token_reserve", 6144)
        )
        self._template_token_reserve = int(
            multi_turn_config.get("template_token_reserve", 32)
        )
        context_cleanup_config = config.get("context_cleanup")
        if context_cleanup_config is None:
            context_cleanup_config = multi_turn_config.get("context_cleanup", {})
        if context_cleanup_config is None:
            context_cleanup_config = {}
        self._context_cleanup_enabled = bool(
            context_cleanup_config.get("enabled", False)
        )
        self._context_cleanup_target_tokens = int(
            context_cleanup_config.get("target_context_tokens", 20000)
        )
        self._context_cleanup_template_margin = int(
            context_cleanup_config.get(
                "template_margin_tokens", self._template_token_reserve
            )
        )
        self._next_turn_reserve = (
            self._max_reasoning_tokens_per_turn
            + self._max_tool_call_tokens_per_turn
            + self._tool_response_token_reserve
            + self._context_cleanup_template_margin
        )
        if (
            self._max_new_tokens_per_turn <= 0
            or self._max_reasoning_tokens_per_turn <= 0
            or self._max_tool_call_tokens_per_turn <= 0
            or self._tool_response_token_reserve <= 0
            or self._template_token_reserve <= 0
            or self._context_cleanup_target_tokens <= 0
            or self._context_cleanup_template_margin < 0
        ):
            raise ValueError(
                "multi_turn token budgets must be positive: "
                "max_new_tokens_per_turn, max_reasoning_tokens_per_turn, "
                "max_tool_call_tokens_per_turn, tool_response_token_reserve, "
                "template_token_reserve, context_cleanup.target_context_tokens "
                "and context_cleanup.template_margin_tokens"
            )
        if self._max_reasoning_tokens_per_turn + self._max_tool_call_tokens_per_turn > self._max_new_tokens_per_turn:
            raise ValueError(
                "reasoning and tool-call token caps must fit within "
                "max_new_tokens_per_turn"
            )
        self._device_mesh_cpu = device_mesh
        os.environ.setdefault("SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK", "true")

        (
            self._tool_schemas,
            self._tool_map,
            self._tool_call_parser_type,
            self._sgl_tools,
            self._function_call_parser,
        ) = self._initialize_tools(config, tokenizer)
        # If turn on `free_cache_engine`, SGLang engine's KV cache
        # will be freed after each `generate_sequences` call.
        assert not (not config.enforce_eager and config.free_cache_engine), "disable CUDA graph (enforce_eager = False) if free cache engine"

        logger.info(f"tool_schemas: {self._tool_schemas}, tool_map: {self._tool_map}, tool_call_parser_type: {self._tool_call_parser_type}, sgl_tools: {self._sgl_tools}, function_call_parser: {self._function_call_parser}")

        self._init_distributed_env(device_mesh_cpu=device_mesh, **kwargs)

        self._verify_config(model_hf_config=model_hf_config)
        # initialize the inference engine
        self._init_inference_engine(trust_remote_code, actor_module, port)

        self._init_sampling_params(**kwargs)

        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id

    def _init_distributed_env(self, device_mesh_cpu, **kwargs):
        self._device_mesh_cpu = device_mesh_cpu
        os.environ.setdefault("SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK", "true")
        self.tensor_parallel_size = self.config.get("tensor_model_parallel_size", 1)
        assert self.tensor_parallel_size <= dist.get_world_size(), "tensor parallel size should be less than or equal to the world size"
        self.train_tp = kwargs.get("train_tp", None)
        if self.train_tp is not None:
            # deployed with megatron
            os.environ["CUDA_TIMER_STREAM_KAFKA_ENABLE"] = "0"
            os.environ["MEGATRON_IMPORT_TIMERS"] = "0"
            train_tp = kwargs.get("train_tp", None)
            num_tp_per_train_tp = train_tp // self.tensor_parallel_size
            sglang_ps.initialize_parallel_state(
                tensor_model_parallel_size=self.tensor_parallel_size,
                num_tp_per_train_tp=num_tp_per_train_tp,
            )

        tp_size = self.tensor_parallel_size
        world_size = int(os.getenv("WORLD_SIZE", "-1"))

        # init device mesh
        if self._device_mesh_cpu is None:
            device_mesh_kwargs = dict(
                mesh_shape=(world_size // tp_size, tp_size, 1),
                mesh_dim_names=["dp", "tp", "pp"],
            )

            self._device_mesh_cpu = init_device_mesh("cpu", **device_mesh_kwargs)

        self._rank = self._device_mesh_cpu.get_rank()
        self._tp_rank = self._device_mesh_cpu["tp"].get_local_rank()
        self._tp_size = self._device_mesh_cpu["tp"].size()
        if self._rank == 0:
            logger.info(f"_init_distributed_env: :tp_world: {self._tp_size}, global_world: {world_size}")
        # get tp_rank of this process in this tp group
        visible_devices = [None] * self._device_mesh_cpu.size(1)

        torch.distributed.all_gather_object(visible_devices, os.environ["CUDA_VISIBLE_DEVICES"], self._device_mesh_cpu.get_group("tp"))
        self.visible_devices_set = set(",".join(visible_devices).split(","))
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(sorted(list(self.visible_devices_set)))

    def _verify_config(self, model_hf_config):
        if not self.config.get("max_model_len", None):
            self.config.max_model_len = self.config.prompt_length + self.config.response_length
        assert self.config.max_model_len >= self.config.prompt_length + self.config.response_length, f"""max_model_len should be greater than total sequence length (prompt_length + response_length): 
            {self.config.max_model_len} >= {self.config.prompt_length} + {self.config.response_length}"""
        max_position_embeddings = getattr(model_hf_config, "max_position_embeddings", None)
        if max_position_embeddings is None:
            text_config = getattr(model_hf_config, "text_config", None)
            max_position_embeddings = getattr(text_config, "max_position_embeddings", None)
        assert max_position_embeddings is not None, "model config must expose max_position_embeddings"
        assert max_position_embeddings >= self.config.max_model_len, "model context length should be greater than total sequence length"
        # currently max_turns stand for max number of tool calls
        if self.config.multi_turn.max_turns is None:
            self.config.multi_turn.max_turns = self.config.max_model_len // 3

    def _init_inference_engine(self, trust_remote_code, actor_module, port):
        # initialize the inference engine
        nnodes = -(-self._tp_size // len(self.visible_devices_set))
        if nnodes > 1:
            ip = get_ip()
            port = get_open_port() if port is None else port
            [ip, port] = broadcast_pyobj(
                [ip, port],
                rank=self._rank,
                dist_group=self._device_mesh_cpu.get_group("tp"),
                src=self._device_mesh_cpu["tp"].mesh[0].item(),
                force_cpu_device=False,
            )
            dist_init_addr = f"[{ip}]:{port}" if is_ipv6(ip) else f"{ip}:{port}"
        else:
            dist_init_addr = None

        load_format = "dummy" if self.config.load_format.startswith("dummy") else self.config.load_format
        tp_size_per_node = self._tp_size // nnodes
        node_rank = self._tp_rank // tp_size_per_node
        first_rank_in_node = self._tp_rank % tp_size_per_node == 0

        if first_rank_in_node:
            rank = dist.get_rank()
            os.environ["SGLANG_BLOCK_NONZERO_RANK_CHILDREN"] = "0"
            self._engine = AsyncEngine(
                model_path=actor_module,
                dtype=self.config.dtype,
                mem_fraction_static=self.config.gpu_memory_utilization,
                enable_memory_saver=True,
                base_gpu_id=0,
                gpu_id_step=1,
                tp_size=self._tp_size,
                node_rank=node_rank,
                load_format=load_format,
                dist_init_addr=dist_init_addr,
                nnodes=nnodes,
                trust_remote_code=trust_remote_code,
                # NOTE(linjunrong): add rank to prevent SGLang generate same port inside PortArgs.init_new
                # when random.seed is being set during training
                port=30000 + rank,
                # NOTE(Chenyang): if you want to debug the SGLang engine output
                # please set the following parameters
                # Otherwise, it will make the engine run too slow
                # log_level="INFO",
                # log_requests=True,
                # log_requests_level=2,
                # max_running_requests=1,
            )
        else:
            self._engine = None

        self.sharding_manager = None
        self.is_sleep = True

    def _init_sampling_params(self, **kwargs):
        kwargs = dict(
            n=1,
            max_new_tokens=self.config.response_length,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            repetition_penalty=1.0,
            stop=None,
            no_stop_trim=False,
        )
        # supporting adding any sampling params from the config file
        for k in self.config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = self.config.get(k)
        self.sampling_params = kwargs

    def _initialize_tools(self, config, tokenizer):
        """Initialize tools from configuration.

        Args:
            config: Configuration object containing tool-related settings,
                    specifically `config.multi_turn.tool_config_path`.
            tokenizer: The tokenizer instance used for parsing tool calls from
                       the model's generated text.

        Returns:
            tuple: A tuple containing:
                - tool_schemas (list[dict]): OpenAI-formatted JSON schemas
                  defining each tool's capabilities.
                - tool_map (dict[str, BaseTool]): A dictionary mapping tool
                  names to their executable `BaseTool` objects.
                - tool_call_parser_type (str): The identifier for the specific
                  parser type (e.g., 'json_mode', 'tool_code') used to extract
                  tool calls.
                - sgl_tools (list[sglang.srt.openai_api.protocol.Tool]): Tool
                  definitions optimized for SGLang's internal engine.
                - function_call_parser (sglang.srt.function_call_parser.FunctionCallParser):
                  The active parser instance responsible for extracting
                  structured tool calls from model outputs.
        """
        if config.multi_turn.tool_config_path is None:
            return [], {}, None, [], None

        import importlib.util
        import sys

        from omegaconf import OmegaConf

        from verl.tools.schemas import OpenAIFunctionToolSchema

        def initialize_tools_from_config(tools_config) -> list:
            tool_list = []

            for tool_config in tools_config.tools:
                cls_name = tool_config.class_name
                module_name, class_name = cls_name.rsplit(".", 1)

                if module_name not in sys.modules:
                    spec = importlib.util.find_spec(module_name)
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = module
                    spec.loader.exec_module(module)
                else:
                    module = sys.modules[module_name]

                tool_cls = getattr(module, class_name)

                tool_schema_dict = OmegaConf.to_container(tool_config.tool_schema, resolve=True)
                tool_schema = OpenAIFunctionToolSchema.model_validate(tool_schema_dict)

                tool = tool_cls(
                    config=OmegaConf.to_container(tool_config.config, resolve=True),
                    tool_schema=tool_schema,
                )
                tool_list.append(tool)

            return tool_list

        tools_config_file = config.multi_turn.tool_config_path
        tools_config = OmegaConf.load(tools_config_file)
        tool_list = initialize_tools_from_config(tools_config)
        logger.info(f"Initialize tools from configuration.: tool_list: {tool_list}")
        tool_schemas = [tool.get_openai_tool_schema().model_dump() for tool in tool_list]
        tool_map = {tool.name: tool for tool in tool_list}
        tool_call_parser_type = get_tool_call_parser_type(tokenizer, config)
        sgl_tools = [Tool.model_validate(tool_schema) for tool_schema in tool_schemas]
        function_call_parser = FunctionCallParser(
            sgl_tools,
            tool_call_parser_type,
        )

        return (
            tool_schemas,
            tool_map,
            tool_call_parser_type,
            sgl_tools,
            function_call_parser,
        )

    @contextmanager
    def update_sampling_params(self, **kwargs):
        """
        Temporarily updates the model's sampling parameters for the
        duration of a `with` block. Parameters are automatically fall
          back to their original values upon exiting the block.

        Args:
            **kwargs: Keyword arguments representing sampling parameters
                    to be updated. Only parameters that already exist in
                    `self.sampling_params` will be updated.
        """
        # Store original values of parameters that will be updated
        old_sampling_params_args = {key: self.sampling_params[key] for key in kwargs if key in self.sampling_params}

        # Update sampling parameters with new values
        for key, value in kwargs.items():
            if key in self.sampling_params:
                self.sampling_params[key] = value

        try:
            yield
            # Yield and execute the code within the 'with' block
        finally:
            # Always restore original values, even if an error
            # occurred in the `with` block
            for key, value in old_sampling_params_args.items():
                self.sampling_params[key] = value

    @GPUMemoryLogger(role="sglang rollout", logger=logger)
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        if self.config.multi_turn.enable:
            return self._req_level_generate_sequences(prompts, **kwargs)
        return self._batch_level_generate_sequences(prompts, **kwargs)

    @GPUMemoryLogger(role="sglang rollout", logger=logger)
    @torch.no_grad()
    def _batch_level_generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """Generates sequences for a batch of prompts.
        For single-turn generation, all prompts are processed in one request.
        For multi-turn generation, each prompt is processed separately via
        `_generate_req_level_sequences` for better tool calling control.
        `_generate_batch_level_sequences` involves:
        1.  Extracting and pre-processing prompt token IDs from the input
            `prompts`. This includes handling padding and preparing raw
            token ID lists.
        2.  Preparing inputs for the SGLang engine, including multi-modal
            data if present.
        3.  Invoking the SGLang engine (`self._engine.async_generate`,
            an async coroutine) with the batch of processed inputs and
            specified sampling parameters on the master TP rank.
        4.  Broadcasting the results from the master TP rank to all
            other TP ranks.
        5.  Post-processing the engine's output to format the generated
            token IDs and (if applicable) log probabilities.
        6.  Constructing the final sequences by concatenating original
            prompts with the generated responses.
        7.  Updating attention masks and position IDs to reflect the full
            concatenated sequences.
        8.  If `self.config.free_cache_engine` is true, the SGLang engine's
            KV cache is flushed after generation on the master TP rank.
        Args:
            prompts: A `DataProto` object containing the batch of
              input prompts, including tensor data (like `input_ids`,
              `attention_mask`) and meta-information (like `eos_token_id`,
              `do_sample`).
            **kwargs: Additional keyword arguments that can override the
              default sampling parameters (e.g., `temperature`, `top_p`,
              `max_new_tokens`). These are temporarily applied using
              `update_sampling_params`.
        Returns:
            DataProto: A `DataProto` object containing the batch of
              generated sequences. This includes tensors for `prompts`
              (original input IDs), `responses` (generated token IDs),
              `input_ids` (concatenated prompt and response),
              `attention_mask`, and `position_ids` for the full
              sequences.
        Note that when `n > 1`, each prompt generates multiple sequences,
        so we need to replicate its non-tensor data (i.e. raw prompts,
        messages, reward scores, etc.) n times to match the expanded
        tensor data. This is done in the `_non_tensor_batch` dictionary.
        """
        # input ids: (bs, prompt_length), left-padded
        idx = prompts.batch["input_ids"]
        # attention_mask: (bs, seq_length), left-padded
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        # used to generate attention mask for the
        # response based on EOS token position
        eos_token_id = prompts.meta_info["eos_token_id"]

        batch_size = idx.size(0)

        # Extract non-tensor data
        non_tensor_batch = prompts.non_tensor_batch
        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)],
                dtype=object,
            )

        if "multi_modal_data" in non_tensor_batch:
            sglang_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(
                non_tensor_batch.pop("raw_prompt_ids"),
                non_tensor_batch.pop("multi_modal_data"),
            ):
                sglang_inputs.append(
                    {
                        "prompt_token_ids": raw_prompt_ids,
                        "multi_modal_data": multi_modal_data,
                        "image_data": (multi_modal_data.get("image", None) if isinstance(multi_modal_data, dict) else None),
                    }
                )
        else:
            sglang_inputs = [{"prompt_token_ids": raw_prompt_ids} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")]

        # Ensure token IDs are lists or numpy arrays
        for input_data in sglang_inputs:
            if isinstance(input_data["prompt_token_ids"], np.ndarray):
                input_data["prompt_token_ids"] = input_data["prompt_token_ids"].tolist()
            elif not isinstance(input_data["prompt_token_ids"], list):
                raise TypeError(f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}")

        # Extract token IDs and image data for SGLang Engine
        idx_list = [input_data["prompt_token_ids"] for input_data in sglang_inputs]
        image_list = [input_data.get("image_data", None) for input_data in sglang_inputs]

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        if not do_sample:
            kwargs = dict(
                n=1,
                presence_penalty=0.0,
                frequency_penalty=0.0,
                repetition_penalty=1.0,
                temperature=0,
                top_p=1,
                top_k=-1,
                ignore_eos=False,
                min_new_tokens=0,
                max_new_tokens=self.config.response_length,
                skip_special_tokens=True,
                spaces_between_special_tokens=True,
            )
        elif is_validate:
            kwargs = dict(
                top_k=self.config.val_kwargs.top_k,
                top_p=self.config.val_kwargs.top_p,
                temperature=self.config.val_kwargs.temperature,
                n=1,  # if validate, already repeat in ray_trainer
            )

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            # print(f"{self.sampling_params=}")
            if self._tp_rank == 0:
                loop = asyncio.get_event_loop()
                output = loop.run_until_complete(
                    self._engine.async_generate(
                        prompt=None,  # because we have already convert it to prompt token id
                        sampling_params=self.sampling_params,
                        return_logprob=True,
                        input_ids=idx_list,
                        image_data=image_list,
                    )
                )
            else:
                output = None

            # Most naive implementation, can extract tensor and send via gloo if too slow
            # dist.barrier()
            [output] = broadcast_pyobj(
                data=[output],
                rank=self._rank,
                dist_group=self._device_mesh_cpu["tp"].get_group(),
                src=self._device_mesh_cpu["tp"].mesh[0].item(),
                force_cpu_device=False,
            )
            out = _post_process_outputs(self.tokenizer, output)

            response = out[0].to(idx.device)
            rollout_log_probs = out[1].to(idx.device)

            if response.shape[1] < self.config.response_length:
                response = pad_sequence_to_length(response, self.config.response_length, self.pad_token_id)
                rollout_log_probs = pad_sequence_to_length(rollout_log_probs, self.config.response_length, self.pad_token_id)

            # utilize current sampling params
            if self.sampling_params.get("n", 1) > 1 and do_sample:
                idx = idx.repeat_interleave(self.sampling_params["n"], dim=0)
                attention_mask = attention_mask.repeat_interleave(self.sampling_params["n"], dim=0)
                position_ids = position_ids.repeat_interleave(self.sampling_params["n"], dim=0)
                batch_size = batch_size * self.sampling_params["n"]
                _non_tensor_batch = {}
                for key, val in non_tensor_batch.items():
                    _non_tensor_batch[key] = np.repeat(val, self.sampling_params["n"], axis=0)
            else:
                _non_tensor_batch = non_tensor_batch
            seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,  # here input_ids become the whole sentences
                "rollout_log_probs": rollout_log_probs,  # we will recompute old log prob with actor
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )

        # free cache engine
        if self.config.free_cache_engine and self._engine is not None:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(self._engine.flush_cache())

        return DataProto(batch=batch, non_tensor_batch=_non_tensor_batch)

    def _compute_generation_budget(
        self,
        req: AsyncRolloutRequest,
        generation_prompt_ids: list[int],
        *,
        max_new_tokens_per_turn: int | None = None,
        phase: str = "generation",
    ) -> int:
        active_segment = getattr(req, "active_segment", None)
        if active_segment is not None:
            fragment_prompt_tokens = int(
                active_segment.get("start_input", len(req.prompt_ids))
            )
        else:
            fragment_prompt_tokens = len(req.prompt_ids)
        current_response_tokens = max(
            0, len(generation_prompt_ids) - fragment_prompt_tokens
        )
        fragment_response_limit = min(
            int(req.max_response_len),
            max(0, int(req.max_model_len) - fragment_prompt_tokens),
        )
        phase_cap = int(max_new_tokens_per_turn or self._max_new_tokens_per_turn)
        max_new_tokens, budget_clamped = compute_generation_budget(
            generation_prompt_tokens=len(generation_prompt_ids),
            current_response_tokens=current_response_tokens,
            max_model_len=req.max_model_len,
            max_response_len=fragment_response_limit,
            max_new_tokens_per_turn=phase_cap,
            tool_response_token_reserve=self._tool_response_token_reserve,
            template_token_reserve=self._template_token_reserve,
            has_tools=bool(req.tool_schemas),
        )
        req.record_length_event(
            phase,
            context_tokens=len(generation_prompt_ids),
            response_tokens=current_response_tokens,
            generation_prompt_tokens=len(generation_prompt_ids),
            current_response_tokens=current_response_tokens,
            remaining_response_tokens=max(
                0, fragment_response_limit - current_response_tokens
            ),
            remaining_context_tokens=max(0, req.max_model_len - len(generation_prompt_ids)),
            max_new_tokens=max_new_tokens,
            budget_clamped=budget_clamped,
            fragment_prompt_tokens=fragment_prompt_tokens,
            fragment_response_limit=fragment_response_limit,
        )
        return max_new_tokens

    @staticmethod
    def _assert_request_budget(req: AsyncRolloutRequest) -> None:
        context_tokens = len(req.input_ids)
        active_segment = getattr(req, "active_segment", None)
        if active_segment is not None:
            fragment_prompt_tokens = int(
                active_segment.get("start_input", len(req.prompt_ids))
            )
        else:
            fragment_prompt_tokens = len(req.prompt_ids)
        response_tokens = max(0, context_tokens - fragment_prompt_tokens)
        response_limit = min(
            int(req.max_response_len),
            max(0, int(req.max_model_len) - fragment_prompt_tokens),
        )
        if context_tokens > req.max_model_len or response_tokens > response_limit:
            raise RolloutLengthExceededError(
                f"request {req.request_id} exceeded cumulative token budget: "
                f"context={context_tokens}/{req.max_model_len}, "
                f"response={response_tokens}/{req.max_response_len}"
            )

    def _maybe_cleanup_context(
        self,
        req: AsyncRolloutRequest,
        *,
        reason: str,
        force: bool = False,
        require_next_turn: bool = True,
    ) -> dict:
        fallback_reserve = (
            int(getattr(self, "_max_reasoning_tokens_per_turn", 2560))
            + int(getattr(self, "_max_tool_call_tokens_per_turn", 512))
            + int(getattr(self, "_tool_response_token_reserve", 6144))
            + int(getattr(self, "_template_token_reserve", 32))
        )
        return req.maybe_cleanup_context(
            self.tokenizer,
            enabled=getattr(self, "_context_cleanup_enabled", False),
            target_context_tokens=getattr(self, "_context_cleanup_target_tokens", 20000),
            next_turn_reserve=getattr(self, "_next_turn_reserve", fallback_reserve),
            reason=reason,
            force=force,
            require_next_turn=require_next_turn,
        )

    @staticmethod
    def _is_length_exception(exc: Exception) -> bool:
        if isinstance(exc, RolloutLengthExceededError):
            return True
        text = str(exc).lower()
        return any(
            marker in text
            for marker in ("max_model_len", "max_response_len", "max_prompt_len")
        ) and ("exceed" in text or "length" in text)

    async def _release_tools_for_request(self, req: AsyncRolloutRequest) -> None:
        for name, tool_kwargs in req.tools_kwargs.items():
            tool = self._tool_map.get(name)
            if tool is None:
                continue
            try:
                await tool.release(
                    req.request_id,
                    **tool_kwargs.get("release_kwargs", {}),
                )
            except Exception:
                logger.exception(
                    "Failed to release quarantined rollout tool %s for %s",
                    name,
                    req.request_id,
                )

    async def _quarantine_user_metrics(
        self,
        req: AsyncRolloutRequest,
    ) -> dict[str, float]:
        """Capture API usage accumulated before an invalid rollout is released."""
        metric_names = (*TRAVEL_USER_METRIC_NAMES, *TRAVEL_ACTOR_ASPECT_METRIC_NAMES)
        metrics = {name: 0.0 for name in metric_names}
        for name in req.tools_kwargs:
            tool = self._tool_map.get(name)
            getter = getattr(tool, "get_reward_metadata", None)
            if getter is None:
                continue
            try:
                metadata = await getter(req.request_id)
            except Exception:
                logger.exception(
                    "Failed to capture quarantined rollout telemetry from %s for %s",
                    name,
                    req.request_id,
                )
                continue
            if not isinstance(metadata, dict):
                continue
            for metric_name in metric_names:
                try:
                    metrics[metric_name] += float(metadata.get(metric_name, 0.0))
                except (TypeError, ValueError):
                    logger.warning(
                        "Ignoring non-numeric quarantined metric %s from %s for %s",
                        metric_name,
                        name,
                        req.request_id,
                    )
        return metrics

    async def _quarantine_invalid_request(
        self,
        req: AsyncRolloutRequest,
        exc: Exception,
        *,
        phase: str,
        finish_reason: FinishReasonTypeEnum,
    ) -> AsyncRolloutRequest:
        user_metrics = await self._quarantine_user_metrics(req)
        await self._release_tools_for_request(req)
        invalid = deepcopy(req)
        invalid.tools_kwargs = {}
        invalid.state = AsyncRolloutRequestStateEnum.COMPLETED
        invalid.input_ids = list(invalid.prompt_ids)
        invalid.attention_mask = [1] * len(invalid.input_ids)
        invalid.position_ids = compute_position_id_with_mask(
            torch.tensor(invalid.attention_mask)
        ).tolist()
        invalid.loss_mask = [0] * len(invalid.input_ids)
        invalid.response_ids = []
        invalid.response_attention_mask = []
        invalid.response_position_ids = []
        invalid.response_loss_mask = []
        invalid.turn_boundaries = []
        invalid.conversation_histories = []
        # Keep the append-only archive and cleanup diagnostics for export, but
        # do not expose an earlier fragment as a trainable response after the
        # request has been quarantined.  Partial fragments remain in a
        # separate archive field so quarantine does not erase their ledger.
        invalid.archive_segment_records = deepcopy(invalid.segment_records)
        invalid.segment_records = []
        invalid.active_segment = None
        invalid.active_turn = {}
        invalid.metrics = {}
        invalid.reward_scores = {
            "interact_with_env": 0.0,
            "interact_with_env_reward_valid": 0.0,
            "interact_with_env_terminal_only": 1.0,
            **{
                f"interact_with_env_{metric_name}": 0.0
                for metric_name in (
                    *TRAVEL_QUALITY_METRIC_NAMES,
                    *TRAVEL_ACTOR_ASPECT_METRIC_NAMES,
                )
            },
            **{
                f"interact_with_env_{metric_name}": value
                for metric_name, value in user_metrics.items()
            },
        }
        invalid.finish_reason = finish_reason.value
        invalid.length_events = deepcopy(req.length_events)
        invalid.record_length_event(
            phase,
            budget_clamped=finish_reason == FinishReasonTypeEnum.LENGTH,
            invalid=True,
        )
        invalid.length_events[-1].update(
            {
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:240],
            }
        )
        logger.warning(
            "Quarantined invalid rollout request %s during %s: %s",
            req.request_id,
            phase,
            exc,
        )
        return invalid

    async def _quarantine_length_exceeded(
        self,
        req: AsyncRolloutRequest,
        exc: Exception,
    ) -> AsyncRolloutRequest:
        return await self._quarantine_invalid_request(
            req,
            exc,
            phase="length",
            finish_reason=FinishReasonTypeEnum.LENGTH,
        )

    async def _quarantine_template_alignment(
        self,
        req: AsyncRolloutRequest,
        exc: Exception,
    ) -> AsyncRolloutRequest:
        return await self._quarantine_invalid_request(
            req,
            exc,
            phase="template_alignment",
            finish_reason=FinishReasonTypeEnum.STOP,
        )

    async def _quarantine_tool_protocol(
        self,
        req: AsyncRolloutRequest,
        exc: Exception,
    ) -> AsyncRolloutRequest:
        return await self._quarantine_invalid_request(
            req,
            exc,
            phase="tool_protocol",
            finish_reason=FinishReasonTypeEnum.STOP,
        )

    def _validate_tool_calls(self, req: AsyncRolloutRequest, tool_calls) -> None:
        unknown_names = sorted(
            {
                str(tool_call.function.name)
                for tool_call in tool_calls
                if tool_call.function.name not in self._tool_map
                or tool_call.function.name not in req.tools_kwargs
            }
        )
        if unknown_names:
            available_names = sorted(set(self._tool_map) & set(req.tools_kwargs))
            raise RolloutProtocolError(
                f"request {req.request_id} generated unknown tool name(s) "
                f"{unknown_names}; available tools are {available_names}"
            )

    def _parse_tool_call_output(
        self,
        req: AsyncRolloutRequest,
        content: str,
    ) -> list[OpenAIFunctionToolCall]:
        if not self._function_call_parser or not self._function_call_parser.has_tool_call(content):
            raise RolloutProtocolError(
                f"request {req.request_id} did not emit a tool call after </think>"
            )
        try:
            normed_content, tool_calls = self._function_call_parser.parse_non_stream(content)
        except (JSONDecodeError, AttributeError) as exc:
            raise RolloutProtocolError(
                f"request {req.request_id} emitted an invalid tool-call payload"
            ) from exc
        if str(normed_content or "").strip():
            raise RolloutProtocolError(
                f"request {req.request_id} emitted non-tool text after </think>"
            )
        parsed_tool_calls = []
        for tool_call in tool_calls:
            function, has_decode_error = OpenAIFunctionCallSchema.from_openai_function_parsed_schema(
                OpenAIFunctionParsedSchema(
                    name=tool_call.name,
                    arguments=tool_call.parameters,
                )
            )
            if has_decode_error:
                continue
            parsed_tool_calls.append(
                OpenAIFunctionToolCall(
                    id=str(tool_call.tool_index),
                    function=function,
                )
            )
        if not parsed_tool_calls:
            raise RolloutProtocolError(
                f"request {req.request_id} emitted no decodable tool call"
            )
        self._validate_tool_calls(req, parsed_tool_calls)
        return parsed_tool_calls

    @staticmethod
    async def _gather_rollout_requests(coroutines):
        """Let sibling requests settle before propagating an infrastructure error.

        A fail-fast gather can leave SGLang requests running while the sharding
        manager starts releasing inference memory. Waiting for every sibling
        prevents cleanup from racing an active scheduler request.
        """
        results = await asyncio.gather(*coroutines, return_exceptions=True)
        for result in results:
            if isinstance(result, BaseException):
                raise result
        return results

    async def _semaphore_wrapped_rollout(self, sem, req, do_sample, is_validate, **kwargs):
        async with sem:
            try:
                return await self._async_rollout_a_request(
                    req, do_sample, is_validate, **kwargs
                )
            except Exception as exc:
                if isinstance(exc, RolloutTemplateAlignmentError):
                    return await self._quarantine_template_alignment(req, exc)
                if isinstance(exc, RolloutProtocolError):
                    return await self._quarantine_tool_protocol(req, exc)
                if not self._is_length_exception(exc):
                    raise
                return await self._quarantine_length_exceeded(req, exc)

    async def _async_rollout_a_request(
        self,
        req: AsyncRolloutRequest,
        do_sample: bool = True,
        is_validate: bool = False,
        **kwargs,
    ) -> AsyncRolloutRequest:
        assert self._tp_rank == 0, "only the master process can call this function"
        _req = deepcopy(req)
        finish_reason_type = None
        output = None

        turn_boundaries = []
        conversation_histories = []
        
        current_turns = 0
        while current_turns < self.config.multi_turn.max_turns:
            if _req.state == AsyncRolloutRequestStateEnum.PENDING:
                await self._handle_pending_state(_req)
                _req.record_length_event("initial_prompt")
                _req.state = AsyncRolloutRequestStateEnum.RUNNING
            elif _req.state == AsyncRolloutRequestStateEnum.TOOL_CALLING:
                if _req.messages[-1].tool_calls is not None:
                    parsed_tool_calls = _req.messages[-1].tool_calls
                    self._validate_tool_calls(_req, parsed_tool_calls)
                    tool_call_results = await asyncio.gather(
                        *[
                            self._tool_map[tool_call.function.name].execute(
                                _req.request_id,
                                tool_call.function.arguments,
                                current_turns,
                                **_req.tools_kwargs[tool_call.function.name].get("execute_kwargs", {}),
                            )
                            for tool_call in parsed_tool_calls
                        ]
                    )
                    before_tool_response_tokens = len(_req.input_ids)
                    _req.add_tool_response_messages(
                        self.tokenizer,
                        [result[0] for result in tool_call_results],
                        tool_call_ids=[tool_call.id for tool_call in parsed_tool_calls],
                        names=[tool_call.function.name for tool_call in parsed_tool_calls],
                    )
                    _req.record_length_event(
                        "tool_response",
                        tool_response_tokens=len(_req.input_ids) - before_tool_response_tokens,
                        tool_response_count=len(tool_call_results),
                    )
                    overall_stop = False
                    for tool_call, result in zip(parsed_tool_calls, tool_call_results):
                        # Keep compatibility with ordinary three-field tools;
                        # InteractTool additionally returns public control
                        # metadata and a stop flag.
                        _resp, reward, metrics = result[0], result[1], result[-1]
                        is_done = bool(result[2]) if len(result) >= 6 else False
                        choice = result[3] if len(result) >= 6 else tool_call.function.name
                        content = result[4] if len(result) >= 6 else ""
                        _req.update_metrics(metrics, tool_call.function.name)
                        conversation_histories[-1]["choice"] = choice
                        conversation_histories[-1]["reward"] = reward
                        conversation_histories[-1]["content"] = content
                        turn_event = metrics.get("turn_event") if isinstance(metrics, dict) else None
                        if isinstance(turn_event, dict):
                            conversation_histories[-1].setdefault("turn_events", []).append(
                                dict(turn_event)
                            )
                        if is_done:
                            overall_stop = True
                    if self._context_cleanup_enabled:
                        _req.complete_turn(
                            conversation_histories[-1].get("turn_events", []),
                            conversation_histories[-1],
                        )
                    cleanup_result = self._maybe_cleanup_context(
                        _req,
                        reason="after_tool_response",
                        force=len(_req.input_ids) > _req.max_model_len,
                        require_next_turn=not overall_stop,
                    )
                    if cleanup_result.get("attempted"):
                        _req.record_length_event(
                            "context_cleanup",
                            cleanup_success=bool(cleanup_result.get("success", False)),
                            cleanup_released_tokens=int(
                                cleanup_result.get("released_tokens", 0)
                            ),
                            cleanup_after_context_tokens=int(
                                cleanup_result.get("after_context_tokens", len(_req.input_ids))
                            ),
                        )
                    legacy_response_over_limit = (
                        not self._context_cleanup_enabled
                        and len(_req.input_ids) - len(_req.prompt_ids) > _req.max_response_len
                    )
                    if len(_req.input_ids) > _req.max_model_len or legacy_response_over_limit:
                        raise RolloutLengthExceededError(
                            f"request {_req.request_id} exceeded cumulative token budget "
                            f"after tool response: context={len(_req.input_ids)}/"
                            f"{_req.max_model_len}, response="
                            f"{len(_req.input_ids) - len(_req.prompt_ids)}/{_req.max_response_len}"
                        )
                    if overall_stop or (
                        not self._context_cleanup_enabled
                        and len(_req.input_ids) >= self.config.max_model_len
                    ):
                        finish_reason_type = FinishReasonTypeEnum.STOP
                        break
                    _req.state = AsyncRolloutRequestStateEnum.RUNNING
                else:
                    raise ValueError(f"Unexpected tool calling last message state: {_req.messages[-1]}")
            elif _req.state == AsyncRolloutRequestStateEnum.RUNNING:
                # The cleanup check runs before a new generation marker is
                # appended.  It cannot interrupt a half-generated reasoning
                # or tool-call phase.
                cleanup_result = self._maybe_cleanup_context(
                    _req,
                    reason="before_next_turn",
                    require_next_turn=True,
                )
                if cleanup_result.get("attempted"):
                    _req.record_length_event(
                        "context_cleanup",
                        cleanup_success=bool(cleanup_result.get("success", False)),
                        cleanup_released_tokens=int(
                            cleanup_result.get("released_tokens", 0)
                        ),
                        cleanup_after_context_tokens=int(
                            cleanup_result.get(
                                "after_context_tokens", len(_req.input_ids)
                            )
                        ),
                    )
                # The engine call computes the remaining cumulative budget and
                # raises a typed length error when no safe turn can fit.
                turn_boundaries.append(len(_req.input_ids))
                conversation_histories.append({
                    "reward": 0.0,
                    "choice": "action",
                    "content": "",
                    "turn_idx": len(conversation_histories),
                })
                if self._context_cleanup_enabled:
                    _req.begin_turn(current_turns)
                if _req.tool_schemas and self._function_call_parser:
                    reasoning_prompt_ids = _req.get_generation_prompt_ids(self.tokenizer)
                    if self._context_cleanup_enabled:
                        _req.ensure_active_segment(reasoning_prompt_ids)
                    reasoning_output = await self._handle_engine_call(
                        _req,
                        do_sample,
                        is_validate,
                        generation_prompt_ids=reasoning_prompt_ids,
                        max_new_tokens_per_turn=self._max_reasoning_tokens_per_turn,
                        phase="reasoning_generation",
                        sampling_overrides={"stop": "</think>", "no_stop_trim": True},
                        **kwargs,
                    )
                    reasoning_text = str(reasoning_output["text"] or "")
                    reasoning_finish = FinishReasonTypeEnum.from_str(
                        reasoning_output["meta_info"]["finish_reason"]["type"]
                    )
                    _req.record_model_output(
                        "reasoning_generation",
                        reasoning_text,
                        finish_reason=reasoning_finish.value,
                    )
                    if "</think>" in reasoning_text:
                        reasoning_content, trailing_content = reasoning_text.split("</think>", 1)
                        if trailing_content.strip():
                            raise RolloutProtocolError(
                                f"request {_req.request_id} emitted text after the reasoning stop marker"
                            )
                        forced_reasoning_end = False
                    else:
                        reasoning_content = reasoning_text
                        forced_reasoning_end = True
                        _req.record_length_event(
                            "forced_reasoning_end",
                            reasoning_limit_reached=reasoning_finish == FinishReasonTypeEnum.LENGTH,
                        )

                    tool_call_prompt_ids = _req.build_tool_call_prompt_ids(
                        self.tokenizer,
                        reasoning_content,
                    )
                    tool_output = await self._handle_engine_call(
                        _req,
                        do_sample,
                        is_validate,
                        generation_prompt_ids=tool_call_prompt_ids,
                        max_new_tokens_per_turn=self._max_tool_call_tokens_per_turn,
                        phase="tool_call_generation",
                        **kwargs,
                    )
                    tool_content = str(tool_output["text"] or "")
                    tool_finish = FinishReasonTypeEnum.from_str(
                        tool_output["meta_info"]["finish_reason"]["type"]
                    )
                    _req.record_model_output(
                        "tool_call_generation",
                        tool_content,
                        finish_reason=tool_finish.value,
                    )
                    try:
                        parsed_tool_calls = self._parse_tool_call_output(_req, tool_content)
                    except RolloutProtocolError:
                        if tool_finish == FinishReasonTypeEnum.LENGTH:
                            raise RolloutLengthExceededError(
                                f"request {_req.request_id} reached the tool-call generation cap "
                                f"{self._max_tool_call_tokens_per_turn} before completing the protocol"
                            ) from None
                        raise
                    _req.add_assistant_message(
                        self.tokenizer,
                        "",
                        tool_calls=parsed_tool_calls,
                        reasoning_content=reasoning_content,
                        forced_reasoning_end=forced_reasoning_end,
                    )
                    self._assert_request_budget(_req)
                    current_turns += 1
                    finish_reason_type = FinishReasonTypeEnum.TOOL_CALL
                    _req.state = AsyncRolloutRequestStateEnum.TOOL_CALLING
                else:
                    generation_prompt_ids = _req.get_generation_prompt_ids(self.tokenizer)
                    if self._context_cleanup_enabled:
                        _req.ensure_active_segment(generation_prompt_ids)
                    output = await self._handle_engine_call(
                        _req,
                        do_sample,
                        is_validate,
                        generation_prompt_ids=generation_prompt_ids,
                        **kwargs,
                    )
                    content = output["text"]
                    finish_reason_type = FinishReasonTypeEnum.from_str(
                        output["meta_info"]["finish_reason"]["type"]
                    )
                    _req.record_model_output(
                        "generation",
                        str(content or ""),
                        finish_reason=finish_reason_type.value,
                    )
                    current_turns += 1
                    _req.add_assistant_message(self.tokenizer, content)
                    self._assert_request_budget(_req)
                    if finish_reason_type == FinishReasonTypeEnum.LENGTH:
                        raise RolloutLengthExceededError(
                            f"request {_req.request_id} reached the per-turn generation cap "
                            f"{self._max_new_tokens_per_turn}"
                        )
                    if self._function_call_parser and self._function_call_parser.has_tool_call(content):
                        raise RolloutProtocolError(
                            f"request {_req.request_id} emitted a tool call without a tool schema"
                        )
                    if self._context_cleanup_enabled:
                        _req.complete_turn([], conversation_histories[-1])
                    break

        if current_turns >= self.config.multi_turn.max_turns:
            finish_reason_type = FinishReasonTypeEnum.STOP

        # Calculate the reward for each tool
        async def calc_reward_and_release_fn(name: str, tool: BaseTool):
            reward = await tool.calc_reward(_req.request_id, **_req.tools_kwargs[name].get("calc_reward_kwargs", {}))
            metadata = {}
            if hasattr(tool, "get_reward_metadata"):
                metadata = await tool.get_reward_metadata(_req.request_id)
            await tool.release(_req.request_id, **_req.tools_kwargs[name].get("release_kwargs", {}))
            return name, reward, metadata

        tool_reward_tasks = []
        for name in _req.tools_kwargs.keys():
            tool = self._tool_map[name]
            tool_reward_tasks.append(calc_reward_and_release_fn(name, tool))
        tool_reward_results = await asyncio.gather(*tool_reward_tasks)
        tool_reward_scores = {}
        for name, reward, metadata in tool_reward_results:
            try:
                tool_reward_scores[name] = float(reward)
            except (TypeError, ValueError):
                tool_reward_scores[name] = reward
            if metadata:
                tool_reward_scores[f"{name}_reward_valid"] = float(bool(metadata.get("reward_valid", False)))
                tool_reward_scores[f"{name}_terminal_only"] = float(bool(metadata.get("terminal_only", False)))
                for metric_name in TRAVEL_REWARD_METRIC_NAMES:
                    if metric_name in metadata:
                        tool_reward_scores[f"{name}_{metric_name}"] = float(metadata[metric_name])
        _req.finalize(
            self.tokenizer,
            tool_reward_scores,
            turn_boundaries,
            conversation_histories,
            finish_reason_type or FinishReasonTypeEnum.STOP,
        )

        return _req

    async def _handle_engine_call(
        self,
        _req: AsyncRolloutRequest,
        do_sample: bool,
        is_validate: bool,
        override_n: bool = True,
        generation_prompt_ids: list[int] | None = None,
        max_new_tokens_per_turn: int | None = None,
        phase: str = "generation",
        sampling_overrides: dict | None = None,
        **kwargs,
    ) -> dict:
        if generation_prompt_ids is None:
            generation_prompt_ids = _req.get_generation_prompt_ids(self.tokenizer)
        else:
            generation_prompt_ids = list(generation_prompt_ids)
        max_new_tokens = self._compute_generation_budget(
            _req,
            generation_prompt_ids,
            max_new_tokens_per_turn=max_new_tokens_per_turn,
            phase=phase,
        )
        if not do_sample:
            kwargs = dict(
                n=1,
                presence_penalty=0.0,
                frequency_penalty=0.0,
                repetition_penalty=1.0,
                temperature=0,
                top_p=1,
                top_k=-1,
                ignore_eos=False,
                min_new_tokens=0,
                max_new_tokens=max_new_tokens,
                skip_special_tokens=True,
                spaces_between_special_tokens=True,
            )
        elif is_validate:
            # TODO: try **
            kwargs = {
                "top_k": self.config.val_kwargs.top_k,
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "n": 1,  # if validate, already repeat in ray_trainer
            }
        else:
            # In training, make rollout temperature 1.0, encourage more diverse responses
            kwargs = {
                "top_k": -1,
                "top_p": 1,
                "temperature": 1.0,
                "n": 1,
            }
        kwargs["max_new_tokens"] = max_new_tokens
        if sampling_overrides:
            kwargs.update(sampling_overrides)
        if "n" not in kwargs or (kwargs["n"] > 1 and override_n):  # group size is supported in preprocess
            kwargs["n"] = 1
        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            output = await self._engine.async_generate(
                input_ids=generation_prompt_ids,
                sampling_params=self.sampling_params,
                return_logprob=False,
            )
        return output

    async def _extract_actor_aspects(self, _req: AsyncRolloutRequest):
        """Run one inference-only aspect planning call before environment creation."""

        from travelgym.env.actor_aspects import (
            ActorAspectExtractionResult,
            build_actor_aspect_messages,
            first_user_requirement,
            new_actor_aspect_telemetry,
            parse_actor_aspect_response,
        )

        telemetry = new_actor_aspect_telemetry()
        messages = build_actor_aspect_messages(
            first_user_requirement(_req.messages)
        )
        telemetry["actor_aspect_extraction_calls"] = 1
        started = time.perf_counter()
        try:
            try:
                prompt_ids = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                prompt_ids = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                )
            if isinstance(prompt_ids, dict):
                prompt_ids = prompt_ids.get("input_ids", [])
            if hasattr(prompt_ids, "tolist"):
                prompt_ids = prompt_ids.tolist()
            if prompt_ids and isinstance(prompt_ids[0], (list, tuple)):
                prompt_ids = prompt_ids[0]
            prompt_ids = [int(token_id) for token_id in prompt_ids]
            telemetry["actor_aspect_extraction_prompt_tokens"] = len(prompt_ids)
            with self.update_sampling_params(
                n=1,
                top_k=-1,
                top_p=1.0,
                temperature=0.0,
                max_new_tokens=getattr(
                    self, "_actor_aspect_extraction_max_tokens", 128
                ),
                skip_special_tokens=True,
                spaces_between_special_tokens=True,
            ):
                output = await self._engine.async_generate(
                    input_ids=prompt_ids,
                    sampling_params=self.sampling_params,
                    return_logprob=False,
                )
            meta_info = output.get("meta_info", {}) if isinstance(output, dict) else {}
            if isinstance(meta_info, dict):
                completion_tokens = meta_info.get("completion_tokens", 0) or 0
                total_tokens = meta_info.get("total_tokens", 0) or 0
            else:
                completion_tokens = 0
                total_tokens = 0
            if not completion_tokens and isinstance(output, dict):
                output_ids = output.get("output_ids", []) or []
                completion_tokens = len(output_ids)
            if not total_tokens:
                total_tokens = len(prompt_ids) + int(completion_tokens)
            telemetry["actor_aspect_extraction_completion_tokens"] = int(completion_tokens)
            telemetry["actor_aspect_extraction_total_tokens"] = int(total_tokens)
            telemetry["actor_aspect_extraction_wall_time_seconds"] = (
                time.perf_counter() - started
            )
            text = output.get("text", "") if isinstance(output, dict) else ""
            return parse_actor_aspect_response(text).with_telemetry(telemetry)
        except Exception:
            telemetry["actor_aspect_extraction_errors"] = 1
            telemetry["actor_aspect_extraction_wall_time_seconds"] = (
                time.perf_counter() - started
            )
            return ActorAspectExtractionResult(
                format_error="actor_call_failed",
            ).with_telemetry(telemetry)

    async def _handle_pending_state(self, _req: AsyncRolloutRequest) -> AsyncRolloutRequest:
        if _req.tool_schemas is not None:
            extraction_enabled = bool(
                getattr(self, "_enable_actor_aspect_extraction", False)
            )
            actor_aspect_result = None
            if extraction_enabled:
                actor_aspect_result = await self._extract_actor_aspects(_req)
            tool_creation_coroutines = []
            created_tools = []
            for tool_schema in _req.tool_schemas:
                tool = self._tool_map[tool_schema.function.name]
                create_kwargs = dict(
                    _req.tools_kwargs[tool.name].get("create_kwargs", {})
                )
                create_kwargs["max_turns"] = self.config["multi_turn"]["max_turns"]
                create_kwargs["model_name"] = self.config["multi_turn"]["model_name"]
                if tool.name == "interact_with_env":
                    create_kwargs["enable_actor_aspect_extraction"] = (
                        extraction_enabled
                    )
                    if actor_aspect_result is not None:
                        create_kwargs["actor_aspect_result"] = actor_aspect_result.to_dict()
                tool_creation_coroutines.append(tool.create(_req.request_id, **create_kwargs))
                created_tools.append(tool)
            await asyncio.gather(*tool_creation_coroutines)
            initial_contexts = []
            for tool in created_tools:
                getter = getattr(tool, "get_initial_context", None)
                if getter is None:
                    continue
                context = str(getter(_req.request_id)).strip()
                if context:
                    initial_contexts.append(context)
            if initial_contexts:
                _req.inject_initial_user_context(
                    self.tokenizer,
                    "\n\n".join(initial_contexts),
                )

    @GPUMemoryLogger(role="sglang rollout", logger=logger)
    @torch.no_grad()
    def generate_sequences_with_tools(self, prompts: DataProto, **kwargs) -> DataProto:
        logger.warning(
            "`generate_sequences_with_tools` is deprecated, please use `generate_sequences(...)`",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._req_level_generate_sequences(prompts, **kwargs)
    
    @GPUMemoryLogger(role="sglang rollout", logger=logger)
    @torch.no_grad()
    def _req_level_generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # Async rollout with tools support
        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        tgt_device = prompts.batch["input_ids"].device

        sorted_output_req_list = None
        rollout_error = None

        try:
            if self._tp_rank == 0:

                print(f"[Rank {self._tp_rank}, {self._rank}] Starting async rollout with prompts, number: {len(prompts)}")

                req_list = self._preprocess_prompt_to_async_rollout_requests(
                    prompts,
                    n=1 if is_validate else self.config.n,
                )

                print(f"[Rank {self._rank}] Preprocessed {len(req_list)} requests for async rollout")

                sem = asyncio.Semaphore(128)  # Limit the concurrency of async requests to ensure the stability of the system
                loop = asyncio.get_event_loop()
                output_req_list = loop.run_until_complete(
                    self._gather_rollout_requests(
                        [
                            self._semaphore_wrapped_rollout(sem, req, do_sample, is_validate, **kwargs)
                            for req in req_list
                        ]
                    )
                )
                
                sorted_output_req_list = sorted(
                    output_req_list, key=lambda x: (x.batch_data_id, x.rollout_offset)
                )
        
        except Exception as e:
            logger.exception(f"[Rank {self._rank}] Rollout failed: {e}")
            rollout_error = f"{type(e).__name__}: {e}"
            sorted_output_req_list = None

        # Broadcast the failure as data first so every TP rank exits together
        # instead of turning a rank-0 error into a later NoneType exception.
        rollout_error, sorted_output_req_list = broadcast_pyobj(
            data=[rollout_error, sorted_output_req_list],
            rank=self._rank,
            dist_group=self._device_mesh_cpu["tp"].get_group(),
            src=self._device_mesh_cpu["tp"].mesh[0].item(),
            force_cpu_device=False,
        )
        if rollout_error is not None:
            raise RuntimeError(f"SGLang rollout failed on the source rank: {rollout_error}")
        # Construct the batch data
        prompt_ids, response_ids = [], []
        prompt_attention_mask, response_attention_mask = [], []
        prompt_position_ids, response_position_ids = [], []
        prompt_loss_mask, response_loss_mask = [], []
        messages = []
        conversation_histories = []  # Add conversation histories for new algorithm
        reward_scores = []
        turn_boundaries_list = []
        length_summaries = []
        segment_records_list = []
        cleanup_events_list = []
        archive_messages_list = []
        archive_model_outputs_list = []
        archive_segment_records_list = []
        archive_turns_list = []
        length_events_list = []
        for req in sorted_output_req_list:
            assert req.state == AsyncRolloutRequestStateEnum.COMPLETED, f"Request {req.request_id} is not completed"
            assert len(req.input_ids) == len(req.attention_mask) == len(req.position_ids) == len(req.loss_mask), f"""Request {req.request_id} has different length of 
                {len(req.input_ids)=}, {len(req.attention_mask)=}, {len(req.position_ids)=}, {len(req.loss_mask)=}"""
            error_message_lines = [
                f"""Request {req.request_id} has input_ids length {len(req.input_ids)}
                    greater than max_model_len {self.config.max_model_len}""",
                f"Decoded input_ids: {self.tokenizer.decode(req.input_ids)}",
                f"Decoded prompt_ids: {self.tokenizer.decode(req.prompt_ids)}",
                f"Decoded response_ids: {self.tokenizer.decode(req.response_ids)}",
                f"Messages: {req.messages}",
                f"Max model length: {req.max_model_len}",
            ]
            error_message = "\n".join(error_message_lines)
            assert len(req.input_ids) <= self.config.max_model_len, error_message

            prompt_ids.append(torch.tensor(req.prompt_ids, dtype=torch.int, device=tgt_device))
            if len(req.response_ids) > self.config.response_length:
                raise RuntimeError(
                    f"Request {req.request_id} produced {len(req.response_ids)} response tokens, "
                    f"exceeding configured response_length={self.config.response_length}; "
                    "the trajectory is rejected instead of being silently padded or truncated"
                )
            response_ids.append(torch.tensor(req.response_ids, dtype=torch.int, device=tgt_device))
            prompt_attention_mask.append(torch.tensor(req.prompt_attention_mask, dtype=torch.int, device=tgt_device))
            response_attention_mask.append(torch.tensor(req.response_attention_mask, dtype=torch.int, device=tgt_device))
            prompt_position_ids.append(torch.tensor(req.prompt_position_ids, dtype=torch.int, device=tgt_device))
            response_position_ids.append(torch.tensor(req.response_position_ids, dtype=torch.int, device=tgt_device))
            prompt_loss_mask.append(torch.tensor(req.prompt_loss_mask, dtype=torch.int, device=tgt_device))
            response_loss_mask.append(torch.tensor(req.response_loss_mask, dtype=torch.int, device=tgt_device))
            if self._context_cleanup_enabled:
                messages.append(
                    {
                        # Keep the historical key for existing consumers, and
                        # add explicit archive/active views so an export does
                        # not look like one uninterrupted prompt after
                        # compaction.
                        "messages": req.messages,
                        "archive_messages": req._archive_message_dicts(),
                        "archive_model_outputs": req.archive_model_outputs,
                        "archive_segment_records": req.archive_segment_records,
                        "active_messages": req._message_dicts(),
                    }
                )
            else:
                messages.append({"messages": req.messages})
            conversation_histories.append(req.conversation_histories)
            reward_scores.append(req.reward_scores)
            segment_records_list.append(req.segment_records)
            cleanup_events_list.append(req.cleanup_events)
            archive_messages_list.append(req._archive_message_dicts())
            archive_model_outputs_list.append(req.archive_model_outputs)
            archive_segment_records_list.append(req.archive_segment_records)
            archive_turns_list.append(req.archive_turns)
            length_events_list.append(req.length_events)
            length_summaries.append(
                req.length_summary(finish_reason=req.finish_reason)
            )

            # Convert turn boundaries to tensor format.  Segmented rows have
            # their own local boundaries; the legacy request-level boundaries
            # are in the pre-compaction input coordinate system and cannot be
            # projected onto the last fragment.
            response_length = len(req.response_ids)
            turn_boundary_tensor = torch.zeros(response_length, dtype=torch.int, device=tgt_device)
            if req.segment_records:
                local_boundaries = req.segment_records[-1].get("turn_boundaries", [])
                for boundary_pos in local_boundaries:
                    boundary_pos = int(boundary_pos)
                    if 0 <= boundary_pos < response_length:
                        turn_boundary_tensor[boundary_pos] = 1
                if response_length and not bool(turn_boundary_tensor.any()):
                    turn_boundary_tensor[0] = 1
            elif hasattr(req, 'turn_boundaries') and req.turn_boundaries:
                prompt_length = len(req.prompt_ids)
                # Convert turn boundaries from input_ids space to response_ids space
                for boundary_pos in req.turn_boundaries:
                    # Turn boundaries are positions in input_ids, convert to response_ids positions
                    response_pos = boundary_pos - prompt_length
                    if 0 <= response_pos < response_length:
                        turn_boundary_tensor[response_pos] = 1
                # Ensure first turn always starts at position 0 if no boundary set
                if response_length:
                    assert turn_boundary_tensor[0] == 1, f"First turn boundary should be at position 0, but got {turn_boundary_tensor[0]}"
            turn_boundaries_list.append(turn_boundary_tensor)

        prompt_ids = pad_sequence(
            prompt_ids,
            batch_first=True,
            padding_value=self.pad_token_id,
            padding_side="left",
        )
        if prompt_ids.shape[1] < self.config.prompt_length:
            prompt_ids = pad_sequence_to_length(prompt_ids, self.config.prompt_length, self.pad_token_id, left_pad=True)
        response_ids = pad_sequence(response_ids, batch_first=True, padding_value=self.pad_token_id)
        if response_ids.shape[1] < self.config.response_length:
            response_ids = pad_sequence_to_length(response_ids, self.config.response_length, self.pad_token_id)
        prompt_attention_mask = pad_sequence(
            prompt_attention_mask,
            batch_first=True,
            padding_value=0,
            padding_side="left",
        )
        if prompt_attention_mask.shape[1] < self.config.prompt_length:
            prompt_attention_mask = pad_sequence_to_length(prompt_attention_mask, self.config.prompt_length, 0, left_pad=True)
        response_attention_mask = pad_sequence(response_attention_mask, batch_first=True, padding_value=0)
        if response_attention_mask.shape[1] < self.config.response_length:
            response_attention_mask = pad_sequence_to_length(response_attention_mask, self.config.response_length, 0)
        prompt_position_ids = pad_sequence(prompt_position_ids, batch_first=True, padding_value=0, padding_side="left")
        if prompt_position_ids.shape[1] < self.config.prompt_length:
            prompt_position_ids = pad_sequence_to_length(prompt_position_ids, self.config.prompt_length, 0, left_pad=True)
        response_length = response_ids.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=response_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(len(sorted_output_req_list), 1)
        response_position_ids = prompt_position_ids[:, -1:] + delta_position_id
        prompt_loss_mask = pad_sequence(prompt_loss_mask, batch_first=True, padding_value=0, padding_side="left")
        if prompt_loss_mask.shape[1] < self.config.prompt_length:
            prompt_loss_mask = pad_sequence_to_length(prompt_loss_mask, self.config.prompt_length, 0, left_pad=True)
        response_loss_mask = pad_sequence(response_loss_mask, batch_first=True, padding_value=0)
        if response_loss_mask.shape[1] < self.config.response_length:
            response_loss_mask = pad_sequence_to_length(response_loss_mask, self.config.response_length, 0)

        # Pad turn boundaries to match response sequence length only
        turn_boundaries = pad_sequence(turn_boundaries_list, batch_first=True, padding_value=0)
        if turn_boundaries.shape[1] < self.config.response_length:
            turn_boundaries = pad_sequence_to_length(turn_boundaries, self.config.response_length, 0)

        input_ids = torch.cat((prompt_ids, response_ids), dim=-1)
        attention_mask = torch.cat((prompt_attention_mask, response_attention_mask), dim=-1)
        position_ids = torch.cat((prompt_position_ids, response_position_ids), dim=-1)
        loss_mask = torch.cat((prompt_loss_mask, response_loss_mask), dim=-1)

        # Construct the batch data
        batch = TensorDict(
            {
                "prompts": prompt_ids,
                "responses": response_ids,
                "input_ids": input_ids,  # here input_ids become the whole sentences
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "loss_mask": loss_mask,
                "turn_boundaries": turn_boundaries,  # Add turn boundaries to batch
            },
            batch_size=len(sorted_output_req_list),
        )

        # free cache engine
        if self.config.free_cache_engine and self._engine is not None and self._tp_rank == 0:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(self._engine.flush_cache()) 

        length_monitor = {}
        if length_summaries:
            for key in length_summaries[0]:
                values = [float(summary[key]) for summary in length_summaries]
                length_monitor[f"mean_{key}"] = float(np.mean(values))
                length_monitor[f"max_{key}"] = float(np.max(values))
            length_monitor["invalid_rollouts"] = float(
                sum(summary.get("invalid_rollout", 0.0) for summary in length_summaries)
            )
        non_tensor_batch = {
            "messages": make_1d_object_array(messages),
            # Keep one complete history as one object per task row.  A
            # nested np.array construction changes rank when a retry
            # contains one row with a uniform history length.
            "conversation_histories": make_1d_object_array(conversation_histories),
            "reward_scores": make_1d_object_array(reward_scores),
        }
        if self._context_cleanup_enabled:
            non_tensor_batch.update(
                {
                    "segment_records": make_1d_object_array(segment_records_list),
                    "cleanup_events": make_1d_object_array(cleanup_events_list),
                    "archive_messages": make_1d_object_array(archive_messages_list),
                    "archive_model_outputs": make_1d_object_array(archive_model_outputs_list),
                    "archive_segment_records": make_1d_object_array(archive_segment_records_list),
                    "archive_turns": make_1d_object_array(archive_turns_list),
                    "length_events": make_1d_object_array(length_events_list),
                }
            )
        return DataProto(
            batch=batch,
            non_tensor_batch=non_tensor_batch,
            meta_info={
                "travel_rollout_length": length_monitor,
                "travel_context_cleanup_enabled": bool(self._context_cleanup_enabled),
            },
        )

    def _preprocess_prompt_to_async_rollout_requests(self, prompts: DataProto, n: int) -> list[AsyncRolloutRequest]:
        assert "raw_prompt" in prompts.non_tensor_batch, "need data.return_raw_chat=True, due to no official way do parse_messages"
        req_list = []
        for data_idx, raw_prompt in enumerate(prompts.non_tensor_batch["raw_prompt"]):
            for rollout_offset in range(n):
                if self._tool_schemas:
                    _tools_kwargs = prompts.non_tensor_batch["tools_kwargs"][data_idx]
                    _tool_schemas = [self._tool_map[k].get_openai_tool_schema() for k in _tools_kwargs.keys()]
                    _input_ids = None
                    _attention_mask = None
                    
                    # Pass max_turns to tool creation (environment will be created in tool.create())
                    for tool_name in _tools_kwargs.keys():
                        if "create_kwargs" not in _tools_kwargs[tool_name]:
                            _tools_kwargs[tool_name]["create_kwargs"] = {}
                        _tools_kwargs[tool_name]["create_kwargs"]["max_turns"] = self.config["multi_turn"]["max_turns"]
                else:
                    _input_ids = _pre_process_inputs(self.pad_token_id, prompts.batch["input_ids"][data_idx])
                    _attention_mask = _pre_process_inputs(0, prompts.batch["attention_mask"][data_idx])
                    _tools_kwargs = {}
                    _tool_schemas = None

                response_token_buffer = int(self.config.multi_turn.get("response_token_buffer", 0) or 0)
                req = AsyncRolloutRequest(
                    batch_data_id=data_idx,
                    rollout_offset=rollout_offset,
                    request_id=str(uuid4()),
                    state=AsyncRolloutRequestStateEnum.PENDING,
                    messages=raw_prompt.tolist(),
                    tool_schemas=_tool_schemas,
                    tools_kwargs=_tools_kwargs,
                    input_ids=_input_ids,
                    response_ids=[],
                    attention_mask=_attention_mask,
                    response_attention_mask=[],
                    response_position_ids=[],
                    response_loss_mask=[],
                    reward_scores={},
                    max_prompt_len=self.config.prompt_length,
                    max_response_len=self.config.response_length + response_token_buffer,
                    max_model_len=min(self.config.max_model_len, self.config.prompt_length + self.config.response_length),
                    use_inference_chat_template=self.config.multi_turn.use_inference_chat_template,
                    enable_tokenization_sanity_check=self.config.multi_turn.enable_tokenization_sanity_check,
                    tokenizer=self.tokenizer,
                )

                error_message = f"Request {req.request_id} has mismatched lengths: input_ids={len(req.input_ids)}, attention_mask={len(req.attention_mask)}, position_ids={len(req.position_ids)}, loss_mask={len(req.loss_mask)}"
                assert len(req.input_ids) == len(req.attention_mask) == len(req.position_ids) == len(req.loss_mask), error_message

                req_list.append(req)

        return req_list

    async def chat_completion(self, json_request):
        assert self._tp_rank == 0, "only called in tp rank 0"
        _input_ids = []
        _attention_mask = []
        _position_ids = []
        _tool_schemas = []
        _tools_kwargs = {}

        req = AsyncRolloutRequest(
            request_id=str(uuid4()),
            state=AsyncRolloutRequestStateEnum.PENDING,
            messages=[Message.model_validate(msg) for msg in json_request["messages"]],
            tools=_tool_schemas,
            tools_kwargs=_tools_kwargs,
            max_prompt_len=self.config.prompt_length,
            input_ids=_input_ids,
            prompt_ids=_input_ids,
            response_ids=[],
            attention_mask=_attention_mask,
            prompt_attention_mask=_attention_mask,
            response_attention_mask=[],
            position_ids=_position_ids,
            prompt_position_ids=_position_ids,
            response_position_ids=[],
            loss_mask=[0] * len(_input_ids),
            prompt_loss_mask=[0] * len(_input_ids),
            response_loss_mask=[],
            reward_scores={},
            max_response_len=self.config.response_length,
            max_model_len=min(self.config.max_model_len, self.config.prompt_length + self.config.response_length),
            tokenizer=self.tokenizer,
        )

        # json_request already contains sampling_params
        output = await self._handle_engine_call(req, True, False, False, **json_request)
        # it can be Dict or AsyncIterator[Dict]
        if isinstance(output, dict):
            outputs = [output]
        else:
            outputs = output

        # build openai chat completion format
        choices = []
        id = None
        for i, content in enumerate(outputs):
            choices.append(
                {
                    "index": i,
                    "message": {
                        "role": "assistant",
                        "content": content["text"],
                    },
                    "finish_reason": content["meta_info"]["finish_reason"]["type"],
                }
            )
            id = content["meta_info"]["id"]

        return {
            "id": "chatcmpl-" + id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": json_request.get("model", "sglang_model"),
            "choices": choices,
        }

        # this function is left for uniform train-inference resharding

    async def wake_up(self):
        if not self.is_sleep:
            return
        await self.sharding_manager.wake_up()  # pylint: disable=C2801
        self.is_sleep = False

    # this function is left for uniform train-inference resharding
    async def sleep(self):
        if self.is_sleep:
            return
        await self.sharding_manager.sleep()
        self.is_sleep = True
