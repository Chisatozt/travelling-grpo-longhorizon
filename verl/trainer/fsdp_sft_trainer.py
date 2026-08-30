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
A lightweight one-file FSDP SFT Trainer
TODO(zhangchi.usc1992)
- Add calculation of mfu
- Add validation
"""

import os

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import logging
import re
import json
import math
from contextlib import nullcontext

import hydra
import torch
import torch.distributed
from peft import LoraConfig, TaskType, get_peft_model
from tensordict import TensorDict
from torch import nn, optim
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import CPUOffload, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel

import verl.utils.hdfs_io as hdfs_io
from verl.utils.dataset import SFTDataset
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.device import get_device_name, get_torch_device, is_cuda_available, is_npu_available
from verl.utils.distributed import destroy_global_process_group, initialize_global_process_group
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    apply_fsdp2,
    fsdp2_load_full_state_dict,
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    init_fn,
    fsdp2_clip_grad_norm_
)
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import get_cosine_schedule_with_warmup, get_wsd_schedule_with_warmup
from verl.utils.py_functional import convert_to_regular_types
from verl.utils.tracking import Tracking
from verl.utils.ulysses import (
    gather_outpus_and_unpad,
    get_ulysses_sequence_parallel_world_size,
    ulysses_pad_and_slice_inputs,
)
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_SFT_LOGGING_LEVEL", "WARN"))


def extract_step(path):
    match = re.search(r"global_step_(\d+)", path)
    if match:
        return int(match.group(1))
    return None


class FSDPSFTTrainer:
    def __init__(self, config, device_mesh: DeviceMesh, ulysses_device_mesh: DeviceMesh, tokenizer, train_dataset: Dataset, val_dataset: Dataset):
        self.config = config
        self.device_mesh = device_mesh
        self.ulysses_device_mesh = ulysses_device_mesh
        self.sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)
        self.tokenizer = tokenizer
        if self.config.data.chat_template is not None:
            raise ValueError("Apply Chat template from config is not supported yet.")

        # normalize dp size
        self._normalize_config_bsz()

        # Set sequence parallel size
        self.config.ulysses_sequence_parallel_size = getattr(self.config, "ulysses_sequence_parallel_size", 1)
        self.use_remove_padding = getattr(self.config, "use_remove_padding", False)
        if self.device_mesh.get_rank() == 0:
            print(f"Using sequence parallel size: {self.config.ulysses_sequence_parallel_size}")
            print(f"Using remove padding: {self.use_remove_padding}")

        self._build_dataloader(train_dataset, val_dataset)
        # build model
        self._build_model_optimizer()

        # TODO: add checkpoint manager
        if self.device_mesh.get_rank() == 0:
            print(self.config)
        self.device_name = get_device_name()

    def _normalize_config_bsz(self):
        dp_size = self.device_mesh.size(0) if not self.ulysses_device_mesh else self.ulysses_device_mesh.size(0)
        if self.device_mesh.get_rank() == 0:
            print(f"Normalize batch size by dp {dp_size}")

        assert self.config.data.train_batch_size % dp_size == 0, f"Global batch size {self.config.data.train_batch_size} is not divisible by dp size {dp_size}"

        self.config.data.train_batch_size //= dp_size

        assert self.config.data.train_batch_size % self.config.data.micro_batch_size_per_gpu == 0

    def _build_dataloader(self, train_dataset, val_dataset):
        # build dataset
        config = self.config
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        # build dataloader
        # Use data parallel rank and size instead of global rank and world size

        # If doing SP, we need to use the local rank and size
        if self.config.ulysses_sequence_parallel_size > 1:
            rank = self.ulysses_device_mesh.get_local_rank("dp")
            world_size = self.ulysses_device_mesh.size(0)
            if self.ulysses_device_mesh.get_rank() == 0:
                print(f"Using SP rank {rank} and size {world_size} for data distribution")
                print("Each SP rank gets different data, but the same data WITHIN the same rank")
        else:
            rank = self.device_mesh.get_rank()
            world_size = self.device_mesh.size()
        if self.device_mesh.get_rank() == 0:
            print(f"Using FSDP rank {rank} and size {world_size} for data distribution")

        # Keep every trajectory, including a final partial batch.  Dropping
        # the tail would make the exact task-group split and sample weights
        # depend on world size.
        self.train_sampler = DistributedSampler(self.train_dataset, shuffle=True, num_replicas=world_size, rank=rank, drop_last=False)
        self.train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=config.data.train_batch_size,
            sampler=self.train_sampler,
            num_workers=8,
            pin_memory=True,
            drop_last=False,
        )

        self.val_sampler = DistributedSampler(self.val_dataset, shuffle=False, num_replicas=world_size, rank=rank, drop_last=False)
        self.val_dataloader = DataLoader(
            dataset=self.val_dataset,
            batch_size=config.data.micro_batch_size_per_gpu,
            sampler=self.val_sampler,
            num_workers=8,
            pin_memory=True,
            drop_last=False,
        )

    def _build_model_optimizer(self):
        # TODO (zhangchi.usc1992):
        # 1. support pretrain from random weights
        # 2. support init directly from sharded weights
        local_model_path = copy_to_local(src=self.config.model.partial_pretrain, verbose=True)

        if self.config.model.get("external_lib", None) is not None:
            # This is used to import external_lib into the huggingface systems
            import importlib

            importlib.import_module(self.config.model.external_lib)

        log_gpu_memory_usage("Before model allocation", logger=logger)

        trust_remote_code = self.config.model.trust_remote_code
        torch_dtype = self.config.model.fsdp_config.get("model_dtype", "fp32")
        torch_dtype = PrecisionType.to_dtype(torch_dtype)
        # load config first
        config = AutoConfig.from_pretrained(local_model_path, trust_remote_code=trust_remote_code)
        self.model_config = config
        if self.config.ulysses_sequence_parallel_size > 1:
            assert self.use_remove_padding, "Sequence parallel is only supported when remove_padding is enabled"

        # This may be very large
        init_context = get_init_weight_context_manager(use_meta_tensor=not config.tie_word_embeddings, mesh=self.device_mesh)

        with init_context():
            self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
                local_model_path,
                config=config,
                torch_dtype=torch_dtype,
                attn_implementation="flash_attention_2",
                trust_remote_code=trust_remote_code,
            )

            if self.use_remove_padding or self.config.ulysses_sequence_parallel_size > 1:
                from verl.models.transformers.monkey_patch import apply_monkey_patch

                apply_monkey_patch(model=self.model, ulysses_sp_size=self.config.ulysses_sequence_parallel_size)

            # Apply Liger kernel if use_liger is enabled
            if self.config.model.get("use_liger", False):
                from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance

                _apply_liger_kernel_to_instance(model=self.model)

            if self.config.model.get("lora_rank", 0) > 0:
                self.model.enable_input_require_grads()
                # Convert config to regular Python types before creating PEFT model
                lora_config = {
                    "task_type": TaskType.CAUSAL_LM,
                    "r": self.config.model.lora_rank,
                    "lora_alpha": self.config.model.lora_alpha,
                    "target_modules": convert_to_regular_types(self.config.model.target_modules),
                    "bias": "none",
                }
                self.model = get_peft_model(self.model, LoraConfig(**lora_config))

        if self.config.model.enable_gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        log_gpu_memory_usage("After model allocation", logger=logger)

        mixed_precision = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32)

        auto_wrap_policy = get_fsdp_wrap_policy(
            self.model,
            config=self.config.model.fsdp_config.wrap_policy,
            is_lora=self.config.model.get("lora_rank", 0) > 0,
        )
        if self.device_mesh.get_rank() == 0:
            print(auto_wrap_policy)

        if not self.config.model.fsdp_config.cpu_offload:
            cpu_offload = None
        else:
            cpu_offload = CPUOffload(offload_params=self.config.model.fsdp_config.offload_params)

        fsdp_strategy = self.config.model.strategy
        if fsdp_strategy == "fsdp":
            self.fsdp_model = FSDP(
                self.model,
                cpu_offload=cpu_offload,
                param_init_fn=init_fn,
                use_orig_params=False,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_torch_device().current_device(),
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                mixed_precision=mixed_precision,
                sync_module_states=True,
                device_mesh=self.device_mesh,
                forward_prefetch=False,
            )
        elif fsdp_strategy == "fsdp2":
            assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
            mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                                             cast_forward_inputs=True)

            fsdp_kwargs = {
                "mesh": self.device_mesh,
                "mp_policy": mp_policy,
                "offload_policy": cpu_offload,
                "reshard_after_forward": True,
            }
            full_state = self.model.state_dict()
            apply_fsdp2(self.model, fsdp_kwargs, self.config.model.fsdp_config)
            fsdp2_load_full_state_dict(self.model, full_state, self.device_mesh, cpu_offload)
            self.fsdp_model = self.model
        else:
            raise NotImplementedError(f"not implement {fsdp_strategy}")

        log_gpu_memory_usage("After FSDP wrapping", logger=logger)

        self.optimizer = optim.AdamW(
            self.fsdp_model.parameters(),
            lr=self.config.optim.lr,
            betas=self.config.optim.betas,
            weight_decay=self.config.optim.weight_decay,
        )

        log_gpu_memory_usage("After initialize optimizer", logger=logger)

        self.steps_per_epoch = len(self.train_dataloader)
        self.total_steps = self.steps_per_epoch * self.config.trainer.total_epochs

        if self.device_mesh.get_rank() == 0:
            print(f"Number of steps/epoch {self.steps_per_epoch}, number of epochs {self.config.trainer.total_epochs}, total number of steps {self.total_steps}")

        num_warmup_steps = int(self.total_steps * self.config.optim.warmup_steps_ratio)

        if not hasattr(self.config.optim, "lr_scheduler") or self.config.optim.lr_scheduler == "cosine":
            self.lr_scheduler = get_cosine_schedule_with_warmup(optimizer=self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=self.total_steps)
        elif self.config.optim.lr_scheduler == "wsd":
            self.lr_scheduler = get_wsd_schedule_with_warmup(optimizer=self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=self.total_steps)
        else:
            raise ValueError(f"Unknown lr scheduler: {self.config.optim.lr_scheduler}")

    def _compute_loss_and_backward(self, batch, do_backward=True, normalization_denominator=None):
        """Compute loss with optional sequence parallelism and remove padding features"""
        use_sp = self.use_remove_padding and self.config.ulysses_sequence_parallel_size > 1

        # Move inputs to GPU and prepare loss mask
        input_ids = batch["input_ids"].to(self.device_name)
        attention_mask = batch["attention_mask"].to(self.device_name)
        position_ids = batch["position_ids"].to(self.device_name)
        loss_mask = batch.pop("loss_mask")[:, :-1].reshape(-1).to(self.device_name)
        # These masks are produced from the same Qwen native token stream as
        # ``loss_mask``.  They are diagnostics only: optimization still uses
        # the authoritative Assistant mask, while the span masks report
        # reasoning/tool-call NLL without re-tokenising hand-sliced text.
        span_masks = {}
        for field in ("reasoning_loss_mask", "tool_call_loss_mask"):
            value = batch.get(field, None)
            if value is None:
                span_masks[field] = torch.zeros_like(loss_mask)
            else:
                value = value.to(self.device_name, dtype=torch.float32)
                span_masks[field] = value[:, :-1].reshape(-1)
        # Canonical Travel SFT carries one scalar weight per complete
        # trajectory (strict/recoverable=1, partial=0.5).  It stays outside
        # the chat template and scales both token numerator and denominator.
        sample_weight = batch.get("sample_weight", None)
        if sample_weight is None:
            sample_weight = torch.ones((input_ids.shape[0],), dtype=torch.float32, device=self.device_name)
        else:
            sample_weight = sample_weight.to(self.device_name, dtype=torch.float32).reshape(-1)
            if sample_weight.numel() != input_ids.shape[0]:
                raise ValueError(f"sample_weight batch dimension {sample_weight.numel()} != {input_ids.shape[0]}")
        sequence_loss_width = loss_mask.view(input_ids.shape[0], -1).shape[1]
        token_weight = loss_mask.view(input_ids.shape[0], -1) * sample_weight[:, None]
        token_weight = token_weight.reshape(-1)
        for field in span_masks:
            span_masks[field] = span_masks[field] * sample_weight.repeat_interleave(sequence_loss_width)
        loss_fct = nn.CrossEntropyLoss(reduction="none")

        # Context manager for sequence parallel if needed
        context = self.sharding_manager if use_sp else nullcontext()
        with context, torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            if not use_sp:
                # Standard forward pass without sequence parallel
                labels = input_ids[:, 1:].contiguous()
                output = self.fsdp_model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=False)
                logits = output.logits

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels.contiguous()
                # Flatten the tokens
                shift_logits = shift_logits.view(-1, self.model.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                token_losses = loss_fct(shift_logits, shift_labels)
                loss = token_losses * token_weight.to(token_losses.device)
            else:
                # IMPORTANT: We have a big assumption here, so we can shard the SAME sequence across SP ranks
                # i.e., each GPU has <1 sequence, and each SP group has 1 sequence
                # 1. All SP ranks will receive the *SAME* batch
                # 2. Different SP groups will receive *DIFFERENT* batches
                # This is implemented by the DistributedSampler

                batch_size, seqlen = input_ids.shape
                # Remove padding
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # Unpad position_ids to align rotary
                position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                # Pad and slice inputs for sequence parallelism
                input_ids_rmpad_sliced, position_ids_rmpad_padded, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, position_ids_rmpad, sp_size=get_ulysses_sequence_parallel_world_size())
                # For computing loss
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)
                input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(input_ids_rmpad_rolled, None, get_ulysses_sequence_parallel_world_size())
                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # Forward pass
                output = self.fsdp_model(
                    input_ids=input_ids_rmpad_sliced,
                    attention_mask=None,  # Not needed with flash attention varlen
                    position_ids=position_ids_rmpad_padded,
                    use_cache=False,
                )

                # Compute loss locally then aggregate
                logits_rmpad = output.logits.squeeze(0)
                input_ids_rmpad_rolled = input_ids_rmpad_rolled.to(logits_rmpad.device)
                loss = loss_fct(logits_rmpad, input_ids_rmpad_rolled)
                # Gather and unpad for sequence parallelism
                loss = gather_outpus_and_unpad(loss, gather_dim=0, unpad_dim=0, padding_size=pad_size)

                # This is the loss collected from all ulysses ranks
                full_loss = pad_input(hidden_states=loss.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen)
                full_loss = full_loss.squeeze(-1)[:, :-1]  # Remove last token's loss
                full_loss = full_loss.reshape(-1)
                token_weight = token_weight.to(full_loss.device)
                token_losses = full_loss
                loss = token_losses * token_weight

            # Keep loss diagnostics at the same token granularity as the
            # optimization mask.  ``trajectory_macro_loss`` is the mean of
            # per-trajectory masked NLLs, so a long trajectory cannot drown
            # out shorter complete examples in the validation report.
            if not use_sp:
                token_losses = token_losses.to(token_weight.device)
            flat_token_losses = token_losses.reshape(-1)
            flat_token_weight = token_weight.to(flat_token_losses.device)
            batch_size = int(input_ids.shape[0])
            per_sample_losses = flat_token_losses.view(batch_size, -1)
            per_sample_weights = flat_token_weight.view(batch_size, -1)
            sample_numerators = torch.sum(per_sample_losses * per_sample_weights, dim=1)
            sample_denominators = torch.sum(per_sample_weights, dim=1)
            valid_samples = sample_denominators > 0
            macro_values = torch.where(
                valid_samples,
                sample_numerators / (sample_denominators + 1e-8),
                torch.zeros_like(sample_numerators),
            )
            self._last_loss_details = {
                "weighted_nll_sum": float(torch.sum(sample_numerators.detach()).item()),
                "weighted_supervised_tokens": float(torch.sum(sample_denominators.detach()).item()),
                "supervised_tokens": float(torch.sum(flat_token_weight.detach() > 0).item()),
                "trajectory_macro_sum": float(torch.sum(macro_values[valid_samples].detach()).item()),
                "trajectory_count": float(torch.sum(valid_samples.detach()).item()),
            }
            for field, key in (("reasoning_loss_mask", "reasoning"), ("tool_call_loss_mask", "tool_call")):
                span_weight = span_masks[field].to(flat_token_losses.device)
                self._last_loss_details[f"{key}_nll_sum"] = float(
                    torch.sum(flat_token_losses * span_weight).detach().item()
                )
                self._last_loss_details[f"{key}_supervised_tokens"] = float(
                    torch.sum(span_weight).detach().item()
                )

            valid_token_this_rank = torch.sum(token_weight)
            # Always use a global weighted-token denominator.  During gradient
            # accumulation the caller computes it once over the *complete*
            # batch and passes it to every micro-batch; normalising each
            # micro-batch independently would make the result depend on how
            # examples happened to be split across GPUs.
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                if normalization_denominator is None:
                    denominator = valid_token_this_rank.detach().clone()
                    torch.distributed.all_reduce(denominator)
                else:
                    denominator = torch.as_tensor(
                        normalization_denominator,
                        device=valid_token_this_rank.device,
                        dtype=valid_token_this_rank.dtype,
                    ).detach()
                dp_size = self.ulysses_device_mesh.size("dp") if use_sp else torch.distributed.get_world_size()
            else:
                denominator = (
                    valid_token_this_rank.detach()
                    if normalization_denominator is None
                    else torch.as_tensor(normalization_denominator, device=valid_token_this_rank.device, dtype=valid_token_this_rank.dtype).detach()
                )
                dp_size = 1
            loss = torch.sum(loss) / (denominator + 1e-8) * dp_size

            if do_backward:
                loss.backward()
            return loss

    def training_step(self, batch: TensorDict):
        self.fsdp_model.train()

        log_gpu_memory_usage("Before optimizer zero_grad", logger=logger)

        self.optimizer.zero_grad()

        log_gpu_memory_usage("After optimizer zero_grad", logger=logger)

        micro_batches = batch.split(self.config.data.micro_batch_size_per_gpu)
        n_micro_batches = len(micro_batches)
        step_loss = 0
        # Compute one denominator for the complete effective batch.  The
        # numerator is accumulated by backward() over micro-batches, so no
        # additional 1/N micro-batch scaling is applied below.
        full_loss_mask = batch["loss_mask"][:, :-1].to(dtype=torch.float32)
        full_sample_weight = batch.get("sample_weight", None)
        if full_sample_weight is None:
            full_sample_weight = torch.ones((full_loss_mask.shape[0],), dtype=torch.float32)
        else:
            full_sample_weight = full_sample_weight.to(dtype=torch.float32).reshape(-1).cpu()
        global_denominator = torch.sum(full_loss_mask.cpu() * full_sample_weight[:, None])
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            global_denominator = global_denominator.to(self.device_name)
            torch.distributed.all_reduce(global_denominator)
        loss_details = []
        for micro_batch in micro_batches:
            loss = self._compute_loss_and_backward(
                batch=micro_batch,
                normalization_denominator=global_denominator,
            )
            step_loss += loss.item()
            loss_details.append(dict(getattr(self, "_last_loss_details", {})))
        if n_micro_batches > 1:
            step_loss /= n_micro_batches

        if self.config.model.strategy == 'fsdp':
            grad_norm = self.fsdp_model.clip_grad_norm_(max_norm=self.config.optim.clip_grad)
        elif self.config.model.strategy == 'fsdp2':
            grad_norm = fsdp2_clip_grad_norm_(self.fsdp_model.parameters(), max_norm=self.config.optim.clip_grad)
        else:
            raise NotImplementedError(f"not implement {self.config.model.strategy}")

        log_gpu_memory_usage("Before optimizer step", logger=logger)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: grad_norm is not finite: {grad_norm}")
            self.optimizer.zero_grad()
        else:
            self.optimizer.step()

        log_gpu_memory_usage("After optimizer step", logger=logger)

        self.lr_scheduler.step()

        # reduce loss across dp ranks
        lr = self.lr_scheduler.get_last_lr()[0]

        log_gpu_memory_usage("After offload weights", logger=logger)

        # Aggregate diagnostics independently from the autograd loss.  This
        # keeps masked-token NLL and trajectory-macro loss meaningful even
        # when a batch is split into multiple micro-batches or DP ranks.
        detail_keys = (
            "weighted_nll_sum",
            "weighted_supervised_tokens",
            "supervised_tokens",
            "trajectory_macro_sum",
            "trajectory_count",
            "reasoning_nll_sum",
            "reasoning_supervised_tokens",
            "tool_call_nll_sum",
            "tool_call_supervised_tokens",
        )
        detail_totals = {
            key: float(sum(float(item.get(key, 0.0)) for item in loss_details))
            for key in detail_keys
        }
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            detail_tensor = torch.tensor(
                [detail_totals[key] for key in detail_keys],
                dtype=torch.float64,
                device=self.device_name,
            )
            torch.distributed.all_reduce(detail_tensor, op=torch.distributed.ReduceOp.SUM)
            detail_totals = {
                key: float(value)
                for key, value in zip(detail_keys, detail_tensor.detach().cpu().tolist())
            }

        step_loss = torch.tensor(step_loss).to(self.device_name)
        if is_cuda_available:
            torch.distributed.all_reduce(step_loss, op=torch.distributed.ReduceOp.AVG)
        elif is_npu_available:
            torch.distributed.all_reduce(step_loss)
            step_loss /= self.device_mesh.size(0)
        supervised_tokens = float(full_loss_mask.sum().item())
        weighted_tokens = float((full_loss_mask * full_sample_weight[:, None]).sum().item())
        masked_token_nll = (
            detail_totals["weighted_nll_sum"] / detail_totals["weighted_supervised_tokens"]
            if detail_totals["weighted_supervised_tokens"] > 0
            else 0.0
        )
        trajectory_macro_loss = (
            detail_totals["trajectory_macro_sum"] / detail_totals["trajectory_count"]
            if detail_totals["trajectory_count"] > 0
            else 0.0
        )
        reasoning_nll = (
            detail_totals["reasoning_nll_sum"] / detail_totals["reasoning_supervised_tokens"]
            if detail_totals["reasoning_supervised_tokens"] > 0 else 0.0
        )
        tool_call_nll = (
            detail_totals["tool_call_nll_sum"] / detail_totals["tool_call_supervised_tokens"]
            if detail_totals["tool_call_supervised_tokens"] > 0 else 0.0
        )
        return {
            "train/loss": step_loss.detach().item(),
            "train/masked_token_nll": float(masked_token_nll),
            "train/perplexity": float(math.exp(min(50.0, masked_token_nll))) if masked_token_nll else 1.0,
            "train/trajectory_macro_loss": float(trajectory_macro_loss),
            "train/lr(1e-3)": lr * 1e3,
            "train/supervised_tokens": detail_totals["supervised_tokens"] or supervised_tokens,
            "train/weighted_supervised_tokens": detail_totals["weighted_supervised_tokens"] or weighted_tokens,
            "train/reasoning_nll": float(reasoning_nll),
            "train/tool_call_nll": float(tool_call_nll),
        }

    def validation_step(self, batch: TensorDict):
        self.fsdp_model.eval()
        with torch.no_grad():
            loss = self._compute_loss_and_backward(batch, do_backward=False)
            if is_cuda_available:
                torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.AVG)
            elif is_npu_available:
                torch.distributed.all_reduce(loss)
                loss /= self.device_mesh.size(0)
        return loss

    def save_checkpoint(self, step, alias=None):
        # save checkpoint
        path = os.path.join(self.config.trainer.default_local_dir, f"global_step_{step}")

        fsdp_strategy = self.config.model.strategy
        if fsdp_strategy == "fsdp":
            # FSDP1 checkpoint saving
            from torch.distributed.fsdp import FullStateDictConfig, StateDictType

            cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(self.fsdp_model, StateDictType.FULL_STATE_DICT, cfg):
                state_dict = self.fsdp_model.state_dict()

            # save huggingface model
            if self.device_mesh.get_rank() == 0:
                os.makedirs(path, exist_ok=True)
                self.model.save_pretrained(path, state_dict=state_dict)
                self.tokenizer.save_pretrained(path)
        elif fsdp_strategy == "fsdp2":
            # FSDP2 checkpoint saving
            from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

            # Get full state dict with FSDP2
            options = StateDictOptions(full_state_dict=True, cpu_offload=True)
            state_dict = get_model_state_dict(self.fsdp_model, options=options)

            # save huggingface model
            if self.device_mesh.get_rank() == 0:
                os.makedirs(path, exist_ok=True)
                self.model.save_pretrained(path, state_dict=state_dict)
                self.model_config.save_pretrained(path)
                self.tokenizer.save_pretrained(path)
        else:
            raise NotImplementedError(f"not implement {fsdp_strategy}")

        # Optimizer/scheduler/RNG state is needed for a faithful resume.  Keep
        # it beside the model; the HF directory remains directly loadable for
        # inference.
        if self.device_mesh.get_rank() == 0:
            torch.save(
                {
                    "step": int(step),
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.lr_scheduler.state_dict(),
                    "python_rng": __import__("random").getstate(),
                    "torch_rng": torch.get_rng_state(),
                },
                os.path.join(path, "trainer_state.pt"),
            )
            with open(os.path.join(path, "checkpoint_metadata.json"), "w", encoding="utf-8") as handle:
                json.dump({"step": int(step), "alias": alias, "max_length": int(self.config.data.max_length), "enable_thinking": True}, handle, indent=2)
            if alias:
                # A tiny pointer avoids duplicating a potentially multi-GB
                # model while still giving callers stable epoch aliases.
                with open(os.path.join(self.config.trainer.default_local_dir, f"{alias}.json"), "w", encoding="utf-8") as handle:
                    json.dump({"path": path, "step": int(step)}, handle, indent=2)
                alias_dir = os.path.join(self.config.trainer.default_local_dir, str(alias))
                os.makedirs(alias_dir, exist_ok=True)
                with open(os.path.join(alias_dir, "checkpoint_reference.json"), "w", encoding="utf-8") as handle:
                    json.dump({"path": path, "step": int(step)}, handle, indent=2)

        # Copy to HDFS if configured
        if self.device_mesh.get_rank() == 0 and self.config.trainer.default_hdfs_dir:
            hdfs_io.makedirs(self.config.trainer.default_hdfs_dir, exist_ok=True)
            hdfs_io.copy(src=path, dst=self.config.trainer.default_hdfs_dir, dirs_exist_ok=True)

        torch.distributed.barrier()

    def fit(self):
        rank = self.device_mesh.get_rank()
        tracking = None
        if rank == 0:
            tracking = Tracking(
                project_name=self.config.trainer.project_name,
                experiment_name=self.config.trainer.experiment_name,
                default_backend=self.config.trainer.logger,
                config=self.config,
            )

        global_step = 0
        last_valid_metric = None
        validation_history = []
        try:
            from verl.utils.sft_metrics import structured_action_metrics
            train_protocol = structured_action_metrics(
                [{"messages": value} for value in getattr(self.train_dataset, "messages", [])]
            )
            val_protocol = structured_action_metrics(
                [{"messages": value} for value in getattr(self.val_dataset, "messages", [])]
            )
        except Exception:
            train_protocol = val_protocol = {"tool_parse_rate": 0.0, "structured_choice_rate": 0.0, "structured_content_rate": 0.0}
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps
        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        for epoch in range(self.config.trainer.total_epochs):
            self.train_sampler.set_epoch(epoch=epoch)
            for data in tqdm(
                self.train_dataloader,
                total=self.steps_per_epoch,
                desc=f"Epoch {epoch + 1}/{self.config.trainer.total_epochs}",
                disable=rank != 0,
            ):
                global_step += 1
                data = TensorDict(data, batch_size=[data["input_ids"].shape[0]]).to(self.device_name)
                train_metric = self.training_step(data)
                if rank == 0 and tracking is not None:
                    train_log = {f"sft/train/{key.split('/', 1)[-1]}": value for key, value in train_metric.items()}
                    train_log.update({f"sft/train/{key}": value for key, value in train_protocol.items() if key != "tool_call_count"})
                    tracking.log(data=train_log, step=global_step)
                if global_step >= self.total_training_steps:
                    break

            # Exactly one validation pass at every epoch end.  Step-level
            # test_freq validation is intentionally ignored for this trainer.
            val_losses = []
            val_details = []
            for val_data in self.val_dataloader:
                val_data = TensorDict(val_data, batch_size=[val_data["input_ids"].shape[0]]).to(self.device_name)
                val_losses.append(self.validation_step(val_data))
                val_details.append(dict(getattr(self, "_last_loss_details", {})))
            if val_losses:
                val_loss = torch.mean(torch.stack(val_losses))
            else:
                val_loss = torch.tensor(0.0, device=self.device_name)
            if torch.distributed.is_available() and torch.distributed.is_initialized() and val_details:
                detail_keys = (
                    "weighted_nll_sum",
                    "weighted_supervised_tokens",
                    "supervised_tokens",
                    "trajectory_macro_sum",
                    "trajectory_count",
                    "reasoning_nll_sum",
                    "reasoning_supervised_tokens",
                    "tool_call_nll_sum",
                    "tool_call_supervised_tokens",
                )
                detail_tensor = torch.tensor(
                    [sum(float(item.get(key, 0.0)) for item in val_details) for key in detail_keys],
                    dtype=torch.float64,
                    device=self.device_name,
                )
                torch.distributed.all_reduce(detail_tensor, op=torch.distributed.ReduceOp.SUM)
                val_details = [{key: float(value) for key, value in zip(detail_keys, detail_tensor.detach().cpu().tolist())}]
            if rank == 0:
                val_weighted_nll = sum(float(item.get("weighted_nll_sum", 0.0)) for item in val_details)
                val_weighted_tokens = sum(float(item.get("weighted_supervised_tokens", 0.0)) for item in val_details)
                val_trajectory_sum = sum(float(item.get("trajectory_macro_sum", 0.0)) for item in val_details)
                val_trajectory_count = sum(float(item.get("trajectory_count", 0.0)) for item in val_details)
                val_masked_nll = val_weighted_nll / val_weighted_tokens if val_weighted_tokens > 0 else float(val_loss.detach().item())
                val_trajectory_macro = val_trajectory_sum / val_trajectory_count if val_trajectory_count > 0 else val_masked_nll
                val_metric = {
                    "sft/val/masked_token_nll": float(val_masked_nll),
                    "sft/val/perplexity": float(math.exp(min(50.0, val_masked_nll))) if val_masked_nll else 1.0,
                    "sft/val/trajectory_macro_loss": float(val_trajectory_macro),
                    "sft/val/epoch": float(epoch + 1),
                    "step": int(global_step),
                    "sft/val/reasoning_nll": float(
                        sum(float(item.get("reasoning_nll_sum", 0.0)) for item in val_details)
                        / sum(float(item.get("reasoning_supervised_tokens", 0.0)) for item in val_details)
                        if sum(float(item.get("reasoning_supervised_tokens", 0.0)) for item in val_details) > 0 else 0.0
                    ),
                    "sft/val/tool_call_nll": float(
                        sum(float(item.get("tool_call_nll_sum", 0.0)) for item in val_details)
                        / sum(float(item.get("tool_call_supervised_tokens", 0.0)) for item in val_details)
                        if sum(float(item.get("tool_call_supervised_tokens", 0.0)) for item in val_details) > 0 else 0.0
                    ),
                    "sft/val/supervised_tokens": float(sum(float(item.get("supervised_tokens", 0.0)) for item in val_details)),
                    "sft/val/weighted_supervised_tokens": float(val_weighted_tokens),
                }
                val_metric.update({f"sft/val/{key}": value for key, value in val_protocol.items() if key != "tool_call_count"})
                if validation_history:
                    previous = validation_history[-1]
                    val_metric["sft/val/protocol_non_degraded"] = float(all(
                        float(val_metric.get(f"sft/val/{key}", 0.0)) + 1e-12 >= float(previous.get(f"sft/val/{key}", 0.0))
                        for key in ("tool_parse_rate", "structured_choice_rate", "structured_content_rate")
                    ))
                else:
                    val_metric["sft/val/protocol_non_degraded"] = 1.0
                validation_history.append(val_metric)
                if tracking is not None:
                    tracking.log(data=val_metric, step=global_step)
                last_valid_metric = val_metric
            torch.distributed.barrier()

            # The experiment contract keeps every epoch checkpoint so a
            # later audit can compare validation and mask diagnostics. The
            # legacy ``save_each_epoch`` switch is intentionally ignored:
            # disabling it would make a run non-reproducible across configs.
            self.save_checkpoint(step=global_step, alias=f"epoch_{epoch + 1}")
            if global_step >= self.total_training_steps:
                break

        # Keep explicit last/best pointers rather than deleting epoch folders.
        if rank == 0:
            root = os.path.join(self.config.trainer.default_local_dir)
            os.makedirs(root, exist_ok=True)
            if last_valid_metric is not None:
                with open(os.path.join(root, "last_checkpoint.json"), "w", encoding="utf-8") as handle:
                    json.dump({"step": global_step, "metrics": last_valid_metric}, handle, indent=2)
                os.makedirs(os.path.join(root, "last"), exist_ok=True)
                with open(os.path.join(root, "last", "checkpoint_reference.json"), "w", encoding="utf-8") as handle:
                    json.dump({"path": os.path.join(root, f"global_step_{global_step}"), "step": global_step}, handle, indent=2)
                candidates = [item for item in validation_history if item.get("sft/val/protocol_non_degraded", 1.0) >= 1.0]
                best = min(candidates or validation_history, key=lambda item: item.get("sft/val/masked_token_nll", float("inf"))) if validation_history else last_valid_metric
                with open(os.path.join(root, "best_checkpoint.json"), "w", encoding="utf-8") as handle:
                    json.dump({"step": best.get("step", global_step), "metrics": best}, handle, indent=2)
                os.makedirs(os.path.join(root, "best"), exist_ok=True)
                with open(os.path.join(root, "best", "checkpoint_reference.json"), "w", encoding="utf-8") as handle:
                    json.dump({"path": os.path.join(root, f"global_step_{best.get('step', global_step)}"), "step": best.get("step", global_step)}, handle, indent=2)
            print(f"Final validation metrics: {last_valid_metric}")
        if tracking is not None:
            tracking.finish()


def run_sft(config):
    device_name = get_device_name()
    local_rank, rank, world_size = initialize_global_process_group()

    device_mesh = init_device_mesh(device_type=device_name, mesh_shape=(world_size,), mesh_dim_names=("fsdp",))
    dp_size = world_size // config.ulysses_sequence_parallel_size
    ulysses_device_mesh = init_device_mesh(device_type=device_name, mesh_shape=(dp_size, config.ulysses_sequence_parallel_size), mesh_dim_names=("dp", "sp"))
    # build tokenizer and datasets first
    from verl.utils import hf_tokenizer

    local_model_path = copy_to_local(src=config.model.partial_pretrain, verbose=True)
    tokenizer = hf_tokenizer(local_model_path, trust_remote_code=config.model.trust_remote_code)
    train_dataset = create_sft_dataset(config.data.train_files, config.data, tokenizer)
    val_dataset = create_sft_dataset(config.data.val_files, config.data, tokenizer)

    trainer = FSDPSFTTrainer(config=config, device_mesh=device_mesh, ulysses_device_mesh=ulysses_device_mesh, tokenizer=tokenizer, train_dataset=train_dataset, val_dataset=val_dataset)

    trainer.fit()

    destroy_global_process_group()


@hydra.main(config_path="config", config_name="sft_trainer", version_base=None)
def main(config):
    run_sft(config)


def create_sft_dataset(data_paths, data_config, tokenizer):
    """Create a dataset."""
    # build dataset
    # First check if a custom dataset class is specified
    if data_config.custom_cls.get("path", None):
        from verl.utils.import_utils import load_extern_type

        dataset_cls = load_extern_type(data_config.custom_cls.path, data_config.custom_cls.name)
    # Then check if multi-turn dataset should be used
    elif data_config.get("multiturn", {}).get("enable", False):
        dataset_cls = MultiTurnSFTDataset
    # Default to single-turn dataset
    else:
        dataset_cls = SFTDataset

    # Create datasets based on the selected class
    dataset = dataset_cls(parquet_files=data_paths, tokenizer=tokenizer, config=data_config)
    return dataset


if __name__ == "__main__":
    main()
