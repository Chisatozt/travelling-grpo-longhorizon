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
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import uuid
from collections import defaultdict
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Any, Optional, Type

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_reward_component_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.validation_baseline import (
    ValidationBaselineError,
    load_step0_validation_metrics,
    resolve_validation_baseline_path,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.rollout_health import validate_initial_rollout_health
from verl.trainer.ppo.segmented_rollout import (
    expand_segmented_batch,
    has_segmented_rollouts,
)
from verl.trainer.ppo.validation_passes import (
    PUBLIC_VALIDATION_METRICS,
    aggregate_validation_attempts,
)
from verl.utils.checkpoint.checkpoint_manager import BaseCheckpointManager, find_latest_ckpt_path, get_best_score
from verl.utils.debug.performance import _timer
from verl.utils.metric import (
    reduce_metrics,
)
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.trainer.ppo.experiment_integrity import (
    ExperimentIntegrityError,
    capture_rng_state,
    restore_rng_state,
    resolve_training_step_policy,
    validate_process_run_until_step,
    validate_total_training_steps,
)

WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0) for node, node_info in node_available_resources.items()}

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])
        if total_available_gpus < total_required_gpus:
            raise ValueError(f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}")

        # check each resource pool can be satisfied, O(#resource_pools * #nodes)
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes}" + "cannot be satisfied in this ray cluster")


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl", multi_turn=False):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    if "segment_response_mask" in data.batch.keys():
        response_mask = data.batch["loss_mask"] if multi_turn else data.batch["segment_response_mask"]
    elif multi_turn:
        loss_mask = data.batch["loss_mask"]
        response_mask = loss_mask[:, -response_length:]
    else:
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty)  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    if "segment_response_mask" in data.batch.keys():
        return data.batch["segment_response_mask"]
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def _extract_terminal_reward_metadata(data: DataProto):
    """Read trainer-only terminal scores emitted by interaction rollouts.

    ``reward_scores`` is a non-tensor object array because tool names are
    dynamic.  This adapter deliberately ignores conversation turn rewards and
    any ground-truth fields.  TravelGym rows use the explicit terminal scalar;
    non-terminal metadata falls back to the framework's ordinary token-level
    path without changing the TravelGym contract.
    """
    raw = data.non_tensor_batch.get("reward_scores")
    if raw is None:
        return None, None
    if hasattr(raw, "tolist"):
        raw = raw.tolist()
    if not isinstance(raw, (list, tuple)):
        raw = [raw]
    expected_rows = int(data.batch["responses"].shape[0])
    if len(raw) != expected_rows:
        # A malformed/misaligned object-array must not be interpreted as a
        # terminal reward vector.  The caller will use the ordinary fallback.
        return None, None
    values, valid = [], []
    fallback_scores = data.batch["token_level_rewards"].sum(dim=-1) if "token_level_rewards" in data.batch else None
    if fallback_scores is not None:
        fallback_scores = fallback_scores.detach().reshape(-1)
    terminal_flags = []
    for item in raw:
        while isinstance(item, (list, tuple)) and len(item) == 1:
            item = item[0]
        if not isinstance(item, Mapping):
            return None, None
        value = item.get("interact_with_env")
        if value is None:
            # This batch may be for a non-interaction tool.
            return None, None
        validity = item.get("interact_with_env_reward_valid", 1.0)
        try:
            row_valid = bool(float(validity))
        except (TypeError, ValueError):
            row_valid = False
        terminal_only = item.get("interact_with_env_terminal_only")
        if terminal_only is None:
            # Older rollout records do not declare the protocol.  Falling
            # back to the pre-terminal path is safer than guessing that a
            # legacy score is a TravelGym terminal score.
            return None, None
        try:
            is_terminal = bool(float(terminal_only))
        except (TypeError, ValueError):
            is_terminal = False
        terminal_flags.append(is_terminal)
        if is_terminal:
            try:
                values.append(float(value))
            except (TypeError, ValueError):
                values.append(0.0)
            valid.append(row_valid)
        else:
            if fallback_scores is None or len(values) >= fallback_scores.numel():
                return None, None
            # Non-terminal metadata uses the scalar emitted by token-level
            # reward construction; keep TravelGym's terminal path private and
            # explicit.
            values.append(float(fallback_scores[len(values)].item()))
            valid.append(row_valid)
    if not terminal_flags:
        return None, None
    device = data.batch["token_level_rewards"].device if "token_level_rewards" in data.batch else data.batch["responses"].device
    dtype = data.batch["token_level_rewards"].dtype if "token_level_rewards" in data.batch else torch.float32
    return torch.tensor(values, device=device, dtype=dtype), torch.tensor(valid, device=device, dtype=torch.bool)


_TERMINAL_REWARD_COMPONENT_SPECS = (
    ("correct_completion", 3.00),
    ("coverage_adjusted_answer_quality", 0.30),
    ("coverage_adjusted_legal_chain_rate", 0.20),
    ("hidden_preference_hit_rate", 0.15),
    ("efficiency", 0.05),
    ("policy_penalty", -1.00),
    ("redundant_action_penalty", -1.00),
    ("incomplete_penalty", -1.00),
    ("zero_answer_penalty", -1.00),
    ("max_steps_penalty", -1.00),
)


def _extract_terminal_reward_components(
    data: DataProto,
    terminal_scores: torch.Tensor,
) -> tuple[tuple[str, ...], torch.Tensor] | None:
    """Reconstruct the exact TravelGym terminal score as signed components."""
    batch_size = int(terminal_scores.numel())
    reward_versions = data.non_tensor_batch.get("reward_version")
    if reward_versions is None:
        return None
    versions = np.asarray(reward_versions, dtype=object).reshape(-1)
    if versions.size != batch_size or any(
        str(version) != "travelgym-terminal-v2" for version in versions
    ):
        return None
    columns = []
    names = []
    for name, coefficient in _TERMINAL_REWARD_COMPONENT_SPECS:
        raw = data.non_tensor_batch.get(name)
        if raw is None:
            return None
        try:
            values = np.asarray(raw, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError):
            return None
        if values.size != batch_size or not np.isfinite(values).all():
            return None
        names.append(name)
        columns.append(
            torch.tensor(
                values,
                device=terminal_scores.device,
                dtype=terminal_scores.dtype,
            )
            * (float(coefficient) / 3.70)
        )
    components = torch.stack(columns, dim=-1)
    # terminal_reward is clipped to [-1, 1]. Keep clipping as an explicit
    # residual so component sums remain exactly equal to the optimized score.
    clip_residual = terminal_scores.reshape(-1) - components.sum(dim=-1)
    components = torch.cat([components, clip_residual.unsqueeze(-1)], dim=-1)
    return tuple(names) + ("clip_residual",), components


def _aligned_reward_extra_info_arrays(
    reward_extra_infos: Mapping[str, list],
    *,
    batch_size: int,
) -> dict[str, np.ndarray]:
    """Convert row-wise diagnostics only after proving batch alignment."""
    arrays: dict[str, np.ndarray] = {}
    mismatches: list[str] = []
    for key, values in reward_extra_infos.items():
        try:
            actual = len(values)
        except TypeError:
            actual = -1
        if actual != batch_size:
            mismatches.append(f"{key}={actual}")
            continue
        arrays[key] = np.asarray(values)
    if mismatches:
        details = ", ".join(sorted(mismatches))
        raise ValueError(
            "reward metadata must contain exactly one value per trajectory: "
            f"batch_size={batch_size}; mismatched lengths: {details}"
        )
    return arrays


def _apply_turn_credit_stage(
    original_advantages: torch.Tensor,
    credited_advantages: torch.Tensor,
    response_mask: torch.Tensor,
    diagnostics: dict[str, float],
    *,
    stage: str,
    conservation_groups=None,
    conservation_atol: float = 1.0e-5,
    conservation_rtol: float = 1.0e-6,
) -> torch.Tensor:
    """Select train or shadow advantages with a strict conservation contract."""
    if stage not in {"shadow", "train"}:
        raise ValueError(f"turn credit application requires shadow/train stage, got {stage}")
    if (
        not np.isfinite(conservation_atol)
        or not np.isfinite(conservation_rtol)
        or conservation_atol < 0
        or conservation_rtol < 0
    ):
        raise ValueError("turn credit conservation tolerances must be finite and non-negative")

    if conservation_groups is None:
        conservation = core_algos.turn_credit_conservation_stats(
            original_advantages,
            credited_advantages,
            response_mask,
        )
    else:
        conservation = core_algos.segmented_turn_credit_conservation_stats(
            original_advantages,
            credited_advantages,
            response_mask,
            conservation_groups,
        )
    abs_error = float(conservation["absolute_error"].detach().cpu())
    relative_error = float(conservation["relative_error"].detach().cpu())
    mean_token_error = float(conservation["mean_token_error"].detach().cpu())
    finite = bool(conservation["finite"].detach().cpu())
    diagnostics.update(
        {
            "conservation_error": abs_error,
            "conservation_abs_error": abs_error,
            "conservation_relative_error": relative_error,
            "conservation_mean_token_error": mean_token_error,
            "conservation_finite": float(finite),
            "conservation_atol": float(conservation_atol),
            "conservation_rtol": float(conservation_rtol),
            "applied": 0.0,
        }
    )

    if stage == "shadow":
        return original_advantages
    fallback_rows = int(diagnostics.get("fallback_row_count", 0.0))
    if fallback_rows:
        raise ValueError(
            "component turn credit is in train mode but quality events are missing "
            f"for {fallback_rows} rollout row(s)"
        )
    preprojection_abs_error = float(diagnostics.get("preprojection_abs_error", abs_error))
    preprojection_relative_error = float(
        diagnostics.get("preprojection_relative_error", relative_error)
    )
    preprojection_finite = bool(diagnostics.get("preprojection_finite", finite))
    final_conserved = finite and (
        abs_error <= conservation_atol or relative_error <= conservation_rtol
    )
    preprojection_conserved = preprojection_finite and (
        preprojection_abs_error <= conservation_atol
        or preprojection_relative_error <= conservation_rtol
    )
    if not (preprojection_conserved and final_conserved):
        raise RuntimeError(
            "turn credit conservation failed in train mode: "
            f"preprojection_abs_error={preprojection_abs_error:.9g}, "
            f"preprojection_relative_error={preprojection_relative_error:.9g}, "
            f"final_abs_error={abs_error:.9g} (atol={conservation_atol:.9g}), "
            f"final_relative_error={relative_error:.9g} (rtol={conservation_rtol:.9g})"
        )
    diagnostics["applied"] = 1.0
    return credited_advantages


def _segment_metadata(data: DataProto) -> tuple[list, list, list]:
    """Extract one fragment's public histories and global turn ids per row."""
    raw_records = data.non_tensor_batch.get("segment_records", [])
    if hasattr(raw_records, "tolist"):
        raw_records = raw_records.tolist()
    histories, global_indices, record_ids = [], [], []
    for item in raw_records:
        if isinstance(item, Mapping):
            records = [item]
        elif isinstance(item, (list, tuple)):
            records = [record for record in item if isinstance(record, Mapping)]
        else:
            records = []
        record = records[0] if records else {}
        histories.append(record.get("conversation_history", []))
        global_indices.append(record.get("global_turn_indices", []))
        record_ids.append(record)
    return histories, global_indices, record_ids


def _unique_segment_rows(segment_trajectory_ids) -> tuple[list[int], list[str]]:
    first_rows: list[int] = []
    unique_ids: list[str] = []
    seen: set[str] = set()
    for row, value in enumerate(segment_trajectory_ids):
        key = str(value)
        if key in seen:
            continue
        seen.add(key)
        first_rows.append(row)
        unique_ids.append(key)
    return first_rows, unique_ids


def _compute_segmented_grpo_advantage(
    data: DataProto,
    *,
    adv_estimator,
    num_repeat: int,
    multi_turn: bool,
    turn_level_method: str,
    norm_adv_by_std_in_grpo: bool,
    config,
) -> DataProto:
    """Compute GRPO once per rollout, then map the result to its fragments."""
    response_mask = data.batch["response_mask"]
    grpo_mask = data.batch["loss_mask"] if multi_turn else response_mask
    segment_ids = data.non_tensor_batch.get("segment_trajectory_uid")
    if segment_ids is None:
        raise ValueError("segmented rollouts require segment_trajectory_uid metadata")
    segment_ids = [str(value) for value in np.asarray(segment_ids, dtype=object).reshape(-1)]
    first_rows, unique_segment_ids = _unique_segment_rows(segment_ids)
    unique_rows = torch.tensor(first_rows, device=data.batch["token_level_rewards"].device)
    unique_uids = np.asarray(data.non_tensor_batch["uid"], dtype=object)[first_rows]
    terminal_scores, reward_valid = _extract_terminal_reward_metadata(data)
    if terminal_scores is not None:
        terminal_scores_unique = terminal_scores[unique_rows]
        reward_valid_unique = reward_valid[unique_rows]
        from verl.trainer.ppo.dynamic_sampling import (
            resolve_reward_spread_thresholds,
            select_reward_varying_groups,
        )

        expected_group_size = max(2, int(num_repeat or 1))
        numerical_epsilon, min_reward_spread = resolve_reward_spread_thresholds(
            (config or {}).get("dynamic_sampling", {})
        )
        kept_indices, sampling_stats = select_reward_varying_groups(
            unique_uids,
            terminal_scores_unique.detach().cpu().tolist(),
            reward_valid=reward_valid_unique.detach().cpu().tolist(),
            expected_group_size=expected_group_size,
            numerical_epsilon=numerical_epsilon,
            min_reward_spread=min_reward_spread,
        )
        if data.meta_info.get("travel_skip_update", False):
            kept_indices = []
            sampling_stats = dict(sampling_stats)
            sampling_stats["bounded_retry_exhausted"] = True
        data.meta_info["terminal_sampling_stats"] = sampling_stats
        unique_reward = torch.zeros(
            (len(first_rows), 1),
            device=data.batch["token_level_rewards"].device,
            dtype=data.batch["token_level_rewards"].dtype,
        )
        unique_mask = torch.ones_like(unique_reward)
        unique_advantage, _ = core_algos.compute_terminal_group_advantage(
            token_level_rewards=unique_reward,
            response_mask=unique_mask,
            index=unique_uids,
            terminal_scores=terminal_scores_unique,
            reward_valid=reward_valid_unique,
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        keep_ids = {
            unique_segment_ids[index]
            for index in kept_indices
            if 0 <= int(index) < len(unique_segment_ids)
        }
        scalar_by_segment_id = {
            unique_segment_ids[index]: unique_advantage[index, 0]
            for index in range(len(unique_segment_ids))
        }
        advantages = torch.zeros_like(data.batch["token_level_rewards"])
        trainable_rows = []
        for row, segment_id in enumerate(segment_ids):
            scalar = scalar_by_segment_id.get(segment_id)
            is_trainable = segment_id in keep_ids and scalar is not None
            trainable_rows.append(is_trainable)
            if is_trainable:
                advantages[row] = scalar * grpo_mask[row]
        returns = advantages.clone()
    else:
        # Generic reward managers do not provide TravelGym's terminal ledger.
        # Aggregate each original rollout once so repeated fragments cannot
        # change its GRPO group statistics.
        unique_rewards = torch.stack(
            [data.batch["token_level_rewards"][
                [row for row, value in enumerate(segment_ids) if value == segment_id]
            ].sum() for segment_id in unique_segment_ids]
        ).reshape(-1, 1)
        unique_mask = torch.ones_like(unique_rewards)
        unique_reward_advantage, _ = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=unique_rewards,
            response_mask=unique_mask,
            index=unique_uids,
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        scalar_by_segment_id = {
            segment_id: unique_reward_advantage[index, 0]
            for index, segment_id in enumerate(unique_segment_ids)
        }
        advantages = torch.stack(
            [scalar_by_segment_id[segment_id] * grpo_mask[row] for row, segment_id in enumerate(segment_ids)]
        )
        returns = advantages.clone()
        terminal_scores = None
        reward_valid = None

    turn_credit_config = (config or {}).get("turn_credit", {})
    if not isinstance(turn_credit_config, Mapping):
        turn_credit_config = {}
    turn_mode = str(turn_credit_config.get("method", turn_level_method or "off"))
    credit_stage = str(
        turn_credit_config.get(
            "stage",
            (config or {}).get("turn_credit_stage", "off"),
        )
    ).casefold()
    valid_turn_modes = {
        "off",
        "none",
        "equalized",
        "r2g",
        "em",
        "component_attribution",
        "behavior_component",
        "behavior_delta",
    }
    if turn_mode.casefold() not in valid_turn_modes:
        raise ValueError(f"invalid turn credit method: {turn_mode}")
    if credit_stage not in {"off", "shadow", "train"}:
        raise ValueError(f"invalid turn credit stage: {credit_stage}")
    if (
        terminal_scores is not None
        and turn_mode.casefold() not in {"off", "none"}
        and credit_stage in {"shadow", "train"}
    ):
        histories, global_indices, _ = _segment_metadata(data)
        if turn_mode.casefold() in {
            "component_attribution",
            "behavior_component",
            "behavior_delta",
        }:
            component_payload = _extract_terminal_reward_components(data, terminal_scores)
            if component_payload is None:
                if credit_stage == "train":
                    raise ValueError(
                        "component turn credit requires complete TravelGym reward metadata"
                    )
                credited_advantages = advantages
                diagnostics = {"component_metadata_missing": 1.0}
            else:
                component_names, reward_components = component_payload
                unique_components = reward_components[unique_rows]
                unique_component_advantages = core_algos.compute_terminal_component_advantages(
                    terminal_scores[unique_rows],
                    unique_components,
                    unique_uids,
                    reward_valid=reward_valid[unique_rows],
                    norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                )
                component_by_id = {
                    unique_segment_ids[index]: unique_component_advantages[index]
                    for index in range(len(unique_segment_ids))
                }
                component_advantages = torch.stack(
                    [component_by_id[segment_id] for segment_id in segment_ids]
                )
                train_mask = torch.tensor(
                    [segment_id in keep_ids for segment_id in segment_ids],
                    device=advantages.device,
                    dtype=advantages.dtype,
                )
                component_advantages = component_advantages * train_mask.unsqueeze(-1)
                routing = turn_credit_config.get("routing", {})
                if OmegaConf.is_config(routing):
                    routing = OmegaConf.to_container(routing, resolve=True)
                credited_advantages, diagnostics = core_algos.redistribute_segmented_behavior_component_turn_credit(
                    advantages,
                    grpo_mask,
                    data.batch["turn_boundaries"],
                    segment_ids,
                    histories,
                    global_indices,
                    component_advantages,
                    component_names,
                    mix_ratio=float(turn_credit_config.get("mix_ratio", 0.30)),
                    routing=routing,
                )
        else:
            credited_advantages = core_algos.redistribute_segmented_terminal_turn_credit(
                advantages,
                grpo_mask,
                data.batch["turn_boundaries"],
                segment_ids,
                histories,
                global_indices,
            )
            conservation = core_algos.segmented_turn_credit_conservation_stats(
                advantages,
                credited_advantages,
                grpo_mask,
                segment_ids,
            )
            diagnostics = {
                "conservation_error": float(conservation["absolute_error"].detach().cpu()),
                "conservation_abs_error": float(conservation["absolute_error"].detach().cpu()),
                "conservation_relative_error": float(conservation["relative_error"].detach().cpu()),
                "conservation_mean_token_error": float(conservation["mean_token_error"].detach().cpu()),
                "conservation_finite": float(conservation["finite"].detach().cpu()),
            }
        advantages = _apply_turn_credit_stage(
            advantages,
            credited_advantages,
            grpo_mask,
            diagnostics,
            stage=credit_stage,
            conservation_groups=segment_ids,
            conservation_atol=float(turn_credit_config.get("conservation_atol", 1.0e-5)),
            conservation_rtol=float(turn_credit_config.get("conservation_rtol", 1.0e-6)),
        )
        returns = advantages.clone()
        data.meta_info["turn_credit_conservation_error"] = diagnostics.get(
            "conservation_abs_error", diagnostics.get("conservation_error", 0.0)
        )
        data.meta_info["turn_credit_applied"] = diagnostics.get("applied", 0.0)
        data.meta_info["turn_credit_diagnostics"] = diagnostics

    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
    return data


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1, multi_turn=False, turn_level_method="Equalized", trajectory_score_method="Sum", norm_adv_by_std_in_grpo=True, config=None):
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator: The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.
        turn_level_method (str, optional): Legacy method selector; behavior-component routing is configured under algorithm.turn_credit.
        trajectory_score_method (str, optional): Method to compute trajectory score, "Sum" or "R2G". Defaults to "Sum".
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    if has_segmented_rollouts(data) and adv_estimator in {
        AdvantageEstimator.GRPO,
        AdvantageEstimator.GRPO_MULTITURN,
    }:
        return _compute_segmented_grpo_advantage(
            data,
            adv_estimator=adv_estimator,
            num_repeat=num_repeat,
            multi_turn=multi_turn,
            turn_level_method=turn_level_method,
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            config=config,
        )
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.get("pf_ppo_reweight_method", "pow"),
                config.get("pf_ppo_weight_pow", 2.0),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]
        if multi_turn:
            # If multi-turn, replace the mask with the relevant part of loss_mask
            # Get length from the initial response mask
            response_length = grpo_calculation_mask.size(1)
            # This mask is the one intended for GRPO
            grpo_calculation_mask = data.batch["loss_mask"][:, -response_length:]
        terminal_scores, reward_valid = _extract_terminal_reward_metadata(data)
        if terminal_scores is not None:
            from verl.trainer.ppo.dynamic_sampling import (
                resolve_reward_spread_thresholds,
                select_reward_varying_groups,
            )
            expected_group_size = max(2, int(num_repeat or 1))
            numerical_epsilon, min_reward_spread = resolve_reward_spread_thresholds(
                (config or {}).get("dynamic_sampling", {})
            )
            kept_indices, sampling_stats = select_reward_varying_groups(
                data.non_tensor_batch["uid"],
                terminal_scores.detach().cpu().tolist(),
                reward_valid=reward_valid.detach().cpu().tolist(),
                expected_group_size=expected_group_size,
                numerical_epsilon=numerical_epsilon,
                min_reward_spread=min_reward_spread,
            )
            if data.meta_info.get("travel_skip_update", False):
                kept_indices = []
                sampling_stats = dict(sampling_stats)
                sampling_stats["bounded_retry_exhausted"] = True
            data.meta_info["terminal_sampling_stats"] = sampling_stats
            advantages, returns = core_algos.compute_terminal_group_advantage(
                token_level_rewards=data.batch["token_level_rewards"],
                response_mask=grpo_calculation_mask,
                index=data.non_tensor_batch["uid"],
                terminal_scores=terminal_scores,
                reward_valid=reward_valid,
                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            )
            train_mask = torch.zeros(advantages.size(0), device=advantages.device, dtype=advantages.dtype)
            if kept_indices:
                train_mask[kept_indices] = 1.0
            advantages = advantages * train_mask.unsqueeze(-1)
            returns = returns * train_mask.unsqueeze(-1)
        else:
            advantages, returns = core_algos.compute_grpo_outcome_advantage(
                token_level_rewards=data.batch["token_level_rewards"],
                response_mask=grpo_calculation_mask,
                index=data.non_tensor_batch["uid"],
                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.GRPO_MULTITURN:
        grpo_calculation_mask = data.batch["response_mask"]
        if multi_turn:
            # If multi-turn, replace the mask with the relevant part of loss_mask
            response_length = grpo_calculation_mask.size(1)  # Get length from the initial response mask
            grpo_calculation_mask = data.batch["loss_mask"][:, -response_length:]  # This mask is the one intended for GRPO
        terminal_scores, reward_valid = _extract_terminal_reward_metadata(data)
        if terminal_scores is not None:
            from verl.trainer.ppo.dynamic_sampling import (
                resolve_reward_spread_thresholds,
                select_reward_varying_groups,
            )
            expected_group_size = max(2, int(num_repeat or 1))
            numerical_epsilon, min_reward_spread = resolve_reward_spread_thresholds(
                (config or {}).get("dynamic_sampling", {})
            )
            kept_indices, sampling_stats = select_reward_varying_groups(
                data.non_tensor_batch["uid"],
                terminal_scores.detach().cpu().tolist(),
                reward_valid=reward_valid.detach().cpu().tolist(),
                expected_group_size=expected_group_size,
                numerical_epsilon=numerical_epsilon,
                min_reward_spread=min_reward_spread,
            )
            if data.meta_info.get("travel_skip_update", False):
                kept_indices = []
                sampling_stats = dict(sampling_stats)
                sampling_stats["bounded_retry_exhausted"] = True
            data.meta_info["terminal_sampling_stats"] = sampling_stats
            advantages, returns = core_algos.compute_terminal_group_advantage(
                token_level_rewards=data.batch["token_level_rewards"],
                response_mask=grpo_calculation_mask,
                index=data.non_tensor_batch["uid"],
                terminal_scores=terminal_scores,
                reward_valid=reward_valid,
                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            )
            train_mask = torch.zeros(advantages.size(0), device=advantages.device, dtype=advantages.dtype)
            if kept_indices:
                train_mask[kept_indices] = 1.0
            advantages = advantages * train_mask.unsqueeze(-1)
            returns = returns * train_mask.unsqueeze(-1)
            # Turn credit is strictly opt-in and runs only after terminal
            # group-relative advantages have been determined.
            turn_credit_config = (config or {}).get("turn_credit", {})
            if not isinstance(turn_credit_config, Mapping):
                turn_credit_config = {}
            turn_mode = str(
                turn_credit_config.get("method", turn_level_method or "off")
            )
            credit_stage = str(
                turn_credit_config.get(
                    "stage",
                    (config or {}).get("turn_credit_stage", "off"),
                )
            ).casefold()
            valid_turn_modes = {
                "off",
                "none",
                "equalized",
                "r2g",
                "em",
                "component_attribution",
                "behavior_component",
                "behavior_delta",
            }
            if turn_mode.casefold() not in valid_turn_modes:
                raise ValueError(f"invalid turn credit method: {turn_mode}")
            if credit_stage not in {"off", "shadow", "train"}:
                raise ValueError(f"invalid turn credit stage: {credit_stage}")
            if turn_mode.casefold() not in {"off", "none"} and credit_stage in {"shadow", "train"}:
                histories = data.non_tensor_batch.get("conversation_histories", [])
                if hasattr(histories, "tolist"):
                    histories = histories.tolist()
                histories = [h[0] if isinstance(h, list) and len(h) == 1 else h for h in histories]
                if turn_mode.casefold() in {
                    "component_attribution",
                    "behavior_component",
                    "behavior_delta",
                }:
                    component_payload = _extract_terminal_reward_components(
                        data,
                        terminal_scores,
                    )
                    if component_payload is None:
                        if credit_stage == "train":
                            raise ValueError(
                                "component turn credit requires complete TravelGym reward metadata"
                            )
                        credited_advantages = advantages
                        diagnostics = {"component_metadata_missing": 1.0}
                    else:
                        component_names, reward_components = component_payload
                        component_advantages = core_algos.compute_terminal_component_advantages(
                            terminal_scores,
                            reward_components,
                            data.non_tensor_batch["uid"],
                            reward_valid=reward_valid,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                        )
                        component_advantages = component_advantages * train_mask.unsqueeze(-1)
                        routing = turn_credit_config.get("routing", {})
                        if OmegaConf.is_config(routing):
                            routing = OmegaConf.to_container(routing, resolve=True)
                        credited_advantages, diagnostics = (
                            core_algos.redistribute_behavior_component_turn_credit(
                                advantages,
                                grpo_calculation_mask,
                                data.batch["turn_boundaries"],
                                histories,
                                component_advantages,
                                component_names,
                                mix_ratio=float(turn_credit_config.get("mix_ratio", 0.30)),
                                routing=routing,
                            )
                        )
                else:
                    credited_advantages = core_algos.redistribute_terminal_turn_credit(
                        advantages,
                        grpo_calculation_mask,
                        data.batch["turn_boundaries"],
                        histories,
                    )
                    diagnostics = {
                        "conservation_error": float(
                            core_algos.turn_credit_conservation_error(
                                advantages,
                                credited_advantages,
                                grpo_calculation_mask,
                            ).detach().cpu()
                        )
                    }
                advantages = _apply_turn_credit_stage(
                    advantages,
                    credited_advantages,
                    grpo_calculation_mask,
                    diagnostics,
                    stage=credit_stage,
                    conservation_atol=float(turn_credit_config.get("conservation_atol", 1.0e-5)),
                    conservation_rtol=float(turn_credit_config.get("conservation_rtol", 1.0e-6)),
                )
                returns = advantages.clone()
                data.meta_info["turn_credit_conservation_error"] = diagnostics["conservation_abs_error"]
                data.meta_info["turn_credit_applied"] = diagnostics["applied"]
                data.meta_info["turn_credit_diagnostics"] = diagnostics
        else:
            # Generic non-terminal batches retain the ordinary API; the
            # TravelGym path above computes terminal-only scores.
            conversation_histories = data.non_tensor_batch.get("conversation_histories", [])
            if hasattr(conversation_histories, "tolist"):
                conversation_histories = conversation_histories.tolist()
            conversation_histories = [
                c[0] if isinstance(c, (list, tuple)) and len(c) == 1 else c
                for c in conversation_histories
            ]
            data_sources = data.non_tensor_batch.get("data_source", [])
            if hasattr(data_sources, "tolist"):
                data_sources = data_sources.tolist()
            advantages, returns = core_algos.compute_grpo_multiturn_advantage(
                token_level_rewards=data.batch["token_level_rewards"],
                response_mask=grpo_calculation_mask,
                turn_boundaries=data.batch["turn_boundaries"],
                conversation_histories=conversation_histories,
                data_sources=data_sources,
                index=data.non_tensor_batch["uid"],
                gamma=gamma,
                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                turn_level_method="off",
                trajectory_score_method="Terminal",
            )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        smoke_val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name="cuda",
    ):
        """Initialize distributed PPO trainer with Ray backend."""

        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn
        self.smoke_val_dataset = smoke_val_dataset

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name
        self.validation_generations_logger = ValidationGenerationsLogger()

        # Passive trainer-side hard-case observer.  It only records complete
        # valid all-zero TravelGym groups; it is deliberately absent from
        # Actor inputs, sampling, rewards and advantage computation.
        self.hard_case_pool = None
        hard_case_config = config.algorithm.get("hard_case_pool", {})
        if bool(hard_case_config.get("enable", False)):
            from verl.trainer.ppo.hard_case_pool import HardCasePool

            pool_path = hard_case_config.get("path") or os.path.join(
                str(config.trainer.get("default_local_dir", "checkpoints")), "hard_case_pool.json"
            )
            self.hard_case_pool = HardCasePool(
                pool_path,
                threshold=int(hard_case_config.get("threshold", 3)),
                reward_version=str(hard_case_config.get("reward_version", "travelgym-terminal-v2")),
                enabled=True,
                rank=0,
            )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
            AdvantageEstimator.GRPO,
            AdvantageEstimator.GRPO_PASSK,
            AdvantageEstimator.REINFORCE_PLUS_PLUS,
            AdvantageEstimator.REMAX,
            AdvantageEstimator.RLOO,
            AdvantageEstimator.OPO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
            AdvantageEstimator.GRPO_MULTITURN,
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError

        self._validate_config()
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)
        self.best_valid_score = -1

    @staticmethod
    def _stable_grpo_namespace(metrics: Mapping[str, Any], *, validation: bool = False) -> dict[str, Any]:
        """Map legacy VERL names into the experiment's stable namespaces."""
        prefix = "grpo/val/" if validation else "grpo/train/"
        result = {}
        for key, value in metrics.items():
            text = str(key)
            if text.startswith("grpo/"):
                result[text] = value
            else:
                is_validation_metric = validation or text.startswith(("val", "smoke20/"))
                target_prefix = "grpo/val/" if is_validation_metric else prefix
                text = text.replace("training/", "", 1).replace("val-core/", "", 1).replace("val-aux/", "", 1)
                result[target_prefix + text] = value
        return result

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes
        if config.actor_rollout_ref.actor.strategy == "megatron":
            model_parallel_size = config.actor_rollout_ref.actor.megatron.tensor_model_parallel_size * config.actor_rollout_ref.actor.megatron.pipeline_model_parallel_size
            assert n_gpus % (model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size) == 0, f"n_gpus ({n_gpus}) must be divisible by model_parallel_size ({model_parallel_size}) times context_parallel_size ({config.actor_rollout_ref.actor.megatron.context_parallel_size})"
            megatron_dp = n_gpus // (model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size)
            minimal_bsz = megatron_dp * config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu
        else:
            minimal_bsz = n_gpus

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % minimal_bsz == 0, f"real_train_batch_size ({real_train_batch_size}) must be divisible by minimal possible batch size ({minimal_bsz})"

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            settings = {
                "actor_rollout_ref.actor": "micro_batch_size",
                "critic": "micro_batch_size",
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove '{name}.{param}' because only '*_{param_per_gpu}'" + "is supported (the former is deprecated).")

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.actor.ppo_micro_batch_size,
                config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                "actor_rollout_ref.actor",
            )

            if self.use_reference_policy:
                # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
                check_mutually_exclusive(
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                    "actor_rollout_ref.ref",
                )

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.rollout",
            )

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu, "critic")

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu, "reward_model")

        # Actor
        # check if train_batch_size is larger than ppo_mini_batch_size
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.actor_rollout_ref.actor.ppo_mini_batch_size
            sp_size = config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        assert config.actor_rollout_ref.actor.loss_agg_mode in [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        ], f"Invalid loss_agg_mode: {config.actor_rollout_ref.actor.loss_agg_mode}"

        if config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
            print("NOTICE: You have both enabled in-reward kl and kl loss.")

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size
            sp_size = config.critic.get("ulysses_sequence_parallel_size", 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == "fsdp" and (config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1) > 1 or config.actor_rollout_ref.ref.get("ulysses_sequence_parallel_size", 1) > 1):
            assert config.actor_rollout_ref.model.use_remove_padding, "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."

        if self.use_critic and config.critic.strategy == "fsdp":
            if config.critic.get("ulysses_sequence_parallel_size", 1) > 1:
                assert config.critic.model.use_remove_padding, "When using sequence parallelism for critic, you must enable `use_remove_padding`."

        if config.data.get("val_batch_size", None) is not None:
            print("WARNING: val_batch_size is deprecated." + " Validation datasets are sent to inference engines as a whole batch," + " which will schedule the memory themselves.")

        # check eval config
        try:
            validation_pass_k = int(config.trainer.get("validation_pass_k", 1))
        except (TypeError, ValueError) as exc:
            raise ExperimentIntegrityError(
                "trainer.validation_pass_k must be a positive integer"
            ) from exc
        if validation_pass_k < 1:
            raise ExperimentIntegrityError(
                "trainer.validation_pass_k must be a positive integer"
            )
        if bool(config.trainer.get("val_only", False)) and int(
            config.trainer.get("validation_retry_attempts", 0)
        ):
            raise ExperimentIntegrityError(
                "trainer.val_only cannot combine validation_retry_attempts; "
                "pass@k attempts must be independent"
            )
        val_kwargs = config.actor_rollout_ref.rollout.val_kwargs
        validation_do_sample = bool(val_kwargs.get("do_sample", False))
        if validation_do_sample:
            try:
                validation_temperature = float(val_kwargs.get("temperature", 0.0))
            except (TypeError, ValueError) as exc:
                raise ExperimentIntegrityError(
                    "validation sampling temperature must be a positive finite number"
                ) from exc
            if not np.isfinite(validation_temperature) or validation_temperature <= 0:
                raise ExperimentIntegrityError(
                    "validation sampling temperature must be a positive finite number"
                )
        if validation_pass_k > 1 and not validation_do_sample:
            raise ExperimentIntegrityError(
                "task-level pass@k validation requires val_kwargs.do_sample=true "
                "so attempts are independent"
            )

        # check multi_turn with tool config
        if config.actor_rollout_ref.rollout.multi_turn.enable:
            assert config.actor_rollout_ref.rollout.multi_turn.tool_config_path is not None, "tool_config_path must be set when enabling multi_turn with tool, due to no role-playing support"
            assert config.algorithm.adv_estimator in [AdvantageEstimator.GRPO, AdvantageEstimator.GRPO_MULTITURN], "only GRPO is tested for multi-turn with tool"

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                task_pool_name=self.config.data.get(
                    "task_pool_train_name", self.config.data.get("task_pool_name", None)
                ),
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                task_pool_name=self.config.data.get(
                    "task_pool_val_name", self.config.data.get("task_pool_name", None)
                ),
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )
        self.smoke_val_dataloader = None
        if self.smoke_val_dataset is not None:
            self.smoke_val_dataloader = StatefulDataLoader(
                dataset=self.smoke_val_dataset,
                batch_size=len(self.smoke_val_dataset),
                num_workers=self.config.data.get("dataloader_num_workers", 8),
                shuffle=False,
                drop_last=False,
                collate_fn=collate_fn,
            )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: {len(self.val_dataloader)}")

        # Production and the two overfit diagnostics have separate, fixed
        # optimizer horizons and legal process stop points.
        (
            self.training_profile,
            expected_total_steps,
            self.allowed_run_until_steps,
        ) = resolve_training_step_policy(
            self.config.trainer.get("experiment_profile", "production")
        )
        total_training_steps = validate_total_training_steps(
            self.config.trainer.get("total_training_steps", None),
            expected=expected_total_steps,
        )
        self.run_until_step = validate_process_run_until_step(
            self.config.trainer.get("run_until_step", None),
            total_training_steps=total_training_steps,
            allowed_steps=self.allowed_run_until_steps,
        )
        configured_milestones = self.config.trainer.get(
            "milestones", list(self.allowed_run_until_steps)
        )
        self.milestones = sorted({int(value) for value in configured_milestones if 0 < int(value) <= total_training_steps})
        configured_save_freq = self.config.trainer.get("save_freq", -1)
        try:
            self.save_freq = -1 if configured_save_freq is None else int(configured_save_freq)
        except (TypeError, ValueError) as exc:
            raise ExperimentIntegrityError(
                f"trainer.save_freq must be -1 or a positive integer, got {configured_save_freq!r}"
            ) from exc
        if self.save_freq == 0 or self.save_freq < -1:
            raise ExperimentIntegrityError(
                f"trainer.save_freq must be -1 or a positive integer, got {self.save_freq}"
            )
        if self.device_name == "cuda" and self.run_until_step == 0:
            raise ExperimentIntegrityError("run_until_step=0 is only useful for a dry-run, not GPU training")

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(
        self,
        inputs,
        outputs,
        scores,
        reward_extra_infos_dict,
        dump_path,
        rollout_metadata=None,
    ):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v
        for key, values in (rollout_metadata or {}).items():
            if len(values) == n:
                base_data[key] = values.tolist() if hasattr(values, "tolist") else values

        with open(filename, "w") as f:
            for i in range(n):
                entry = {k: v[i] for k, v in base_data.items()}
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"Dumped generations to {filename}")

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _generate_validation_batch_once(self, test_gen_batch: DataProto):
        """Generate exactly one validation response for every input row.

        This helper is deliberately separate from invalid-row retries.  A
        pass@k evaluator must count each fresh generation as an attempt; a
        retry inside one attempt would silently change the denominator.
        """
        padded_batch, pad_size = pad_dataproto_to_divisor(
            test_gen_batch, self.actor_rollout_wg.world_size
        )
        if not self.async_rollout_mode:
            padded_output = self.actor_rollout_wg.generate_sequences(padded_batch)
        else:
            self.async_rollout_manager.wake_up()
            try:
                padded_output = self.async_rollout_manager.generate_sequences(
                    padded_batch
                )
            finally:
                self.async_rollout_manager.sleep()
        generated = unpad_dataproto(padded_output, pad_size=pad_size)
        if len(generated) != len(test_gen_batch):
            raise ExperimentIntegrityError(
                "validation rollout changed the task-row count: "
                f"requested={len(test_gen_batch)}, generated={len(generated)}"
            )
        # Rollout diagnostics are trainer-private training metrics.  A
        # validation output is unioned with its source batch, whose meta_info
        # must remain stable across attempts/batches; retaining per-rollout
        # timing/length values would make that union fail on the next attempt.
        generated.meta_info.pop("timing", None)
        generated.meta_info.pop("travel_rollout_length", None)
        return generated

    def _generate_validation_batch_with_retries(self, test_gen_batch: DataProto):
        """Retry invalid interaction rows with fresh validation rollouts."""
        max_retries = max(
            0, int(self.config.trainer.get("validation_retry_attempts", 0))
        )
        output = None
        pending_indices = list(range(len(test_gen_batch)))
        pending_batch = test_gen_batch
        initial_invalid_rows = 0
        retried_rows = 0
        final_invalid_rows = 0

        for attempt in range(max_retries + 1):
            generated = self._generate_validation_batch_once(pending_batch)

            if output is None:
                output = generated
            else:
                for key in output.batch.keys():
                    output.batch[key][pending_indices] = generated.batch[key]
                for key in output.non_tensor_batch:
                    output.non_tensor_batch[key][pending_indices] = (
                        generated.non_tensor_batch[key]
                    )

            _, reward_valid = _extract_terminal_reward_metadata(output)
            if reward_valid is None:
                # Generic validation has no interaction validity contract.
                pending_indices = []
                break
            pending_indices = (
                (~reward_valid).nonzero(as_tuple=False).flatten().cpu().tolist()
            )
            final_invalid_rows = len(pending_indices)
            if attempt == 0:
                initial_invalid_rows = final_invalid_rows
            if not pending_indices or attempt == max_retries:
                break

            retried_rows += len(pending_indices)
            print(
                "[validation-retry] "
                f"retry={attempt + 1}/{max_retries} "
                f"invalid_rows={len(pending_indices)}"
            )
            pending_batch = test_gen_batch.select_idxs(pending_indices)

        if output is None:
            raise RuntimeError("validation generation produced no output")
        stats = {
            "retried_rows": retried_rows,
            "recovered_rows": initial_invalid_rows - final_invalid_rows,
            "final_invalid_rows": final_invalid_rows,
        }
        if retried_rows or final_invalid_rows:
            print(f"[validation-retry] summary={stats}")
        return output, stats

    @staticmethod
    def _validation_values(batch: DataProto, key: str) -> list:
        values = batch.non_tensor_batch.get(key)
        if values is None:
            return []
        if hasattr(values, "tolist"):
            values = values.tolist()
        if not isinstance(values, (list, tuple)):
            values = [values]
        return list(values)

    @classmethod
    def _validation_task_keys(cls, batch: DataProto) -> list[str]:
        """Return strict task identities for task-level pass accounting."""
        task_ids = cls._validation_values(batch, "task_id")
        if not task_ids:
            reward_models = cls._validation_values(batch, "reward_model")
            task_ids = [
                item.get("id") if isinstance(item, Mapping) else None
                for item in reward_models
            ]
        if len(task_ids) != len(batch):
            raise ExperimentIntegrityError(
                "task-level validation requires one reward_model.id/task_id per row: "
                f"rows={len(batch)}, task_ids={len(task_ids)}"
            )

        env_names = cls._validation_values(batch, "_travel_source_env_name")
        if env_names and len(env_names) != len(batch):
            raise ExperimentIntegrityError(
                "validation task provenance is misaligned: "
                f"rows={len(batch)}, env_names={len(env_names)}"
            )

        task_keys: list[str] = []
        seen: set[str] = set()
        for index, raw_task_id in enumerate(task_ids):
            if isinstance(raw_task_id, Mapping):
                raw_task_id = raw_task_id.get("id")
            task_id = str(raw_task_id or "").strip()
            if not task_id:
                raise ExperimentIntegrityError(
                    f"validation row {index} has no stable task identity"
                )
            env_name = ""
            if env_names:
                env_name = str(env_names[index] or "").strip()
            task_key = f"{env_name}::{task_id}" if env_name else task_id
            if task_key in seen:
                raise ExperimentIntegrityError(
                    f"validation batch contains duplicate task identity {task_key!r}"
                )
            seen.add(task_key)
            task_keys.append(task_key)
        return task_keys

    def _prepare_validation_generation(self, test_batch: DataProto):
        input_ids = test_batch.batch["input_ids"]
        input_texts = [
            self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids
        ]
        batch_keys_to_pop = [
            key for key in ("input_ids", "attention_mask", "position_ids")
            if key in test_batch.batch.keys()
        ]
        non_tensor_batch_keys_to_pop = [
            key
            for key in (
                "raw_prompt_ids",
                "_travel_source_env_name",
                "multi_modal_data",
                "raw_prompt",
                "tools_kwargs",
            )
            if key in test_batch.non_tensor_batch
        ]
        test_gen_batch = test_batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
        )
        test_gen_batch.meta_info = {
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "recompute_log_prob": False,
            "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
            "validate": True,
        }
        print(f"validation pass meta info: {test_gen_batch.meta_info}")
        return input_texts, test_gen_batch

    @staticmethod
    def _public_validation_scalar(value: Any):
        while isinstance(value, (list, tuple)) and len(value) == 1:
            value = value[0]
        if hasattr(value, "item"):
            try:
                value = value.item()
            except (TypeError, ValueError):
                pass
        if isinstance(value, (str, bool, int, float)):
            return value
        if isinstance(value, np.generic):
            return value.item()
        return str(value)

    @staticmethod
    def _validation_json_value(value: Any):
        """Convert rollout ledgers to JSON without flattening their structure."""
        if isinstance(value, np.ndarray):
            return RayPPOTrainer._validation_json_value(value.tolist())
        if isinstance(value, np.generic):
            return value.item()
        if hasattr(value, "model_dump"):
            return RayPPOTrainer._validation_json_value(value.model_dump(exclude_none=True))
        if isinstance(value, Mapping):
            return {
                str(key): RayPPOTrainer._validation_json_value(child)
                for key, child in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [RayPPOTrainer._validation_json_value(child) for child in value]
        if value is None or isinstance(value, (str, bool, int, float)):
            return value
        return str(value)

    @staticmethod
    def _validation_number(value: Any, default: float = 0.0) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return default
        return number if np.isfinite(number) else default

    def _dump_validation_passes(
        self,
        records: list[Mapping[str, Any]],
        summary: Mapping[str, Any],
        dump_path: str,
        *,
        pass_k: int,
        smoke: bool,
        early_stop: bool,
    ) -> None:
        os.makedirs(dump_path, exist_ok=True)
        prefix = f"{self.global_steps}_pass{pass_k}"
        generations_path = os.path.join(dump_path, f"{prefix}.jsonl")
        allowed = set(PUBLIC_VALIDATION_METRICS) | {
            "task_id",
            "attempt",
            "input",
            "output",
            "score",
            "reward_valid",
            "reward_version",
            "segment_records",
            "cleanup_events",
            "archive_messages",
            "archive_model_outputs",
            "archive_segment_records",
            "archive_turns",
            "length_events",
        }
        with open(generations_path, "w", encoding="utf-8") as handle:
            for record in records:
                payload = {}
                for key, value in record.items():
                    if key not in allowed:
                        continue
                    if key in {
                        "segment_records",
                        "cleanup_events",
                        "archive_messages",
                        "archive_model_outputs",
                        "archive_segment_records",
                        "archive_turns",
                        "length_events",
                    }:
                        payload[key] = self._validation_json_value(value)
                    else:
                        payload[key] = self._public_validation_scalar(value)
                payload["step"] = int(self.global_steps)
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

        rollout_config = self.config.actor_rollout_ref.rollout
        multi_turn_config = rollout_config.get("multi_turn", {})
        context_cleanup_config = multi_turn_config.get("context_cleanup", {}) or {}
        protocol = {
            "schema_version": "travelgym-native-validation-v1",
            "backend": str(rollout_config.get("name", "unknown")),
            "native_two_stage": bool(multi_turn_config.get("enable", False)),
            "template_prefill": "<think>" if bool(multi_turn_config.get("enable_thinking", False)) else "",
            "reasoning_max_tokens": int(multi_turn_config.get("max_reasoning_tokens_per_turn", 0)),
            "tool_call_max_tokens": int(multi_turn_config.get("max_tool_call_tokens_per_turn", 0)),
            "max_new_tokens_per_turn": int(multi_turn_config.get("max_new_tokens_per_turn", 0)),
            "tool_response_token_reserve": int(multi_turn_config.get("tool_response_token_reserve", 0)),
            "template_token_reserve": int(multi_turn_config.get("template_token_reserve", 0)),
            "context_cleanup_enabled": bool(
                context_cleanup_config.get("enabled", False)
            ),
            "context_cleanup_target_tokens": int(
                context_cleanup_config.get("target_context_tokens", 20000)
            ),
            "context_cleanup_template_margin_tokens": int(
                context_cleanup_config.get("template_margin_tokens", 32)
            ),
            "next_turn_reserve": int(
                multi_turn_config.get("max_reasoning_tokens_per_turn", 0)
                + multi_turn_config.get("max_tool_call_tokens_per_turn", 0)
                + multi_turn_config.get("tool_response_token_reserve", 0)
                + context_cleanup_config.get("template_margin_tokens", 32)
            ),
            "tool_call_parser": str(multi_turn_config.get("tool_call_parser", "")),
            "do_sample": bool(rollout_config.val_kwargs.get("do_sample", False)),
            "temperature": float(rollout_config.val_kwargs.get("temperature", 0.0)),
            "top_p": float(rollout_config.val_kwargs.get("top_p", 1.0)),
            "top_k": int(rollout_config.val_kwargs.get("top_k", -1)),
            "forced_reasoning_end": True,
            "forced_reasoning_end_loss_mask": 0,
            "pass_k": int(pass_k),
            "task_level_early_stop": bool(early_stop),
            "validation_retry_attempts": 0,
            "split": "validation_smoke" if smoke else "validation",
            "eos_token_id": int(self.tokenizer.eos_token_id),
            "pad_token_id": int(self.tokenizer.pad_token_id),
        }
        summary_path = os.path.join(dump_path, f"{prefix}_summary.json")
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "schema_version": "travelgym-validation-pass-summary-v1",
                    "step": int(self.global_steps),
                    "protocol": protocol,
                    "summary": dict(summary),
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )
            handle.write("\n")

        # The corrected SFT smoke20 result can be consumed directly by the
        # formal GRPO launcher as its immutable step-0 baseline.  Keep this
        # opt-in and fail closed on an existing target so an old/generic HTTP
        # result cannot be silently replaced or silently reused.
        baseline_output_path = self.config.trainer.get(
            "validation_baseline_output_path", None
        )
        if baseline_output_path and self.global_steps == 0:
            baseline_path = os.path.abspath(
                os.path.expanduser(str(baseline_output_path))
            )
            if os.path.exists(baseline_path):
                raise ExperimentIntegrityError(
                    f"validation baseline output already exists: {baseline_path}"
                )
            os.makedirs(os.path.dirname(baseline_path), exist_ok=True)
            metric_prefix = "grpo/val/smoke20/" if smoke else "grpo/val/validation/"
            baseline_metrics = {
                f"{metric_prefix}{key}": float(value)
                for key, value in summary.items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            }
            baseline_payload = {
                "schema_version": "travelgym-grpo-validation-baseline-v1",
                "step": 0,
                "source": "native_validation",
                "protocol": protocol,
                "step0_metrics": baseline_metrics,
            }
            temporary_path = f"{baseline_path}.tmp"
            with open(temporary_path, "w", encoding="utf-8") as handle:
                json.dump(baseline_payload, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            os.replace(temporary_path, baseline_path)
            print(f"Wrote immutable step-0 validation baseline to {baseline_path}")
        print(f"Dumped validation pass generations to {generations_path}")
        print(f"Dumped validation pass summary to {summary_path}")

    def _validate_task_passes(self, *, pass_k: int, smoke: bool = False):
        """Run native validation as task-level pass@k with optional early stop."""
        try:
            pass_k = int(pass_k)
        except (TypeError, ValueError) as exc:
            raise ExperimentIntegrityError(
                f"trainer.validation_pass_k must be a positive integer, got {pass_k!r}"
            ) from exc
        if pass_k < 1:
            raise ExperimentIntegrityError(
                f"trainer.validation_pass_k must be a positive integer, got {pass_k!r}"
            )
        retry_attempts = int(self.config.trainer.get("validation_retry_attempts", 0))
        if retry_attempts:
            raise ExperimentIntegrityError(
                "task-level pass@k validation requires "
                "trainer.validation_retry_attempts=0; each attempt is a fresh rollout"
            )
        early_stop = bool(
            self.config.trainer.get("validation_task_level_early_stop", True)
        )
        dataloader = self.smoke_val_dataloader if smoke else self.val_dataloader
        if dataloader is None:
            dataloader = self.val_dataloader
        expected_tasks = len(getattr(dataloader, "dataset", []))
        attempts_by_task: dict[str, list[dict[str, Any]]] = {}
        seen_task_keys: set[str] = set()
        all_inputs: list[str] = []
        all_outputs: list[str] = []
        all_scores: list[float] = []
        all_records: list[dict[str, Any]] = []

        for test_data in dataloader:
            original_batch = DataProto.from_single_dict(test_data)
            task_keys = self._validation_task_keys(original_batch)
            duplicate_keys = seen_task_keys.intersection(task_keys)
            if duplicate_keys:
                raise ExperimentIntegrityError(
                    "validation dataloader yielded a task more than once: "
                    f"{sorted(duplicate_keys)[:3]}"
                )
            seen_task_keys.update(task_keys)
            pending_indices = list(range(len(original_batch)))

            for attempt_number in range(1, pass_k + 1):
                if not pending_indices:
                    break
                attempt_batch = original_batch.select_idxs(pending_indices)
                attempt_keys = [task_keys[index] for index in pending_indices]
                input_texts, test_gen_batch = self._prepare_validation_generation(
                    attempt_batch
                )
                generated = self._generate_validation_batch_once(test_gen_batch)
                output_ids = generated.batch["responses"]
                output_texts = [
                    self.tokenizer.decode(ids, skip_special_tokens=True)
                    for ids in output_ids
                ]
                evaluated_batch = attempt_batch.union(generated)
                result = self.val_reward_fn(evaluated_batch, return_dict=True)
                reward_tensor = result["reward_tensor"]
                if int(reward_tensor.shape[0]) != len(pending_indices):
                    raise ExperimentIntegrityError(
                        "validation reward row count is misaligned: "
                        f"tasks={len(pending_indices)}, rewards={reward_tensor.shape[0]}"
                    )
                scores = reward_tensor.sum(-1).detach().cpu().tolist()
                extra_info = result.get("reward_extra_info", {})
                if not isinstance(extra_info, Mapping):
                    raise ExperimentIntegrityError(
                        "native pass@k validation requires row-aligned reward_extra_info"
                    )
                extra: dict[str, list] = {}
                for key, values in extra_info.items():
                    if hasattr(values, "tolist"):
                        values = values.tolist()
                    if not isinstance(values, (list, tuple)) or len(values) != len(pending_indices):
                        raise ExperimentIntegrityError(
                            "validation reward metadata is not row-aligned: "
                            f"{key} has {len(values) if hasattr(values, '__len__') else 'unknown'} rows, "
                            f"expected {len(pending_indices)}"
                        )
                    extra[str(key)] = list(values)
                missing = {
                    "completion_success",
                    "reward_valid",
                }.difference(extra)
                if missing:
                    raise ExperimentIntegrityError(
                        "native pass@k validation is missing public reward fields: "
                        f"{sorted(missing)}"
                    )
                if "terminal_reward" not in extra:
                    extra["terminal_reward"] = list(scores)

                for local_index, (global_index, task_key) in enumerate(
                    zip(pending_indices, attempt_keys)
                ):
                    record: dict[str, Any] = {
                        "task_id": task_key,
                        "attempt": int(attempt_number),
                        "input": input_texts[local_index],
                        "output": output_texts[local_index],
                        "score": float(scores[local_index]),
                    }
                    for metric in PUBLIC_VALIDATION_METRICS:
                        if metric in extra:
                            record[metric] = self._public_validation_scalar(
                                extra[metric][local_index]
                            )
                    for metric in ("reward_valid", "reward_version"):
                        if metric in extra:
                            record[metric] = self._public_validation_scalar(
                                extra[metric][local_index]
                            )
                    for key in (
                        "segment_records",
                        "cleanup_events",
                        "archive_messages",
                        "archive_model_outputs",
                        "archive_segment_records",
                        "archive_turns",
                        "length_events",
                    ):
                        values = generated.non_tensor_batch.get(key)
                        if values is not None and local_index < len(values):
                            record[key] = self._validation_json_value(
                                values[local_index]
                            )
                    attempts_by_task.setdefault(task_key, []).append(record)
                    all_records.append(record)
                    all_inputs.append(record["input"])
                    all_outputs.append(record["output"])
                    all_scores.append(record["score"])

                if early_stop:
                    pending_indices = [
                        global_index
                        for local_index, global_index in enumerate(pending_indices)
                        if self._validation_number(
                            extra["completion_success"][local_index]
                        )
                        != 1.0
                    ]
                elif attempt_number < pass_k:
                    pending_indices = list(range(len(original_batch)))
                else:
                    pending_indices = []

                completed_tasks = sum(
                    any(
                        self._validation_number(row.get("completion_success"))
                        == 1.0
                        for row in rows
                    )
                    for rows in attempts_by_task.values()
                )
                print(
                    "[validation-pass] "
                    f"attempt={attempt_number}/{pass_k} "
                    f"completed_tasks={completed_tasks}/{expected_tasks or len(seen_task_keys)} "
                    f"pending_in_batch={len(pending_indices)}"
                )

        if expected_tasks and len(attempts_by_task) != expected_tasks:
            raise ExperimentIntegrityError(
                "validation task count changed during pass@k evaluation: "
                f"expected={expected_tasks}, observed={len(attempts_by_task)}"
            )
        if not attempts_by_task:
            raise ExperimentIntegrityError("native validation produced no task attempts")

        summary = aggregate_validation_attempts(
            attempts_by_task,
            pass_k=pass_k,
        )
        # Keep a compact, stable view alongside the richer mean@N/mean_best
        # validation table so SwanLab can plot the final reward and every
        # public terminal-reward component directly.
        component_infos = {
            "reward": [record["score"] for record in all_records],
        }
        for metric in PUBLIC_VALIDATION_METRICS:
            component_infos[metric] = [record.get(metric, 0.0) for record in all_records]
        summary.update(compute_reward_component_metrics(component_infos))
        if self.global_steps == 0 and bool(
            self.config.trainer.get("initial_rollout_health_gate", False)
        ):
            validate_initial_rollout_health(
                all_outputs,
                response_token_limit=int(self.config.data.max_response_length),
            )
        self._maybe_log_val_generations(
            inputs=all_inputs,
            outputs=all_outputs,
            scores=all_scores,
        )
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_validation_passes(
                all_records,
                summary,
                str(val_data_dir),
                pass_k=pass_k,
                smoke=smoke,
                early_stop=early_stop,
            )

        prefix = "smoke20/" if smoke else "validation/"
        return {f"{prefix}{key}": value for key, value in summary.items()}

    def _validate(self, *, save_best: bool = True):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []
        validation_rollout_metadata: dict[str, list] = defaultdict(list)
        rollout_metadata_keys = (
            "segment_records",
            "cleanup_events",
            "archive_messages",
            "archive_model_outputs",
            "archive_segment_records",
            "archive_turns",
            "length_events",
        )
        validation_retry_stats = {
            "retried_rows": 0,
            "recovered_rows": 0,
            "final_invalid_rows": 0,
        }

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # repeat test batch
            test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True)

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            # The source variant is a private task-pool provenance field used
            # only for loader/HCP auditing.  Do not carry it into validation
            # generation where generic workers could accidentally serialize
            # it alongside the Actor prompt.
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "_travel_source_env_name"]
            if "multi_modal_data" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            test_gen_batch = test_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            test_output_gen_batch, retry_stats = (
                self._generate_validation_batch_with_retries(test_gen_batch)
            )
            for key, value in retry_stats.items():
                validation_retry_stats[key] += value
            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)
            for key in rollout_metadata_keys:
                values = test_output_gen_batch.non_tensor_batch.get(key)
                if values is None or len(values) != len(output_texts):
                    validation_rollout_metadata[key].extend(
                        [None] * len(output_texts)
                    )
                else:
                    validation_rollout_metadata[key].extend(
                        [self._validation_json_value(value) for value in values]
                    )

            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))
        
        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
                rollout_metadata=validation_rollout_metadata,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        if self.global_steps == 0 and bool(
            self.config.trainer.get("initial_rollout_health_gate", False)
        ):
            validate_initial_rollout_health(
                sample_outputs,
                response_token_limit=int(self.config.data.max_response_length),
            )

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (var_name == core_var) and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"]) and (f"@{n_max}" in metric_name):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val
        for key, value in validation_retry_stats.items():
            metric_dict[f"val-aux/runtime/{key}"] = value
        metric_dict.update(compute_reward_component_metrics(reward_extra_infos_dict))
        
        # get all the metrics start with "val-core"
        core_metrics = [metric_dict[pfx] for pfx in metric_dict.keys() if pfx.startswith("val-core")]
        # get the average value of the core metrics
        avg_core_metric = sum(core_metrics) / len(core_metrics) if core_metrics else 0
        # Smoke validation is a diagnostic view of the fixed 20-task subset;
        # it must never silently become the experiment's selected/final
        # checkpoint.  Callers can therefore disable the upstream best-save
        # side effect while retaining the normal full-validation behavior.
        if save_best and avg_core_metric > self.best_valid_score:
            self.best_valid_score = avg_core_metric
            self._save_checkpoint(valid_save_best=True, valid_best_score=avg_core_metric)

        return metric_dict

    def _validate_smoke(self):
        """Run the fixed validation_smoke subset when a dataset was supplied."""
        if self.smoke_val_dataloader is None:
            return self._validate()
        original = self.val_dataloader
        self.val_dataloader = self.smoke_val_dataloader
        try:
            metrics = self._validate(save_best=False)
        finally:
            self.val_dataloader = original
        return {f"smoke20/{key}": value for key, value in metrics.items()}

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy], config=self.config.actor_rollout_ref, role="ref")
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls, device_name=self.device_name, **wg_kwargs)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.workers.rollout.async_server import AsyncLLMServerManager

            self.async_rollout_mode = True
            self.async_rollout_manager = AsyncLLMServerManager(
                config=self.config,
                worker_group=self.actor_rollout_wg,
            )

    def _save_checkpoint(self, valid_save_best=False, valid_best_score=-1):
        if not valid_save_best:
            # path: given_path + `/global_step_{global_steps}` + `/actor`
            local_global_step_folder = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")
        else:
            local_global_step_folder = os.path.join(self.config.trainer.default_local_dir, f"best_global_step_{self.global_steps}")

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")
        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        # remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        # if remove_previous_ckpt_in_save:
        #     print("Warning: remove_previous_ckpt_in_save is deprecated," + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead")
        # max_actor_ckpt_to_keep = self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        # max_critic_ckpt_to_keep = self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=None)

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=None)

        # save dataloader
        BaseCheckpointManager.local_mkdir(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # Persist the passive Hard Case Pool alongside the model/data state so
        # resume reproduces the zero-group streak exactly.  The pool remains a
        # trainer-private audit artifact; no admitted row is injected into a
        # sampler or training batch.
        if self.hard_case_pool is not None and self.hard_case_pool.rank == 0:
            pool_state_path = os.path.join(local_global_step_folder, "hard_case_pool_state.json")
            temporary_pool_state = pool_state_path + ".tmp"
            with open(temporary_pool_state, "w", encoding="utf-8") as handle:
                json.dump(self.hard_case_pool.state_dict(), handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            os.replace(temporary_pool_state, pool_state_path)
        # Persist RNG state so resumed sampling remains reproducible.
        if getattr(self.actor_rollout_wg, "world_size", 1) == 1 or getattr(self.actor_rollout_wg, "rank", 0) == 0:
            try:
                torch.save(capture_rng_state(), os.path.join(local_global_step_folder, "rng_state.pt"))
            except Exception as exc:
                print(f"Warning: failed to save RNG state: {exc}")

        if not valid_save_best:
            with open(os.path.join(self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"), "w") as f:
                f.write(str(self.global_steps))
        else:
            # save best score
            with open(os.path.join(self.config.trainer.default_local_dir, "best_score.txt"), "w") as f:
                f.write(str(self.global_steps) + ", " + str(valid_best_score))
    
    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load best score
        best_score = get_best_score(self.config.trainer.default_local_dir)
        if best_score is not None:
            self.best_valid_score = best_score

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        # load actor
        self.actor_rollout_wg.load_checkpoint(actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)

        if self.hard_case_pool is not None:
            pool_state_path = os.path.join(global_step_folder, "hard_case_pool_state.json")
            if not os.path.isfile(pool_state_path):
                raise ExperimentIntegrityError(
                    f"checkpoint Hard Case Pool state is missing: {pool_state_path}"
                )
            try:
                with open(pool_state_path, "r", encoding="utf-8") as handle:
                    pool_state = json.load(handle)
                if not isinstance(pool_state, Mapping):
                    raise ExperimentIntegrityError("Hard Case Pool state must be an object")
                self.hard_case_pool.load_state_dict(pool_state)
            except ExperimentIntegrityError:
                raise
            except (OSError, ValueError, TypeError) as exc:
                raise ExperimentIntegrityError(
                    f"unable to restore Hard Case Pool state: {exc}"
                ) from exc

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            raise ExperimentIntegrityError(
                f"checkpoint dataloader state is missing: {dataloader_local_path}"
            )
        rng_path = os.path.join(global_step_folder, "rng_state.pt")
        if os.path.isfile(rng_path):
            try:
                restore_rng_state(torch.load(rng_path, weights_only=False))
            except Exception as exc:
                raise ExperimentIntegrityError(f"unable to restore RNG state: {exc}") from exc
        else:
            raise ExperimentIntegrityError(f"checkpoint RNG state is missing: {rng_path}")

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst, k_partitions=world_size, equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix)
        metrics.update(global_balance_stats)

    @staticmethod
    def _hard_case_metadata(batch: DataProto, key: str, nested_key: str | None = None):
        """Read trainer-private per-prompt provenance without exposing it."""
        values = batch.non_tensor_batch.get(key)
        if values is None:
            return None
        if hasattr(values, "tolist"):
            values = values.tolist()
        if not isinstance(values, (list, tuple)):
            values = [values]
        result = []
        for value in values:
            if nested_key is not None and isinstance(value, Mapping):
                value = value.get(nested_key)
            if value is None:
                return None
            result.append(str(value))
        return result

    @staticmethod
    def _hard_case_task_keys(batch: DataProto):
        """Return private ``env_name::task_id`` keys for Hard Case auditing.

        The parquet loader retains the source composition in the private
        ``_travel_source_env_name`` column. Prefixing here prevents identical
        task IDs from different TravelGym variants from sharing a zero-reward
        streak, while keeping the field out of the generated prompt.
        """

        task_ids = RayPPOTrainer._hard_case_metadata(batch, "task_id")
        if task_ids is None:
            task_ids = RayPPOTrainer._hard_case_metadata(batch, "reward_model", nested_key="id")
        if task_ids is None:
            return None
        env_names = RayPPOTrainer._hard_case_metadata(batch, "_travel_source_env_name")
        if env_names is not None and len(env_names) != len(task_ids):
            return None
        from verl.trainer.ppo.hard_case_pool import compose_task_key

        return [
            compose_task_key(task_id, env_names[index] if env_names is not None else None)
            for index, task_id in enumerate(task_ids)
        ]

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # TravelGym's terminal-only GRPO path may request bounded resampling.
        # The wrapper is installed on the driver after workers are created, so
        # it can retry complete rollout batches without changing environment
        # state or exposing private reward fields to the policy.
        dynamic_sampling_config = self.config.algorithm.get("dynamic_sampling", {})
        if bool(dynamic_sampling_config.get("enable", False)) and not bool(
            self.config.trainer.get("val_only", False)
        ):
            from verl.trainer.ppo.dynamic_sampling import install_verl_bounded_sampler

            group_size = int(self.config.actor_rollout_ref.rollout.n)
            install_verl_bounded_sampler(
                self.actor_rollout_wg,
                dynamic_sampling_config,
                group_size=group_size,
            )
            if self.async_rollout_mode:
                install_verl_bounded_sampler(
                    self.async_rollout_manager,
                    dynamic_sampling_config,
                    group_size=group_size,
                )

        # load checkpoint before doing anything
        self._load_checkpoint()

        # A validation-only job is an explicit native-evaluation mode.  It
        # must run before baseline reuse and before any PPO horizon checks, so
        # it cannot accidentally start training or silently skip live eval.
        if bool(self.config.trainer.get("val_only", False)):
            if self.val_reward_fn is None:
                raise ExperimentIntegrityError(
                    "trainer.val_only requires a validation reward function"
                )
            pass_k = int(self.config.trainer.get("validation_pass_k", 1))
            val_metrics = self._validate_task_passes(
                pass_k=pass_k,
                smoke=bool(self.config.trainer.get("initial_validation_smoke", False)),
            )
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Validation-only metrics: {val_metrics}")
            logger.log(
                data=self._stable_grpo_namespace(val_metrics, validation=True),
                step=self.global_steps,
            )
            return

        # Reuse the completed merged-SFT evaluation as the step-0 Validation
        # record for formal GRPO; this avoids a second rollout.
        baseline_path = self.config.trainer.get("initial_validation_baseline_path", None)
        validation_disabled = bool(self.config.trainer.get("disable_validation", False))
        val_before_train = bool(self.config.trainer.get("val_before_train", True))
        if (
            not validation_disabled
            and baseline_path
            and not val_before_train
            and self.global_steps == 0
        ):
            try:
                baseline_metrics = load_step0_validation_metrics(baseline_path)
            except (FileNotFoundError, ValidationBaselineError, OSError) as exc:
                resolved = resolve_validation_baseline_path(baseline_path)
                raise ExperimentIntegrityError(
                    f"configured step-0 validation baseline is unusable: {resolved}"
                ) from exc
            pprint(
                "Initial validation baseline reused: "
                f"{resolve_validation_baseline_path(baseline_path)}"
            )
            logger.log(data=baseline_metrics, step=self.global_steps)


        # perform validation before training
        # currently, we only support validation using the reward_function.
        elif (
            not validation_disabled
            and self.val_reward_fn is not None
            and self.config.trainer.get("val_before_train", True)
        ):
            # TravelGym's formal run uses the fixed 20-task smoke view at
            # both boundaries.  Keep the generic trainer's full-validation
            # default unless the experiment opts into this diagnostic view.
            validation_pass_k = int(self.config.trainer.get("validation_pass_k", 1))
            if validation_pass_k > 1:
                val_metrics = self._validate_task_passes(
                    pass_k=validation_pass_k,
                    smoke=bool(self.config.trainer.get("initial_validation_smoke", False)),
                )
            elif bool(self.config.trainer.get("initial_validation_smoke", False)):
                val_metrics = self._validate_smoke()
            else:
                val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=self._stable_grpo_namespace(val_metrics, validation=True), step=self.global_steps)

        # Reuse the profile policy resolved during trainer initialization.
        run_until_step = validate_process_run_until_step(
            self.config.trainer.get("run_until_step", None),
            total_training_steps=self.total_training_steps,
            allowed_steps=self.allowed_run_until_steps,
        )
        if self.global_steps >= run_until_step:
            print(f"run_until_step={run_until_step} already reached; no rollout started")
            return
        progress_bar = tqdm(total=run_until_step, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # Stable per-prompt IDs are needed by terminal GRPO grouping
                # and bounded resampling.  They are trainer metadata only and
                # never enter the public environment observation.
                prompt_uids = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))],
                    dtype=object,
                )
                batch.non_tensor_batch["uid"] = prompt_uids

                # Capture private provenance before generation pops tool/raw
                # prompt fields.  These arrays are used only by HardCasePool.
                hard_case_task_ids = self._hard_case_task_keys(batch)
                hard_case_sources = self._hard_case_metadata(batch, "data_source")

                # pop those keys for generation
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                # Keep UID with the generation input so the bounded sampler
                # can align retry candidates; it is restored on the training
                # batch immediately before repeating each prompt n times.
                # Source variant is private provenance used only by the
                # Hard Case audit above; do not carry it through rollout
                # workers or any Actor-facing metadata.
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "uid", "_travel_source_env_name"]
                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                pad_id = self.tokenizer.pad_token_id
                # just after creating `gen_batch`
                max_len = self.config.data.max_prompt_length
                for key in ("input_ids", "attention_mask", "position_ids"):
                    t = gen_batch.batch[key]
                    if t.size(1) > max_len:
                        # Do not silently cut a Qwen3.5 native prompt.  The
                        # system/tool schema and public control history must
                        # remain token-identical to the rollout template;
                        # overlength inputs are an explicit configuration/data
                        # error and should be quarantined or given a larger
                        # prompt budget upstream.
                        raise ValueError(
                            f"{key} prompt width {t.size(1)} exceeds "
                            f"data.max_prompt_length={max_len}; truncation=error"
                        )
                    if t.size(1) < max_len:
                        pad_val = pad_id if key != "attention_mask" else 0
                        delta = max_len - t.size(1)
                        gen_batch.batch[key] = torch.nn.functional.pad(t, (0, delta), value=pad_val)
                    else:
                        gen_batch.batch[key] = t

                print("input_ids.shape", gen_batch.batch["input_ids"].shape)
                print("max prompt len in this chunk:", gen_batch.batch["input_ids"].ne(pad_id).sum(-1).max().item())
                print("min prompt len in this chunk:", gen_batch.batch["input_ids"].ne(pad_id).sum(-1).min().item())

                is_last_step = self.global_steps >= run_until_step

                with _timer("step", timing_raw):
                    # generate a batch
                    with _timer("gen", timing_raw):
                        if not self.async_rollout_mode:
                            print(f"gen_batch: {gen_batch.batch['input_ids']}")
                            print(f"gen_batch: {gen_batch.batch['attention_mask']}")
                            print(f"gen_batch: {gen_batch.batch['position_ids']}")
                            print(f"gen_batch keys: {gen_batch.batch.keys()}")
                            print(f"gen_batch shape: {gen_batch.batch['input_ids'].shape}")
                            print(f"gen_batch shape: {gen_batch.batch['attention_mask'].shape}")
                            print(f"gen_batch shape: {gen_batch.batch['position_ids'].shape}")

                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        else:
                            self.async_rollout_manager.wake_up()
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                            self.async_rollout_manager.sleep()
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)
                        # Bounded TravelGym retries are trainer-private.  Log
                        # only aggregate diagnostics; never copy reward
                        # metadata or simulator labels into the rollout text.
                        dynamic_info = gen_batch_output.meta_info.get("travel_dynamic_sampling")
                        if isinstance(dynamic_info, Mapping):
                            metrics.update({
                                "training/dynamic_sampling_batches": int(dynamic_info.get("sampled_batches", 0)),
                                "training/dynamic_sampling_accepted_groups": int(dynamic_info.get("accepted_groups", 0)),
                                "training/dynamic_sampling_constant_groups": int(dynamic_info.get("constant_reward_group_count", 0)),
                                "training/dynamic_sampling_insufficient_spread_groups": int(
                                    dynamic_info.get("insufficient_reward_spread_group_count", 0)
                                ),
                                "training/dynamic_sampling_invalid_groups": int(dynamic_info.get("invalid_group_count", 0)),
                            })
                        rollout_length_info = gen_batch_output.meta_info.get("travel_rollout_length")
                        if isinstance(rollout_length_info, Mapping):
                            metrics.update({
                                f"training/rollout_length_{key}": float(value)
                                for key, value in rollout_length_info.items()
                                if isinstance(value, (int, float))
                            })
                        if gen_batch_output.meta_info.get("travel_skip_update", False):
                            metrics["training/dynamic_sampling_skipped_update"] = 1.0
                        if self.hard_case_pool is not None and hard_case_task_ids is not None:
                            group_size = int(self.config.actor_rollout_ref.rollout.n)
                            repeated_tasks = np.repeat(np.asarray(hard_case_task_ids, dtype=object), group_size).tolist()
                            repeated_sources = (
                                np.repeat(np.asarray(hard_case_sources, dtype=object), group_size).tolist()
                                if hard_case_sources is not None else None
                            )
                            pool_outcomes = self.hard_case_pool.observe_output(
                                gen_batch_output,
                                task_ids=repeated_tasks,
                                sources=repeated_sources,
                                group_size=group_size,
                                step=int(self.global_steps),
                            )
                            admitted = sum(bool(item.get("admitted")) for item in pool_outcomes)
                            if admitted:
                                metrics["training/hard_case_pool_admitted"] = float(admitted)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with _timer("gen_max", timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            baseline_non_tensor_keys = [
                                key for key in gen_baseline_output.non_tensor_batch
                                if key in batch.non_tensor_batch
                            ]
                            batch.pop(
                                batch_keys=list(gen_baseline_output.batch.keys()),
                                non_tensor_batch_keys=baseline_non_tensor_keys,
                            )

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    # Restore the prompt UID removed above, then repeat it to
                    # align with the rollout responses.
                    batch.non_tensor_batch["uid"] = gen_batch.non_tensor_batch["uid"].copy()
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with _timer("reward", timing_raw):
                        # compute reward model score
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn, gamma=self.config.algorithm.gamma)

                    # Context-cleaned rollouts are expanded only after the
                    # original trajectory reward has been computed.  This
                    # keeps terminal reward and dynamic sampling at the
                    # rollout level while giving every fragment its real
                    # model input for probability recomputation.
                    if has_segmented_rollouts(batch):
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                            future_reward = None
                        batch, reward_tensor, reward_extra_infos_dict = expand_segmented_batch(
                            batch,
                            reward_tensor,
                            reward_extra_infos_dict,
                            max_model_len=int(self.config.actor_rollout_ref.rollout.max_model_len),
                            pad_token_id=int(self.tokenizer.pad_token_id or 0),
                        )
                        batch.batch["response_mask"] = compute_response_mask(batch)
                        batch.meta_info["global_token_num"] = torch.sum(
                            batch.batch["attention_mask"], dim=-1
                        ).tolist()

                    # recompute old_log_probs
                    with _timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        if (
                            has_segmented_rollouts(batch)
                            and self.config.actor_rollout_ref.rollout.multi_turn.enable
                        ):
                            # Entropy is a training diagnostic too: tool
                            # observations are conditioning, not Actor
                            # targets, so do not include them in the token
                            # average for a cleaned fragment.
                            response_masks = batch.batch["loss_mask"]
                        else:
                            response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                        if "rollout_log_probs" in batch.batch.keys():
                            # TODO: we may want to add diff of probs too.
                            rollout_old_log_probs = batch.batch["rollout_log_probs"]
                            actor_old_log_probs = batch.batch["old_log_probs"]
                            attention_mask = batch.batch["attention_mask"]
                            responses = batch.batch["responses"]
                            response_length = responses.size(1)
                            response_mask = (
                                batch.batch["loss_mask"]
                                if has_segmented_rollouts(batch)
                                else attention_mask[:, -response_length:]
                            )

                            rollout_probs = torch.exp(rollout_old_log_probs)
                            actor_probs = torch.exp(actor_old_log_probs)
                            rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
                            rollout_probs_diff = torch.masked_select(rollout_probs_diff, response_mask.bool())
                            rollout_probs_diff_max = torch.max(rollout_probs_diff)
                            rollout_probs_diff_mean = torch.mean(rollout_probs_diff)
                            rollout_probs_diff_std = torch.std(rollout_probs_diff)
                            metrics.update(
                                {
                                    "training/rollout_probs_diff_max": rollout_probs_diff_max.detach().item(),
                                    "training/rollout_probs_diff_mean": rollout_probs_diff_mean.detach().item(),
                                    "training/rollout_probs_diff_std": rollout_probs_diff_std.detach().item(),
                                }
                            )

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer("ref", timing_raw):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer("adv", timing_raw):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async and future_reward is not None:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        # print(f"{list(reward_extra_infos_dict.keys())=}")
                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update(
                                _aligned_reward_extra_info_arrays(
                                    reward_extra_infos_dict,
                                    batch_size=len(batch),
                                )
                            )
                        metrics.update(
                            compute_reward_component_metrics(
                                reward_extra_infos_dict,
                                prefix="training/reward",
                            )
                        )

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # compute advantages, executed on the driver process

                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                            turn_level_method=self.config.actor_rollout_ref.rollout.multi_turn.turn_level_method,
                            trajectory_score_method=self.config.actor_rollout_ref.rollout.multi_turn.trajectory_score_method,
                            config=self.config.algorithm,
                        )
                        sampling_stats = batch.meta_info.get("terminal_sampling_stats")
                        if sampling_stats:
                            metrics.update({
                                "training/terminal_sampling_trainable_groups": sampling_stats.get("trainable_group_count", 0),
                                "training/terminal_sampling_skipped_groups": sampling_stats.get("skipped_group_count", 0),
                                "training/terminal_sampling_constant_groups": sampling_stats.get(
                                    "skip_reason_counts", {}
                                ).get("constant_reward", 0),
                                "training/terminal_sampling_insufficient_spread_groups": sampling_stats.get(
                                    "skip_reason_counts", {}
                                ).get("insufficient_reward_spread", 0),
                                "training/terminal_sampling_kept_rows": sampling_stats.get("kept_rows", 0),
                            })
                        if "turn_credit_conservation_error" in batch.meta_info:
                            metrics["training/turn_credit_conservation_error"] = batch.meta_info["turn_credit_conservation_error"]
                        turn_credit_diagnostics = batch.meta_info.get("turn_credit_diagnostics", {})
                        for name, value in turn_credit_diagnostics.items():
                            if isinstance(value, (int, float)):
                                metrics[f"training/turn_credit_{name}"] = float(value)
                            
                    # update critic
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer("update_actor", timing_raw):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with _timer("dump_rollout_generations", timing_raw):
                            print(batch.batch.keys())
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                                rollout_metadata={
                                    key: batch.non_tensor_batch[key]
                                    for key in (
                                        "segment_records",
                                        "cleanup_events",
                                        "archive_messages",
                                        "archive_model_outputs",
                                        "archive_segment_records",
                                        "archive_turns",
                                        "length_events",
                                    )
                                    if key in batch.non_tensor_batch
                                },
                            )

                    # Periodic saves follow trainer.save_freq (20 for the
                    # TravelGym experiment).  Milestones are deliberately
                    # ordered checkpoint first, then fixed smoke20 validation.
                    # Every checkpoint folder is retained for later manual
                    # comparison; periodic saves do not trigger validation.
                    is_milestone = self.global_steps in self.milestones
                    is_periodic_save = (
                        self.save_freq > 0 and self.global_steps % self.save_freq == 0
                    )
                    if is_milestone or is_periodic_save or is_last_step:
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint(valid_save_best=False)
                        if (
                            not bool(self.config.trainer.get("disable_validation", False))
                            and self.val_reward_fn is not None
                            and (is_milestone or is_last_step)
                        ):
                            with _timer("testing", timing_raw):
                                validation_pass_k = int(
                                    self.config.trainer.get("validation_pass_k", 1)
                                )
                                if validation_pass_k > 1:
                                    val_metrics: dict = self._validate_task_passes(
                                        pass_k=validation_pass_k,
                                        smoke=bool(is_milestone),
                                    )
                                else:
                                    val_metrics = (
                                        self._validate_smoke()
                                        if is_milestone
                                        else self._validate()
                                    )
                                if is_last_step:
                                    last_val_metrics = val_metrics
                            metrics.update(val_metrics)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=self._stable_grpo_namespace(metrics, validation=False), step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1
                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    if self.training_profile == "production" and self.run_until_step >= self.total_training_steps and bool(self.config.trainer.get("wait_for_selected_grpo_checkpoint", True)):
                        marker = os.path.join(self.config.trainer.default_local_dir, "WAITING_FOR_SELECTED_GRPO_CHECKPOINT")
                        with open(marker, "w", encoding="utf-8") as handle:
                            handle.write("Provide SELECTED_GRPO_CHECKPOINT manually before launching final200.\n")
                    return
