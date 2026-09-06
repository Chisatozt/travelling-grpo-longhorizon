# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
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
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO-like algorithms.
"""

__all__ = [
    "get_adv_estimator_fn",
    "AdvantageEstimator",
    "compute_terminal_group_advantage",
    "compute_grpo_multiturn_advantage",
    "redistribute_terminal_turn_credit",
    "redistribute_segmented_terminal_turn_credit",
    "compute_terminal_component_advantages",
    "redistribute_behavior_component_turn_credit",
    "redistribute_segmented_behavior_component_turn_credit",
    "turn_credit_conservation_error",
    "turn_credit_conservation_stats",
    "segmented_turn_credit_conservation_stats",
]

from collections import defaultdict
from enum import Enum

import numpy as np
import torch

import verl.utils.torch_functional as verl_F

ADV_ESTIMATOR_REGISTRY = {}

def register_adv_est(name_or_enum):
    """Decorator to register a advantage estimator function with a given name.

    Args:
        name_or_enum: `(str)` or `(AdvantageEstimator)`
            The name or enum of the advantage estimator.

    """
    def decorator(fn):
        name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum
        if name in ADV_ESTIMATOR_REGISTRY and ADV_ESTIMATOR_REGISTRY[name] != fn:
            raise ValueError(f"Adv estimator {name} has already been registered: {ADV_ESTIMATOR_REGISTRY[name]} vs {fn}")
        ADV_ESTIMATOR_REGISTRY[name] = fn
        return fn
    return decorator

def get_adv_estimator_fn(name_or_enum):
    """Get the advantage estimator function with a given name.

    Args:
        name_or_enum: `(str)` or `(AdvantageEstimator)`
            The name or enum of the advantage estimator.

    Returns:
        `(callable)`: The advantage estimator function.
    """
    name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum
    if name not in ADV_ESTIMATOR_REGISTRY:
        raise ValueError(f"Unknown advantage estimator simply: {name}")
    return ADV_ESTIMATOR_REGISTRY[name]

class AdvantageEstimator(str, Enum):
    """Using an enumeration class to avoid spelling errors in adv_estimator.

    Note(haibin.lin): this enum class is immutable after creation. Extending this
    enum for new estimators may not be necessary since users can always just call
    `verl.trainer.ppo.core_algos.register` with string name for a custom advantage
    estimator instead.
    """

    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REINFORCE_PLUS_PLUS_BASELINE = "reinforce_plus_plus_baseline"
    REMAX = "remax"
    RLOO = "rloo"
    OPO = "opo"
    GRPO_PASSK = "grpo_passk"
    GRPO_MULTITURN = "grpo_multiturn"


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        pass


def get_kl_controller(kl_ctrl):
    if kl_ctrl.type == "fixed":
        return FixedKLController(kl_coef=kl_ctrl.kl_coef)
    elif kl_ctrl.type == "adaptive":
        assert kl_ctrl.horizon > 0, f"horizon must be larger than 0. Got {kl_ctrl.horizon}"
        return AdaptiveKLController(init_kl_coef=kl_ctrl.kl_coef, target_kl=kl_ctrl.target_kl, horizon=kl_ctrl.horizon)
    else:
        raise NotImplementedError

@register_adv_est(AdvantageEstimator.GAE) # or simply: @register_adv_est("gae")
def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor,
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        values: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma is `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, response_mask)
    return advantages, returns


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
@register_adv_est(AdvantageEstimator.GRPO) # or simply: @register_adv_est("grpo")
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
):
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length)
        norm_adv_by_std_in_grpo: (bool)
            whether to scale the GRPO advantage.
            If True, the advantage is scaled by the std, as in the original GRPO.
            If False, the advantage is not scaled, as in Dr.GRPO (https://arxiv.org/abs/2503.20783).

    Returns:
        advantages: `(torch.Tensor)`
            shape is (bs, response_length)
        Returns: `(torch.Tensor)`
            shape is (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]
        scores = scores.unsqueeze(-1) * response_mask

    print(f"Normal GRPO Advantage: {scores.shape}")
    print(scores)
    
    return scores, scores

@register_adv_est(AdvantageEstimator.GRPO_PASSK) # or simply: @register_adv_est("grpo_passk")
def compute_grpo_passk_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config = None,
    **kwargs,
):
    """
    Compute advantage for Pass@k using a GRPO-style outcome reward formulation.
    Only the best response per group gets a non-zero advantage: r_max - r_second_max.

    Implemented as described in https://arxiv.org/abs/2503.19595.

    Args:
        token_level_rewards: (bs, response_length)
        response_mask: (bs, response_length)
        index: (bs,) → group ID per sample
        epsilon: float for numerical stability
        config: (dict) algorithm settings, which contains "norm_adv_by_std_in_grpo"

    Returns:
        advantages: (bs, response_length)
        returns: (bs, response_length)
    """
    assert config is not None
    # if True, normalize advantage by std within group
    norm_adv_by_std_in_grpo = config.get("norm_adv_by_std_in_grpo", True)
    scores = token_level_rewards.sum(dim=-1)  # (bs,)
    advantages = torch.zeros_like(scores)

    id2scores = defaultdict(list)
    id2indices = defaultdict(list)

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            idx = index[i]
            id2scores[idx].append(scores[i])
            id2indices[idx].append(i)

        for idx in id2scores:
            rewards = torch.stack(id2scores[idx])  # (k,)
            if rewards.numel() < 2:
                raise ValueError(f"Pass@k requires at least 2 samples per group. Got {rewards.numel()} for group {idx}.")
            topk, topk_idx = torch.topk(rewards, 2)
            r_max, r_second_max = topk[0], topk[1]
            i_max = id2indices[idx][topk_idx[0].item()]
            advantage = r_max - r_second_max
            if norm_adv_by_std_in_grpo:
                std = torch.std(rewards)
                advantage = advantage / (std + epsilon)
            advantages[i_max] = advantage

    advantages = advantages.unsqueeze(-1) * response_mask
    return advantages, advantages

@register_adv_est(AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE) # or simply: @register_adv_est("reinforce_plus_plus_baseline")
def compute_reinforce_plus_plus_baseline_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: torch.Tensor,
                                                           epsilon: float = 1e-6, config=None, **kwargs):
    """
    Compute advantage for RF++-baseline (https://arxiv.org/abs/2501.03262), operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (dict) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2mean[index[i]]

        scores = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
        scores = verl_F.masked_whiten(scores, response_mask) * response_mask

    return scores, scores

@register_adv_est(AdvantageEstimator.RLOO) # or simply: @register_adv_est("rloo")
def compute_rloo_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: np.ndarray,
                                   epsilon: float = 1e-6, config=None, **kwargs):
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (dict) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            response_num = len(id2score[index[i]])
            if response_num > 1:
                scores[i] = scores[i] * response_num / (response_num - 1) - id2mean[index[i]] * response_num / (response_num - 1)
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores

@register_adv_est(AdvantageEstimator.OPO) # or simply: @register_adv_est("opo")
def compute_opo_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: np.ndarray, epsilon: float = 1e-6,
                                  config=None, **kwargs):
    """
    Compute advantage for OPO based on https://arxiv.org/pdf/2505.23585

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (dict) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = response_mask.sum(dim=-1)
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2len = defaultdict(list)
    id2bsl = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
            id2len[index[i]].append(response_length[i])

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2bsl[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                score_tensor = torch.tensor(id2score[idx])
                len_tensor = torch.tensor(id2len[idx])
                id2bsl[idx] = (len_tensor * score_tensor).sum() / len_tensor.sum()
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2bsl[index[i]]
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores

@register_adv_est(AdvantageEstimator.REINFORCE_PLUS_PLUS) # or simply: @register_adv_est("reinforce_plus_plus")
def compute_reinforce_plus_plus_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, config=None, **kwargs):
    """
    Compute advantage for REINFORCE++.
    This implementation is based on the paper: https://arxiv.org/abs/2501.03262

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (dict) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    assert config is not None
    gamma = config.gamma
    with torch.no_grad():
        returns = torch.zeros_like(token_level_rewards)
        running_return = 0

        for t in reversed(range(token_level_rewards.shape[1])):
            running_return = token_level_rewards[:, t] + gamma * running_return
            returns[:, t] = running_return
            # Reset after EOS
            running_return = running_return * response_mask[:, t]

        advantages = verl_F.masked_whiten(returns, response_mask)
        advantages = advantages * response_mask

    return advantages, returns

@register_adv_est(AdvantageEstimator.REMAX) # or simply: @register_adv_est("remax")
def compute_remax_outcome_advantage(token_level_rewards: torch.Tensor, reward_baselines: torch.Tensor, response_mask: torch.Tensor, config=None, **kwargs):
    """
    Compute advantage for ReMax, operating only on Outcome reward
    This implementation is based on the paper: https://arxiv.org/abs/2310.10505
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        reward_baselines: `(torch.Tensor)`
            shape: (bs,)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (dict) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = (token_level_rewards * response_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
        advantages = returns - reward_baselines.unsqueeze(-1) * response_mask

    return advantages, returns


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str):
    """
    Aggregate the loss matrix into a scalar.

    Args:
        loss_mat: `(torch.Tensor)`:
            shape: (bs, response_length)
        loss_mask: `(torch.Tensor)`:
            shape: (bs, response_length)
        loss_agg_mode: (str) choices:
            method to aggregate the loss matrix into a scalar.
    Returns:
        loss: `a scalar torch.Tensor`
            aggregated loss
    """
    if loss_agg_mode == "token-mean":
        loss = verl_F.masked_mean(loss_mat, loss_mask)
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # token-sum
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1)  # token-mean
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-sum-norm":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        loss = torch.sum(seq_losses) / loss_mask.shape[-1]  # The divisor
        # (loss_mask.shape[-1]) should ideally be constant
        # throughout training to well-replicate the DrGRPO paper.
        # TODO: Perhaps add user-defined normalizer argument to
        # agg_loss to ensure divisor stays constant throughout.
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    return loss


def compute_policy_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    loss_agg_mode: str = "token-mean",
):
    """
    Compute the clipped policy objective and related metrics for PPO.

    Adapted from
    https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        cliprange (float, optional):
            Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
            Defaults to None (must be provided).
        cliprange_low (float, optional):
            Lower clip range for dual-clip PPO. Defaults to same as `cliprange`.
        cliprange_high (float, optional):
            Upper clip range for dual-clip PPO. Defaults to same as `cliprange`.
        clip_ratio_c (float, optional):
            Lower bound of the ratio for dual-clip PPO. See https://arxiv.org/pdf/1912.09729.
            Defaults to 3.0.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
    """
    assert clip_ratio_c > 1.0, "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0," + f" but get the value: {clip_ratio_c}."

    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask)

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


def compute_entropy_loss(logits, response_mask, loss_agg_mode: str = "token-mean"):
    """Compute categorical entropy loss (For backward compatibility)

    Args:
        logits (torch.Tensor): shape is (bs, response_length, vocab_size)
        response_mask (torch.Tensor): shape is (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    token_entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = agg_loss(loss_mat=token_entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    return entropy_loss


def compute_value_loss(vpreds: torch.Tensor, returns: torch.Tensor, values: torch.Tensor, response_mask: torch.Tensor, cliprange_value: float, loss_agg_mode: str = "token-mean"):
    """
    Compute the clipped value-function loss for PPO.

    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (torch.FloatTensor):
            Predicted values from the value head, shape (batch_size, response_length).
        values (torch.FloatTensor):
            Old (baseline) values from the value head, shape (batch_size, response_length).
        returns (torch.FloatTensor):
            Ground-truth returns, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the value loss calculation.
        cliprange_value (float):
            Clip range for value prediction updates.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".

    Returns:
        vf_loss (torch.FloatTensor):
            A scalar tensor containing the aggregated value-function loss.
        vf_clipfrac (float):
            Fraction of elements where the clipped loss was used.
    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns) ** 2
    vf_losses2 = (vpredclipped - returns) ** 2
    clipped_vf_losses = torch.max(vf_losses1, vf_losses2)
    vf_loss = agg_loss(loss_mat=clipped_vf_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), response_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104
    See more description in http://joschu.net/blog/kl-approx.html

    Args:
        logprob:
        ref_logprob:

    Returns:

    """
    if kl_penalty in ("kl", "k1"):
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty in ("mse", "k2"):
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty in ("low_var_kl", "k3"):
        kl = ref_logprob - logprob
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError


def compute_turn_credits(conversation_history, gamma=0.8, turn_level_method="Equalized", trajectory_score_method="Sum"):
    """Compatibility helper for inspecting turn evidence.

    TravelGym no longer derives a trajectory score from per-turn rewards.  The
    returned score is therefore always the explicit terminal reward when one is
    present, and otherwise zero; the list is only a diagnostic weighting hint.
    """
    if not conversation_history:
        return [], 0.0
    terminal_reward = 0.0
    for item in conversation_history:
        if isinstance(item, dict) and "terminal_reward" in item:
            terminal_reward = float(item["terminal_reward"])
    if turn_level_method == "Equalized":
        credits = [1.0] * len(conversation_history)
    elif turn_level_method in {"R2G", "EM"}:
        # Preserve the old diagnostic API without allowing those values to
        # become training rewards.
        credits = [1.0] * len(conversation_history)
    else:
        raise ValueError(f"Invalid turn_level_method: {turn_level_method}")
    if trajectory_score_method not in {"Sum", "R2G", "Terminal"}:
        raise ValueError(f"Invalid trajectory_score_method: {trajectory_score_method}")
    return credits, terminal_reward


def compute_terminal_group_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    *,
    terminal_scores: torch.Tensor | None = None,
    reward_valid: torch.Tensor | None = None,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
):
    """Compute GRPO advantages from one terminal score per trajectory.

    Invalid infrastructure rewards are excluded from the group baseline and
    receive an all-zero advantage.  No turn reward, candidate metric, or
    hidden label is consulted here.
    """
    with torch.no_grad():
        scores = terminal_scores if terminal_scores is not None else token_level_rewards.sum(dim=-1)
        scores = scores.to(device=token_level_rewards.device, dtype=token_level_rewards.dtype).reshape(-1)
        bsz, response_length = token_level_rewards.shape
        if scores.numel() != bsz:
            raise ValueError("terminal_scores must have one value per response")
        valid = torch.ones(bsz, dtype=torch.bool, device=scores.device)
        if reward_valid is not None:
            valid = reward_valid.to(device=scores.device).reshape(-1).bool()
            if valid.numel() != bsz:
                raise ValueError("reward_valid must have one value per response")
        # Treat non-finite terminal values as infrastructure-invalid even if a
        # producer forgot to set the explicit validity bit.  This prevents NaN
        # baselines/advantages from being reintroduced by a zero mask later.
        valid = valid & torch.isfinite(scores)

        id2valid = defaultdict(list)
        for i in range(bsz):
            if bool(valid[i]):
                id2valid[index[i]].append(i)
        means, stds = {}, {}
        for group, positions in id2valid.items():
            values = scores[positions]
            means[group] = values.mean()
            stds[group] = values.std(unbiased=False) if values.numel() > 1 else torch.tensor(1.0, device=scores.device, dtype=scores.dtype)

        advantages = torch.zeros((bsz, response_length), device=token_level_rewards.device, dtype=token_level_rewards.dtype)
        for i in range(bsz):
            if not bool(valid[i]) or index[i] not in means:
                continue
            centered = scores[i] - means[index[i]]
            if norm_adv_by_std_in_grpo:
                centered = centered / (stds[index[i]] + epsilon)
            advantages[i] = centered * response_mask[i]
        return advantages, advantages.clone()


def redistribute_terminal_turn_credit(
    terminal_advantages: torch.Tensor,
    response_mask: torch.Tensor,
    turn_boundaries: torch.Tensor,
    conversation_histories: list | None = None,
    *,
    choice_weights: dict[str, float] | None = None,
) -> torch.Tensor:
    """Optionally redistribute a decided terminal advantage conservatively.

    The sum of token credits for each response is conserved exactly.  This
    function creates no new reward and must only run after terminal GRPO
    advantages are fixed.  ``conversation_histories`` contains public choices
    only; no preference IDs or correctness labels are read.
    """
    weights = {"search": 0.25, "action": 0.45, "answer": 0.30}
    if choice_weights:
        weights.update({str(k): float(v) for k, v in choice_weights.items()})
    output = torch.zeros_like(terminal_advantages)
    for row in range(terminal_advantages.shape[0]):
        valid_positions = torch.where(response_mask[row].bool())[0].tolist()
        if not valid_positions:
            continue
        starts = torch.where(turn_boundaries[row].bool())[0].tolist()
        starts = sorted(set(starts)) or [valid_positions[0]]
        starts = [position for position in starts if position < response_mask.shape[1]]
        if not starts or starts[0] > valid_positions[0]:
            starts.insert(0, valid_positions[0])
        ends = starts[1:] + [response_mask.shape[1]]
        history = conversation_histories[row] if conversation_histories and row < len(conversation_histories) else []
        if isinstance(history, dict):
            history = [history]
        turn_weights = []
        for turn_idx, (start, end) in enumerate(zip(starts, ends)):
            choice = ""
            if turn_idx < len(history) and isinstance(history[turn_idx], dict):
                choice = str(history[turn_idx].get("choice", "")).casefold()
            turn_weights.append(max(0.0, weights.get(choice, 1.0)))
        total_weight = sum(turn_weights) or float(len(turn_weights))
        base_total = terminal_advantages[row, valid_positions].sum()
        for (start, end), turn_weight in zip(zip(starts, ends), turn_weights):
            token_positions = [pos for pos in valid_positions if start <= pos < end]
            if not token_positions:
                continue
            share = base_total * (turn_weight / total_weight)
            output[row, token_positions] = share / float(len(token_positions))
    return output



def compute_terminal_component_advantages(
    terminal_scores: torch.Tensor,
    reward_components: torch.Tensor,
    index: np.ndarray,
    *,
    reward_valid: torch.Tensor | None = None,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
) -> torch.Tensor:
    """Center terminal reward components with one shared GRPO denominator.

    Each row of reward_components must sum to its terminal score. Using the
    terminal-score standard deviation for every component guarantees that the
    component advantages sum back to the ordinary trajectory advantage.
    """
    scores = terminal_scores.reshape(-1)
    components = reward_components.to(device=scores.device, dtype=scores.dtype)
    if components.ndim != 2 or components.shape[0] != scores.numel():
        raise ValueError("reward_components must have shape [batch, components]")
    valid = torch.ones(scores.numel(), dtype=torch.bool, device=scores.device)
    if reward_valid is not None:
        valid = reward_valid.to(device=scores.device).reshape(-1).bool()
    valid = valid & torch.isfinite(scores) & torch.isfinite(components).all(dim=-1)
    if not torch.allclose(
        components.sum(dim=-1)[valid],
        scores[valid],
        atol=2e-5,
        rtol=2e-5,
    ):
        raise ValueError("terminal reward components must sum to terminal_scores")

    output = torch.zeros_like(components)
    id2valid = defaultdict(list)
    for row in range(scores.numel()):
        if bool(valid[row]):
            id2valid[index[row]].append(row)
    for rows in id2valid.values():
        values = scores[rows]
        denominator = (
            values.std(unbiased=False) + epsilon
            if norm_adv_by_std_in_grpo and values.numel() > 1
            else torch.tensor(1.0, device=scores.device, dtype=scores.dtype)
        )
        component_mean = components[rows].mean(dim=0)
        output[rows] = (components[rows] - component_mean) / denominator
        target = (values - values.mean()) / denominator
        # Put floating-point closure error in the explicit residual component.
        output[rows, -1] += target - output[rows].sum(dim=-1)
    return output


def _coalesce_turn_event(history_item) -> dict:
    if not isinstance(history_item, dict):
        return {}
    raw_events = history_item.get("turn_events", [])
    if isinstance(raw_events, dict):
        raw_events = [raw_events]
    events = [event for event in raw_events if isinstance(event, dict)]
    if not events:
        return {
            "choice": str(history_item.get("choice", "")).casefold(),
            "aspect": "",
            "accepted": True,
            "has_quality_event": False,
        }
    merged = {
        "choice": str(history_item.get("choice", events[-1].get("choice", ""))).casefold(),
        "aspect": str(events[-1].get("aspect", "")),
        "accepted": all(bool(event.get("accepted", False)) for event in events),
        "has_quality_event": True,
    }
    boolean_fields = (
        "invalid_call",
        "new_search",
        "useful_action",
        "no_gain_action",
        "duplicate_action",
        "redundant_action",
        "completed_aspect",
        "correct_answer",
        "best_answer",
        "legal_answer",
        "wrong_answer",
        "terminated",
        "truncated",
    )
    for field in boolean_fields:
        merged[field] = any(bool(event.get(field, False)) for event in events)
    merged["new_preference_count"] = sum(
        max(0, int(event.get("new_preference_count", 0) or 0))
        for event in events
    )
    merged["termination_reason"] = str(events[-1].get("termination_reason", ""))
    return merged


def _normalized_turn_weights(weight_by_turn: dict[int, float], turn_count: int) -> list[float]:
    values = [max(0.0, float(weight_by_turn.get(idx, 0.0))) for idx in range(turn_count)]
    total = sum(values)
    if total <= 0:
        return [0.0] * turn_count
    return [value / total for value in values]


def _answer_chain_turn_weights(
    events: list[dict],
    *,
    answer_flag: str,
    search_share: float,
    action_share: float,
    answer_share: float,
) -> list[float]:
    routed: dict[int, float] = defaultdict(float)
    answer_indices = [idx for idx, event in enumerate(events) if bool(event.get(answer_flag, False))]
    for answer_idx in answer_indices:
        aspect = str(events[answer_idx].get("aspect", ""))
        searches = [
            idx
            for idx in range(answer_idx)
            if events[idx].get("aspect") == aspect and bool(events[idx].get("new_search", False))
        ]
        actions = [
            idx
            for idx in range(answer_idx)
            if events[idx].get("aspect") == aspect and bool(events[idx].get("useful_action", False))
        ]
        if searches:
            routed[searches[0]] += search_share
        else:
            routed[answer_idx] += search_share
        if actions:
            per_action = action_share / len(actions)
            for idx in actions:
                routed[idx] += per_action
        else:
            routed[answer_idx] += action_share
        routed[answer_idx] += answer_share
    return _normalized_turn_weights(routed, len(events))


def _behavior_component_turn_weights(
    component_name: str,
    component_advantage: float,
    events: list[dict],
    routing: dict | None = None,
) -> list[float]:
    routing = routing or {}
    turn_count = len(events)
    if turn_count == 0:
        return []
    answer_routes = {
        "correct_completion": ("correct_answer", (0.15, 0.25, 0.60)),
        "coverage_adjusted_answer_quality": ("best_answer", (0.10, 0.35, 0.55)),
        "coverage_adjusted_legal_chain_rate": ("legal_answer", (0.25, 0.35, 0.40)),
    }
    custom_routes = routing.get("causal_routing", {}) if isinstance(routing, dict) else {}
    if component_name in answer_routes and component_advantage >= 0:
        flag, default_split = answer_routes[component_name]
        split = custom_routes.get(component_name, default_split)
        if len(split) != 3:
            raise ValueError(f"causal route for {component_name} must contain three shares")
        weights = _answer_chain_turn_weights(
            events,
            answer_flag=flag,
            search_share=float(split[0]),
            action_share=float(split[1]),
            answer_share=float(split[2]),
        )
        if any(weights):
            return weights

    useful = {
        idx: max(1.0, float(event.get("new_preference_count", 0) or 0))
        for idx, event in enumerate(events)
        if bool(event.get("useful_action", False))
    }
    progress = {
        idx: 1.0
        for idx, event in enumerate(events)
        if bool(event.get("accepted", False))
        and (
            bool(event.get("new_search", False))
            or bool(event.get("useful_action", False))
            or bool(event.get("completed_aspect", False))
        )
    }
    offenders = {
        idx: 1.0
        for idx, event in enumerate(events)
        if bool(event.get("invalid_call", False))
        or bool(event.get("wrong_answer", False))
    }
    redundant = {
        idx: 1.0
        for idx, event in enumerate(events)
        if bool(event.get("redundant_action", False))
        or bool(event.get("duplicate_action", False))
    }
    wasteful = dict(offenders)
    for idx, value in redundant.items():
        wasteful[idx] = wasteful.get(idx, 0.0) + value
    # A first no-gain question has grace under the dedicated redundancy
    # penalty, but it still explains wasted budget/efficiency. Give it half
    # the blame of an explicitly invalid, duplicate, or redundant turn.
    for idx, event in enumerate(events):
        if bool(event.get("no_gain_action", False)) and idx not in redundant:
            wasteful[idx] = wasteful.get(idx, 0.0) + 0.5

    if component_name == "hidden_preference_hit_rate" and useful:
        return _normalized_turn_weights(useful, turn_count)
    if component_name == "policy_penalty" and component_advantage < 0 and offenders:
        weighted = {
            idx: 2.0 if bool(events[idx].get("wrong_answer", False)) else 1.0
            for idx in offenders
        }
        return _normalized_turn_weights(weighted, turn_count)
    if component_name == "redundant_action_penalty" and component_advantage < 0 and redundant:
        return _normalized_turn_weights(redundant, turn_count)
    if component_name in {"incomplete_penalty", "zero_answer_penalty", "max_steps_penalty"}:
        if component_advantage < 0:
            routed = {idx: 0.70 / len(wasteful) for idx in wasteful} if wasteful else {}
            routed[turn_count - 1] = routed.get(turn_count - 1, 0.0) + 0.30
            if not wasteful:
                routed[turn_count - 1] = 1.0
            return _normalized_turn_weights(routed, turn_count)
        completed = {
            idx: 1.0
            for idx, event in enumerate(events)
            if bool(event.get("completed_aspect", False))
        }
        return _normalized_turn_weights(completed or progress, turn_count)
    if component_name == "clip_residual":
        return _normalized_turn_weights({turn_count - 1: 1.0}, turn_count)
    if component_advantage < 0:
        negative_targets = wasteful or {
            idx: 1.0
            for idx, event in enumerate(events)
            if bool(event.get("wrong_answer", False))
        }
        if negative_targets:
            return _normalized_turn_weights(negative_targets, turn_count)
        return _normalized_turn_weights({turn_count - 1: 1.0}, turn_count)
    return _normalized_turn_weights(progress or {idx: 1.0 for idx in range(turn_count)}, turn_count)


def redistribute_behavior_component_turn_credit(
    terminal_advantages: torch.Tensor,
    response_mask: torch.Tensor,
    turn_boundaries: torch.Tensor,
    conversation_histories: list | None,
    component_advantages: torch.Tensor,
    component_names: list[str] | tuple[str, ...],
    *,
    mix_ratio: float = 0.30,
    routing: dict | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Route group-relative reward components to causally responsible turns.

    The output preserves the original per-response token-advantage sum. Turn
    totals are divided by each turns trainable-token count so verbose
    reasoning does not receive more credit merely for being longer.
    """
    if not 0.0 <= float(mix_ratio) <= 1.0:
        raise ValueError("turn credit mix_ratio must be between 0 and 1")
    if component_advantages.shape != (terminal_advantages.shape[0], len(component_names)):
        raise ValueError("component_advantages and component_names are misaligned")

    behavior = torch.zeros_like(terminal_advantages)
    diagnostic_values: dict[str, list[float]] = defaultdict(list)
    for row in range(terminal_advantages.shape[0]):
        valid_positions = torch.where(response_mask[row].bool())[0].tolist()
        if not valid_positions:
            continue
        starts = sorted(set(torch.where(turn_boundaries[row].bool())[0].tolist()))
        starts = [position for position in starts if position < response_mask.shape[1]]
        if not starts or starts[0] > valid_positions[0]:
            starts.insert(0, valid_positions[0])
        ends = starts[1:] + [response_mask.shape[1]]
        token_positions_by_turn = [
            [pos for pos in valid_positions if start <= pos < end]
            for start, end in zip(starts, ends)
        ]
        nonempty = [idx for idx, positions in enumerate(token_positions_by_turn) if positions]
        if not nonempty:
            continue

        history = conversation_histories[row] if conversation_histories and row < len(conversation_histories) else []
        if isinstance(history, dict):
            history = [history]
        events = [
            _coalesce_turn_event(history[idx] if idx < len(history) else {})
            for idx in range(len(starts))
        ]
        if not any(bool(event.get("has_quality_event", False)) for event in events):
            behavior[row] = terminal_advantages[row]
            diagnostic_values["fallback_rows"].append(1.0)
            continue

        turn_credits = torch.zeros(
            len(starts),
            device=terminal_advantages.device,
            dtype=terminal_advantages.dtype,
        )
        for component_idx, component_name in enumerate(component_names):
            value = component_advantages[row, component_idx]
            weights = _behavior_component_turn_weights(
                str(component_name),
                float(value.detach().cpu()),
                events,
                routing,
            )
            # A generated turn can become entirely untrainable after
            # masking (for example a forced close tag). Never strand component
            # mass on such a turn; renormalize over trainable turns only.
            weights = _normalized_turn_weights(
                {
                    idx: weights[idx]
                    for idx in nonempty
                    if weights and idx < len(weights) and weights[idx] > 0
                },
                len(starts),
            )
            if not any(weights):
                weights = _normalized_turn_weights(
                    {idx: 1.0 for idx in nonempty},
                    len(starts),
                )
            turn_credits += value * torch.tensor(
                weights,
                device=turn_credits.device,
                dtype=turn_credits.dtype,
            )

        total_tokens = float(len(valid_positions))
        for turn_idx, token_positions in enumerate(token_positions_by_turn):
            if not token_positions:
                continue
            behavior[row, token_positions] = (
                total_tokens * turn_credits[turn_idx] / float(len(token_positions))
            )

        for turn_idx, event in enumerate(events):
            value = float(turn_credits[turn_idx].detach().cpu())
            if event.get("useful_action"):
                diagnostic_values["useful_action_credit"].append(value)
            if event.get("redundant_action") or event.get("duplicate_action"):
                diagnostic_values["redundant_action_credit"].append(value)
            if event.get("wrong_answer"):
                diagnostic_values["wrong_answer_credit"].append(value)
            if event.get("new_search"):
                diagnostic_values["new_search_credit"].append(value)

    # Record the raw routing error so train mode can distinguish harmless
    # float rounding from a real component/routing mismatch before projection.
    preprojection = turn_credit_conservation_stats(
        terminal_advantages,
        behavior,
        response_mask,
    )
    # Close the routed and final mixed sums in float64, then put the residual
    # on one trainable token. Spreading a sub-ULP correction over thousands of
    # float32 tokens can otherwise leave a visible 1e-3 trajectory-sum error.
    behavior = _project_trajectory_credit_sum(
        behavior,
        terminal_advantages,
        response_mask,
    )
    mixed = (1.0 - float(mix_ratio)) * terminal_advantages + float(mix_ratio) * behavior
    mixed = _project_trajectory_credit_sum(
        mixed,
        terminal_advantages,
        response_mask,
    )
    diagnostics = {
        f"mean_{name}": float(sum(values) / len(values))
        for name, values in diagnostic_values.items()
        if values
    }
    diagnostics["fallback_row_count"] = float(len(diagnostic_values.get("fallback_rows", [])))
    diagnostics["mix_ratio"] = float(mix_ratio)
    for event_name in (
        "useful_action_credit",
        "redundant_action_credit",
        "wrong_answer_credit",
        "new_search_credit",
    ):
        raw_name = f"mean_{event_name}"
        if raw_name in diagnostics:
            diagnostics[f"effective_{raw_name}"] = float(mix_ratio) * diagnostics[raw_name]
    conservation = turn_credit_conservation_stats(
        terminal_advantages,
        mixed,
        response_mask,
    )
    diagnostics.update(
        {
            "conservation_error": float(conservation["absolute_error"].detach().cpu()),
            "conservation_abs_error": float(conservation["absolute_error"].detach().cpu()),
            "conservation_relative_error": float(conservation["relative_error"].detach().cpu()),
            "conservation_mean_token_error": float(conservation["mean_token_error"].detach().cpu()),
            "conservation_finite": float(conservation["finite"].detach().cpu()),
            "preprojection_abs_error": float(preprojection["absolute_error"].detach().cpu()),
            "preprojection_relative_error": float(preprojection["relative_error"].detach().cpu()),
            "preprojection_finite": float(preprojection["finite"].detach().cpu()),
        }
    )
    return mixed, diagnostics


def _segmented_turn_units(
    row: int,
    response_mask: torch.Tensor,
    turn_boundaries: torch.Tensor,
    history_item,
    global_turn_indices=None,
) -> list[dict]:
    """Map one fragment's local boundary positions to global turn units."""
    valid_positions = torch.where(response_mask[row].bool())[0].tolist()
    if not valid_positions:
        return []
    starts = sorted(set(torch.where(turn_boundaries[row].bool())[0].tolist()))
    starts = [position for position in starts if position < response_mask.shape[1]]
    if not starts or starts[0] > valid_positions[0]:
        starts.insert(0, valid_positions[0])
    ends = starts[1:] + [response_mask.shape[1]]
    history = history_item
    if isinstance(history, dict):
        history = [history]
    if not isinstance(history, (list, tuple)):
        history = []
    indices = global_turn_indices
    if isinstance(indices, torch.Tensor):
        indices = indices.detach().cpu().tolist()
    if not isinstance(indices, (list, tuple)):
        indices = []
    units = []
    for local_idx, (start, end) in enumerate(zip(starts, ends)):
        positions = [position for position in valid_positions if start <= position < end]
        if not positions:
            continue
        item = history[local_idx] if local_idx < len(history) else {}
        if not isinstance(item, dict):
            item = {}
        if local_idx < len(indices):
            global_idx = int(indices[local_idx])
        elif item.get("turn_idx") is not None:
            global_idx = int(item["turn_idx"])
        else:
            global_idx = local_idx
        units.append(
            {
                "global_turn": global_idx,
                "row": row,
                "positions": positions,
                "history": item,
            }
        )
    return units


def _group_segment_indices(segment_trajectory_ids) -> dict[object, list[int]]:
    groups: dict[object, list[int]] = defaultdict(list)
    for row, trajectory_id in enumerate(segment_trajectory_ids):
        if isinstance(trajectory_id, np.ndarray):
            trajectory_id = trajectory_id.item()
        try:
            groups[trajectory_id].append(row)
        except TypeError:
            groups[str(trajectory_id)].append(row)
    return groups


def _ordered_segment_units(
    rows: list[int],
    response_mask: torch.Tensor,
    turn_boundaries: torch.Tensor,
    segment_histories,
    segment_global_turn_indices,
) -> list[dict]:
    units = []
    for row in rows:
        history = segment_histories[row] if segment_histories is not None else []
        indices = (
            segment_global_turn_indices[row]
            if segment_global_turn_indices is not None
            else None
        )
        units.extend(
            _segmented_turn_units(
                row,
                response_mask,
                turn_boundaries,
                history,
                indices,
            )
        )
    units.sort(key=lambda item: (int(item["global_turn"]), int(item["row"])))
    return units


def _segment_history_events(units: list[dict]) -> list[dict]:
    return [_coalesce_turn_event(unit.get("history", {})) for unit in units]


def _segmented_choice_turn_weights(units: list[dict], choice_weights: dict[str, float] | None) -> list[float]:
    weights = {"search": 0.25, "action": 0.45, "answer": 0.30}
    if choice_weights:
        weights.update({str(key): float(value) for key, value in choice_weights.items()})
    values = []
    for event in _segment_history_events(units):
        values.append(max(0.0, weights.get(str(event.get("choice", "")).casefold(), 1.0)))
    total = sum(values)
    return [value / total for value in values] if total > 0 else [1.0 / len(units)] * len(units)


def redistribute_segmented_terminal_turn_credit(
    terminal_advantages: torch.Tensor,
    response_mask: torch.Tensor,
    turn_boundaries: torch.Tensor,
    segment_trajectory_ids,
    segment_histories=None,
    segment_global_turn_indices=None,
    *,
    choice_weights: dict[str, float] | None = None,
) -> torch.Tensor:
    """Redistribute terminal credit across local fragments using global turns.

    ``segment_trajectory_ids`` is deliberately separate from the GRPO group
    UID: it identifies one original rollout, so splitting a rollout cannot
    create another group member or another terminal reward.
    """
    if terminal_advantages.shape != response_mask.shape:
        raise ValueError("terminal_advantages and response_mask must have identical shapes")
    if turn_boundaries.shape != response_mask.shape:
        raise ValueError("turn_boundaries and response_mask must have identical shapes")
    if len(segment_trajectory_ids) != terminal_advantages.shape[0]:
        raise ValueError("segment trajectory ids must align with the batch")
    output = torch.zeros_like(terminal_advantages)
    for trajectory_id, rows in _group_segment_indices(segment_trajectory_ids).items():
        del trajectory_id
        units = _ordered_segment_units(
            rows,
            response_mask,
            turn_boundaries,
            segment_histories,
            segment_global_turn_indices,
        )
        if not units:
            for row in rows:
                output[row] = terminal_advantages[row]
            continue
        base_total = sum(
            terminal_advantages[unit["row"], unit["positions"]].sum()
            for unit in units
        )
        weights = _segmented_choice_turn_weights(units, choice_weights)
        for unit, weight in zip(units, weights):
            positions = unit["positions"]
            share = base_total * float(weight)
            output[unit["row"], positions] = share / float(len(positions))
    return output


def redistribute_segmented_behavior_component_turn_credit(
    terminal_advantages: torch.Tensor,
    response_mask: torch.Tensor,
    turn_boundaries: torch.Tensor,
    segment_trajectory_ids,
    segment_histories,
    segment_global_turn_indices,
    component_advantages: torch.Tensor,
    component_names: list[str] | tuple[str, ...],
    *,
    mix_ratio: float = 0.30,
    routing: dict | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Route behavior components over global turns while preserving fragments."""
    if not 0.0 <= float(mix_ratio) <= 1.0:
        raise ValueError("turn credit mix_ratio must be between 0 and 1")
    if component_advantages.ndim != 2 or component_advantages.shape != (
        terminal_advantages.shape[0],
        len(component_names),
    ):
        raise ValueError("component_advantages and component_names are misaligned")

    behavior = torch.zeros_like(terminal_advantages)
    diagnostics: dict[str, float] = {"fallback_row_count": 0.0}
    for _, rows in _group_segment_indices(segment_trajectory_ids).items():
        units = _ordered_segment_units(
            rows,
            response_mask,
            turn_boundaries,
            segment_histories,
            segment_global_turn_indices,
        )
        if not units:
            for row in rows:
                behavior[row] = terminal_advantages[row]
                diagnostics["fallback_row_count"] += 1.0
            continue
        events = _segment_history_events(units)
        if not any(bool(event.get("has_quality_event", False)) for event in events):
            for row in rows:
                behavior[row] = terminal_advantages[row]
                diagnostics["fallback_row_count"] += 1.0
            continue

        nonempty = list(range(len(units)))
        turn_credits = torch.zeros(
            len(units),
            device=terminal_advantages.device,
            dtype=terminal_advantages.dtype,
        )
        source_row = rows[0]
        for component_idx, component_name in enumerate(component_names):
            value = component_advantages[source_row, component_idx]
            weights = _behavior_component_turn_weights(
                str(component_name),
                float(value.detach().cpu()),
                events,
                routing,
            )
            filtered = {
                idx: weights[idx]
                for idx in nonempty
                if idx < len(weights) and weights[idx] > 0
            }
            normalized = _normalized_turn_weights(filtered, len(units))
            if not any(normalized):
                normalized = _normalized_turn_weights(
                    {idx: 1.0 for idx in nonempty}, len(units)
                )
            turn_credits += value * torch.tensor(
                normalized,
                device=turn_credits.device,
                dtype=turn_credits.dtype,
            )

        total_tokens = float(
            sum(len(unit["positions"]) for unit in units)
        )
        for unit_idx, unit in enumerate(units):
            positions = unit["positions"]
            behavior[unit["row"], positions] = (
                total_tokens * turn_credits[unit_idx] / float(len(positions))
            )

    preprojection = segmented_turn_credit_conservation_stats(
        terminal_advantages,
        behavior,
        response_mask,
        segment_trajectory_ids,
    )
    behavior = _project_segmented_trajectory_credit_sum(
        behavior,
        terminal_advantages,
        response_mask,
        segment_trajectory_ids,
    )
    mixed = (1.0 - float(mix_ratio)) * terminal_advantages + float(mix_ratio) * behavior
    mixed = _project_segmented_trajectory_credit_sum(
        mixed,
        terminal_advantages,
        response_mask,
        segment_trajectory_ids,
    )
    conservation = segmented_turn_credit_conservation_stats(
        terminal_advantages,
        mixed,
        response_mask,
        segment_trajectory_ids,
    )
    diagnostics.update(
        {
            "mix_ratio": float(mix_ratio),
            "conservation_error": float(conservation["absolute_error"].detach().cpu()),
            "conservation_abs_error": float(conservation["absolute_error"].detach().cpu()),
            "conservation_relative_error": float(conservation["relative_error"].detach().cpu()),
            "conservation_mean_token_error": float(conservation["mean_token_error"].detach().cpu()),
            "conservation_finite": float(conservation["finite"].detach().cpu()),
            "preprojection_abs_error": float(preprojection["absolute_error"].detach().cpu()),
            "preprojection_relative_error": float(preprojection["relative_error"].detach().cpu()),
            "preprojection_finite": float(preprojection["finite"].detach().cpu()),
        }
    )
    return mixed, diagnostics


def _project_trajectory_credit_sum(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Project each response onto the reference token-credit sum.

    Sums are measured in float64 and the residual is assigned to the
    smallest-magnitude trainable token. Two passes account for the final cast
    back to the candidate dtype without perturbing every token in a long row.
    """
    if candidate.shape != reference.shape or candidate.shape != response_mask.shape:
        raise ValueError("turn credit tensors and response_mask must have identical shapes")
    projected = candidate.clone()
    mask = response_mask.bool()
    for row in range(projected.shape[0]):
        positions = torch.where(mask[row])[0]
        if positions.numel() == 0:
            continue
        row_values = projected[row, positions]
        anchor = positions[torch.argmin(torch.abs(row_values))]
        target = reference[row, positions].to(torch.float64).sum()
        for _ in range(2):
            actual = projected[row, positions].to(torch.float64).sum()
            residual = target - actual
            projected[row, anchor] = (
                projected[row, anchor].to(torch.float64) + residual
            ).to(projected.dtype)
    return projected


def turn_credit_conservation_stats(
    original: torch.Tensor,
    redistributed: torch.Tensor,
    response_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return float64 trajectory-sum conservation diagnostics.

    Relative error is normalized by the larger masked L1 signal, rather than
    by the signed sum, because positive and negative token credit can cancel.
    """
    if original.shape != redistributed.shape or original.shape != response_mask.shape:
        raise ValueError("turn credit tensors and response_mask must have identical shapes")
    mask = response_mask.bool()
    original64 = torch.where(mask, original.to(torch.float64), 0.0)
    redistributed64 = torch.where(mask, redistributed.to(torch.float64), 0.0)
    absolute_by_row = torch.abs(original64.sum(dim=-1) - redistributed64.sum(dim=-1))
    signal_scale = torch.maximum(
        original64.abs().sum(dim=-1),
        redistributed64.abs().sum(dim=-1),
    ).clamp_min(1.0)
    token_count = mask.sum(dim=-1).clamp_min(1).to(torch.float64)
    finite = (
        torch.isfinite(original64).all()
        & torch.isfinite(redistributed64).all()
        & torch.isfinite(absolute_by_row).all()
    )
    return {
        "absolute_error": absolute_by_row.max(),
        "relative_error": (absolute_by_row / signal_scale).max(),
        "mean_token_error": (absolute_by_row / token_count).max(),
        "finite": finite.to(torch.float64),
    }


def segmented_turn_credit_conservation_stats(
    original: torch.Tensor,
    redistributed: torch.Tensor,
    response_mask: torch.Tensor,
    segment_trajectory_ids,
) -> dict[str, torch.Tensor]:
    """Check conservation after one rollout has been split into fragments."""
    if original.shape != redistributed.shape or original.shape != response_mask.shape:
        raise ValueError("turn credit tensors and response_mask must have identical shapes")
    if len(segment_trajectory_ids) != original.shape[0]:
        raise ValueError("segment trajectory ids must align with the batch")
    mask = response_mask.bool()
    original64 = torch.where(mask, original.to(torch.float64), 0.0)
    redistributed64 = torch.where(mask, redistributed.to(torch.float64), 0.0)
    original_by_group: dict[object, torch.Tensor] = {}
    redistributed_by_group: dict[object, torch.Tensor] = {}
    signal_by_group: dict[object, torch.Tensor] = {}
    for row, trajectory_id in enumerate(segment_trajectory_ids):
        if isinstance(trajectory_id, np.ndarray):
            trajectory_id = trajectory_id.item()
        try:
            key = trajectory_id
            hash(key)
        except TypeError:
            key = str(trajectory_id)
        zero = torch.zeros((), device=original.device, dtype=torch.float64)
        original_by_group[key] = original_by_group.get(key, zero) + original64[row].sum()
        redistributed_by_group[key] = redistributed_by_group.get(key, zero) + redistributed64[row].sum()
        signal_by_group[key] = signal_by_group.get(key, zero) + torch.maximum(
            original64[row].abs().sum(), redistributed64[row].abs().sum()
        )
    if not original_by_group:
        zero = torch.zeros((), device=original.device, dtype=torch.float64)
        return {
            "absolute_error": zero,
            "relative_error": zero,
            "mean_token_error": zero,
            "finite": torch.tensor(1.0, device=original.device, dtype=torch.float64),
        }
    original_values = torch.stack(list(original_by_group.values()))
    redistributed_values = torch.stack(
        [redistributed_by_group[key] for key in original_by_group]
    )
    signal_scale = torch.stack(
        [signal_by_group[key] for key in original_by_group]
    ).clamp_min(1.0)
    error = torch.abs(original_values - redistributed_values)
    finite = (
        torch.isfinite(original_values).all()
        & torch.isfinite(redistributed_values).all()
        & torch.isfinite(error).all()
    )
    return {
        "absolute_error": error.max(),
        "relative_error": (error / signal_scale).max(),
        "mean_token_error": error.mean(),
        "finite": finite.to(torch.float64),
    }


def _project_segmented_trajectory_credit_sum(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    response_mask: torch.Tensor,
    segment_trajectory_ids,
) -> torch.Tensor:
    """Project credit sums per original rollout, not per fragment row."""
    projected = candidate.clone()
    groups = defaultdict(list)
    for row, trajectory_id in enumerate(segment_trajectory_ids):
        try:
            hash(trajectory_id)
            key = trajectory_id
        except TypeError:
            key = str(trajectory_id)
        groups[key].append(row)
    for rows in groups.values():
        positions = [
            (row, position)
            for row in rows
            for position in torch.where(response_mask[row].bool())[0].tolist()
        ]
        if not positions:
            continue
        target = sum(reference[row, position].to(torch.float64) for row, position in positions)
        for _ in range(2):
            actual = sum(projected[row, position].to(torch.float64) for row, position in positions)
            residual = target - actual
            row, position = min(
                positions,
                key=lambda item: abs(float(projected[item[0], item[1]].detach().cpu())),
            )
            projected[row, position] = (
                projected[row, position].to(torch.float64) + residual
            ).to(projected.dtype)
    return projected


def turn_credit_conservation_error(
    original: torch.Tensor,
    redistributed: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Return the float64 maximum absolute per-response conservation error."""
    return turn_credit_conservation_stats(
        original,
        redistributed,
        response_mask,
    )["absolute_error"]


def compute_grpo_multiturn_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    turn_boundaries: torch.Tensor,
    conversation_histories: list,
    data_sources: list,
    index: np.ndarray,
    gamma: float = 0.8,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    turn_level_method: str = "Equalized", # "Equalized" or "R2G" or "EM"
    trajectory_score_method: str = "Sum", # "Sum" or "R2G"
    turn_credit_stage: str = "off",
):
    """Backward-compatible wrapper using terminal-only GRPO semantics.

    The old implementation averaged per-turn rewards to form a trajectory
    score.  That path is intentionally removed; turn credit is an optional
    post-processing step over the already computed terminal advantage.
    """
    advantages, returns = compute_terminal_group_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
    )
    if turn_level_method not in {"off", "none", "Equalized", "R2G", "EM"}:
        raise ValueError(f"Invalid turn_level_method: {turn_level_method}")
    stage = str(turn_credit_stage).casefold()
    if stage not in {"off", "shadow", "train"}:
        raise ValueError(f"Invalid turn_credit_stage: {turn_credit_stage}")
    if turn_level_method not in {"off", "none"} and stage in {"shadow", "train"}:
        advantages = redistribute_terminal_turn_credit(
            advantages,
            response_mask,
            turn_boundaries,
            conversation_histories,
        )
        returns = advantages.clone()
    return advantages, returns


def compute_pf_ppo_reweight_data(
    data,
    reweight_method: str = "pow",
    weight_pow: float = 2.0,
):
    """Reweight the data based on the token_level_scores.

    Args:
        data: DataProto object, containing batch, non_tensor_batch and meta_info
        reweight_method: str, choices: "pow", "max_min", "max_random"
        weight_pow: float, the power of the weight

    Returns:

    """

    @torch.no_grad()
    def compute_weights(scores: torch.Tensor, reweight_method: str, weight_pow: float) -> torch.Tensor:
        if reweight_method == "pow":
            weights = torch.pow(torch.abs(scores), weight_pow)
        elif reweight_method == "max_min":
            max_score = torch.max(scores)
            min_score = torch.min(scores)
            weights = torch.where((scores == max_score) | (scores == min_score), 1.0, 0.0)
        elif reweight_method == "max_random":
            max_score = torch.max(scores)
            weights = torch.where(scores == max_score, 0.4, 0.1)
        else:
            raise ValueError(f"Unsupported reweight_method: {reweight_method}")
        return weights

    scores = data.batch["token_level_scores"].sum(dim=-1)
    weights = compute_weights(scores, reweight_method, weight_pow)
    weights = torch.clamp(weights + 1e-8, min=1e-8)

    batch_size = scores.shape[0]
    sample_indices = torch.multinomial(weights, batch_size, replacement=True)

    resampled_batch = {key: tensor[sample_indices] for key, tensor in data.batch.items()}

    sample_indices_np = sample_indices.numpy()
    resampled_non_tensor_batch = {}
    for key, array in data.non_tensor_batch.items():
        if isinstance(array, np.ndarray):
            resampled_non_tensor_batch[key] = array[sample_indices_np]
        else:
            resampled_non_tensor_batch[key] = [array[i] for i in sample_indices_np]

    resampled_meta_info = {}
    for key, value in data.meta_info.items():
        if isinstance(value, list) and len(value) == batch_size:
            resampled_meta_info[key] = [value[i] for i in sample_indices_np]
        else:
            resampled_meta_info[key] = value

    from copy import deepcopy

    resampled_data = deepcopy(data)
    resampled_data.batch = type(data.batch)(resampled_batch)
    resampled_data.batch.batch_size = data.batch.batch_size
    resampled_data.non_tensor_batch = resampled_non_tensor_batch
    resampled_data.meta_info = resampled_meta_info

    return resampled_data
