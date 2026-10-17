# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from tensordict import TensorDict
from typing import Any

from rsl_rl.base.base_algorithm import BaseAlgorithm
from rsl_rl.ppo.modules import ActorCritic
from rsl_rl.ppo.storage import PpoRolloutStorage


class PPO(BaseAlgorithm):
    """Proximal Policy Optimization algorithm (https://arxiv.org/abs/1707.06347)."""

    policy: ActorCritic
    """The actor critic module."""

    def __init__(
        self,
        policy: ActorCritic,
        storage: PpoRolloutStorage,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        learning_rate: float = 0.001,
        max_grad_norm: float = 1.0,
        use_clipped_value_loss: bool = True,
        schedule: str = "adaptive",
        desired_kl: float = 0.01,
        normalize_advantage_per_mini_batch: bool = False,
        device: str = "cpu",
        # Distributed training parameters
        multi_gpu_cfg: dict[str, int] | None = None,
        # Symmetry parameters
        symmetry: Any | None = None,
    ) -> None:
        """Configure feed-forward PPO and optional symmetry augmentation."""
        super().__init__(policy, learning_rate, device, multi_gpu_cfg)
        if num_mini_batches < 1 or num_learning_epochs < 1:
            raise ValueError("Mini-batch and epoch counts must be positive.")
        # Symmetry components
        self.symmetry = symmetry
        self.symmetry_augment_func = self.symmetry.augment_batch if self.symmetry is not None else None

        # PPO components
        # Create the optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

        # Add storage
        self.storage = storage
        self.transition = storage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch

    def act(self, obs: TensorDict) -> torch.Tensor:
        # Compute the actions and values
        self.transition.actions = self.policy.act(obs).detach()
        self.transition.values = self.policy.evaluate(obs).detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        # Record observations before env.step()
        self.transition.observations = obs
        return self.transition.actions

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, Any]
    ) -> None:
        # Update the normalizers
        self.policy.update_normalization(obs)

        # Record the rewards and dones
        # Note: We clone here because later on we bootstrap the rewards based on timeouts
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        # Bootstrapping on time outs
        if "time_outs" in extras:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device), 1
            )

        # Record the transition
        self.storage.add_transition(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def compute_returns(self, obs: TensorDict) -> None:
        st = self.storage
        # Compute value for the last step
        last_values = self.policy.evaluate(obs).detach()
        # Compute returns and advantages
        advantage = 0
        for step in reversed(range(st.num_transitions_per_env)):
            # If we are at the last step, bootstrap the return value
            next_values = last_values if step == st.num_transitions_per_env - 1 else st.values[step + 1]
            # 1 if we are not in a terminal state, 0 otherwise
            next_is_not_terminal = 1.0 - st.dones[step].float()
            # TD error: r_t + gamma * V(s_{t+1}) - V(s_t)
            delta = st.rewards[step] + next_is_not_terminal * self.gamma * next_values - st.values[step]
            # Advantage: A(s_t, a_t) = delta_t + gamma * lambda * A(s_{t+1}, a_{t+1})
            advantage = delta + next_is_not_terminal * self.gamma * self.lam * advantage
            # Return: R_t = A(s_t, a_t) + V(s_t)
            st.returns[step] = advantage + st.values[step]
        # Compute the advantages
        st.advantages = st.returns - st.values
        # Normalize the advantages if per minibatch normalization is not used
        if not self.normalize_advantage_per_mini_batch:
            st.advantages = (st.advantages - st.advantages.mean()) / (st.advantages.std(unbiased=st.advantages.numel() > 1) + 1e-8)

    def loss_terms(self, batch: tuple) -> tuple[torch.Tensor, ...]:
        """Compute PPO losses, entropy, and KL from a feed-forward mini-batch."""
        obs, actions, target_values, advantages, returns, old_log_prob, old_mu, old_sigma = batch
        original_samples = actions.shape[0]
        # Normalize before augmentation so duplicating samples does not change variance.
        if self.normalize_advantage_per_mini_batch:
            advantages = (advantages - advantages.mean()) / (
                advantages.std(unbiased=advantages.numel() > 1) + 1e-8
            )
        if self.symmetry_augment_func is not None:
            obs, actions, target_values, advantages, returns, old_log_prob, old_mu, old_sigma = (
                self.symmetry_augment_func(
                    obs, actions, target_values, advantages, returns, old_log_prob, old_mu, old_sigma
                )
            )
        target_values = target_values.squeeze(-1)
        advantages = advantages.squeeze(-1)
        returns = returns.squeeze(-1)
        old_log_prob = old_log_prob.squeeze(-1)
        self.policy.act(obs)
        values = self.policy.evaluate(obs).squeeze(-1)
        log_prob = self.policy.get_actions_log_prob(actions)
        ratio = torch.exp(log_prob - old_log_prob)
        surrogate = torch.maximum(
            -advantages * ratio,
            -advantages * ratio.clamp(1 - self.clip_param, 1 + self.clip_param),
        ).mean()
        if self.use_clipped_value_loss:
            clipped_values = target_values + (values - target_values).clamp(-self.clip_param, self.clip_param)
            value_loss = torch.maximum((values - returns).square(), (clipped_values - returns).square()).mean()
        else:
            value_loss = (returns - values).square().mean()
        entropy = self.policy.entropy[:original_samples].mean()
        with torch.no_grad():
            mu = self.policy.action_mean[:original_samples]
            sigma = self.policy.action_std[:original_samples]
            old_mu, old_sigma = old_mu[:original_samples], old_sigma[:original_samples]
            kl = (
                torch.log(sigma / old_sigma + 1e-5)
                + (old_sigma.square() + (old_mu - mu).square()) / (2 * sigma.square())
                - 0.5
            ).sum(-1)
            kl_mean = kl.mean()
        loss = surrogate + self.value_loss_coef * value_loss - self.entropy_coef * entropy
        return loss, value_loss, surrogate, entropy, kl_mean

    def _adapt_learning_rate(self, kl: torch.Tensor) -> None:
        """Adjust and synchronize the optimizer learning rate from original-sample KL."""
        if self.desired_kl is None or self.schedule != "adaptive":
            return
        if self.is_multi_gpu:
            torch.distributed.all_reduce(kl, op=torch.distributed.ReduceOp.SUM)
            kl /= self.gpu_world_size
        if self.gpu_global_rank == 0:
            if kl > self.desired_kl * 2:
                self.learning_rate = max(1e-5, self.learning_rate / 1.5)
            elif kl < self.desired_kl / 2 and kl > 0:
                self.learning_rate = min(1e-2, self.learning_rate * 1.5)
        if self.is_multi_gpu:
            rate = torch.tensor(self.learning_rate, device=self.device)
            torch.distributed.broadcast(rate, src=0)
            self.learning_rate = rate.item()
        for group in self.optimizer.param_groups:
            group["lr"] = self.learning_rate

    def update(self) -> dict[str, float]:
        """Optimize PPO batches and weight reported losses by sample count."""
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        totals = torch.zeros(3, device=self.device)
        valid_samples = torch.zeros((), device=self.device)
        for batch in generator:
            loss, value, surrogate, entropy, kl = self.loss_terms(batch)
            self._adapt_learning_rate(kl)
            self.optimizer.zero_grad()
            loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            count = batch[1].shape[0]
            totals += torch.stack((value.detach(), surrogate.detach(), entropy.detach())) * count
            valid_samples += count
        values = (totals / valid_samples.clamp_min(1)).tolist()
        self.storage.clear()
        return dict(value=values[0], surrogate=values[1], entropy=values[2])
