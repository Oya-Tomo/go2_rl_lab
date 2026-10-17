# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO rollout storage and mini-batch generation."""

from __future__ import annotations

import torch
from collections.abc import Generator
from tensordict import TensorDict

from rsl_rl.base.storage import RolloutStorage


class PpoRolloutStorage(RolloutStorage):
    """Rollout storage containing PPO-specific value and policy statistics."""

    class Transition(RolloutStorage.Transition):
        def __init__(self) -> None:
            super().__init__()
            self.values: torch.Tensor | None = None
            self.actions_log_prob: torch.Tensor | None = None
            self.action_mean: torch.Tensor | None = None
            self.action_sigma: torch.Tensor | None = None

    def __init__(
        self,
        num_envs: int,
        num_transitions_per_env: int,
        obs: TensorDict,
        actions_shape: tuple[int] | list[int],
        device: str = "cpu",
    ) -> None:
        super().__init__(num_envs, num_transitions_per_env, obs, actions_shape, device)
        self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)

    def _add_algorithm_transition(self, transition: RolloutStorage.Transition) -> None:
        if not isinstance(transition, self.Transition):
            raise TypeError(f"Expected {self.Transition.__name__}, got {type(transition).__name__}.")
        if transition.values is None:
            raise ValueError("PPO transition values must be populated before storage.")
        if transition.actions_log_prob is None:
            raise ValueError("PPO transition action log probabilities must be populated before storage.")
        if transition.action_mean is None:
            raise ValueError("PPO transition action means must be populated before storage.")
        if transition.action_sigma is None:
            raise ValueError("PPO transition action standard deviations must be populated before storage.")
        self.values[self.step].copy_(transition.values)
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)

    def mini_batch_generator(self, num_mini_batches: int, num_epochs: int = 8) -> Generator:
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        observations = self.observations.flatten(0, 1)
        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

        for _ in range(num_epochs):
            for index in range(num_mini_batches):
                start = index * mini_batch_size
                stop = (index + 1) * mini_batch_size
                batch_idx = indices[start:stop]
                yield (
                    observations[batch_idx],
                    actions[batch_idx],
                    values[batch_idx],
                    advantages[batch_idx],
                    returns[batch_idx],
                    old_actions_log_prob[batch_idx],
                    old_mu[batch_idx],
                    old_sigma[batch_idx],
                )
