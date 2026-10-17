# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Algorithm-independent rollout-buffer primitives."""

from __future__ import annotations

import torch
from tensordict import TensorDict


class RolloutStorage:
    """Store transition fields shared by rollout-based algorithms.

    Algorithm packages extend this class with their own transition fields and
    batch generators instead of adding algorithm switches to the common buffer.
    """

    class Transition:
        """Fields common to a single environment transition."""

        def __init__(self) -> None:
            self.observations: TensorDict | None = None
            self.actions: torch.Tensor | None = None
            self.rewards: torch.Tensor | None = None
            self.dones: torch.Tensor | None = None

        def clear(self) -> None:
            self.__init__()

    def __init__(
        self,
        num_envs: int,
        num_transitions_per_env: int,
        obs: TensorDict,
        actions_shape: tuple[int] | list[int],
        device: str = "cpu",
    ) -> None:
        self.device = device
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.actions_shape = actions_shape

        self.observations = TensorDict(
            {key: torch.zeros(num_transitions_per_env, *value.shape, device=device) for key, value in obs.items()},
            batch_size=[num_transitions_per_env, num_envs],
            device=self.device,
        )
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()

        self.step = 0

    def add_transition(self, transition: Transition) -> None:
        if self.step >= self.num_transitions_per_env:
            raise OverflowError("Rollout buffer overflow! You should call clear() before adding new transitions.")
        if transition.observations is None:
            raise ValueError("Transition observations must be populated before storage.")
        if transition.actions is None:
            raise ValueError("Transition actions must be populated before storage.")
        if transition.rewards is None:
            raise ValueError("Transition rewards must be populated before storage.")
        if transition.dones is None:
            raise ValueError("Transition dones must be populated before storage.")

        self.observations[self.step].copy_(transition.observations)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self._add_algorithm_transition(transition)
        self.step += 1

    def clear(self) -> None:
        self.step = 0

    def _add_algorithm_transition(self, transition: Transition) -> None:
        """Copy fields owned by a concrete algorithm storage."""
