# Copyright (c) 2026 Tianyang Wu
# SPDX-License-Identifier: MIT

"""Teacher/student rollout storage for MoE-CTS."""

from __future__ import annotations

import torch
from collections.abc import Generator
from functools import partial
from tensordict import TensorDict

from rsl_rl.base.storage import RolloutStorage


class MoeCtsRolloutStorage(RolloutStorage):
    """PPO rollout data arranged as contiguous teacher and student segments."""

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
        teacher_num_envs: int,
        num_transitions_per_env: int,
        obs: TensorDict,
        actions_shape: tuple[int, ...] | list[int],
        device: str = "cpu",
    ) -> None:
        if not 0 < teacher_num_envs < num_envs:
            raise ValueError(
                f"teacher_num_envs must be in (0, num_envs), got {teacher_num_envs} for {num_envs} environments."
            )
        super().__init__(num_envs, num_transitions_per_env, obs, actions_shape, device)
        self.teacher_num_envs = int(teacher_num_envs)
        self.student_num_envs = int(num_envs - teacher_num_envs)
        self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)

    def _add_algorithm_transition(self, transition: RolloutStorage.Transition) -> None:
        if not isinstance(transition, self.Transition):
            raise TypeError(f"Expected {self.Transition.__name__}, got {type(transition).__name__}.")
        required = {
            "values": transition.values,
            "actions_log_prob": transition.actions_log_prob,
            "action_mean": transition.action_mean,
            "action_sigma": transition.action_sigma,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(f"MoE-CTS transition fields must be populated before storage: {missing}.")
        assert transition.values is not None
        assert transition.actions_log_prob is not None
        assert transition.action_mean is not None
        assert transition.action_sigma is not None
        self.values[self.step].copy_(transition.values)
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)

    def generate_mini_batch_indices(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate reusable shuffled indices for teacher and student samples."""
        teacher_samples = self.teacher_num_envs * self.num_transitions_per_env
        student_samples = self.student_num_envs * self.num_transitions_per_env
        teacher_indices = torch.randperm(teacher_samples, requires_grad=False, device=self.device)
        student_indices = teacher_samples + torch.randperm(
            student_samples,
            requires_grad=False,
            device=self.device,
        )
        return teacher_indices, student_indices

    def mini_batch_generator(
        self,
        num_mini_batches: int,
        num_epochs: int = 8,
        batch_indices: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Generator:
        teacher_samples = self.teacher_num_envs * self.num_transitions_per_env
        student_samples = self.student_num_envs * self.num_transitions_per_env
        if teacher_samples % num_mini_batches != 0 or student_samples % num_mini_batches != 0:
            raise ValueError(
                "Teacher and student rollout sample counts must both be divisible by num_mini_batches; "
                f"got {teacher_samples}, {student_samples}, and {num_mini_batches}."
            )

        teacher_batch_size = teacher_samples // num_mini_batches
        student_batch_size = student_samples // num_mini_batches
        if batch_indices is None:
            batch_indices = self.generate_mini_batch_indices()
        teacher_indices, student_indices = batch_indices
        if teacher_indices.shape != (teacher_samples,) or student_indices.shape != (student_samples,):
            raise ValueError(
                "MoE-CTS batch indices do not match the rollout sample counts; "
                f"got {tuple(teacher_indices.shape)} and {tuple(student_indices.shape)}, "
                f"expected {(teacher_samples,)} and {(student_samples,)}."
            )

        observations = self.observations.transpose(0, 1).flatten(0, 1)
        arrays = (
            self.actions.transpose(0, 1).flatten(0, 1),
            self.values.transpose(0, 1).flatten(0, 1),
            self.advantages.transpose(0, 1).flatten(0, 1),
            self.returns.transpose(0, 1).flatten(0, 1),
            self.actions_log_prob.transpose(0, 1).flatten(0, 1),
            self.mu.transpose(0, 1).flatten(0, 1),
            self.sigma.transpose(0, 1).flatten(0, 1),
        )

        def select_segmented_batch(data: TensorDict | torch.Tensor, slices: tuple[tuple[int, int], tuple[int, int]]):
            (teacher_start, teacher_stop), (student_start, student_stop) = slices
            return torch.cat(
                [
                    data[teacher_indices[teacher_start:teacher_stop]],
                    data[student_indices[student_start:student_stop]],
                ],
                dim=0,
            ).detach()

        for _ in range(num_epochs):
            for batch_index in range(num_mini_batches):
                slices = (
                    (batch_index * teacher_batch_size, (batch_index + 1) * teacher_batch_size),
                    (batch_index * student_batch_size, (batch_index + 1) * student_batch_size),
                )
                get_batch = partial(select_segmented_batch, slices=slices)
                yield (
                    get_batch(observations),
                    *(get_batch(array) for array in arrays),
                )


__all__ = ["MoeCtsRolloutStorage"]
