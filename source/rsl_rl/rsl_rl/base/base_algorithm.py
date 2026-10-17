# Copyright (c) 2026 Tianyang Wu
#
# SPDX-License-Identifier: MIT

"""Algorithm interface and behavior shared by all RSL-RL runners."""

from __future__ import annotations

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from tensordict import TensorDict
from typing import Any, Callable


class BaseAlgorithm(ABC):
    """Define the algorithm contract required by :class:`BaseRunner`."""

    policy: nn.Module
    """Trainable policy module owned by the algorithm."""

    optimizer: torch.optim.Optimizer
    """Optimizer used for checkpointing and training."""

    learning_rate: float
    """Current optimizer learning rate exposed to the runner logger."""

    def __init__(
        self,
        policy: nn.Module,
        learning_rate: float,
        device: str = "cpu",
        multi_gpu_cfg: dict[str, int] | None = None,
    ) -> None:
        self.device = device
        self.policy = policy
        self.policy.to(self.device)
        self.learning_rate = learning_rate

        self.is_multi_gpu = multi_gpu_cfg is not None
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

    @abstractmethod
    def act(self, obs: TensorDict) -> torch.Tensor:
        """Compute actions and record the pre-step transition data."""

    @abstractmethod
    def process_env_step(
        self,
        obs: TensorDict,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict[str, Any],
    ) -> None:
        """Record environment results and finalize one rollout transition."""

    @abstractmethod
    def compute_returns(self, obs: TensorDict) -> None:
        """Prepare rollout targets before the policy update."""

    @abstractmethod
    def update(self) -> dict[str, float]:
        """Update the policy and return scalar losses for logging."""

    def get_inference_policy(self, device: str | None = None) -> Callable[[TensorDict], torch.Tensor]:
        """Return the policy inference callable on the requested device."""
        self.eval_mode()
        if device is not None:
            self.policy.to(device)

        inference_policy = getattr(self.policy, "act_inference", None)
        if not callable(inference_policy):
            raise TypeError(f"Policy '{type(self.policy).__name__}' must implement act_inference().")
        return inference_policy

    def train_mode(self) -> None:
        """Put the policy in training mode."""
        self.policy.train()

    def eval_mode(self) -> None:
        """Put the policy in evaluation mode."""
        self.policy.eval()

    @property
    def action_std(self) -> torch.Tensor | None:
        """Return the current action standard deviation when the policy exposes it."""
        action_std = getattr(self.policy, "action_std", None)
        return action_std if isinstance(action_std, torch.Tensor) else None

    def broadcast_parameters(self) -> None:
        """Broadcast policy parameters from the primary distributed worker."""
        model_params = [self.policy.state_dict()]
        torch.distributed.broadcast_object_list(model_params, src=0)
        self.policy.load_state_dict(model_params[0])

    def reduce_parameters(self) -> None:
        """Average policy gradients across all distributed workers."""
        grads = [parameter.grad.view(-1) for parameter in self.policy.parameters() if parameter.grad is not None]
        all_grads = torch.cat(grads)
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        offset = 0
        for parameter in self.policy.parameters():
            if parameter.grad is not None:
                numel = parameter.numel()
                parameter.grad.data.copy_(all_grads[offset : offset + numel].view_as(parameter.grad.data))
                offset += numel


__all__ = ["BaseAlgorithm"]
