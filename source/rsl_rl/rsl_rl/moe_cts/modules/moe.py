# Copyright (c) 2026 Tianyang Wu
# SPDX-License-Identifier: MIT

"""Mixture-of-experts network components used by MoE-CTS."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as functional


class CatELU(nn.Module):
    """Apply ELU to a tensor and its negation, doubling the feature dimension."""

    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.elu = nn.ELU(inplace=inplace)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.dim() < 2:
            raise ValueError(f"CatELU expects a feature dimension, got shape {tuple(value.shape)}.")
        return torch.cat((self.elu(value), self.elu(-value)), dim=-1)


class L2Norm(nn.Module):
    """Normalize the last dimension to unit L2 norm."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return functional.normalize(value, p=2.0, dim=-1)


class SimNorm(nn.Module):
    """Normalize fixed-size latent groups with a simplex softmax."""

    def __init__(self, group_dim: int = 8) -> None:
        super().__init__()
        self.group_dim = int(group_dim)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.shape[-1] % self.group_dim != 0:
            raise ValueError(
                f"SimNorm input dimension {value.shape[-1]} must be divisible by group_dim {self.group_dim}."
            )
        shape = value.shape
        normalized = functional.softmax(value.reshape(*shape[:-1], -1, self.group_dim), dim=-1)
        return normalized.reshape(*shape)

    def __repr__(self) -> str:
        return f"SimNorm(group_dim={self.group_dim})"


def get_activation(name: str) -> nn.Module:
    """Resolve the activation names supported by the reference MoE implementation."""
    activations: dict[str, type[nn.Module]] = {
        "elu": nn.ELU,
        "selu": nn.SELU,
        "relu": nn.ReLU,
        "crelu": nn.ReLU,
        "lrelu": nn.LeakyReLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "cat_elu": CatELU,
    }
    try:
        return activations[name.lower()]()
    except KeyError as exc:
        raise ValueError(f"Unsupported MoE activation {name!r}; expected one of {sorted(activations)}.") from exc


class MLP(nn.Module):
    """Reference-compatible MLP with an explicit ``network`` state-dict prefix."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int | tuple[int, ...] | list[int],
        hidden_dims: tuple[int, ...] | list[int],
        activation: str = "elu",
        last_activation: str | None = None,
    ) -> None:
        super().__init__()
        if not hidden_dims:
            raise ValueError("MoE MLP requires at least one hidden layer.")

        layers: list[nn.Module] = []
        last_dim = int(input_dim)
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, int(hidden_dim)))
            layers.append(get_activation(activation))
            last_dim = int(hidden_dim) * (2 if activation == "cat_elu" else 1)

        if isinstance(output_dim, int):
            layers.append(nn.Linear(last_dim, output_dim))
        else:
            output_shape = tuple(int(dim) for dim in output_dim)
            layers.append(nn.Linear(last_dim, math.prod(output_shape)))
            layers.append(nn.Unflatten(dim=-1, unflattened_size=output_shape))

        if last_activation is not None:
            layers.append(get_activation(last_activation))
        self.network = nn.Sequential(*layers)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)


class Experts(nn.Module):
    """Generate all expert outputs from a shared backbone and grouped heads."""

    def __init__(
        self,
        expert_num: int,
        input_dim: int,
        backbone_hidden_dims: tuple[int, ...] | list[int],
        expert_hidden_dim: int,
        output_dim: int,
        activation: str = "elu",
    ) -> None:
        super().__init__()
        self.expert_num = int(expert_num)
        self.output_dim = int(output_dim)
        self.backbone = MLP(
            input_dim,
            self.expert_num * int(expert_hidden_dim),
            backbone_hidden_dims,
            activation,
            last_activation=activation,
        )
        activation_multiplier = 2 if activation == "cat_elu" else 1
        self.experts = nn.Conv1d(
            in_channels=self.expert_num * int(expert_hidden_dim) * activation_multiplier,
            out_channels=self.expert_num * self.output_dim,
            kernel_size=1,
            groups=self.expert_num,
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        shared_features = self.backbone(value).unsqueeze(-1)
        expert_outputs = self.experts(shared_features).squeeze(-1)
        return expert_outputs.reshape(-1, self.expert_num, self.output_dim)


class MoE(nn.Module):
    """Dense mixture of experts with a learned softmax gating network."""

    def __init__(
        self,
        expert_num: int,
        input_dim: int,
        hidden_dims: tuple[int, ...] | list[int],
        output_dim: int,
        activation: str = "elu",
    ) -> None:
        super().__init__()
        if len(hidden_dims) < 2:
            raise ValueError("MoE hidden_dims must contain a backbone and an expert hidden dimension.")
        self.experts = Experts(
            expert_num=expert_num,
            input_dim=input_dim,
            backbone_hidden_dims=hidden_dims[:-1],
            expert_hidden_dim=hidden_dims[-1],
            output_dim=output_dim,
            activation=activation,
        )
        self.gating_network = nn.Sequential(
            MLP(input_dim, expert_num, hidden_dims, activation),
            nn.Softmax(dim=-1),
        )

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = self.gating_network(value)
        expert_outputs = self.experts(value)
        output = torch.sum(weights.unsqueeze(-1) * expert_outputs, dim=1)
        return output, weights


__all__ = ["CatELU", "Experts", "L2Norm", "MLP", "MoE", "SimNorm", "get_activation"]
