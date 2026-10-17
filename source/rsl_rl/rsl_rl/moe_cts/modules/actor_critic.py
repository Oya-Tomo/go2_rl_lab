# Copyright (c) 2026 Tianyang Wu
# SPDX-License-Identifier: MIT

"""Teacher/student actor-critic model for MoE-CTS."""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal
from typing import Any, NoReturn

from rsl_rl.base.networks import EmpiricalNormalization

from .moe import L2Norm, MLP, MoE, SimNorm


class StudentMoEEncoder(nn.Module):
    """Encode observation history through a dense mixture of experts."""

    def __init__(
        self,
        expert_num: int,
        input_dim: int,
        hidden_dims: tuple[int, ...] | list[int],
        output_dim: int,
        activation: str = "elu",
        norm_type: str = "l2norm",
    ) -> None:
        super().__init__()
        self.norm_layer = _make_latent_normalizer(norm_type)
        self.moe = MoE(
            expert_num=expert_num,
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            activation=activation,
        )

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent, weights = self.moe(obs)
        return self.norm_layer(latent), weights


class ActorCriticMoECTS(nn.Module):
    """Concurrent teacher/student actor-critic with a student MoE encoder."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        critic_hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        teacher_encoder_hidden_dims: tuple[int, ...] | list[int] = (512, 256),
        student_encoder_hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
        expert_num: int = 8,
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        latent_dim: int = 32,
        norm_type: str = "l2norm",
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if kwargs:
            raise TypeError(f"Unexpected ActorCriticMoECTS arguments: {sorted(kwargs)}")
        required_groups = {"policy", "critic", "single_obs"}
        missing_groups = required_groups.difference(obs.keys())
        if missing_groups:
            raise ValueError(f"ActorCriticMoECTS is missing observation groups: {sorted(missing_groups)}")
        if norm_type not in {"l2norm", "simnorm"}:
            raise ValueError(f"Unsupported latent normalization {norm_type!r}.")

        self.num_actions = int(num_actions)
        self.obs_groups = obs_groups
        self.num_actor_obs = _observation_dimension(obs, obs_groups["policy"], "policy")
        num_critic_obs = _observation_dimension(obs, obs_groups["critic"], "critic")
        self.num_single_obs = int(obs["single_obs"].shape[-1])

        self.teacher_encoder = nn.Sequential(
            MLP(num_critic_obs, latent_dim, teacher_encoder_hidden_dims, activation=activation),
            _make_latent_normalizer(norm_type),
        )
        self.student_moe_encoder = StudentMoEEncoder(
            expert_num=expert_num,
            input_dim=self.num_actor_obs,
            hidden_dims=student_encoder_hidden_dims,
            output_dim=latent_dim,
            activation=activation,
            norm_type=norm_type,
        )

        self.state_dependent_std = bool(state_dependent_std)
        actor_output_dim: int | list[int] = [2, self.num_actions] if self.state_dependent_std else self.num_actions
        self.actor = MLP(latent_dim + self.num_single_obs, actor_output_dim, actor_hidden_dims, activation)
        self.critic = MLP(latent_dim + num_critic_obs, 1, critic_hidden_dims, activation)

        self.actor_obs_normalization = bool(actor_obs_normalization)
        self.critic_obs_normalization = bool(critic_obs_normalization)
        if self.actor_obs_normalization:
            self.actor_obs_normalizer: nn.Module = EmpiricalNormalization(self.num_actor_obs)
            self.single_obs_normalizer: nn.Module = EmpiricalNormalization(self.num_single_obs)
        else:
            self.actor_obs_normalizer = nn.Identity()
            self.single_obs_normalizer = nn.Identity()
        self.critic_obs_normalizer: nn.Module = (
            EmpiricalNormalization(num_critic_obs) if self.critic_obs_normalization else nn.Identity()
        )

        self.noise_std_type = noise_std_type
        if self.state_dependent_std:
            output_layer = self.actor.network[-2]
            if not isinstance(output_layer, nn.Linear):
                raise TypeError("State-dependent action noise requires a linear actor output layer.")
            nn.init.zeros_(output_layer.weight[self.num_actions :])
            if noise_std_type == "scalar":
                nn.init.constant_(output_layer.bias[self.num_actions :], init_noise_std)
            elif noise_std_type == "log":
                nn.init.constant_(output_layer.bias[self.num_actions :], torch.log(torch.tensor(init_noise_std + 1e-7)))
            else:
                raise ValueError(f"Unsupported standard-deviation type {noise_std_type!r}.")
        elif noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(self.num_actions))
        elif noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(self.num_actions)))
        else:
            raise ValueError(f"Unsupported standard-deviation type {noise_std_type!r}.")

        self.distribution: Normal | None = None
        Normal.set_default_validate_args(False)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        return None

    def forward(self, *args: Any, **kwargs: Any) -> NoReturn:
        raise NotImplementedError("Use act(), act_inference(), or evaluate() with ActorCriticMoECTS.")

    @property
    def action_mean(self) -> torch.Tensor:
        return self._distribution().mean

    @property
    def action_std(self) -> torch.Tensor:
        return self._distribution().stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self._distribution().entropy().sum(dim=-1)

    def _distribution(self) -> Normal:
        if self.distribution is None:
            raise RuntimeError("Action distribution has not been initialized; call act() first.")
        return self.distribution

    def _update_distribution(self, latent_and_obs: torch.Tensor) -> None:
        if self.state_dependent_std:
            actor_output = self.actor(latent_and_obs)
            if self.noise_std_type == "scalar":
                mean, std = torch.unbind(actor_output, dim=-2)
            else:
                mean, log_std = torch.unbind(actor_output, dim=-2)
                std = torch.exp(log_std)
        else:
            mean = self.actor(latent_and_obs)
            if self.noise_std_type == "scalar":
                std = self.std.expand_as(mean)
            else:
                std = torch.exp(self.log_std).expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, obs: TensorDict, is_teacher: bool, **kwargs: Any) -> torch.Tensor:
        single_obs = self.single_obs_normalizer(obs["single_obs"])
        if is_teacher:
            critic_obs = self.critic_obs_normalizer(self.get_critic_obs(obs))
            latent = self.teacher_encoder(critic_obs)
        else:
            with torch.no_grad():
                actor_obs = self.actor_obs_normalizer(self.get_actor_obs(obs))
                latent, _ = self.student_moe_encoder(actor_obs)
        self._update_distribution(torch.cat([latent, single_obs], dim=-1))
        return self._distribution().sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        single_obs = self.single_obs_normalizer(obs["single_obs"])
        actor_obs = self.actor_obs_normalizer(self.get_actor_obs(obs))
        latent, _ = self.student_moe_encoder(actor_obs)
        actor_output = self.actor(torch.cat([latent, single_obs], dim=-1))
        return actor_output[..., 0, :] if self.state_dependent_std else actor_output

    def evaluate(self, obs: TensorDict, is_teacher: bool, **kwargs: Any) -> torch.Tensor:
        critic_obs = self.critic_obs_normalizer(self.get_critic_obs(obs))
        if is_teacher:
            latent = self.teacher_encoder(critic_obs)
        else:
            actor_obs = self.actor_obs_normalizer(self.get_actor_obs(obs))
            latent, _ = self.student_moe_encoder(actor_obs)
        return self.critic(torch.cat([latent.detach(), critic_obs], dim=-1))

    def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        return torch.cat([obs[group] for group in self.obs_groups["policy"]], dim=-1)

    def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        return torch.cat([obs[group] for group in self.obs_groups["critic"]], dim=-1)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self._distribution().log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.actor_obs_normalization:
            self.actor_obs_normalizer.update(self.get_actor_obs(obs))  # type: ignore[attr-defined]
            self.single_obs_normalizer.update(obs["single_obs"])  # type: ignore[attr-defined]
        if self.critic_obs_normalization:
            self.critic_obs_normalizer.update(self.get_critic_obs(obs))  # type: ignore[attr-defined]

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], strict: bool = True) -> bool:
        super().load_state_dict(state_dict, strict=strict)
        return True


def _make_latent_normalizer(norm_type: str) -> nn.Module:
    if norm_type == "l2norm":
        return L2Norm()
    if norm_type == "simnorm":
        return SimNorm()
    raise ValueError(f"Unsupported latent normalization {norm_type!r}.")


def _observation_dimension(obs: TensorDict, groups: list[str], role: str) -> int:
    dimension = 0
    for group in groups:
        if group not in obs.keys():
            raise ValueError(f"Unknown {role} observation group {group!r}.")
        if len(obs[group].shape) != 2:
            raise ValueError(f"ActorCriticMoECTS only supports flat observations, got {group}: {obs[group].shape}.")
        dimension += int(obs[group].shape[-1])
    return dimension


__all__ = ["ActorCriticMoECTS", "StudentMoEEncoder"]
