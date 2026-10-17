# Copyright (c) 2026 Tianyang Wu
# SPDX-License-Identifier: MIT

"""Concurrent teacher/student PPO with a mixture-of-experts student encoder."""

from __future__ import annotations

from collections.abc import Generator
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from tensordict import TensorDict

from rsl_rl.base.base_algorithm import BaseAlgorithm

from .modules import ActorCriticMoECTS
from .storage import MoeCtsRolloutStorage


class MoECTS(BaseAlgorithm):
    """Concurrent Teacher-Student algorithm with a mixture-of-experts student."""

    policy: ActorCriticMoECTS

    def __init__(
        self,
        policy: ActorCriticMoECTS,
        storage: MoeCtsRolloutStorage,
        num_envs: int,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        betas: tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0.0,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        load_balance_coef: float = 0.01,
        learning_rate: float = 0.001,
        student_encoder_learning_rate: float = 0.001,
        max_grad_norm: float = 1.0,
        use_clipped_value_loss: bool = True,
        schedule: str = "adaptive",
        desired_kl: float = 0.01,
        teacher_env_ratio: float = 0.75,
        normalize_advantage_per_mini_batch: bool = False,
        device: str = "cpu",
        symmetry: Any | None = None,
        multi_gpu_cfg: dict[str, int] | None = None,
    ) -> None:
        if not isinstance(policy, ActorCriticMoECTS):
            raise TypeError(f"MoECTS requires ActorCriticMoECTS, got {type(policy).__name__}.")
        if not 0.0 < teacher_env_ratio < 1.0:
            raise ValueError(f"teacher_env_ratio must be in (0, 1), got {teacher_env_ratio}.")

        super().__init__(policy, learning_rate, device, multi_gpu_cfg)
        self.storage = storage
        self.transition = MoeCtsRolloutStorage.Transition()
        self.symmetry = symmetry

        expected_teacher_envs = max(int(num_envs * teacher_env_ratio), 1)
        if storage.teacher_num_envs != expected_teacher_envs:
            raise ValueError(
                f"Storage has {storage.teacher_num_envs} teacher environments, expected {expected_teacher_envs}."
            )
        self.teacher_num_envs = expected_teacher_envs
        self.student_num_envs = int(num_envs - expected_teacher_envs)
        if self.student_num_envs < 1:
            raise ValueError("MoE-CTS requires at least one student environment.")
        self.teacher_env_idxs, self.student_env_idxs = _split_environment_indices(
            num_envs,
            self.teacher_num_envs,
            self.device,
        )

        optimizer_groups: list[dict[str, Any]] = [
            {"params": self.policy.teacher_encoder.parameters()},
            {"params": self.policy.critic.parameters()},
            {"params": self.policy.actor.parameters()},
        ]
        if hasattr(self.policy, "std"):
            optimizer_groups.append({"params": [self.policy.std]})
        elif hasattr(self.policy, "log_std"):
            optimizer_groups.append({"params": [self.policy.log_std]})
        self.optimizer = optim.Adam(
            optimizer_groups,
            lr=learning_rate,
            betas=betas,
            weight_decay=weight_decay,
        )
        self.optimizer_stu_enc = optim.Adam(
            self.policy.student_moe_encoder.parameters(),
            lr=student_encoder_learning_rate,
            betas=betas,
            weight_decay=weight_decay,
        )

        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.load_balance_coef = load_balance_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch

    def act(self, obs: TensorDict) -> torch.Tensor:
        teacher_results = self._policy_results(obs[self.teacher_env_idxs], is_teacher=True)
        student_results = self._policy_results(obs[self.student_env_idxs], is_teacher=False)
        results = [torch.cat([teacher, student], dim=0) for teacher, student in zip(teacher_results, student_results)]
        self.transition.actions = results[0]
        self.transition.values = results[1]
        self.transition.actions_log_prob = results[2]
        self.transition.action_mean = results[3]
        self.transition.action_sigma = results[4]
        self.transition.observations = torch.cat(
            [obs[self.teacher_env_idxs], obs[self.student_env_idxs]],
            dim=0,
        )

        reordered_actions = torch.zeros_like(self.transition.actions)
        reordered_actions[self.teacher_env_idxs] = self.transition.actions[: self.teacher_num_envs]
        reordered_actions[self.student_env_idxs] = self.transition.actions[self.teacher_num_envs :]
        return reordered_actions

    def _policy_results(self, obs: TensorDict, is_teacher: bool) -> tuple[torch.Tensor, ...]:
        actions = self.policy.act(obs, is_teacher)
        return (
            actions.detach(),
            self.policy.evaluate(obs, is_teacher).detach(),
            self.policy.get_actions_log_prob(actions).detach(),
            self.policy.action_mean.detach(),
            self.policy.action_std.detach(),
        )

    def process_env_step(
        self,
        obs: TensorDict,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict[str, Any],
    ) -> None:
        self.policy.update_normalization(obs)
        teacher_indices = self.teacher_env_idxs
        student_indices = self.student_env_idxs
        rewards = rewards.clone()
        self.transition.rewards = torch.cat([rewards[teacher_indices], rewards[student_indices]], dim=0)
        self.transition.dones = torch.cat([dones[teacher_indices], dones[student_indices]], dim=0)

        if "time_outs" in extras:
            time_outs = extras["time_outs"].to(self.device)
            reordered_time_outs = torch.cat([time_outs[teacher_indices], time_outs[student_indices]], dim=0)
            assert self.transition.values is not None
            assert self.transition.rewards is not None
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * reordered_time_outs.unsqueeze(1),
                1,
            )

        self.storage.add_transition(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def compute_returns(self, obs: TensorDict) -> None:
        storage = self.storage
        last_values = torch.cat(
            [
                self.policy.evaluate(obs[self.teacher_env_idxs], is_teacher=True).detach(),
                self.policy.evaluate(obs[self.student_env_idxs], is_teacher=False).detach(),
            ],
            dim=0,
        )
        advantage: torch.Tensor | float = 0.0
        for step in reversed(range(storage.num_transitions_per_env)):
            next_values = last_values if step == storage.num_transitions_per_env - 1 else storage.values[step + 1]
            next_is_not_terminal = 1.0 - storage.dones[step].float()
            delta = storage.rewards[step] + next_is_not_terminal * self.gamma * next_values - storage.values[step]
            advantage = delta + next_is_not_terminal * self.gamma * self.lam * advantage
            storage.returns[step] = advantage + storage.values[step]
        storage.advantages = storage.returns - storage.values
        if not self.normalize_advantage_per_mini_batch:
            storage.advantages = (storage.advantages - storage.advantages.mean()) / (
                storage.advantages.std() + 1e-8
            )

    def update(self) -> dict[str, float]:
        teacher_samples = (
            self.teacher_num_envs * self.storage.num_transitions_per_env // self.num_mini_batches
        )
        student_samples = (
            self.student_num_envs * self.storage.num_transitions_per_env // self.num_mini_batches
        )
        batch_indices = self.storage.generate_mini_batch_indices()
        num_aug = 1
        original_mask = torch.ones(teacher_samples + student_samples, dtype=torch.bool, device=self.device)
        if self.symmetry is not None:
            num_aug = int(self.symmetry.num_aug)
            original_mask = self.symmetry.get_original_mask(teacher_samples, student_samples, self.device)

        generator = self._mini_batch_generator(teacher_samples, batch_indices)
        augmented_teacher_samples = teacher_samples * num_aug
        augmented_student_samples = student_samples * num_aug
        totals, num_updates = self._update_ppo_batches(
            generator,
            augmented_teacher_samples,
            augmented_student_samples,
            original_mask,
        )
        if num_updates == 0:
            raise RuntimeError("MoE-CTS received no PPO mini-batches.")

        # The distributed gradient reducer walks every policy parameter. Clear the
        # PPO gradients so the student-only pass cannot reduce stale actor/critic
        # gradients from the final PPO mini-batch.
        self.optimizer.zero_grad(set_to_none=True)
        student_totals, student_updates = self._update_student_batches(
            self._mini_batch_generator(teacher_samples, batch_indices),
            augmented_teacher_samples,
        )
        if student_updates != num_updates:
            raise RuntimeError(
                f"MoE-CTS generated {num_updates} PPO mini-batches but {student_updates} student mini-batches."
            )

        totals.update(student_totals)
        self.storage.clear()
        return {name: value / num_updates for name, value in totals.items()}

    def _update_ppo_batches(
        self,
        generator: Generator,
        teacher_samples: int,
        student_samples: int,
        original_mask: torch.Tensor,
    ) -> tuple[dict[str, float], int]:
        totals = {"value": 0.0, "surrogate": 0.0, "entropy": 0.0}
        num_updates = 0
        for batch in generator:
            (
                obs_batch,
                actions_batch,
                target_values_batch,
                advantages_batch,
                returns_batch,
                old_actions_log_prob_batch,
                old_mu_batch,
                old_sigma_batch,
            ) = batch
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (
                        advantages_batch.std() + 1e-8
                    )

            teacher_results = self._batch_policy_results(
                obs_batch[:teacher_samples], actions_batch[:teacher_samples], True
            )
            student_stop = teacher_samples + student_samples
            student_results = self._batch_policy_results(
                obs_batch[teacher_samples:student_stop],
                actions_batch[teacher_samples:student_stop],
                False,
            )
            actions_log_prob_batch, value_batch, mu_batch, sigma_batch, entropy_batch = [
                torch.cat([teacher, student], dim=0)
                for teacher, student in zip(teacher_results, student_results)
            ]

            self._adapt_learning_rate(mu_batch, sigma_batch, old_mu_batch, old_sigma_batch, original_mask)

            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio,
                1.0 - self.clip_param,
                1.0 + self.clip_param,
            )
            surrogate_losses = torch.max(surrogate, surrogate_clipped)
            surrogate_loss = (
                surrogate_losses[:teacher_samples].mean() + surrogate_losses[teacher_samples:student_stop].mean()
            )

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param,
                    self.clip_param,
                )
                value_loss = torch.max(
                    (value_batch - returns_batch).pow(2),
                    (value_clipped - returns_batch).pow(2),
                ).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch[original_mask].mean()
            )
            self.optimizer.zero_grad()
            loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            parameters = (parameter for group in self.optimizer.param_groups for parameter in group["params"])
            nn.utils.clip_grad_norm_(parameters, self.max_grad_norm)
            self.optimizer.step()

            totals["value"] += value_loss.item()
            totals["surrogate"] += surrogate_loss.item()
            totals["entropy"] += entropy_batch.mean().item()
            num_updates += 1
        return totals, num_updates

    def _update_student_batches(
        self,
        generator: Generator,
        teacher_samples: int,
    ) -> tuple[dict[str, float], int]:
        totals = {"mean_latent_loss": 0.0, "mean_load_balance_loss": 0.0}
        student_updates = 0
        for batch in generator:
            obs_batch = batch[0]
            actor_obs = self.policy.actor_obs_normalizer(self.policy.get_actor_obs(obs_batch))
            student_latent, gating_weights = self.policy.student_moe_encoder(actor_obs[teacher_samples:])
            with torch.no_grad():
                critic_obs = self.policy.critic_obs_normalizer(self.policy.get_critic_obs(obs_batch))
                teacher_latent = self.policy.teacher_encoder(critic_obs[teacher_samples:])
            latent_loss = (teacher_latent - student_latent).pow(2).mean()
            mean_usage = gating_weights.mean(dim=0)
            target_usage = torch.full_like(mean_usage, 1.0 / gating_weights.shape[1])
            load_balance_loss = (mean_usage - target_usage).pow(2).mean()
            student_loss = latent_loss + self.load_balance_coef * load_balance_loss

            self.optimizer_stu_enc.zero_grad()
            student_loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            nn.utils.clip_grad_norm_(self.policy.student_moe_encoder.parameters(), self.max_grad_norm)
            self.optimizer_stu_enc.step()
            totals["mean_latent_loss"] += latent_loss.item()
            totals["mean_load_balance_loss"] += load_balance_loss.item()
            student_updates += 1
        return totals, student_updates

    def _mini_batch_generator(
        self,
        teacher_samples: int,
        batch_indices: tuple[torch.Tensor, torch.Tensor],
    ) -> Generator:
        generator = self.storage.mini_batch_generator(
            self.num_mini_batches,
            self.num_learning_epochs,
            batch_indices=batch_indices,
        )
        if self.symmetry is not None:
            generator = self.symmetry.augment_moe_cts_batch_generator(generator, teacher_samples)
        return generator

    def _batch_policy_results(
        self,
        obs: TensorDict,
        actions: torch.Tensor,
        is_teacher: bool,
    ) -> tuple[torch.Tensor, ...]:
        self.policy.act(obs, is_teacher)
        return (
            self.policy.get_actions_log_prob(actions),
            self.policy.evaluate(obs, is_teacher),
            self.policy.action_mean,
            self.policy.action_std,
            self.policy.entropy,
        )

    def _adapt_learning_rate(
        self,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        old_mu: torch.Tensor,
        old_sigma: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        if self.desired_kl is None or self.schedule != "adaptive":
            return
        with torch.inference_mode():
            kl = torch.sum(
                torch.log(sigma[mask] / old_sigma[mask] + 1.0e-5)
                + (torch.square(old_sigma[mask]) + torch.square(old_mu[mask] - mu[mask]))
                / (2.0 * torch.square(sigma[mask]))
                - 0.5,
                dim=-1,
            )
            kl_mean = torch.mean(kl)
            if self.is_multi_gpu:
                torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                kl_mean /= self.gpu_world_size
            if self.gpu_global_rank == 0:
                if kl_mean > self.desired_kl * 2.0:
                    self.learning_rate = max(1.0e-5, self.learning_rate / 1.5)
                elif 0.0 < kl_mean < self.desired_kl / 2.0:
                    self.learning_rate = min(1.0e-2, self.learning_rate * 1.5)
            if self.is_multi_gpu:
                learning_rate = torch.tensor(self.learning_rate, device=self.device)
                torch.distributed.broadcast(learning_rate, src=0)
                self.learning_rate = learning_rate.item()
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.learning_rate


def _split_environment_indices(
    num_envs: int,
    teacher_num_envs: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Distribute student environments evenly while preserving the reference 3:1 split."""
    student_num_envs = num_envs - teacher_num_envs
    student_indices = torch.div(
        torch.arange(student_num_envs, device=device) * num_envs,
        student_num_envs,
        rounding_mode="floor",
    )
    student_mask = torch.zeros(num_envs, dtype=torch.bool, device=device)
    student_mask[student_indices] = True
    teacher_indices = torch.arange(num_envs, device=device)[~student_mask]
    if teacher_indices.numel() != teacher_num_envs or student_indices.unique().numel() != student_num_envs:
        raise RuntimeError("Failed to construct the requested teacher/student environment split.")
    return teacher_indices, student_indices


__all__ = ["MoECTS"]
