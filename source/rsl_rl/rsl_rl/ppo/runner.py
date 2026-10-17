# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO runner specialization."""

from __future__ import annotations

from tensordict import TensorDict
from typing import Any

from rsl_rl.base.base_runner import BaseRunner
from rsl_rl.base.env import VecEnv
from rsl_rl.ppo.algorithm import PPO
from rsl_rl.ppo.modules import ActorCritic
from rsl_rl.ppo.storage import PpoRolloutStorage
from rsl_rl.utils import RoboGaugeEvaluator, resolve_callable


class PPORunner(BaseRunner):
    """Construct and train PPO actor-critic policies."""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        super().__init__(env, train_cfg, log_dir, device)

        obs = self.env.get_observations()
        self._robogauge_single_obs_dim = int(obs["single_obs"].shape[-1]) if "single_obs" in obs else None
        self.robogauge_evaluator = RoboGaugeEvaluator(
            task_name=str(self.cfg["robogauge"].get("task_name", "go2_lab_1.3.1")),
            config=self.cfg["robogauge"],
            log_dir=self.logger.log_dir,
            writer=self.logger.writer,
            export_policy=self._export_robogauge_policy,
            experiment_name=str(self.cfg.get("experiment_name", "rsl_rl")),
            disabled=self.logger.disable_logs,
        )

    def _get_default_obs_sets(self) -> list[str]:
        return ["critic"]

    def export_policy_as_jit(
        self,
        path: str,
        normalizer: object | None = None,
        filename: str = "policy.pt",
        **kwargs: Any,
    ) -> None:
        from rsl_rl.ppo.exporter import export_policy_as_jit

        num_single_obs = kwargs.get("num_single_obs")
        num_actions = kwargs.get("num_actions")
        if num_single_obs is None or num_actions is None:
            raise ValueError("PPO JIT export requires num_single_obs and num_actions for observation history.")

        export_policy_as_jit(
            self.alg.policy,
            actor_obs_normalizer=normalizer,
            num_single_obs=int(num_single_obs),
            num_actions=int(num_actions),
            path=path,
            filename=filename,
        )

    def export_policy_as_onnx(
        self,
        path: str,
        normalizer: object | None = None,
        filename: str = "policy.onnx",
        verbose: bool = False,
    ) -> None:
        from rsl_rl.ppo.exporter import export_policy_as_onnx

        export_policy_as_onnx(self.alg.policy, path, normalizer, filename, verbose)

    def _export_robogauge_policy(self, path: str, filename: str) -> None:
        if self._robogauge_single_obs_dim is None:
            raise ValueError("RoboGauge PPO export requires a 'single_obs' observation group.")

        self.export_policy_as_jit(
            path=path,
            normalizer=getattr(self.alg.policy, "actor_obs_normalizer", None),
            filename=filename,
            num_single_obs=self._robogauge_single_obs_dim,
            num_actions=self.env.num_actions,
        )

    def _after_save(self, iteration: int, final: bool) -> None:
        self.robogauge_evaluator.update(iteration, final)

    def _construct_algorithm(self, obs: TensorDict) -> PPO:
        """Construct optional symmetry helpers and algorithm-specific rollout storage."""
        symmetry_cfg = self.alg_cfg.pop("symmetry_cfg")
        actor_critic_class = resolve_callable(self.policy_cfg.pop("class_name"))
        symmetry = (
            resolve_callable(symmetry_cfg["symmetry_class"])(self.env)
            if symmetry_cfg["use_symmetric_augmentation"] else None
        )

        actor_critic: ActorCritic = actor_critic_class(
            obs, self.cfg["obs_groups"], self.env.num_actions, **self.policy_cfg
        ).to(self.device)

        alg_class = resolve_callable(self.alg_cfg.pop("class_name"))
        storage_class = getattr(alg_class, "storage_class", PpoRolloutStorage)
        storage = storage_class(
            self.env.num_envs,
            self.cfg["num_steps_per_env"],
            obs,
            [self.env.num_actions],
            self.device,
        )

        return alg_class(
            actor_critic,
            storage,
            device=self.device,
            symmetry=symmetry,
            **self.alg_cfg,
            multi_gpu_cfg=self.multi_gpu_cfg,
        )

__all__ = ["PPORunner"]
