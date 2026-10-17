# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Algorithm-independent runner lifecycle and rollout loop shared by all algorithms."""

from __future__ import annotations

import os
import time
import torch
from abc import ABC, abstractmethod
from tensordict import TensorDict
from typing import Any, Callable

from rsl_rl.utils import resolve_obs_groups
from rsl_rl.utils.logger import Logger

from .base_algorithm import BaseAlgorithm
from .env import VecEnv


class BaseRunner(ABC):
    """Shared lifecycle for algorithms trained from on-policy environment rollouts."""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        self.cfg = train_cfg
        self.policy_cfg = train_cfg["policy"]
        self.alg_cfg = train_cfg["algorithm"]
        self.device = device
        self.env = env

        self._configure_multi_gpu()

        obs = self.env.get_observations()
        self.cfg["obs_groups"] = resolve_obs_groups(obs, self.cfg["obs_groups"], self._get_default_obs_sets())
        self.alg: BaseAlgorithm = self._construct_algorithm(obs)
        if not isinstance(self.alg, BaseAlgorithm):
            raise TypeError(
                f"Runner '{type(self).__name__}' constructed '{type(self.alg).__name__}', "
                "which must inherit BaseAlgorithm."
            )

        self.logger = Logger(
            log_dir=log_dir,
            cfg=self.cfg,
            env_cfg=self.env.cfg,
            num_envs=self.env.num_envs,
            is_distributed=self.is_distributed,
            gpu_world_size=self.gpu_world_size,
            gpu_global_rank=self.gpu_global_rank,
            device=self.device,
            env_groups=self._get_logger_env_groups(),
        )
        self.current_learning_iteration = 0

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        """Collect rollouts and update the configured algorithm."""
        self._before_learn()

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs = self.env.get_observations().to(self.device)
        self.train_mode()

        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations
        for it in range(start_it, total_it):
            start = time.time()
            with torch.inference_mode():
                for _ in range(self.cfg["num_steps_per_env"]):
                    actions = self.alg.act(obs)
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    self.logger.process_env_step(rewards, dones, extras)

                stop = time.time()
                collect_time = stop - start
                start = stop
                self.alg.compute_returns(obs)

            loss_dict = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            self.logger.log(
                it=it + 1,
                start_it=start_it,
                total_it=total_it,
                collect_time=collect_time,
                learn_time=learn_time,
                loss_dict=loss_dict,
                learning_rate=float(self.alg.learning_rate),
                action_std=self.alg.action_std,
            )

            if it % self.cfg["save_interval"] == 0:
                self.save(
                    os.path.join(self.logger.log_dir, f"model_{it}.pt"),  # type: ignore[arg-type]
                    iteration=it,
                )

        if self.logger.log_dir is not None and not self.logger.disable_logs:
            self.save(
                os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"),
                iteration=self.current_learning_iteration,
                final=True,
            )

        self.logger.close()

    def save(
        self,
        path: str,
        infos: dict | None = None,
        *,
        iteration: int | None = None,
        final: bool = False,
    ) -> None:
        """Save runner and algorithm state."""
        torch.save(self._checkpoint_state(infos), path)
        saved_iteration = self.current_learning_iteration if iteration is None else iteration
        self._after_save(saved_iteration, final)

    def load(self, path: str, load_optimizer: bool = True, map_location: str | None = None) -> dict | None:
        """Restore runner and algorithm state."""
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
        return self._load_checkpoint_state(loaded_dict, load_optimizer)

    def get_inference_policy(self, device: str | None = None) -> Callable[[TensorDict], torch.Tensor]:
        return self.alg.get_inference_policy(device)

    def train_mode(self) -> None:
        self.alg.train_mode()

    def eval_mode(self) -> None:
        self.alg.eval_mode()

    def add_git_repo_to_log(self, repo_file_path: str) -> None:
        self.logger.add_git_repo(repo_file_path)

    def export_policy_as_jit(
        self,
        path: str,
        normalizer: object | None = None,
        filename: str = "policy.pt",
        **kwargs: Any,
    ) -> None:
        """Export the current policy as TorchScript."""
        raise NotImplementedError(f"{type(self).__name__} does not provide a TorchScript exporter.")

    def export_policy_as_onnx(
        self,
        path: str,
        normalizer: object | None = None,
        filename: str = "policy.onnx",
        verbose: bool = False,
    ) -> None:
        """Export the current policy as ONNX."""
        raise NotImplementedError(f"{type(self).__name__} does not provide an ONNX exporter.")

    def _before_learn(self) -> None:
        """Run algorithm-specific validation immediately before training."""
        return None

    def _after_save(self, iteration: int, final: bool) -> None:
        """Run algorithm-specific work after a checkpoint is saved."""
        return None

    def _get_logger_env_groups(self) -> dict[str, torch.Tensor] | None:
        """Return named environment-index groups for algorithm-specific episode metrics."""
        return None

    def _checkpoint_state(self, infos: dict | None) -> dict[str, Any]:
        return {
            "model_state_dict": self.alg.policy.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }

    def _load_checkpoint_state(self, loaded_dict: dict[str, Any], load_optimizer: bool) -> dict | None:
        resumed_training = self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])
        if load_optimizer and resumed_training:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        if resumed_training:
            self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict.get("infos")

    @abstractmethod
    def _get_default_obs_sets(self) -> list[str]:
        """Return non-policy observation sets required by the algorithm."""

    @abstractmethod
    def _construct_algorithm(self, obs: TensorDict) -> BaseAlgorithm:
        """Construct the algorithm, policy, and storage for this runner."""

    def _configure_multi_gpu(self) -> None:
        self.gpu_world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.is_distributed = self.gpu_world_size > 1

        if not self.is_distributed:
            self.gpu_local_rank = 0
            self.gpu_global_rank = 0
            self.multi_gpu_cfg = None
            return

        self.gpu_local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.gpu_global_rank = int(os.getenv("RANK", "0"))
        self.multi_gpu_cfg = {
            "global_rank": self.gpu_global_rank,
            "local_rank": self.gpu_local_rank,
            "world_size": self.gpu_world_size,
        }

        visible_gpus = torch.cuda.device_count()
        device_index = self.gpu_local_rank if self.gpu_local_rank < visible_gpus else 0
        expected_device = f"cuda:{device_index}"
        if self.device != expected_device:
            raise ValueError(
                f"Device '{self.device}' does not match expected device '{expected_device}' "
                f"for local rank '{self.gpu_local_rank}'."
            )
        if self.gpu_local_rank >= self.gpu_world_size:
            raise ValueError(
                f"Local rank '{self.gpu_local_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
            )
        if self.gpu_global_rank >= self.gpu_world_size:
            raise ValueError(
                f"Global rank '{self.gpu_global_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
            )

        torch.distributed.init_process_group(backend="nccl", rank=self.gpu_global_rank, world_size=self.gpu_world_size)
        torch.cuda.set_device(device_index)
