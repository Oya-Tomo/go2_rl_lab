# Copyright (c) 2026 Tianyang Wu
# SPDX-License-Identifier: MIT

"""Runner specialization for MoE-CTS."""

from __future__ import annotations

from tensordict import TensorDict
from typing import Any

from rsl_rl.base.base_runner import BaseRunner
from rsl_rl.base.env import VecEnv
from rsl_rl.utils import RoboGaugeEvaluator, resolve_callable

from .algorithm import MoECTS
from .modules import ActorCriticMoECTS
from .storage import MoeCtsRolloutStorage


class MoeCtsRunner(BaseRunner):
    """Construct, train, export, and evaluate a MoE-CTS policy."""

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

    def _get_logger_env_groups(self) -> dict[str, Any]:
        return {
            "teacher": self.alg.teacher_env_idxs,
            "student": self.alg.student_env_idxs,
        }

    def export_policy_as_jit(
        self,
        path: str,
        normalizer: object | None = None,
        filename: str = "policy.pt",
        **kwargs: Any,
    ) -> None:
        from .exporter import export_policy_as_jit

        policy = self.alg.policy
        export_policy_as_jit(
            policy,
            actor_obs_normalizer=(normalizer if normalizer is not None else policy.actor_obs_normalizer),
            single_obs_normalizer=policy.single_obs_normalizer,
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
        from .exporter import export_policy_as_onnx

        policy = self.alg.policy
        export_policy_as_onnx(
            policy,
            path,
            actor_obs_normalizer=(normalizer if normalizer is not None else policy.actor_obs_normalizer),
            single_obs_normalizer=policy.single_obs_normalizer,
            filename=filename,
            verbose=verbose,
        )

    def _export_robogauge_policy(self, path: str, filename: str) -> None:
        if self._robogauge_single_obs_dim is None:
            raise ValueError("RoboGauge MoE-CTS export requires a 'single_obs' observation group.")
        if self._robogauge_single_obs_dim != self.alg.policy.num_single_obs:
            raise ValueError(
                f"Environment single_obs dimension {self._robogauge_single_obs_dim} does not match "
                f"the policy dimension {self.alg.policy.num_single_obs}."
            )
        self.export_policy_as_jit(
            path=path,
            normalizer=self.alg.policy.actor_obs_normalizer,
            filename=filename,
        )

    def _after_save(self, iteration: int, final: bool) -> None:
        self.robogauge_evaluator.update(iteration, final)

    def _construct_algorithm(self, obs: TensorDict) -> MoECTS:
        symmetry_cfg = self.alg_cfg.pop("symmetry_cfg", None)
        symmetry = None
        if symmetry_cfg is not None and symmetry_cfg["use_symmetric_augmentation"]:
            symmetry = resolve_callable(symmetry_cfg["symmetry_class"])(self.env)

        policy_class = resolve_callable(self.policy_cfg.pop("class_name"))
        policy: ActorCriticMoECTS = policy_class(
            obs,
            self.cfg["obs_groups"],
            self.env.num_actions,
            **self.policy_cfg,
        ).to(self.device)

        teacher_env_ratio = float(self.alg_cfg["teacher_env_ratio"])
        teacher_num_envs = max(int(self.env.num_envs * teacher_env_ratio), 1)
        storage = MoeCtsRolloutStorage(
            self.env.num_envs,
            teacher_num_envs,
            self.cfg["num_steps_per_env"],
            obs,
            [self.env.num_actions],
            self.device,
        )
        algorithm_class = resolve_callable(self.alg_cfg.pop("class_name"))
        return algorithm_class(
            policy,
            storage,
            self.env.num_envs,
            device=self.device,
            symmetry=symmetry,
            **self.alg_cfg,
            multi_gpu_cfg=self.multi_gpu_cfg,
        )

    def _checkpoint_state(self, infos: dict | None) -> dict[str, Any]:
        state = super()._checkpoint_state(infos)
        state["optimizer_stu_enc_state_dict"] = self.alg.optimizer_stu_enc.state_dict()
        return state

    def _load_checkpoint_state(self, loaded_dict: dict[str, Any], load_optimizer: bool) -> dict | None:
        resumed_training = self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])
        if load_optimizer and resumed_training:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            student_optimizer_state = loaded_dict.get("optimizer_stu_enc_state_dict")
            if student_optimizer_state is None:
                raise KeyError("MoE-CTS checkpoint is missing optimizer_stu_enc_state_dict.")
            self.alg.optimizer_stu_enc.load_state_dict(student_optimizer_state)
        if resumed_training:
            self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict.get("infos")


__all__ = ["MoeCtsRunner"]
