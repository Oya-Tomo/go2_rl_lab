# Copyright (c) 2026 Tianyang Wu
# SPDX-License-Identifier: MIT

"""Configuration schema for the MoE-CTS algorithm package."""

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils.configclass import configclass

from rsl_rl.base.base_runner_config import RslRlBaseRunnerCfg
from rsl_rl.utils.robogauge_evaluator import RoboGaugeEvaluatorCfg


@configclass
class MoeCtsSymmetryCfg:
    """Optional symmetry augmentation for segmented teacher/student batches."""

    use_symmetric_augmentation: bool = True
    symmetry_class: str = "go2_rl_lab.utils.symmetry:Go2Symmetry"


@configclass
class RslRlMoeCtsActorCriticCfg:
    """Teacher/student policy-network configuration."""

    class_name: str = "rsl_rl.moe_cts:ActorCriticMoECTS"
    init_noise_std: float = MISSING
    noise_std_type: Literal["scalar", "log"] = "scalar"
    state_dependent_std: bool = False
    actor_obs_normalization: bool = False
    critic_obs_normalization: bool = False
    actor_hidden_dims: list[int] = MISSING
    critic_hidden_dims: list[int] = MISSING
    teacher_encoder_hidden_dims: list[int] = MISSING
    student_encoder_hidden_dims: list[int] = MISSING
    expert_num: int = MISSING
    latent_dim: int = MISSING
    norm_type: Literal["l2norm", "simnorm"] = "l2norm"
    activation: str = MISSING


@configclass
class RslRlMoeCtsAlgorithmCfg:
    """Concurrent teacher/student PPO and latent-distillation configuration."""

    class_name: str = "rsl_rl.moe_cts:MoECTS"
    num_learning_epochs: int = MISSING
    num_mini_batches: int = MISSING
    learning_rate: float = MISSING
    student_encoder_learning_rate: float = MISSING
    schedule: str = MISSING
    gamma: float = MISSING
    lam: float = MISSING
    entropy_coef: float = MISSING
    desired_kl: float = MISSING
    max_grad_norm: float = MISSING
    value_loss_coef: float = MISSING
    load_balance_coef: float = MISSING
    use_clipped_value_loss: bool = MISSING
    clip_param: float = MISSING
    teacher_env_ratio: float = MISSING
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.0
    normalize_advantage_per_mini_batch: bool = False
    symmetry_cfg: MoeCtsSymmetryCfg | None = None


@configclass
class RslRlMoeCtsRunnerCfg(RslRlBaseRunnerCfg):
    """Runner configuration for MoE-CTS."""

    class_name: str = "rsl_rl.moe_cts:MoeCtsRunner"
    policy: RslRlMoeCtsActorCriticCfg = MISSING
    algorithm: RslRlMoeCtsAlgorithmCfg = MISSING
    robogauge: RoboGaugeEvaluatorCfg = RoboGaugeEvaluatorCfg()


__all__ = [
    "MoeCtsSymmetryCfg",
    "RslRlMoeCtsActorCriticCfg",
    "RslRlMoeCtsAlgorithmCfg",
    "RslRlMoeCtsRunnerCfg",
]
