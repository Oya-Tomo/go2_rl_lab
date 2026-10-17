# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils.configclass import configclass

from rsl_rl.base.base_runner_config import RslRlBaseRunnerCfg
from rsl_rl.utils.robogauge_evaluator import RoboGaugeEvaluatorCfg


@configclass
class SymmetryCfg:
    """Symmetry data-augmentation configuration."""

    use_symmetric_augmentation: bool = True
    """Whether to enable symmetry data augmentation."""

    symmetry_class: str = "go2_rl_lab.utils.symmetry:Go2Symmetry"
    """Import path of the task symmetry implementation."""


@configclass
class RslRlPpoActorCriticCfg:
    """Configuration for PPO actor-critic networks."""

    class_name: str = "rsl_rl.ppo:ActorCritic"
    """The policy class name."""

    init_noise_std: float = MISSING
    """The initial action-noise standard deviation."""

    noise_std_type: Literal["scalar", "log"] = "scalar"
    """The action-noise standard-deviation representation."""

    state_dependent_std: bool = False
    """Whether the policy predicts a state-dependent standard deviation."""

    actor_obs_normalization: bool = False
    """Whether to normalize actor observations."""

    critic_obs_normalization: bool = False
    """Whether to normalize critic observations."""

    actor_hidden_dims: list[int] = MISSING
    """The actor's hidden-layer dimensions."""

    critic_hidden_dims: list[int] = MISSING
    """The critic's hidden-layer dimensions."""

    activation: str = MISSING
    """The actor and critic activation function."""


@configclass
class RslRlPpoAlgorithmCfg:
    """Configuration for the PPO algorithm."""

    class_name: str = "rsl_rl.ppo:PPO"
    """The algorithm class name."""

    num_learning_epochs: int = MISSING
    """The number of learning epochs per update."""

    num_mini_batches: int = MISSING
    """Number of mini-batches per learning epoch."""

    learning_rate: float = MISSING
    """The optimizer learning rate."""

    schedule: str = MISSING
    """The learning-rate schedule."""

    gamma: float = MISSING
    """The reward discount factor."""

    lam: float = MISSING
    """The Generalized Advantage Estimation lambda."""

    entropy_coef: float = MISSING
    """The entropy-loss coefficient."""

    desired_kl: float = MISSING
    """The desired KL divergence."""

    max_grad_norm: float = MISSING
    """The maximum gradient norm."""

    value_loss_coef: float = MISSING
    """The value-loss coefficient."""

    use_clipped_value_loss: bool = MISSING
    """Whether to use a clipped value loss."""

    clip_param: float = MISSING
    """The PPO clipping parameter."""

    normalize_advantage_per_mini_batch: bool = False
    """Whether to normalize advantages separately in each mini-batch."""

    symmetry_cfg: SymmetryCfg = SymmetryCfg()
    """Symmetry augmentation, enabled by default with the Go2 implementation."""


@configclass
class RslRlPpoRunnerCfg(RslRlBaseRunnerCfg):
    """Configuration for the PPO runner."""

    class_name: str = "rsl_rl.ppo:PPORunner"
    """Fully qualified runner class."""

    policy: RslRlPpoActorCriticCfg = MISSING
    """The actor-critic policy configuration."""

    algorithm: RslRlPpoAlgorithmCfg = MISSING
    """The PPO algorithm configuration."""

    robogauge: RoboGaugeEvaluatorCfg = RoboGaugeEvaluatorCfg()
    """Asynchronous RoboGauge evaluation configuration."""


__all__ = [
    "RslRlPpoActorCriticCfg",
    "RslRlPpoAlgorithmCfg",
    "RslRlPpoRunnerCfg",
    "SymmetryCfg",
]
