# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base runner configuration shared by all RSL-RL algorithms."""

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils.configclass import configclass


@configclass
class RslRlBaseRunnerCfg:
    """Base configuration for an RSL-RL runner."""

    class_name: str = MISSING
    """Fully qualified runner class used by the generic entry points."""

    seed: int = 42
    """The seed for the experiment."""

    device: str = "cuda:0"
    """The device used by the learning agent."""

    num_steps_per_env: int = MISSING
    """The number of steps collected from each environment per update."""

    max_iterations: int = MISSING
    """The maximum number of learning iterations."""

    obs_groups: dict[str, list[str]] = MISSING
    """Mapping from algorithm observation sets to environment observation groups.

    For example, ``{"policy": ["policy"], "critic": ["critic"]}`` routes
    the environment's ``policy`` observations to the actor and its ``critic``
    observations to the critic.
    """

    clip_actions: float | None = None
    """The action clipping magnitude, or ``None`` to disable clipping."""

    clip_rewards: float | None = None
    """The absolute total-reward clipping magnitude, or ``None`` to disable clipping."""

    save_interval: int = MISSING
    """The number of learning iterations between checkpoints."""

    log_interval: int = 10
    """Iterations averaged between persistent writes; console output remains per iteration."""

    experiment_name: str = MISSING
    """The experiment name."""

    run_name: str = ""
    """Optional suffix appended to the run directory name."""

    logger: Literal["tensorboard", "swanlab", "wandb"] = "tensorboard"
    """The logging backend."""

    wandb_project: str = "go2-rl-lab"
    """The Weights & Biases project name."""

    swanlab_project: str = "go2-rl-lab"
    """The SwanLab project name."""

    resume: bool = False
    """Whether to resume a previous training run."""

    load_run: str = ".*"
    """Run-directory name or regular expression used when resuming."""

    load_checkpoint: str = "model_.*.pt"
    """Checkpoint name or regular expression used when resuming."""
