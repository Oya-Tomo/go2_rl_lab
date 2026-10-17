# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import asdict
from typing import Any

from rsl_rl.utils.tensorboard_utils import TensorboardSummaryWriter

try:
    import wandb
except ModuleNotFoundError:
    raise ModuleNotFoundError("wandb package is required to log to Weights and Biases.") from None


class WandbSummaryWriter(TensorboardSummaryWriter):
    """Summary writer for Weights and Biases."""

    def __init__(self, log_dir: str, flush_secs: int, cfg: dict) -> None:
        super().__init__(log_dir, flush_secs)

        # Build a stable W&B run name from the experiment and optional CLI run name.
        # Keep the timestamped local log directory independent from the W&B display name.
        experiment_name = cfg.get("experiment_name", "").replace("_", "-")
        configured_run_name = cfg.get("run_name", "")
        run_name = ""
        if experiment_name:
            run_name = f"{experiment_name}_{configured_run_name}" if configured_run_name else experiment_name
        run_name += "_" + os.path.split(log_dir)[-1]

        # Get wandb project and entity
        try:
            project = cfg["wandb_project"]
        except KeyError:
            raise KeyError("Please specify wandb_project in the runner config, e.g. legged_gym.") from None
        try:
            entity = os.environ["WANDB_USERNAME"]
        except KeyError:
            entity = None

        # Initialize wandb
        wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            # Stop uploading wandb console logs
            settings=wandb.Settings(console="off"),
        )
        wandb.config.update({"log_dir": log_dir})

    def store_config(self, env_cfg: dict | object, train_cfg: dict) -> None:
        wandb.config.update({"runner_cfg": train_cfg})
        wandb.config.update({"policy_cfg": train_cfg["policy"]})
        wandb.config.update({"alg_cfg": train_cfg["algorithm"]})
        try:
            wandb.config.update({"env_cfg": env_cfg.to_dict()})
        except Exception:
            wandb.config.update({"env_cfg": asdict(env_cfg)})

    def add_scalar(
        self,
        tag: str,
        scalar_value: float,
        global_step: int | None = None,
        walltime: float | None = None,
        new_style: bool = False,
    ) -> None:
        super().add_scalar(
            tag,
            scalar_value,
            global_step=global_step,
            walltime=walltime,
            new_style=new_style,
        )
        wandb.log({tag: scalar_value}, step=global_step)

    def add_custom_scalars(
        self,
        tag_scalar_dict: Mapping[str, Any],
        *,
        step_metric: str,
    ) -> None:
        """Log scalars with a custom W&B x-axis and a TensorBoard global step."""
        super().add_custom_scalars(tag_scalar_dict, step_metric=step_metric)

        defined_step_metrics: set[str] = getattr(self, "_defined_custom_step_metrics", set())
        if step_metric not in defined_step_metrics:
            wandb.define_metric(step_metric, hidden=True)
            defined_step_metrics.add(step_metric)
            self._defined_custom_step_metrics = defined_step_metrics

        metric_steps: dict[str, str] = getattr(self, "_custom_metric_steps", {})
        for tag in tag_scalar_dict:
            if tag == step_metric:
                continue
            configured_step = metric_steps.get(tag)
            if configured_step is None:
                wandb.define_metric(tag, step_metric=step_metric)
                metric_steps[tag] = step_metric
            elif configured_step != step_metric:
                raise ValueError(
                    f"W&B metric {tag!r} is already configured with step metric {configured_step!r}, "
                    f"not {step_metric!r}."
                )
        self._custom_metric_steps = metric_steps

        # Do not pass ``step=`` here: W&B's internal step must follow arrival
        # order, while ``step_metric`` supplies the x-axis for these metrics.
        wandb.log(dict(tag_scalar_dict))

    def stop(self) -> None:
        wandb.finish()

    def save_file(self, path: str) -> None:
        wandb.save(path, base_path=os.path.dirname(path))
