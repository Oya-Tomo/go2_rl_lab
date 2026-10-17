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
    import swanlab
except ModuleNotFoundError:
    raise ModuleNotFoundError("swanlab package is required to log to SwanLab.") from None


def _to_python_scalar(value: Any) -> Any:
    """Convert tensor-like scalars into Python numbers accepted by SwanLab."""
    if hasattr(value, "item"):
        return value.item()
    return value


class SwanlabSummaryWriter(TensorboardSummaryWriter):
    """Summary writer for SwanLab."""

    def __init__(self, log_dir: str, flush_secs: int, cfg: dict) -> None:
        super().__init__(log_dir, flush_secs)

        # Build a stable SwanLab experiment name from the experiment and optional CLI run name.
        # Keep the timestamped local log directory independent from the SwanLab display name.
        experiment_name = cfg.get("experiment_name", "").replace("_", "-")
        configured_run_name = cfg.get("run_name", "")
        run_name = ""
        if experiment_name:
            run_name = f"{experiment_name}_{configured_run_name}" if configured_run_name else experiment_name
        run_name += "_" + os.path.split(log_dir)[-1]

        # Get SwanLab project
        try:
            project = cfg["swanlab_project"]
        except KeyError:
            raise KeyError("Please specify swanlab_project in the runner config, e.g. go2-rl-lab.") from None

        # Initialize SwanLab. The run mode and API host follow SwanLab's own
        # configuration, e.g. SWANLAB_MODE=offline or a self-hosted SWANLAB_API_HOST.
        swanlab.init(
            project=project,
            experiment_name=run_name,
            logdir=log_dir,
            config={"log_dir": log_dir},
        )

    def store_config(self, env_cfg: dict | object, train_cfg: dict) -> None:
        swanlab.config.update({"runner_cfg": train_cfg})
        swanlab.config.update({"policy_cfg": train_cfg["policy"]})
        swanlab.config.update({"alg_cfg": train_cfg["algorithm"]})
        try:
            swanlab.config.update({"env_cfg": env_cfg.to_dict()})
        except Exception:
            swanlab.config.update({"env_cfg": asdict(env_cfg)})

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
        swanlab.log({tag: _to_python_scalar(scalar_value)}, step=global_step)

    def add_custom_scalars(
        self,
        tag_scalar_dict: Mapping[str, Any],
        *,
        step_metric: str,
    ) -> None:
        """Log scalars with the step field as the SwanLab global step and TensorBoard step."""
        super().add_custom_scalars(tag_scalar_dict, step_metric=step_metric)
        global_step = int(tag_scalar_dict[step_metric])
        for tag, scalar_value in tag_scalar_dict.items():
            if tag != step_metric:
                swanlab.log({tag: _to_python_scalar(scalar_value)}, step=global_step)

    def stop(self) -> None:
        swanlab.finish()

    def save_file(self, path: str) -> None:
        try:
            swanlab.save(path, base_path=os.path.dirname(path), policy="now")
        except Exception as exc:
            print(f"[WARN] SwanLab failed to save {path}: {exc}")
