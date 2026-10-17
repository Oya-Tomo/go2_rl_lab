# Copyright (c) 2026 Tianyang Wu
#
# SPDX-License-Identifier: MIT

"""Asynchronous RoboGauge checkpoint evaluation and metric logging."""

from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable, Protocol

import yaml
from isaaclab.utils.configclass import configclass


@configclass
class RoboGaugeEvaluatorCfg:
    """Configuration for asynchronous RoboGauge policy evaluation."""

    task_name: str = "go2_lab_1.3.1"
    """RoboGauge robot task prefix used by the stress evaluator."""

    enabled: bool = False
    """Whether asynchronous RoboGauge evaluation is enabled."""

    port: int = 9973
    """Local RoboGauge server port."""

    submit_interval: int = 500
    """Training iterations between policy submissions."""

    poll_interval: float = 60.0
    """Seconds between result polls while waiting for the final evaluation."""


class _ScalarWriter(Protocol):
    def add_custom_scalars(
        self,
        tag_scalar_dict: Mapping[str, Any],
        *,
        step_metric: str,
    ) -> Any: ...


class _RoboGaugeClient(Protocol):
    response_data: dict[str, Any]

    def wait_until_available(self) -> None: ...

    def submit_task(
        self,
        model_path: str,
        step: int,
        task_name: str,
        experiment_name: str,
    ) -> str | None: ...

    def monitor_tasks(self) -> None: ...


class RoboGaugeEvaluator:
    """Submit exported policies to RoboGauge and persist returned scores."""

    def __init__(
        self,
        task_name: str,
        config: dict[str, Any],
        log_dir: str | None,
        writer: _ScalarWriter | None,
        export_policy: Callable[[str, str], None],
        experiment_name: str,
        *,
        disabled: bool = False,
        client: _RoboGaugeClient | None = None,
    ) -> None:
        self.task_name = task_name
        self.config = config
        self.log_dir = Path(log_dir) if log_dir is not None else None
        self.writer = writer
        self.export_policy = export_policy
        self.experiment_name = experiment_name
        self.submit_interval = int(self.config["submit_interval"])
        self.poll_interval = float(self.config["poll_interval"])
        self.logged_steps: set[int] = set()
        self.client: _RoboGaugeClient | None = None
        self.enabled = False

        if not self.config["enabled"]:
            return
        if disabled or self.log_dir is None:
            print("[INFO] RoboGauge disabled because logging is unavailable on this process.")
            return
        if self.submit_interval < 1:
            raise ValueError(f"RoboGauge submit_interval must be at least 1, got {self.submit_interval}.")
        if self.poll_interval < 0:
            raise ValueError(f"RoboGauge poll_interval must be non-negative, got {self.poll_interval}.")

        try:
            if client is None:
                from robogauge.scripts.client import RoboGaugeClient

                client = RoboGaugeClient(f"http://127.0.0.1:{int(self.config['port'])}")
            client.wait_until_available()
        except Exception as exc:
            message = f"RoboGauge initialization failed: {exc}"
            print(f"[ERROR] {message}")
            raise RuntimeError(message) from exc

        self.client = client
        self.enabled = True

    def update(self, step: int, final: bool = False) -> None:
        """Submit a scheduled policy, collect results, and update scalar logs."""
        if not self.enabled or self.client is None:
            return

        submitted = False
        if step % self.submit_interval == 0 or final:
            submitted = self._submit(step)

        received_steps = self._collect(step)
        if received_steps is None:
            return

        while final and submitted and step not in received_steps:
            print(f"[INFO] Waiting {self.poll_interval:g}s for final RoboGauge result at step {step}...")
            time.sleep(self.poll_interval)
            new_steps = self._collect(step)
            if new_steps is None:
                break
            received_steps.update(new_steps)

    def _submit(self, step: int) -> bool:
        assert self.client is not None
        assert self.log_dir is not None

        try:
            jit_dir = self.log_dir / "jit_models"
            filename = f"policy_jit_{step}.pt"
            self.export_policy(str(jit_dir), filename)
            task_id = self.client.submit_task(
                model_path=str(jit_dir / filename),
                step=step,
                task_name=self.task_name,
                experiment_name=self.experiment_name,
            )
            return task_id is not None
        except Exception as exc:
            print(f"[WARN] RoboGauge submission failed at step {step}: {exc}")
            return False

    def _collect(self, fallback_step: int) -> set[int] | None:
        assert self.client is not None
        assert self.log_dir is not None

        try:
            self.client.monitor_tasks()
        except Exception as exc:
            print(f"[WARN] RoboGauge result collection failed at step {fallback_step}: {exc}")
            return None

        received_steps: set[int] = set()
        for task_id, response in self.client.response_data.items():
            if not isinstance(response, dict):
                print(f"[WARN] RoboGauge returned an invalid response for task {task_id}: {response}")
                continue

            results = response.get("results")
            if not isinstance(results, dict):
                print(f"[WARN] RoboGauge returned empty results for task {task_id}.")
                continue
            scores = results.get("scores")
            if not isinstance(scores, dict):
                print(f"[WARN] RoboGauge results for task {task_id} do not contain scores.")
                continue

            try:
                result_step = int(response.get("step", fallback_step))
            except (TypeError, ValueError):
                print(f"[WARN] RoboGauge returned an invalid step for task {task_id}: {response.get('step')}")
                continue

            received_steps.add(result_step)
            if result_step in self.logged_steps:
                continue

            if self.writer is not None:
                step_metric = "RoboGauge/iteration"
                self.writer.add_custom_scalars(
                    {
                        step_metric: result_step,
                        **{f"RoboGauge/{key}": value for key, value in scores.items()},
                    },
                    step_metric=step_metric,
                )

            results_dir = self.log_dir / "robogauge_results"
            results_dir.mkdir(parents=True, exist_ok=True)
            with (results_dir / f"results_{result_step}.yaml").open("w", encoding="utf-8") as result_file:
                yaml.safe_dump(_to_builtin(results), result_file, allow_unicode=True, sort_keys=False)
            self.logged_steps.add(result_step)

        return received_steps


def _to_builtin(value: Any) -> Any:
    """Convert NumPy/Torch-like values into YAML-safe Python containers and scalars."""
    if isinstance(value, dict):
        return {key: _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    if hasattr(value, "tolist"):
        return _to_builtin(value.tolist())
    if hasattr(value, "item"):
        return _to_builtin(value.item())
    return value


__all__ = ["RoboGaugeEvaluator", "RoboGaugeEvaluatorCfg"]
