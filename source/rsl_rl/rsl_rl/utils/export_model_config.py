# Copyright (c) 2026 Tianyang Wu
#
# SPDX-License-Identifier: MIT

"""Export deployment metadata and training parameters alongside policies."""

from __future__ import annotations

import hashlib
import os
import re
from datetime import datetime
from typing import Any

import yaml

_RUN_TIMESTAMP_PATTERN = re.compile(r"^(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})")


class _TrainingConfigLoader(yaml.SafeLoader):
    """Safely normalize Python-specific tags emitted by Isaac Lab's YAML dumper."""


def _construct_python_tuple(loader: _TrainingConfigLoader, node: yaml.Node) -> list[Any]:
    return loader.construct_sequence(node, deep=True)


def _construct_python_slice(loader: _TrainingConfigLoader, node: yaml.Node) -> dict[str, Any]:
    values = loader.construct_sequence(node, deep=True)
    values.extend([None] * (3 - len(values)))
    return {"start": values[0], "stop": values[1], "step": values[2]}


_TrainingConfigLoader.add_constructor("tag:yaml.org,2002:python/tuple", _construct_python_tuple)
_TrainingConfigLoader.add_constructor("tag:yaml.org,2002:python/object/apply:builtins.slice", _construct_python_slice)


def export_model_config(
    path: str,
    checkpoint_path: str,
    checkpoint_iteration: int,
    task: str,
    experiment_name: str,
    *,
    num_single_obs: int,
    num_actor_obs: int,
    num_actions: int,
    control_dt: float,
    filename: str = "config.yaml",
) -> str:
    """Write deployment metadata and the original training configs to YAML.

    The ``params/agent.yaml`` and ``params/env.yaml`` snapshots next to the
    checkpoint are embedded as portable YAML so playback overrides cannot
    silently replace the parameters that were actually used for training.

    Returns:
        Path to the generated YAML file.
    """
    os.makedirs(path, exist_ok=True)
    checkpoint_path = os.path.abspath(checkpoint_path)
    run_dir = os.path.dirname(checkpoint_path)
    run_directory = os.path.basename(run_dir)
    params_dir = os.path.join(run_dir, "params")
    agent_path = os.path.join(params_dir, "agent.yaml")
    env_path = os.path.join(params_dir, "env.yaml")

    agent_params = _load_yaml_metadata(agent_path)
    env_params = _load_yaml_metadata(env_path)

    saved_experiment_name = _nested_value(agent_params, "experiment_name")
    configured_run_name = _nested_value(agent_params, "run_name")
    num_envs = _as_int(_nested_value(env_params, "scene", "num_envs"))
    num_steps_per_env = _as_int(_nested_value(agent_params, "num_steps_per_env"))
    max_iterations = _as_int(_nested_value(agent_params, "max_iterations"))
    seed = _as_int(_nested_value(agent_params, "seed"))
    physics_class = _nested_value(env_params, "sim", "physics", "class_type")

    completed_learning_iterations = max(int(checkpoint_iteration) + 1, 0)
    environment_steps_per_iteration = None
    estimated_environment_steps = None
    if num_envs is not None and num_steps_per_env is not None:
        environment_steps_per_iteration = num_envs * num_steps_per_env
        estimated_environment_steps = completed_learning_iterations * environment_steps_per_iteration

    history_length = None
    if num_single_obs > 0 and num_actor_obs % num_single_obs == 0:
        history_length = num_actor_obs // num_single_obs

    deployment_model = {
        "onnx_file": "policy.onnx",
        "torchscript_file": "policy.pt",
        "single_observation_size": int(num_single_obs),
        "actor_observation_size": int(num_actor_obs),
        "observation_history_length": history_length,
        "action_size": int(num_actions),
        "control_dt": float(control_dt),
        "command_kind": (
            _nested_value(env_params, "commands", "base_velocity", "cmd_kind") or "command/body/velocity"
        ),
        "command_elements": (
            _nested_value(env_params, "commands", "base_velocity", "element_names")
            or ["lin_vel_x", "lin_vel_y", "ang_vel_z"]
        ),
    }
    command_scale = _nested_value(env_params, "observations", "single_obs", "velocity_commands", "scale")
    if command_scale is None:
        command_scale = _nested_value(env_params, "observations", "policy", "velocity_commands", "scale")
    if command_scale is not None:
        deployment_model["command_observation_scale"] = command_scale

    metadata = {
        "metadata": {
            "format_version": 1,
            "exported_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "training_started_at": _training_start_time(run_directory, params_dir),
            "task": task,
            "experiment_name": saved_experiment_name or experiment_name,
            "run_directory": run_directory,
            "configured_run_name": configured_run_name or None,
        },
        "checkpoint": {
            "file": os.path.relpath(checkpoint_path, path),
            "sha256": _sha256(checkpoint_path),
            "iteration": int(checkpoint_iteration),
        },
        "training_progress": {
            "checkpoint_iteration": int(checkpoint_iteration),
            "completed_learning_iterations": completed_learning_iterations,
            "num_environments": num_envs,
            "rollout_steps_per_environment": num_steps_per_env,
            "environment_steps_per_iteration": environment_steps_per_iteration,
            "estimated_environment_steps": estimated_environment_steps,
            "configured_max_iterations": max_iterations,
        },
        "deployment_model": deployment_model,
        "training_summary": {
            "seed": seed,
            "physics_backend": _physics_backend(physics_class),
            "physics_manager": physics_class,
        },
        "training_parameters": {
            "agent": agent_params or None,
            "environment": env_params or None,
        },
    }

    config_path = os.path.join(path, filename)
    temporary_path = f"{config_path}.tmp"
    with open(temporary_path, "w", encoding="utf-8") as config_file:
        yaml.safe_dump(metadata, config_file, sort_keys=False, allow_unicode=True)
    os.replace(temporary_path, config_path)
    return config_path


def _load_yaml_metadata(path: str) -> dict[str, Any]:
    """Load an Isaac Lab config without constructing arbitrary Python objects."""
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as config_file:
        data = yaml.load(config_file, Loader=_TrainingConfigLoader)
    return data if isinstance(data, dict) else {}


def _nested_value(data: dict[str, Any], *keys: str) -> Any:
    value: Any = data
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _as_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _training_start_time(run_directory: str, params_dir: str) -> str | None:
    match = _RUN_TIMESTAMP_PATTERN.match(run_directory)
    if match is not None:
        start_time = datetime.strptime(match.group(1), "%Y-%m-%d_%H-%M-%S")
        return start_time.astimezone().isoformat(timespec="seconds")

    agent_path = os.path.join(params_dir, "agent.yaml")
    if os.path.isfile(agent_path):
        return datetime.fromtimestamp(os.path.getmtime(agent_path)).astimezone().isoformat(timespec="seconds")
    return None


def _physics_backend(class_type: Any) -> str | None:
    if not isinstance(class_type, str):
        return None
    class_type_lower = class_type.lower()
    if "newton" in class_type_lower:
        return "newton_mjwarp"
    if "physx" in class_type_lower:
        return "isaacsim_physx"
    return class_type


def _sha256(path: str) -> str | None:
    if not os.path.isfile(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
