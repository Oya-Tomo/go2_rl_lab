"""MuJoCo physics, observation and velocity-joystick helpers for Go2 deployment."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import torch
import yaml

DEPLOY_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = DEPLOY_DIR.parents[1]
ASSET_DIR = DEPLOY_DIR / "assets"

# pygame axis indices for the Linux velocity-gamepad mapping.
LEFT_STICK_SIDEWAYS_AXIS = 0
LEFT_STICK_FORWARD_AXIS = 1
RIGHT_STICK_HORIZONTAL_AXIS = 3
JOYSTICK_AXIS_DEADZONE = 0.1


def get_gravity_orientation(quaternion: np.ndarray) -> np.ndarray:
    """Return the gravity direction expressed in the robot frame."""
    qw, qx, qy, qz = quaternion
    return np.array(
        [
            2.0 * (-qz * qx + qw * qy),
            -2.0 * (qz * qy + qw * qx),
            1.0 - 2.0 * (qw * qw + qz * qz),
        ],
        dtype=np.float32,
    )


def quat_rotate_inverse(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Rotate a vector by the inverse of a wxyz quaternion."""
    quaternion = np.asarray(quaternion, dtype=np.float32)
    vector = np.asarray(vector, dtype=np.float32)
    quaternion_w = quaternion[0]
    quaternion_xyz = quaternion[1:]
    return (
        vector * (2.0 * quaternion_w**2 - 1.0)
        - np.cross(quaternion_xyz, vector) * quaternion_w * 2.0
        + quaternion_xyz * np.dot(quaternion_xyz, vector) * 2.0
    )


def pd_control(
    target_position: np.ndarray,
    position: np.ndarray,
    stiffness: np.ndarray,
    velocity: np.ndarray,
    damping: np.ndarray,
) -> np.ndarray:
    """Compute the same explicit position-PD effort used during training."""
    return (target_position - position) * stiffness - velocity * damping


def clip_torque_speed(
    effort: np.ndarray,
    joint_velocity: np.ndarray,
    motor_cfg: dict,
) -> np.ndarray:
    """Apply the Go2HV direction-dependent torque-speed envelope."""
    if not motor_cfg["use_torque_speed_curve"]:
        return effort

    speed_x1 = float(motor_cfg["X1"])
    speed_x2 = float(motor_cfg["X2"])
    effort_y1 = float(motor_cfg["Y1"])
    effort_y2 = float(motor_cfg["Y2"])
    if speed_x2 <= speed_x1:
        raise ValueError("motor.X2 must be greater than motor.X1")

    same_direction = joint_velocity * effort > 0.0
    maximum_effort = np.where(same_direction, effort_y1, effort_y2)
    high_speed_effort = maximum_effort * (
        1.0 - (np.abs(joint_velocity) - speed_x1) / (speed_x2 - speed_x1)
    )
    maximum_effort = np.where(
        np.abs(joint_velocity) < speed_x1,
        maximum_effort,
        np.maximum(high_speed_effort, 0.0),
    )
    return np.clip(effort, -maximum_effort, maximum_effort)


def load_config(path: Path) -> dict:
    """Load one deployment YAML mapping."""
    with path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return config


def resolve_policy_path(raw_path: str, config_path: Path) -> Path:
    """Resolve a policy path relative to the repository or the config file."""
    expanded = raw_path.replace("{GO2_RL_LAB_ROOT_DIR}", str(REPOSITORY_ROOT))
    policy_path = Path(expanded).expanduser()
    if not policy_path.is_absolute():
        repository_relative = REPOSITORY_ROOT / policy_path
        config_relative = config_path.parent / policy_path
        policy_path = repository_relative if repository_relative.exists() else config_relative
    return policy_path.resolve()


def resolve_model_path(config: dict, terrain: str, explicit_xml: str | None) -> Path:
    """Resolve the MuJoCo scene XML from the terrain map or an explicit path."""
    if explicit_xml is not None:
        return Path(explicit_xml).expanduser().resolve()
    terrain_paths = config["terrains"]
    if terrain not in terrain_paths:
        available = ", ".join(sorted(terrain_paths))
        raise ValueError(f"Unknown terrain '{terrain}'. Available terrains: {available}")
    return (ASSET_DIR / terrain_paths[terrain]).resolve()


def initialize_joystick(enabled: bool):
    """Open the first pygame joystick, returning ``(pygame, joystick)`` handles."""
    if not enabled:
        return None, None
    try:
        import pygame
    except ImportError:
        print("[INFO] pygame is not installed; using the command from the config.")
        return None, None

    pygame.init()
    if pygame.joystick.get_count() == 0:
        print("[INFO] No joystick detected; using the command from the config.")
        return pygame, None
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print(f"[INFO] Joystick: {joystick.get_name()}")
    return pygame, joystick


def get_velocity_joystick_command(pygame, joystick, maximum_command: np.ndarray) -> np.ndarray:
    """Map the sticks to a body-frame ``[vx, vy, yaw_rate]`` velocity intent."""
    pygame.event.pump()
    axes = np.array(
        [
            joystick.get_axis(LEFT_STICK_FORWARD_AXIS),
            joystick.get_axis(LEFT_STICK_SIDEWAYS_AXIS),
            joystick.get_axis(RIGHT_STICK_HORIZONTAL_AXIS),
        ]
    )
    axes[np.abs(axes) < JOYSTICK_AXIS_DEADZONE] = 0.0
    return np.array(
        [-axes[0] * maximum_command[0], -axes[1] * maximum_command[1], -axes[2] * maximum_command[2]],
        dtype=np.float32,
    )


def configure_mujoco(model: mujoco.MjModel, config: dict) -> None:
    """Apply the deployment integrator, solver and timestep settings."""
    model.opt.timestep = float(config["simulation_dt"])
    integrators = {
        "euler": mujoco.mjtIntegrator.mjINT_EULER,
        "rk4": mujoco.mjtIntegrator.mjINT_RK4,
        "implicit": mujoco.mjtIntegrator.mjINT_IMPLICIT,
        "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
    }
    solvers = {
        "cg": mujoco.mjtSolver.mjSOL_CG,
        "newton": mujoco.mjtSolver.mjSOL_NEWTON,
    }
    model.opt.integrator = integrators[config["integrator"].lower()]
    model.opt.solver = solvers[config["solver"].lower()]
    model.opt.iterations = int(config["solver_iterations"])
    if hasattr(model.opt, "ls_iterations"):
        model.opt.ls_iterations = int(config["line_search_iterations"])


def get_joint_indices(model: mujoco.MjModel, joint_names: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the qpos, qvel and actuator indices for the named joints."""
    joint_ids = []
    actuator_ids = []
    for joint_name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"Joint '{joint_name}' is missing from the MuJoCo model")
        joint_ids.append(joint_id)

        matching_actuators = np.flatnonzero(model.actuator_trnid[:, 0] == joint_id)
        if matching_actuators.size != 1:
            raise ValueError(f"Expected exactly one actuator for joint '{joint_name}'")
        actuator_ids.append(int(matching_actuators[0]))

    joint_ids_array = np.asarray(joint_ids, dtype=np.int32)
    return (
        model.jnt_qposadr[joint_ids_array],
        model.jnt_dofadr[joint_ids_array],
        np.asarray(actuator_ids, dtype=np.int32),
    )


def extract_policy_action(policy_output) -> torch.Tensor:
    """Return the action tensor from a TorchScript policy output."""
    if isinstance(policy_output, tuple):
        policy_output = policy_output[0]
    if not isinstance(policy_output, torch.Tensor):
        raise TypeError(f"Expected policy to return a Tensor, got {type(policy_output).__name__}")
    return policy_output


__all__ = [
    "ASSET_DIR",
    "DEPLOY_DIR",
    "JOYSTICK_AXIS_DEADZONE",
    "LEFT_STICK_FORWARD_AXIS",
    "LEFT_STICK_SIDEWAYS_AXIS",
    "REPOSITORY_ROOT",
    "RIGHT_STICK_HORIZONTAL_AXIS",
    "clip_torque_speed",
    "configure_mujoco",
    "extract_policy_action",
    "get_gravity_orientation",
    "get_joint_indices",
    "get_velocity_joystick_command",
    "initialize_joystick",
    "load_config",
    "pd_control",
    "quat_rotate_inverse",
    "resolve_model_path",
    "resolve_policy_path",
]
