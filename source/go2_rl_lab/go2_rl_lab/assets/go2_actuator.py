"""Go2 actuator model with motor torque-speed and friction limits."""

from __future__ import annotations

import torch

from isaaclab.actuators import DelayedPDActuator, DelayedPDActuatorCfg
from isaaclab.actuators.actuator_base import resolve_joint_parameter
from isaaclab.utils.configclass import configclass
from isaaclab.utils.types import ArticulationActions


class Go2Actuator(DelayedPDActuator):
    """Apply the Go2 motor torque-speed curve and joint friction."""

    cfg: Go2ActuatorCfg

    def __init__(self, cfg: Go2ActuatorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)

        self._joint_vel = torch.zeros_like(self.computed_effort)
        self._effort_y1 = self._resolve_motor_parameter(cfg.Y1, 1e9)
        self._effort_y2 = self._resolve_motor_parameter(cfg.Y2, cfg.Y1)
        self._velocity_x1 = self._resolve_motor_parameter(cfg.X1, 1e9)
        self._velocity_x2 = self._resolve_motor_parameter(cfg.X2, 1e9)
        self._friction_static = self._resolve_motor_parameter(cfg.Fs, 0.0)
        self._friction_dynamic = self._resolve_motor_parameter(cfg.Fd, 0.0)
        self._activation_vel = self._resolve_motor_parameter(cfg.Va, 0.01)

    def _resolve_motor_parameter(self, value, default) -> torch.Tensor:
        """Resolve motor parameters in actuator joint order."""
        return resolve_joint_parameter(value, default, self.joint_names, self._num_envs, self._device)

    def compute(
        self,
        control_action: ArticulationActions,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> ArticulationActions:
        """Compute and limit joint effort for one simulation step."""
        self._joint_vel[:] = joint_vel
        control_action = super().compute(control_action, joint_pos, joint_vel)

        self.applied_effort -= (
            self._friction_static * torch.tanh(joint_vel / self._activation_vel)
            + self._friction_dynamic * joint_vel
        )

        control_action.joint_positions = None
        control_action.joint_velocities = None
        control_action.joint_efforts = self.applied_effort
        return control_action

    def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
        same_direction = self._joint_vel * effort > 0
        max_effort = torch.where(same_direction, self._effort_y1, self._effort_y2)
        max_effort = torch.where(
            self._joint_vel.abs() < self._velocity_x1,
            max_effort,
            self._compute_effort_limit(max_effort),
        )
        return torch.clip(effort, -max_effort, max_effort)

    def _compute_effort_limit(self, max_effort: torch.Tensor) -> torch.Tensor:
        slope = -max_effort / (self._velocity_x2 - self._velocity_x1)
        limit = slope * (self._joint_vel.abs() - self._velocity_x1) + max_effort
        return limit.clip(min=0.0)


@configclass
class Go2ActuatorCfg(DelayedPDActuatorCfg):
    """Configuration for the Unitree Go2 high-voltage motor."""

    class_type: type = Go2Actuator

    joint_effort_limit: float = 1.0e9
    """Preserve beta2 solver limits; the custom motor model clips effort."""

    X1: float = 13.5
    """Maximum speed at full torque, in rad/s."""

    X2: float = 30.0
    """No-load speed, in rad/s."""

    Y1: float = 20.2
    """Peak torque when torque and speed have the same direction, in N m."""

    Y2: float = 23.4
    """Peak torque when torque and speed have opposite directions, in N m."""

    Fs: float = 0.0
    """Static friction coefficient."""

    Fd: float = 0.0
    """Dynamic friction coefficient."""

    Va: float = 0.01
    """Velocity at which friction is fully activated."""
