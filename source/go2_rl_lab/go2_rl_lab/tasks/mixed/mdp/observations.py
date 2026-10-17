"""Privileged proprioceptive observations for velocity locomotion."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.sensors import ContactSensor
    from isaaclab.envs import ManagerBasedRLEnv


def joint_acc(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return joint accelerations for the selected joints."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_acc.torch[:, asset_cfg.joint_ids]


def foot_contact_force_norm(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return flattened current and historical foot-contact force norms."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_forces = contact_sensor.data.net_forces_w_history.torch[:, :, sensor_cfg.body_ids]
    contact_force_norm = torch.norm(contact_forces, dim=-1)
    max_contact_force_norm = torch.max(contact_force_norm, dim=1).values
    contact_force_norm = torch.cat(
        [max_contact_force_norm.unsqueeze(1), contact_force_norm],
        dim=1,
    )
    return contact_force_norm.flatten(start_dim=-2)


__all__ = ["foot_contact_force_norm", "joint_acc"]
