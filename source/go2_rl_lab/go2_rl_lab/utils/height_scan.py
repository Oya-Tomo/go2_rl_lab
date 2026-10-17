"""Finite ray-caster terrain-height queries for velocity locomotion."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.sensors import RayCaster
    from isaaclab.envs import ManagerBasedRLEnv


def _get_ray_hit_heights(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> tuple[RayCaster, torch.Tensor]:
    """Return the scanner and finite world-frame ray-hit heights."""
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    ray_hit_heights = sensor.data.ray_hits_w.torch[..., 2]
    finite_hits = torch.isfinite(ray_hit_heights)
    fallback_height = torch.where(
        finite_hits,
        ray_hit_heights,
        torch.full_like(ray_hit_heights, torch.inf),
    ).amin(dim=1)
    fallback_height = torch.where(
        torch.isfinite(fallback_height),
        fallback_height,
        sensor.data.pos_w.torch[:, 2] - 1.0,
    )
    ray_hit_heights = torch.where(finite_hits, ray_hit_heights, fallback_height.unsqueeze(1))
    return sensor, ray_hit_heights


def _get_crop_mask(
    sensor: RayCaster,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
) -> torch.Tensor:
    """Return a cached mask that crops the requested local scanner region."""
    cache_key = (tuple(x_range), tuple(y_range))
    crop_masks = getattr(sensor, "_go2_height_crop_masks", None)
    if crop_masks is None:
        crop_masks = {}
        sensor._go2_height_crop_masks = crop_masks
    if cache_key in crop_masks:
        return crop_masks[cache_key]

    local_xy = sensor.ray_starts.torch[0, :, :2]
    resolution = float(getattr(sensor.cfg.pattern_cfg, "resolution", 0.0))
    tolerance = max(1.0e-6, resolution * 1.0e-5)
    crop_mask = (
        (local_xy[:, 0] >= x_range[0] - tolerance)
        & (local_xy[:, 0] <= x_range[1] + tolerance)
        & (local_xy[:, 1] >= y_range[0] - tolerance)
        & (local_xy[:, 1] <= y_range[1] + tolerance)
    )
    if not torch.any(crop_mask):
        raise ValueError(
            "The requested height-scan crop contains no rays: "
            f"x_range={x_range}, y_range={y_range}."
        )

    crop_masks[cache_key] = crop_mask
    return crop_mask


def get_cropped_ray_hit_heights(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
) -> torch.Tensor:
    """Return absolute hit heights cropped from the full RayCaster result."""
    sensor, ray_hit_heights = _get_ray_hit_heights(env, sensor_cfg)
    return ray_hit_heights[:, _get_crop_mask(sensor, x_range, y_range)]


def ray_caster_height_scan(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    offset: float = 0.5,
) -> torch.Tensor:
    """Return finite sensor-relative height observations from the RayCaster."""
    sensor, ray_hit_heights = _get_ray_hit_heights(env, sensor_cfg)
    return sensor.data.pos_w.torch[:, 2].unsqueeze(1) - ray_hit_heights - offset


__all__ = ["get_cropped_ray_hit_heights", "ray_caster_height_scan"]
