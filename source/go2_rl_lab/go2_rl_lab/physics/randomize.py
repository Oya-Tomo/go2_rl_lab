"""Initialization-time randomization helpers for Newton physics models."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, replace

import numpy as np
import warp as wp
from newton import ModelBuilder

from isaaclab.utils.configclass import configclass
from isaaclab_newton.physics.mjwarp_manager_cfg import MJWarpSolverCfg

logger = logging.getLogger(__name__)

_BUILDER_MARKER = "_go2_rl_lab_com_bucket_randomization"


@configclass
class BodyComBucketRandomizationCfg:
    """Fixed per-world center-of-mass offsets applied before model finalization."""

    body_name_pattern: str = r"(?:^|/)Robot/base$"
    """Regular expression matching exactly one body in every Newton world."""

    x_offset_range: tuple[float, float] = (0.0, 0.1)
    """Range of x offsets added to the body's local CoM, in meters."""

    y_offset_range: tuple[float, float] = (0.0, 0.0)
    """Range of y offsets added to the body's local CoM, in meters."""

    z_offset_range: tuple[float, float] = (0.0, 0.0)
    """Range of z offsets added to the body's local CoM, in meters."""

    num_buckets: int = 10
    """Number of continuous intervals per axis; the joint bucket count is its cube."""

    seed: int = 42
    """Seed used to shuffle the balanced bucket assignment across worlds."""


@configclass
class RandomizedMJWarpSolverCfg(MJWarpSolverCfg):
    """MuJoCo Warp configuration with project model-initialization options."""

    body_com_buckets: BodyComBucketRandomizationCfg | None = None
    """Optional fixed per-world CoM buckets. ``None`` disables the feature."""

    sensor_bvh_shape_flags: int | None = None
    """Optional Newton shape-flag mask for the shared sensor BVH.

    ``None`` keeps the default mask. Collision-only height scans exclude
    visual-only terrain outlines without changing their rendering or physics flags.
    """


@dataclass(frozen=True)
class BodyComBucketReport:
    """Summary of an applied bucket assignment."""

    num_worlds: int
    num_intervals_per_axis: int
    num_joint_buckets: int
    axis_interval_edges: tuple[tuple[float, ...], ...]
    joint_bucket_counts: tuple[int, ...]
    sampled_offset_ranges: tuple[tuple[float, float], ...]
    absolute_com_ranges: tuple[tuple[float, float], ...]
    already_applied: bool = False

    @property
    def absolute_x_range(self) -> tuple[float, float]:
        return self.absolute_com_ranges[0]

    @property
    def absolute_y_range(self) -> tuple[float, float]:
        return self.absolute_com_ranges[1]

    @property
    def absolute_z_range(self) -> tuple[float, float]:
        return self.absolute_com_ranges[2]


def _validate_cfg(cfg: BodyComBucketRandomizationCfg) -> None:
    axis_ranges = {
        "x": cfg.x_offset_range,
        "y": cfg.y_offset_range,
        "z": cfg.z_offset_range,
    }
    for axis, offset_range in axis_ranges.items():
        offset_min, offset_max = offset_range
        if not np.isfinite((offset_min, offset_max)).all():
            raise ValueError(f"CoM {axis} offset range must be finite, got {offset_range}.")
        if offset_min > offset_max:
            raise ValueError(f"CoM {axis} offset range must be ordered, got {offset_range}.")
    if cfg.num_buckets < 1:
        raise ValueError(f"CoM bucket count must be positive, got {cfg.num_buckets}.")


def _balanced_bucket_ids(
    num_worlds: int,
    num_buckets: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return a randomized assignment whose bucket counts differ by at most one."""
    bucket_ids = np.arange(num_worlds, dtype=np.int64) % num_buckets
    bucket_ids = rng.permutation(num_buckets)[bucket_ids]
    rng.shuffle(bucket_ids)
    return bucket_ids


def _find_body_per_world(builder: ModelBuilder, pattern: str) -> list[int]:
    matcher = re.compile(pattern)
    matches: list[list[int]] = [[] for _ in range(builder.world_count)]

    for body_id, (body_world, body_label) in enumerate(zip(builder.body_world, builder.body_label)):
        world_id = int(body_world)
        if 0 <= world_id < builder.world_count and body_label is not None and matcher.search(body_label):
            matches[world_id].append(body_id)

    invalid = {world_id: body_ids for world_id, body_ids in enumerate(matches) if len(body_ids) != 1}
    if invalid:
        details = ", ".join(f"world {world_id}: {len(body_ids)} matches" for world_id, body_ids in invalid.items())
        raise ValueError(
            f"Body pattern {pattern!r} must match exactly one body in every Newton world; {details}."
        )
    return [body_ids[0] for body_ids in matches]


def apply_body_com_buckets(
    builder: ModelBuilder,
    cfg: BodyComBucketRandomizationCfg,
) -> BodyComBucketReport:
    """Sample fixed local-CoM offsets from balanced joint 3-D buckets.

    This function must run after world replication and before
    :meth:`newton.ModelBuilder.finalize`. Each world retains its assigned CoM
    for the lifetime of the finalized model, including all episode resets. Each
    axis is split into continuous intervals, and the Cartesian product of the
    three axes is balanced across worlds before sampling within each interval.
    """
    _validate_cfg(cfg)
    if builder.world_count < 1:
        raise ValueError("Cannot apply per-world CoM buckets to a builder without Newton worlds.")

    signature = (
        cfg.body_name_pattern,
        tuple(float(value) for value in cfg.x_offset_range),
        tuple(float(value) for value in cfg.y_offset_range),
        tuple(float(value) for value in cfg.z_offset_range),
        int(cfg.num_buckets),
        int(cfg.seed),
    )
    previous = getattr(builder, _BUILDER_MARKER, None)
    if previous is not None:
        if previous["signature"] != signature:
            raise RuntimeError("CoM buckets were already applied to this builder with a different configuration.")
        return replace(previous["report"], already_applied=True)

    body_ids = _find_body_per_world(builder, cfg.body_name_pattern)
    offset_ranges = (
        cfg.x_offset_range,
        cfg.y_offset_range,
        cfg.z_offset_range,
    )
    interval_edges_by_axis = tuple(
        np.linspace(offset_min, offset_max, cfg.num_buckets + 1, dtype=np.float64)
        for offset_min, offset_max in offset_ranges
    )
    assignment_seed, sampling_seed = np.random.SeedSequence(cfg.seed).spawn(2)
    assignment_rng = np.random.default_rng(assignment_seed)
    sampling_rng = np.random.default_rng(sampling_seed)

    num_joint_buckets = cfg.num_buckets**3
    joint_bucket_ids = _balanced_bucket_ids(
        builder.world_count,
        num_joint_buckets,
        assignment_rng,
    )
    joint_bucket_counts = np.bincount(joint_bucket_ids, minlength=num_joint_buckets)
    interval_ids_by_axis = np.stack(
        np.unravel_index(
            joint_bucket_ids,
            (cfg.num_buckets, cfg.num_buckets, cfg.num_buckets),
        ),
        axis=1,
    )

    unit_samples = sampling_rng.random((builder.world_count, 3))
    sampled_offsets = np.empty((builder.world_count, 3), dtype=np.float64)
    for axis, interval_edges in enumerate(interval_edges_by_axis):
        interval_ids = interval_ids_by_axis[:, axis]
        lower_bounds = interval_edges[interval_ids]
        upper_bounds = interval_edges[interval_ids + 1]
        sampled_offsets[:, axis] = lower_bounds + unit_samples[:, axis] * (upper_bounds - lower_bounds)

    absolute_values_by_axis: tuple[list[float], list[float], list[float]] = ([], [], [])
    for world_id, body_id in enumerate(body_ids):
        original_com = builder.body_com[body_id]
        offsets = sampled_offsets[world_id]
        randomized_com = wp.vec3(
            float(original_com[0]) + offsets[0],
            float(original_com[1]) + offsets[1],
            float(original_com[2]) + offsets[2],
        )
        builder.body_com[body_id] = randomized_com
        for axis in range(3):
            absolute_values_by_axis[axis].append(float(randomized_com[axis]))

    report = BodyComBucketReport(
        num_worlds=builder.world_count,
        num_intervals_per_axis=cfg.num_buckets,
        num_joint_buckets=num_joint_buckets,
        axis_interval_edges=tuple(
            tuple(float(value) for value in interval_edges) for interval_edges in interval_edges_by_axis
        ),
        joint_bucket_counts=tuple(int(value) for value in joint_bucket_counts),
        sampled_offset_ranges=tuple(
            (float(sampled_offsets[:, axis].min()), float(sampled_offsets[:, axis].max())) for axis in range(3)
        ),
        absolute_com_ranges=tuple(
            (min(axis_values), max(axis_values)) for axis_values in absolute_values_by_axis
        ),
    )
    setattr(builder, _BUILDER_MARKER, {"signature": signature, "report": report})

    occupied_joint_buckets = int(np.count_nonzero(joint_bucket_counts))
    logger.info(
        "Applied fixed body CoM buckets before Newton model finalization: "
        "worlds=%d, pattern=%r, configured_offsets="
        "{x=[%.4f, %.4f], y=[%.4f, %.4f], z=[%.4f, %.4f]} m, buckets=%d, "
        "joint_buckets={total=%d, occupied=%d, counts=[%d, %d]}, "
        "sampled_offsets={x=[%.4f, %.4f], y=[%.4f, %.4f], z=[%.4f, %.4f]} m, "
        "absolute={x=[%.4f, %.4f], y=[%.4f, %.4f], z=[%.4f, %.4f]} m",
        report.num_worlds,
        cfg.body_name_pattern,
        cfg.x_offset_range[0],
        cfg.x_offset_range[1],
        cfg.y_offset_range[0],
        cfg.y_offset_range[1],
        cfg.z_offset_range[0],
        cfg.z_offset_range[1],
        report.num_intervals_per_axis,
        report.num_joint_buckets,
        occupied_joint_buckets,
        min(report.joint_bucket_counts),
        max(report.joint_bucket_counts),
        report.sampled_offset_ranges[0][0],
        report.sampled_offset_ranges[0][1],
        report.sampled_offset_ranges[1][0],
        report.sampled_offset_ranges[1][1],
        report.sampled_offset_ranges[2][0],
        report.sampled_offset_ranges[2][1],
        report.absolute_x_range[0],
        report.absolute_x_range[1],
        report.absolute_y_range[0],
        report.absolute_y_range[1],
        report.absolute_z_range[0],
        report.absolute_z_range[1],
    )
    return report


__all__ = [
    "BodyComBucketRandomizationCfg",
    "BodyComBucketReport",
    "RandomizedMJWarpSolverCfg",
    "apply_body_com_buckets",
]
