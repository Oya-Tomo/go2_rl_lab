"""Flat-ground velocity locomotion using the mixed-task PPO baseline."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.utils.configclass import configclass

from go2_rl_lab.tasks.mixed import mdp
from go2_rl_lab.tasks.mixed.env_cfg import (
    CommandsCfg as MixedCommandsCfg,
    RewardsCfg as MixedRewardsCfg,
    RobotEnvCfg as MixedRobotEnvCfg,
)
from go2_rl_lab.utils.terrains import SolidColorTerrainImporter


@configclass
class FlatVelocityCommandCfg(mdp.RLGymVelCommandCfg):
    """Fixed-range Go2 Gym velocity command used by the flat baseline."""

    # Keep the mixed-task sampler and limit-command behavior, but remove all
    # command-range and zero-command schedules for a controlled ablation.
    zero_command_curriculum = None
    command_range_curriculum: list[dict] = []
    terrain_max_command_ranges: dict[str, dict[str, tuple[float, float]]] = {}
    ranges = mdp.RLGymVelCommandCfg.Ranges(
        lin_vel_x=(-1.0, 1.0),
        lin_vel_y=(-1.0, 1.0),
        ang_vel_yaw=(-2.0, 2.0),
    )


@configclass
class FlatVelocityCommandsCfg(MixedCommandsCfg):
    """Command manager configuration for the velocity ablation arm."""

    base_velocity: FlatVelocityCommandCfg = FlatVelocityCommandCfg()


@configclass
class FlatVelocityRobotEnvCfg(MixedRobotEnvCfg):
    """Flat-plane PPO environment with mixed-task velocity commands."""

    commands: FlatVelocityCommandsCfg = FlatVelocityCommandsCfg()
    # Keep the complete mixed reward set and weights byte-for-byte equivalent.
    rewards: MixedRewardsCfg = MixedRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16384
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.class_type = SolidColorTerrainImporter
        self.scene.terrain.visual_material = sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.28, 0.47, 0.67), roughness=1.0
        )
        self.scene.terrain.terrain_generator = None
        self.scene.terrain.max_init_terrain_level = None
        self.curriculum.terrain_levels = None

        # A plane has no generated terrain mesh, so scan the ground prim itself
        # instead of the mesh-backend path inherited from the mixed config.
        self.scene.height_scanner.mesh_prim_paths = [self.scene.terrain.prim_path]


@configclass
class FlatVelocityRobotPlayEnvCfg(FlatVelocityRobotEnvCfg):
    """Deterministic playback configuration for the velocity arm."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.commands.base_velocity = mdp.UniformVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(30.0, 30.0),
            rel_standing_envs=0.0,
            heading_command=False,
            ranges=mdp.UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(1.0, 1.0),
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(0.0, 0.0),
            ),
            debug_vis=True,
        )
