"""Go2 locomotion environment configuration for Isaac Lab physics backends."""

import copy
import os

from newton import ShapeFlags

import isaaclab.sim as sim_utils
from isaaclab_newton.physics import (
    NewtonCfg,
    NewtonCollisionPipelineCfg,
    NewtonShapeCfg,
)
from isaaclab_newton.sensors import ContactSensorCfg as NewtonContactSensorCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.sensors import ContactSensorCfg as PhysXContactSensorCfg

from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from isaaclab_tasks.utils import PresetCfg, preset

from go2_rl_lab.assets.go2 import GO2_CFG as ROBOT_CFG
from go2_rl_lab.physics import (
    BodyComBucketRandomizationCfg,
    RandomizedMJWarpSolverCfg,
)
from go2_rl_lab.tasks.mixed import mdp
from go2_rl_lab.tasks.mixed.terrain import MIXED_TERRAIN_CFG
from go2_rl_lab.tasks.mixed.terrain_mesh import (
    MIXED_MESH_TERRAIN_CFG,
    MixedPlayTerrainImporter,
    MixedTerrainImporter,
)
from isaaclab.envs import ManagerBasedRLEnvCfg
from go2_rl_lab.utils.terrains import HighToLowTerrainImporter, configure_play_sub_terrains

JOINT_NAMES = [
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
]
BASE_LINK_NAME = "base"
FOOT_LINK_NAME = ".*_foot"
BASE_HEIGHT_TARGET = 0.38
BASE_HEIGHT_SCAN_X_RANGE = (-0.2, 0.2)
BASE_HEIGHT_SCAN_Y_RANGE = (-0.15, 0.15)

@configclass
class PhysicsCfg(PresetCfg):
    """Physics backends supported by the Go2 mixed-terrain task."""

    isaacsim_physx = PhysxCfg(
        gpu_max_rigid_patch_count=2 * 1024 * 1024,
        gpu_collision_stack_size=512 * 1024 * 1024,
        enable_external_forces_every_iteration=True,
    )
    newton_mjwarp = NewtonCfg(
        solver_cfg=RandomizedMJWarpSolverCfg(
            class_type="go2_rl_lab.physics.resettable_mjwarp_manager:NewtonMJWarpResettableManager",
            njmax=300,
            nconmax=128,
            cone="pyramidal",
            impratio=1,
            integrator="implicitfast",
            update_data_interval=1,
            use_mujoco_contacts=False,
            sensor_bvh_shape_flags=int(ShapeFlags.COLLIDE_SHAPES),
            body_com_buckets=BodyComBucketRandomizationCfg(
                body_name_pattern=r"(?:^|/)Robot/base$",
                x_offset_range=(0.0, 0.1),
                y_offset_range=(-0.05, 0.05),
                z_offset_range=(0.0, 0.05),
                num_buckets=10,
                seed=42,
            ),
        ),
        collision_cfg=NewtonCollisionPipelineCfg(
            max_triangle_pairs=2_500_000,
        ),
        default_shape_cfg=NewtonShapeCfg(
            margin=0.01,
        ),
        num_substeps=1,
        debug_mode=False,
    )
    default = newton_mjwarp


@configclass
class Go2ContactSensorCfg(PresetCfg):
    """Backend-specific contact sensors with identical task-facing data."""

    isaacsim_physx = PhysXContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )
    newton_mjwarp = NewtonContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )
    default = newton_mjwarp


@configclass
class Go2HeightScannerCfg(RayCasterCfg):
    """Configure terrain-only ray casting for Newton and PhysX."""

    global_world_only: bool = True
    """Limit Newton BVH queries to global terrain, excluding per-environment robots.

    PhysX selects terrain meshes through ``mesh_prim_paths``.
    """


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=MIXED_TERRAIN_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=(
                f"{ISAACLAB_NUCLEUS_DIR}/Materials/"
                "TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl"
            ),
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # sensors
    contact_forces = Go2ContactSensorCfg()
    height_scanner = Go2HeightScannerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(
            resolution=0.1,
            size=(1.6, 1.0),
            ordering="yx",
        ),
        mesh_prim_paths=["/World/ground"],
        max_distance=100.0,
        debug_vis=False,
    )
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class EventCfg:
    """Configuration for domain-randomization events."""

    randomize_rigid_body_mass_base = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=BASE_LINK_NAME),
            "mass_distribution_params": (-1.0, 1.0),
            "operation": "add",
            "recompute_inertia": True,
        },
    )
    randomize_rigid_body_mass_others = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="^(?!.*base).*"),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
            "recompute_inertia": True,
        },
    )
    randomize_com_positions = preset(
        default=None,
        isaacsim_physx=EventTerm(
            func=mdp.randomize_rigid_body_com,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=BASE_LINK_NAME),
                "com_range": {"x": (-0.03, 0.03), "y": (-0.03, 0.03), "z": (-0.03, 0.03)},
            },
        ),
        # MJWarp cannot yet fully rebuild the mass matrix after a runtime CoM change.
        newton_mjwarp=None,
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )
    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.9, 1.1),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    randomize_motor_zero_offset = EventTerm(
        func=mdp.randomize_action_joint_pos_offset,
        mode="reset",
        params={
            "action_term_name": "joint_pos",
            "offset_range": (-0.035, 0.035),
        },
    )
    randomize_push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(4.0, 4.0),
        params={
            "velocity_range": {
                "x": (-0.4, 0.4),
                "y": (-0.4, 0.4),
                "roll": (-0.6, 0.6),
                "pitch": (-0.6, 0.6),
                "yaw": (-0.6, 0.6),
            }
        },
    )
    randomize_rigid_body_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.0, 2.0),
            "dynamic_friction_range": (0.0, 2.0),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.0, 0.2),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.RLGymVelCommandCfg(
        limit_vel_curriculum=[
            {
                "iter": 0,
                "lin_vel_x": [-1, 1],
                "lin_vel_y": [-1, 1],
                "ang_vel_yaw": [-1, 0, 1],
            },
            {
                "iter": 25000,
                "lin_vel_x": [-1, 1],
                "lin_vel_y": [0],
                "ang_vel_yaw": [-1, 0, 1],
            },
        ],
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=JOINT_NAMES,
        scale={".*_hip_joint": 0.25, "^(?!.*_hip_joint).*": 0.25},
        use_default_offset=True,
        clip={".*": (-100.0, 100.0)},
        preserve_order=True,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
            clip=(-100.0, 100.0),
            scale=0.25,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            clip=(-100.0, 100.0),
            scale=(0.5, 1.0, 1.0),
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True)},
            noise=Unoise(n_min=-0.03, n_max=0.03),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True)},
            noise=Unoise(n_min=-2.0, n_max=2.0),
            clip=(-100.0, 100.0),
            scale=0.05,
        )
        actions = ObsTerm(
            func=mdp.last_action,
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        def __post_init__(self):
            self.history_length = 10
            self.enable_corruption = True
            self.concatenate_terms = True
            self.flatten_history_dim = True

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100.0, 100.0), scale=(0.5, 1.0, 1.0))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, clip=(-100.0, 100.0), scale=0.25)
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100.0, 100.0), scale=1.0)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            clip=(-100.0, 100.0),
            scale=(0.5, 1.0, 1.0),
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True)},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True)},
            clip=(-100.0, 100.0),
            scale=0.05,
        )
        actions = ObsTerm(
            func=mdp.last_action,
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_acc = ObsTerm(
            func=mdp.joint_acc,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True)},
            clip=(-100.0, 100.0),
            scale=1.0e-4,
        )
        joint_torque = ObsTerm(
            func=mdp.joint_effort,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True)},
            clip=(-100.0, 100.0),
            scale=0.01,
        )
        contact_force = ObsTerm(
            func=mdp.foot_contact_force_norm,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_LINK_NAME)},
            clip=(-100.0, 100.0),
            scale=1.0e-3,
        )
        height_scan = ObsTerm(
            func=mdp.ray_caster_height_scan,
            params={
                "sensor_cfg": SceneEntityCfg("height_scanner"),
                "offset": 0.5,
            },
            clip=(-1.0, 1.0),
            scale=2.5,
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class SingleObsCfg(PolicyCfg):
        """Current policy observation frame used by the deployment exporter."""

        def __post_init__(self):
            super().__post_init__()
            self.history_length = 1

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    single_obs: SingleObsCfg = SingleObsCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # Target-tracking rewards
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )

    # Regularization rewards
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    joint_acc_l2 = RewTerm(
        func=mdp.clipped_joint_acc_l2,
        weight=preset(default=-3.0e-7, isaacsim_physx=-1.0e-7, newton_mjwarp=-3.0e-7),
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES),
            "max_penalty": 1.0e8,
        },
    )
    joint_power = RewTerm(
        func=mdp.joint_power,
        weight=preset(default=-1.5e-5, isaacsim_physx=-2.0e-5, newton_mjwarp=-1.5e-5),
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES)},
    )
    joint_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=preset(default=-6e-5, isaacsim_physx=-1.0e-4, newton_mjwarp=-6e-5),
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES)},
    )
    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=BASE_LINK_NAME),
            "target_height": BASE_HEIGHT_TARGET,
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "x_range": BASE_HEIGHT_SCAN_X_RANGE,
            "y_range": BASE_HEIGHT_SCAN_Y_RANGE,
        },
    )
    action_rate_l2 = RewTerm(
        func=mdp.clipped_action_rate_l2,
        weight=-0.01,
        params={"max_penalty": 20.0},
    )
    action_smoothness_l2 = RewTerm(
        func=mdp.clipped_action_smoothness_l2,
        weight=-0.01,
        params={"max_penalty": 50.0},
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_thigh|.*_calf"),
            "threshold": preset(default=110.0, isaacsim_physx=5.0, newton_mjwarp=110.0),
        },
    )
    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES)},
    )
    feet_regulation = RewTerm(
        func=mdp.feet_regulation,
        weight=-0.05,
        params={
            "base_height_target": BASE_HEIGHT_TARGET,
            "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_LINK_NAME),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "x_range": BASE_HEIGHT_SCAN_X_RANGE,
            "y_range": BASE_HEIGHT_SCAN_Y_RANGE,
        },
    )
    hip_pos_penalty_l1 = RewTerm(
        func=mdp.joint_pos_penalty_l1,
        weight=-0.05,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_hip_joint"),
            "stand_still_scale": 10.0,
            "stand_cmd_idxs": [1, 2],
        },
    )
    joint_pos_penalty_l1 = RewTerm(
        func=mdp.joint_pos_penalty_l1,
        weight=-0.01,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_(thigh|calf)_joint"),
            "stand_still_scale": 10.0,
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    illegal_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=BASE_LINK_NAME),
            "threshold": 1.0,
        },
    )
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "limit_angle": 2.0,
        },
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.TerrainLevelsVelGym)
    base_linear_velocity_z = CurrTerm(
        func=mdp.gradual_reward_weight_modification,
        params={
            "term_name": "lin_vel_z_l2",
            "initial_weight": -2.0,
            "final_weight": 0.0,
            "start_it": 0,
            "end_it": 1500,
        },
    )
    base_height = CurrTerm(
        func=mdp.gradual_reward_weight_modification,
        params={
            "term_name": "base_height_l2",
            "initial_weight": -1.0,
            "final_weight": -10.0,
            "start_it": 0,
            "end_it": 5000,
        },
    )


@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    terrain_backend: str = "mesh"
    """Default terrain collision backend, overridable with ``GO2_MIXED_TERRAIN``."""

    # Simulation settings
    # Go2 motor curves and delays require the custom Python actuator path.
    sim: SimulationCfg = SimulationCfg(physics=PhysicsCfg(), use_newton_actuators=False)
    # Scene settings
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=16384, env_spacing=0.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 25.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material

        terrain_backend = os.environ.get("GO2_MIXED_TERRAIN", self.terrain_backend)
        if terrain_backend not in ("mesh", "heightfield"):
            raise ValueError(
                "GO2_MIXED_TERRAIN must be 'mesh' or 'heightfield', "
                f"got {terrain_backend!r}."
            )
        self.terrain_backend = terrain_backend
        if terrain_backend == "mesh":
            self.scene.terrain.class_type = MixedTerrainImporter
            self.scene.terrain.terrain_generator = copy.deepcopy(MIXED_MESH_TERRAIN_CFG)
            self.scene.terrain.visual_material = None
            if self.scene.height_scanner is not None:
                self.scene.height_scanner.mesh_prim_paths = [
                    f"{self.scene.terrain.prim_path}/terrain/mesh"
                ]

        # update sensor update periods
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    """Deterministic playback over a configurable subset of the mixed terrains."""

    play_terrain_proportions: dict[str, float] | None = None
    """Playback sub-terrain weights keyed by terrain name.

    Only the listed terrains are generated, and the weights are quantized into
    whole columns, so ``{"flat": 0.75, "wheels": 0.25}`` yields three flat
    columns and one wheels column. ``None`` keeps every sub-terrain with one
    column each.
    """

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        if self.scene.terrain.terrain_generator is not None:
            # Isaac Lab assigns difficulty across rows and terrain types across columns.
            # Playback assigns robots deterministically from the hardest row to the easiest.
            self.scene.terrain.class_type = (
                MixedPlayTerrainImporter if self.terrain_backend == "mesh" else HighToLowTerrainImporter
            )
            terrain_generator = self.scene.terrain.terrain_generator
            terrain_generator.num_rows = 10
            configure_play_sub_terrains(terrain_generator, self.play_terrain_proportions)
            self.scene.terrain.max_init_terrain_level = terrain_generator.num_rows - 1
        # Hold the maximum forward command so that terrain_max_command_ranges
        # clamps the speed per terrain (3.0 on flat, 1.5 on wheels).
        self.commands.base_velocity = mdp.RLGymVelCommandCfg(
            asset_name="robot",
            ranges=mdp.RLGymVelCommandCfg.Ranges(
                lin_vel_x=(3.0, 3.0),
                lin_vel_y=(0.0, 0.0),
                ang_vel_yaw=(0.0, 0.0),
            ),
            dynamic_resample_commands=False,
            command_range_curriculum=[],
            # The hold combo (1, 0, 0) requires a zero direction code for lin_vel_y.
            limit_vel_curriculum=[
                {
                    "iter": 0,
                    "lin_vel_x": [-1, 1],
                    "lin_vel_y": [-1, 0, 1],
                    "ang_vel_yaw": [-1, 0, 1],
                },
            ],
            limit_vel_prob=0.0,
            limit_vel_hold_prob=1.0,
            limit_vel_hold_combo=(1, 0, 0),
            zero_command_curriculum=None,
            resampling_time=30.0,
            debug_vis=True,
        )
        # Keep terrain levels fixed during playback.
        self.curriculum.terrain_levels = None
