"""Velocity-task MDP terms and configurations with explicit imports."""

from isaaclab.envs.mdp import (
    JointPositionActionCfg,
    UniformVelocityCommandCfg,
    bad_orientation,
    base_ang_vel,
    base_lin_vel,
    generated_commands,
    illegal_contact,
    joint_effort,
    joint_pos_limits,
    joint_pos_rel,
    joint_torques_l2,
    joint_vel_rel,
    last_action,
    projected_gravity,
    push_by_setting_velocity,
    randomize_actuator_gains,
    randomize_rigid_body_com,
    randomize_rigid_body_mass,
    randomize_rigid_body_material,
    reset_joints_by_scale,
    time_out,
)

from go2_rl_lab.cmd_utils.rl_gym_cmd import (
    RLGymVelCommandCfg,
)

from go2_rl_lab.tasks.mixed.mdp.curriculums import (
    TerrainLevelsVelGym,
    gradual_reward_weight_modification,
)

from go2_rl_lab.tasks.mixed.mdp.rewards import (
    ang_vel_xy_l2,
    base_height_l2,
    clipped_action_rate_l2,
    clipped_action_smoothness_l2,
    clipped_joint_acc_l2,
    feet_regulation,
    joint_pos_penalty_l1,
    joint_power,
    lin_vel_z_l2,
    track_ang_vel_z_exp,
    track_lin_vel_xy_exp,
    undesired_contacts,
)

from go2_rl_lab.tasks.mixed.mdp.observations import (
    foot_contact_force_norm,
    joint_acc,
)

from go2_rl_lab.tasks.mixed.mdp.events import (
    randomize_action_joint_pos_offset,
    reset_root_state_uniform,
)

from go2_rl_lab.utils.height_scan import (
    ray_caster_height_scan,
)

__all__ = [
    "JointPositionActionCfg",
    "RLGymVelCommandCfg",
    "TerrainLevelsVelGym",
    "UniformVelocityCommandCfg",
    "ang_vel_xy_l2",
    "bad_orientation",
    "base_ang_vel",
    "base_height_l2",
    "base_lin_vel",
    "clipped_action_rate_l2",
    "clipped_action_smoothness_l2",
    "clipped_joint_acc_l2",
    "feet_regulation",
    "foot_contact_force_norm",
    "generated_commands",
    "gradual_reward_weight_modification",
    "illegal_contact",
    "joint_acc",
    "joint_effort",
    "joint_pos_limits",
    "joint_pos_penalty_l1",
    "joint_pos_rel",
    "joint_power",
    "joint_torques_l2",
    "joint_vel_rel",
    "last_action",
    "lin_vel_z_l2",
    "projected_gravity",
    "push_by_setting_velocity",
    "randomize_action_joint_pos_offset",
    "randomize_actuator_gains",
    "randomize_rigid_body_com",
    "randomize_rigid_body_mass",
    "randomize_rigid_body_material",
    "ray_caster_height_scan",
    "reset_joints_by_scale",
    "reset_root_state_uniform",
    "time_out",
    "track_ang_vel_z_exp",
    "track_lin_vel_xy_exp",
    "undesired_contacts",
]
