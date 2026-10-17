"""Isaac Lab articulation configuration for the Unitree Go2."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.configclass import configclass
from isaaclab_tasks.utils import preset

from go2_rl_lab.assets.go2_actuator import Go2ActuatorCfg

if TYPE_CHECKING:
    from pxr import Usd

_DEFAULT_GO2_USD_PATH = Path(__file__).resolve().parent / "usd" / "go2.usd"
GO2_USD_PATH = Path(os.environ.get("GO2_USD_PATH", _DEFAULT_GO2_USD_PATH)).expanduser()
"""Path to the Go2 USD asset. Override it with the ``GO2_USD_PATH`` environment variable."""


def _deactivate_visual_branches(root_prim: Usd.Prim) -> int:
    """Deactivate normalized Go2 ``visuals`` branches before environment replication.
    We found that this can very effectively reduce cuda memory usage.
    """
    from pxr import Usd

    stage = root_prim.GetStage()
    visual_paths = [prim.GetPath() for prim in Usd.PrimRange(root_prim) if prim.GetName() == "visuals"]
    for path in visual_paths:
        stage.GetPrimAtPath(path).SetActive(False)
    return len(visual_paths)


def spawn_go2_from_usd(
    prim_path: str,
    cfg: Go2UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    """Spawn Go2 and optionally remove its visual-only branches before scene cloning."""
    robot_prim = sim_utils.spawn_from_usd(
        prim_path,
        cfg,
        translation=translation,
        orientation=orientation,
        **kwargs,
    )
    if not cfg.visuals_enabled:
        _deactivate_visual_branches(robot_prim)
    return robot_prim


@configclass
class Go2UsdFileCfg(sim_utils.UsdFileCfg):
    """USD spawn configuration used by the Go2 articulation."""

    func: Callable | str = spawn_go2_from_usd
    visuals_enabled: bool = True
    """Whether visual-only branches remain active when the source robot is cloned."""

    activate_contact_sensors: bool = True
    rigid_props = sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        retain_accelerations=False,
        linear_damping=0.0,
        angular_damping=0.0,
        max_linear_velocity=1000.0,
        max_angular_velocity=1000.0,
        max_depenetration_velocity=1.0,
    )
    articulation_props = sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=True,
        solver_position_iteration_count=8,
        solver_velocity_iteration_count=4,
    )


GO2_CFG = ArticulationCfg(
    spawn=Go2UsdFileCfg(usd_path=str(GO2_USD_PATH)),
    soft_joint_pos_limit_factor=0.9,
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.4),
        joint_pos={
            ".*R_hip_joint": -0.1,
            ".*L_hip_joint": 0.1,
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 1.0,
            ".*_calf_joint": -1.5,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "Go2HV": Go2ActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=25.0,
            damping=0.5,
            friction=0.01,
            armature=preset(default=0.01, isaacsim_physx=0.0, newton_mjwarp=0.01),
            min_delay=0,
            max_delay=4,
        ),
    },
)
