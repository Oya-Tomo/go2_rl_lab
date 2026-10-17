"""Small MuJoCo viewer helpers for commanded and measured planar velocity."""

from __future__ import annotations

from typing import Literal

import mujoco
import numpy as np


class MujocoRenderUtils:
    """Draw body-frame commanded and measured velocity arrows."""

    def __init__(self) -> None:
        self.target_velocity: np.ndarray | None = None
        self.mj_data: mujoco.MjData | None = None

    def update(self, target_velocity: np.ndarray, mj_data: mujoco.MjData) -> None:
        self.target_velocity = target_velocity
        self.mj_data = mj_data

    def update_external_rendering(self, handle, target: Literal["viewer", "renderer"]) -> None:
        """Update velocity arrows in a passive viewer or offscreen renderer."""
        if target == "viewer":
            handle.user_scn.ngeom = 0
        if self.target_velocity is None or self.mj_data is None:
            return

        base_position = self.mj_data.qpos[:3]
        base_quaternion = self.mj_data.qpos[3:7]
        offset_world = np.zeros(3)
        mujoco.mju_rotVecQuat(offset_world, np.array([0.0, 0.0, 0.2]), base_quaternion)
        start_position = base_position + offset_world

        target_body = np.array([self.target_velocity[0], self.target_velocity[1], 0.0])
        current_body = np.zeros(3)
        inverse_quaternion = np.zeros(4)
        mujoco.mju_negQuat(inverse_quaternion, base_quaternion)
        mujoco.mju_rotVecQuat(current_body, self.mj_data.qvel[:3], inverse_quaternion)
        current_body[2] = 0.0

        target_world = np.zeros(3)
        current_world = np.zeros(3)
        mujoco.mju_rotVecQuat(target_world, target_body, base_quaternion)
        mujoco.mju_rotVecQuat(current_world, current_body, base_quaternion)

        scene = handle.user_scn if target == "viewer" else handle.scene
        start_index = scene.ngeom
        if target == "renderer":
            scene.ngeom += 2
        self._set_arrow(scene.geoms[start_index], start_position, target_world, [0.0, 1.0, 0.0, 1.0])
        self._set_arrow(scene.geoms[start_index + 1], start_position, current_world, [0.0, 0.0, 1.0, 1.0])
        if target == "viewer":
            scene.ngeom = start_index + 2

    @staticmethod
    def _set_arrow(geom, position: np.ndarray, vector: np.ndarray, color: list[float]) -> None:
        vector_norm = np.linalg.norm(vector)
        display_norm = min(vector_norm * 0.7, 1.0)
        if display_norm < 0.1:
            mujoco.mjv_initGeom(
                geom,
                type=mujoco.mjtGeom.mjGEOM_NONE,
                size=[0.0, 0.0, 0.0],
                pos=position,
                mat=np.eye(3).flatten(),
                rgba=[0.0, 0.0, 0.0, 0.0],
            )
            return

        rotation = np.zeros(9)
        quaternion = np.zeros(4)
        mujoco.mju_quatZ2Vec(quaternion, vector / vector_norm)
        mujoco.mju_quat2Mat(rotation, quaternion)
        rotation = rotation.reshape(3, 3)
        rotation[:, 2] *= display_norm
        mujoco.mjv_initGeom(
            geom,
            type=mujoco.mjtGeom.mjGEOM_ARROW,
            size=[0.02, 0.02, display_norm],
            pos=position,
            mat=rotation.flatten(),
            rgba=color,
        )
