"""Curriculum terms for the Go2 mixed-terrain task."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import CurriculumTermCfg, ManagerTermBase

from go2_rl_lab.utils.terrains import get_assigned_terrain_masks

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from go2_rl_lab.cmd_utils.rl_gym_cmd import RLGymVelCommand


class TerrainLevelsVelGym(ManagerTermBase):
    """Update terrain levels and report overall and per-terrain level statistics."""

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)
        terrain_generator = env.scene.terrain.cfg.terrain_generator
        self._terrain_names = list(terrain_generator.sub_terrains) if terrain_generator is not None else []
        self._terrain_masks = get_assigned_terrain_masks(env)
        self._terrain_has_envs = {
            terrain_name: bool(self._terrain_masks[terrain_name].any().item())
            for terrain_name in self._terrain_names
        }
        self._max_levels_reached = torch.full(
            (len(self._terrain_names),),
            -1.0,
            dtype=torch.float,
            device=env.device,
        )

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
    ) -> dict[str, torch.Tensor]:
        """Promote or demote levels and return current and historical statistics."""
        terrain = env.scene.terrain
        command: RLGymVelCommand = env.command_manager.get_term("base_velocity")

        maximum_distance = command.max_move_distance[env_ids]
        accumulated_commands = command.commands_xy_accumulation[env_ids]

        terrain_generator = terrain.cfg.terrain_generator
        if terrain_generator is None:
            return {"mean_level": torch.tensor(0.0, device=env.device)}

        sub_terrain_border_width = getattr(terrain_generator, "sub_terrain_border_width", 0.0) or 0.0
        terrain_length = max(
            0.0,
            terrain_generator.size[0] - 2.0 * sub_terrain_border_width,
        )

        move_up = maximum_distance > terrain_length / 2.0
        target_distance = (
            torch.linalg.norm(accumulated_commands, dim=1)
            * command.cfg.resampling_time
            * (1.0 - command.zero_command_prob)
        )
        move_down = (maximum_distance < target_distance * 0.5) & ~move_up
        terrain.update_env_origins(env_ids, move_up, move_down)

        levels = terrain.terrain_levels.float()
        statistics = {"mean_level": torch.mean(levels)}
        for terrain_index, terrain_name in enumerate(self._terrain_names):
            terrain_mask = self._terrain_masks[terrain_name]
            if self._terrain_has_envs[terrain_name]:
                terrain_levels = levels[terrain_mask]
                mean_level = torch.mean(terrain_levels)
                current_max_level = torch.max(terrain_levels)
                self._max_levels_reached[terrain_index] = torch.maximum(
                    self._max_levels_reached[terrain_index], current_max_level
                )
            else:
                mean_level = torch.tensor(-1.0, device=env.device)

            statistics[f"{terrain_name}/mean_level"] = mean_level
            statistics[f"{terrain_name}/max_level_reached"] = self._max_levels_reached[terrain_index]
        return statistics


def gradual_reward_weight_modification(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    initial_weight: float,
    final_weight: float,
    start_it: int,
    end_it: int,
    num_steps_per_iter: int = 24,
) -> float:
    """Linearly change a reward weight over a range of training iterations."""
    del env_ids
    if end_it <= start_it:
        raise ValueError(f"end_it must be greater than start_it, got {start_it=} and {end_it=}.")

    current_iteration = env.common_step_counter // num_steps_per_iter
    progress = (current_iteration - start_it) / (end_it - start_it)
    progress = max(0.0, min(float(progress), 1.0))
    weight = initial_weight + progress * (final_weight - initial_weight)

    term_cfg = env.reward_manager.get_term_cfg(term_name)
    term_cfg.weight = weight
    env.reward_manager.set_term_cfg(term_name, term_cfg)
    return weight
