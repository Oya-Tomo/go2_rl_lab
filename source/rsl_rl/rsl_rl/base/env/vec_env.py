# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Vectorized-environment interfaces shared by RSL-RL algorithms."""

from __future__ import annotations

import torch
from abc import ABC, abstractmethod
from tensordict import TensorDict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import gymnasium as gym
    from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv


class VecEnv(ABC):
    """Abstract class for a vectorized environment.

    The vectorized environment is a collection of environments that are synchronized. This means that the same type of
    action is applied to all environments and the same type of observation is returned from all environments.
    """

    num_envs: int
    """Number of environments."""

    num_actions: int
    """Number of actions."""

    max_episode_length: int | torch.Tensor
    """Maximum episode length.

    The maximum episode length can be a scalar or a tensor. If it is a scalar, it is the same for all environments.
    If it is a tensor, it is the maximum episode length for each environment. This is useful for dynamic episode
    lengths.
    """

    episode_length_buf: torch.Tensor
    """Buffer for current episode lengths."""

    device: torch.device | str
    """Device to use."""

    cfg: dict | object
    """Configuration object."""

    @abstractmethod
    def get_observations(self) -> TensorDict:
        """Return the current observations.

        Returns:
            The observations from the environment.
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        """Apply input action to the environment.

        Args:
            actions: Input actions to apply. Shape: (num_envs, num_actions)

        Returns:
            observations: Observations from the environment.
            rewards: Rewards from the environment. Shape: (num_envs,)
            dones: Done flags from the environment. Shape: (num_envs,)
            extras: Extra information from the environment.

        Observations:
            The observations TensorDict usually contains multiple observation groups. The `obs_groups`
            dictionary of the runner configuration specifies which observation groups are used for which
            purpose, i.e., it maps the available observation groups to observation sets. The observation sets
            (keys of the `obs_groups` dictionary) currently used by rsl_rl are:

            - "policy": Specified observation groups are used as input to the actor/student network.
            - "critic": Specified observation groups are used as input to the critic network.

            Incomplete or incorrect configurations are handled in the `resolve_obs_groups()` function in
            `rsl_rl/utils/utils.py`.

        Extras:
            The extras dictionary includes metrics such as the episode reward, episode length, etc. The following
            dictionary keys are used by rsl_rl:

            - "time_outs" (torch.Tensor): Timeouts for the environments. These correspond to terminations that
               happen due to time limits and not due to the environment reaching a terminal state. This is useful
               for environments that have a fixed episode length.

            - "log" (dict[str, float | torch.Tensor]): Additional information for logging and debugging purposes.
               The key should be a string and start with "/" for namespacing. The value can be a scalar or a
               tensor. If it is a tensor, the mean of the tensor is used for logging.
        """
        raise NotImplementedError


class RslRlVecEnvWrapper(VecEnv):
    """Adapt an Isaac Lab environment to the RSL-RL vector-environment interface.

    This wrapper must be the final wrapper in the chain because it does not
    implement the :class:`gymnasium.Wrapper` interface.
    """

    def __init__(
        self,
        env: ManagerBasedRLEnv | DirectRLEnv,
        clip_actions: float | None = None,
        clip_rewards: float | None = None,
    ) -> None:
        """Initialize the wrapper and reset the environment once."""
        import gymnasium as gym
        from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv

        if not isinstance(env.unwrapped, (ManagerBasedRLEnv, DirectRLEnv)):
            raise ValueError(
                "The environment must inherit from ManagerBasedRLEnv or DirectRLEnv. "
                f"Received environment type: {type(env)}"
            )

        self.env = env
        self.clip_actions = clip_actions
        if clip_rewards is not None and clip_rewards <= 0.0:
            raise ValueError(f"clip_rewards must be positive or None. Received: {clip_rewards}")
        self.clip_rewards = clip_rewards
        self.num_envs = self.unwrapped.num_envs
        self.device = self.unwrapped.device
        self.max_episode_length = self.unwrapped.max_episode_length

        if hasattr(self.unwrapped, "action_manager"):
            self.num_actions = self.unwrapped.action_manager.total_action_dim
        else:
            self.num_actions = gym.spaces.flatdim(self.unwrapped.single_action_space)

        self._modify_action_space()
        self.env.reset()

    def __str__(self) -> str:
        """Return the wrapper name and wrapped environment representation."""
        return f"<{type(self).__name__}{self.env}>"

    def __repr__(self) -> str:
        """Return the string representation."""
        return str(self)

    @property
    def cfg(self) -> object:
        """The environment configuration."""
        return self.unwrapped.cfg

    @property
    def render_mode(self) -> str | None:
        """The environment render mode."""
        return self.env.render_mode

    @property
    def observation_space(self) -> gym.Space:
        """The batched observation space."""
        return self.env.observation_space

    @property
    def action_space(self) -> gym.Space:
        """The batched action space."""
        return self.env.action_space

    @classmethod
    def class_name(cls) -> str:
        """Return the wrapper class name."""
        return cls.__name__

    @property
    def unwrapped(self) -> ManagerBasedRLEnv | DirectRLEnv:
        """The base Isaac Lab environment."""
        return self.env.unwrapped

    @property
    def episode_length_buf(self) -> torch.Tensor:
        """The current episode lengths."""
        return self.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:
        """Set the current episode lengths."""
        self.unwrapped.episode_length_buf = value

    def seed(self, seed: int = -1) -> int:
        """Seed the base environment."""
        return self.unwrapped.seed(seed)

    def reset(self) -> tuple[TensorDict, dict]:
        """Reset the environment and return TensorDict observations."""
        obs_dict, extras = self.env.reset()
        return TensorDict(obs_dict, batch_size=[self.num_envs]), extras

    def get_observations(self) -> TensorDict:
        """Return the current observations as a TensorDict."""
        if hasattr(self.unwrapped, "observation_manager"):
            obs_dict = self.unwrapped.observation_manager.compute()
        else:
            obs_dict = self.unwrapped._get_observations()
        return TensorDict(obs_dict, batch_size=[self.num_envs])

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        """Apply actions and return the RSL-RL step tuple."""
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)

        obs_dict, rewards, terminated, truncated, extras = self.env.step(actions)
        obs = TensorDict(obs_dict, batch_size=[self.num_envs])

        # Isaac Lab computes observations and rewards after its termination check. Catch
        # non-finite outputs here, before they are forwarded to the policy and rollout storage.
        invalid_mask, invalid_sources = self._find_nonfinite_envs(obs, rewards)

        # Keep this key present on every step so the logger can aggregate this sparse
        # termination event over its logging window. The value is the fraction of
        # environments terminated by a non-finite output on the current step.
        extras = extras.copy()
        extras["log"] = extras.get("log", {}).copy()
        extras["log"]["Episode_Termination/non_finite"] = invalid_mask.float().mean()

        if invalid_mask.any().item():
            invalid_env_ids = invalid_mask.nonzero(as_tuple=False).squeeze(-1)
            source_summary = ", ".join(
                f"{name}={int(source_mask.sum().item())}"
                for name, source_mask in invalid_sources.items()
                if source_mask.any().item()
            )
            displayed_ids = invalid_env_ids[:16].tolist()
            id_suffix = "..." if len(invalid_env_ids) > len(displayed_ids) else ""
            print(
                f"[WARNING] Non-finite environment output; resetting {len(invalid_env_ids)} env(s). "
                f"env_ids={displayed_ids}{id_suffix}; sources: {source_summary}"
            )

            rewards = rewards.clone()
            rewards[invalid_mask] = 0.0
            terminated = terminated.clone()
            terminated[invalid_mask] = True
            truncated = truncated.clone()
            truncated[invalid_mask] = False

            # The base environment has already passed its automatic-reset stage, so
            # reset the affected environments explicitly and return their fresh state.
            try:
                reset_obs_dict, _ = self.unwrapped.reset(env_ids=invalid_env_ids)
            except TypeError as error:
                raise RuntimeError(
                    "Partial reset after a non-finite output requires an environment whose reset() "
                    "method accepts env_ids."
                ) from error
            self._request_hard_reset(invalid_env_ids)
            reset_obs = TensorDict(reset_obs_dict, batch_size=[self.num_envs])
            obs = obs.clone()
            obs[invalid_env_ids] = reset_obs[invalid_env_ids]

            # Never expose a non-finite value even if the reset itself did not fully
            # recover the simulator state. It will be detected and reset again later.
            for _, value in obs.items(include_nested=True, leaves_only=True):
                if value.is_floating_point() or value.is_complex():
                    value[invalid_env_ids] = torch.nan_to_num(
                        value[invalid_env_ids], nan=0.0, posinf=0.0, neginf=0.0
                    )

        if self.clip_rewards is not None:
            reward_clip_mask = torch.isfinite(rewards) & (torch.abs(rewards) > self.clip_rewards)
            extras["log"]["Diagnostics/reward_clip_fraction"] = reward_clip_mask.float().mean()
            rewards = torch.clamp(rewards, -self.clip_rewards, self.clip_rewards)

        dones = (terminated | truncated).to(dtype=torch.long)
        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated

        return obs, rewards, dones, extras

    def _find_nonfinite_envs(
        self, obs: TensorDict, rewards: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return per-environment masks for non-finite observations and rewards."""
        invalid_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=rewards.device)
        invalid_sources = {}

        for key, value in obs.items(include_nested=True, leaves_only=True):
            if not (value.is_floating_point() or value.is_complex()):
                continue
            source_mask = ~torch.isfinite(value).reshape(self.num_envs, -1).all(dim=1)
            source_name = "/".join(key) if isinstance(key, tuple) else key
            invalid_sources[f"obs/{source_name}"] = source_mask
            invalid_mask |= source_mask

        reward_mask = ~torch.isfinite(rewards).reshape(self.num_envs, -1).all(dim=1)
        invalid_sources["rewards"] = reward_mask
        invalid_mask |= reward_mask
        return invalid_mask, invalid_sources

    def _request_hard_reset(self, env_ids: torch.Tensor) -> None:
        """Request an optional backend-level hard reset after a non-finite output."""
        sim = getattr(self.unwrapped, "sim", None)
        physics_manager = getattr(sim, "physics_manager", None)
        request_hard_reset = getattr(physics_manager, "request_hard_reset", None)
        if request_hard_reset is not None:
            request_hard_reset(env_ids)

    def close(self) -> None:
        """Close the wrapped environment."""
        return self.env.close()

    def _modify_action_space(self) -> None:
        """Restrict the advertised action space to the clipping range."""
        if self.clip_actions is None:
            return

        import gymnasium as gym

        self.unwrapped.single_action_space = gym.spaces.Box(
            low=-self.clip_actions,
            high=self.clip_actions,
            shape=(self.num_actions,),
        )
        self.unwrapped.action_space = gym.vector.utils.batch_space(
            self.unwrapped.single_action_space,
            self.num_envs,
        )
