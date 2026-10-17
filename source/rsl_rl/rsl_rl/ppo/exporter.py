# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO policy exporters."""

import copy
import os
import torch

import onnx


class TorchPolicyHistoryExporter(torch.nn.Module):
    """Export a PPO actor with internal observation history."""

    def __init__(
        self,
        policy: object,
        actor_obs_normalizer: object | None,
        num_single_obs: int,
        num_actions: int,
    ) -> None:
        super().__init__()

        if not hasattr(policy, "actor"):
            raise ValueError("PPO policy does not have an actor module.")

        self.actor = copy.deepcopy(policy.actor)
        self.actor_obs_normalizer = (
            copy.deepcopy(actor_obs_normalizer) if actor_obs_normalizer is not None else torch.nn.Identity()
        )
        self.state_dependent_std = bool(getattr(policy, "state_dependent_std", False))
        self.num_single_obs = int(num_single_obs)
        self.num_actor_obs = int(self.actor[0].in_features)

        if self.num_actor_obs % self.num_single_obs != 0:
            raise ValueError(
                f"num_actor_obs ({self.num_actor_obs}) must be divisible by num_single_obs ({self.num_single_obs})."
            )

        self.history_len = self.num_actor_obs // self.num_single_obs

        # Velocity policies store [vx, vy, yaw_rate] as the third vector block:
        # [base_ang_vel, projected_gravity, velocity_command, joints...].
        num_actions = int(num_actions)
        expected_single_obs = 3 + 3 + 3 + 3 * num_actions
        if self.num_single_obs != expected_single_obs:
            raise ValueError(
                "Unsupported single_obs layout: expected "
                f"3+3+3+3*num_actions={expected_single_obs}, "
                f"got {self.num_single_obs}."
            )
        self.feature_dims = [3, 3, 3, num_actions, num_actions, num_actions]

        self.register_buffer("obs_history", torch.zeros(1, self.num_actor_obs, dtype=torch.float32))

    def forward(self, single_obs: torch.Tensor) -> torch.Tensor:
        """Append one observation frame and return the deterministic action."""
        if single_obs.dim() == 1:
            single_obs = single_obs.unsqueeze(0)
        if single_obs.shape[-1] != self.num_single_obs:
            raise ValueError(f"Expected single_obs last dimension {self.num_single_obs}, got {single_obs.shape[-1]}.")
        if single_obs.shape[0] != 1:
            raise ValueError("TorchScript PPO deployment supports batch size 1 only.")

        next_history = self.obs_history.clone()
        history_offset = 0
        single_offset = 0
        for dim in self.feature_dims:
            block_size = dim * self.history_len
            block_end = history_offset + block_size
            single_end = single_offset + dim
            block = self.obs_history[:, history_offset:block_end]
            shifted_block = torch.cat([block[:, dim:], single_obs[:, single_offset:single_end]], dim=-1)
            next_history[:, history_offset:block_end] = shifted_block
            history_offset = block_end
            single_offset = single_end
        self.obs_history.copy_(next_history)

        actor_output = self.actor(self.actor_obs_normalizer(self.obs_history))
        if self.state_dependent_std:
            return actor_output[..., 0, :]
        return actor_output

    @torch.jit.export
    def reset(self) -> None:
        """Clear the internal observation history."""
        self.obs_history.zero_()

    def export(self, path: str, filename: str) -> None:
        """Write the stateful deployment module to disk."""
        os.makedirs(path, exist_ok=True)
        self.eval()
        self.to("cpu")
        torch.jit.script(self).save(os.path.join(path, filename))


class OnnxPolicyExporter(torch.nn.Module):
    """Export an actor-critic policy as an ONNX model."""

    def __init__(self, policy: object, normalizer: object | None = None, verbose: bool = False) -> None:
        super().__init__()
        self.verbose = verbose
        self.state_dependent_std = getattr(policy, "state_dependent_std", False)
        self.actor = copy.deepcopy(policy.actor)

        self.normalizer = copy.deepcopy(normalizer) if normalizer is not None else torch.nn.Identity()

    def _actor_forward(self, x: torch.Tensor) -> torch.Tensor:
        actions = self.actor(x)
        if self.state_dependent_std:
            return actions[..., 0, :]
        return actions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._actor_forward(self.normalizer(x))

    def export(self, path: str, filename: str) -> None:
        """Write the ONNX model to disk."""
        os.makedirs(path, exist_ok=True)
        export_path = os.path.join(path, filename)
        self.to("cpu")
        self.eval()
        ir_version = 9
        opset_version = 18

        args = torch.zeros(1, self.actor[0].in_features)

        onnx_program = torch.onnx.export(
            self,
            args,
            f=None,
            dynamo=True,
            export_params=True,
            external_data=False,
            opset_version=opset_version,
            verbose=self.verbose,
            input_names=["obs"],
            output_names=["actions"],
        )
        if onnx_program is None:
            raise RuntimeError("The ONNX exporter did not return an ONNXProgram.")

        onnx_program.model.ir_version = ir_version
        onnx.checker.check_model(onnx_program.model_proto)
        onnx_program.save(export_path, external_data=False)

        # PyTorch's dynamo exporter defaults to external weights. Remove a stale
        # sidecar left by an older export now that all weights live in the ONNX file.
        external_data_path = f"{export_path}.data"
        if os.path.isfile(external_data_path):
            os.remove(external_data_path)


def export_policy_as_jit(
    policy: object,
    actor_obs_normalizer: object | None,
    num_single_obs: int,
    num_actions: int,
    path: str,
    filename: str = "policy.pt",
) -> None:
    """Export a PPO policy that builds observation history from single frames."""
    TorchPolicyHistoryExporter(
        policy,
        actor_obs_normalizer=actor_obs_normalizer,
        num_single_obs=num_single_obs,
        num_actions=num_actions,
    ).export(path, filename)


def export_policy_as_onnx(
    policy: object,
    path: str,
    normalizer: object | None = None,
    filename: str = "policy.onnx",
    verbose: bool = False,
) -> None:
    """Export a PPO actor policy as ONNX."""
    OnnxPolicyExporter(policy, normalizer, verbose).export(path, filename)


__all__ = [
    "OnnxPolicyExporter",
    "TorchPolicyHistoryExporter",
    "export_policy_as_jit",
    "export_policy_as_onnx",
]
