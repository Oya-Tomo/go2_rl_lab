# Copyright (c) 2026 Tianyang Wu
# SPDX-License-Identifier: MIT

"""Deployment exporters for the student MoE-CTS policy."""

from __future__ import annotations

import copy
import os
import torch

import onnx


class TorchMoeCtsHistoryExporter(torch.nn.Module):
    """Export student inference with internally maintained term-wise history."""

    def __init__(
        self,
        policy: object,
        actor_obs_normalizer: object | None,
        single_obs_normalizer: object | None,
    ) -> None:
        super().__init__()
        for attribute in ("actor", "student_moe_encoder", "num_actions", "num_single_obs", "num_actor_obs"):
            if not hasattr(policy, attribute):
                raise ValueError(f"MoE-CTS policy is missing required attribute {attribute!r}.")

        self.actor = copy.deepcopy(policy.actor)
        self.student_moe_encoder = copy.deepcopy(policy.student_moe_encoder)
        self.state_dependent_std = bool(policy.state_dependent_std)
        self.num_actions = int(policy.num_actions)
        self.num_single_obs = int(policy.num_single_obs)
        self.num_actor_obs = int(policy.num_actor_obs)
        self.history_len = _history_length(self.num_actor_obs, self.num_single_obs)
        self.feature_dims = _feature_dimensions(self.num_single_obs, self.num_actions)
        self.actor_obs_normalizer = (
            copy.deepcopy(actor_obs_normalizer) if actor_obs_normalizer is not None else torch.nn.Identity()
        )
        self.single_obs_normalizer = (
            copy.deepcopy(single_obs_normalizer) if single_obs_normalizer is not None else torch.nn.Identity()
        )
        self.register_buffer("obs_history", torch.zeros(1, self.num_actor_obs, dtype=torch.float32))

    def forward(self, single_obs: torch.Tensor) -> torch.Tensor:
        if single_obs.dim() == 1:
            single_obs = single_obs.unsqueeze(0)
        if single_obs.shape[-1] != self.num_single_obs:
            raise ValueError(f"Expected single_obs dimension {self.num_single_obs}, got {single_obs.shape[-1]}.")
        if single_obs.shape[0] != 1:
            raise ValueError("TorchScript MoE-CTS deployment supports batch size 1 only.")

        next_history = self.obs_history.clone()
        history_offset = 0
        single_offset = 0
        for dimension in self.feature_dims:
            block_size = dimension * self.history_len
            block_end = history_offset + block_size
            single_end = single_offset + dimension
            block = self.obs_history[:, history_offset:block_end]
            next_history[:, history_offset:block_end] = torch.cat(
                [block[:, dimension:], single_obs[:, single_offset:single_end]],
                dim=-1,
            )
            history_offset = block_end
            single_offset = single_end
        self.obs_history.copy_(next_history)

        normalized_single_obs = self.single_obs_normalizer(single_obs)
        actor_obs = self.actor_obs_normalizer(self.obs_history)
        latent, _ = self.student_moe_encoder(actor_obs)
        actor_output = self.actor(torch.cat([latent, normalized_single_obs], dim=-1))
        return actor_output[..., 0, :] if self.state_dependent_std else actor_output

    @torch.jit.export
    def reset(self) -> None:
        """Clear the internally maintained observation history."""
        self.obs_history.zero_()

    def export(self, path: str, filename: str) -> None:
        os.makedirs(path, exist_ok=True)
        self.eval()
        self.to("cpu")
        torch.jit.script(self).save(os.path.join(path, filename))


class OnnxMoeCtsExporter(torch.nn.Module):
    """Export stateless student inference from a complete observation history."""

    def __init__(
        self,
        policy: object,
        actor_obs_normalizer: object | None,
        single_obs_normalizer: object | None,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        self.verbose = verbose
        self.actor = copy.deepcopy(policy.actor)
        self.student_moe_encoder = copy.deepcopy(policy.student_moe_encoder)
        self.state_dependent_std = bool(policy.state_dependent_std)
        self.num_actions = int(policy.num_actions)
        self.num_single_obs = int(policy.num_single_obs)
        self.num_actor_obs = int(policy.num_actor_obs)
        self.history_len = _history_length(self.num_actor_obs, self.num_single_obs)
        self.feature_dims = _feature_dimensions(self.num_single_obs, self.num_actions)
        self.actor_obs_normalizer = (
            copy.deepcopy(actor_obs_normalizer) if actor_obs_normalizer is not None else torch.nn.Identity()
        )
        self.single_obs_normalizer = (
            copy.deepcopy(single_obs_normalizer) if single_obs_normalizer is not None else torch.nn.Identity()
        )

    def _extract_single_obs(self, history: torch.Tensor) -> torch.Tensor:
        terms: list[torch.Tensor] = []
        offset = 0
        for dimension in self.feature_dims:
            end = offset + dimension * self.history_len
            terms.append(history[:, end - dimension : end])
            offset = end
        return torch.cat(terms, dim=-1)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if history.dim() == 1:
            history = history.unsqueeze(0)
        single_obs = self.single_obs_normalizer(self._extract_single_obs(history))
        actor_obs = self.actor_obs_normalizer(history)
        latent, _ = self.student_moe_encoder(actor_obs)
        actor_output = self.actor(torch.cat([latent, single_obs], dim=-1))
        return actor_output[..., 0, :] if self.state_dependent_std else actor_output

    def export(self, path: str, filename: str) -> None:
        os.makedirs(path, exist_ok=True)
        export_path = os.path.join(path, filename)
        self.to("cpu")
        self.eval()
        onnx_program = torch.onnx.export(
            self,
            torch.zeros(1, self.num_actor_obs),
            f=None,
            dynamo=True,
            export_params=True,
            external_data=False,
            opset_version=18,
            verbose=self.verbose,
            input_names=["obs"],
            output_names=["actions"],
        )
        if onnx_program is None:
            raise RuntimeError("The ONNX exporter did not return an ONNXProgram.")
        onnx_program.model.ir_version = 9
        onnx.checker.check_model(onnx_program.model_proto)
        onnx_program.save(export_path, external_data=False)
        external_data_path = f"{export_path}.data"
        if os.path.isfile(external_data_path):
            os.remove(external_data_path)


def export_policy_as_jit(
    policy: object,
    actor_obs_normalizer: object | None,
    single_obs_normalizer: object | None,
    path: str,
    filename: str = "policy.pt",
) -> None:
    """Export a stateful, single-frame student MoE-CTS policy."""
    TorchMoeCtsHistoryExporter(policy, actor_obs_normalizer, single_obs_normalizer).export(path, filename)


def export_policy_as_onnx(
    policy: object,
    path: str,
    actor_obs_normalizer: object | None = None,
    single_obs_normalizer: object | None = None,
    filename: str = "policy.onnx",
    verbose: bool = False,
) -> None:
    """Export a stateless student MoE-CTS policy using stacked observations."""
    OnnxMoeCtsExporter(policy, actor_obs_normalizer, single_obs_normalizer, verbose).export(path, filename)


def _history_length(num_actor_obs: int, num_single_obs: int) -> int:
    if num_actor_obs % num_single_obs != 0:
        raise ValueError(
            f"num_actor_obs ({num_actor_obs}) must be divisible by num_single_obs ({num_single_obs})."
        )
    return num_actor_obs // num_single_obs


def _feature_dimensions(num_single_obs: int, num_actions: int) -> list[int]:
    dimensions = [3, 3, 3, num_actions, num_actions, num_actions]
    if sum(dimensions) != num_single_obs:
        raise ValueError("Unsupported single_obs layout: expected 3+3+3+3*num_actions.")
    return dimensions


__all__ = [
    "OnnxMoeCtsExporter",
    "TorchMoeCtsHistoryExporter",
    "export_policy_as_jit",
    "export_policy_as_onnx",
]
