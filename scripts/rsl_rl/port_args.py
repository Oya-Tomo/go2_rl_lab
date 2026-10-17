# Copyright (c) 2026 Tianyang Wu
# SPDX-License-Identifier: MIT

"""Command-line configuration for Rerun visualizer ports."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnvCfg


_DEFAULT_RERUN_WEB_PORT = 9090
_DEFAULT_RERUN_GRPC_PORT = 9876


def _parse_port(value: str) -> int:
    """Parse and validate a TCP port supplied on the command line."""
    try:
        port = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid port: {value!r}") from exc

    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def add_rerun_port_args(parser: argparse.ArgumentParser) -> None:
    """Add Rerun port arguments to an argument parser."""
    arg_group = parser.add_argument_group("rerun", description="Arguments for the Rerun visualizer.")
    arg_group.add_argument(
        "--rerun_web_port",
        "--rerun-web-port",
        dest="rerun_web_port",
        type=_parse_port,
        default=None,
        help=f"Rerun Web Viewer port (default: {_DEFAULT_RERUN_WEB_PORT}).",
    )
    arg_group.add_argument(
        "--rerun_grpc_port",
        "--rerun-grpc-port",
        dest="rerun_grpc_port",
        type=_parse_port,
        default=None,
        help=f"Rerun gRPC server port (default: {_DEFAULT_RERUN_GRPC_PORT}).",
    )


def validate_rerun_port_args(parser: argparse.ArgumentParser, args_cli: argparse.Namespace) -> None:
    """Validate relationships between Rerun CLI arguments before launching Isaac Sim."""
    web_port = args_cli.rerun_web_port
    grpc_port = args_cli.rerun_grpc_port
    if web_port is None and grpc_port is None:
        return

    visualizers = args_cli.visualizer or []
    if "rerun" not in visualizers:
        parser.error("Rerun port arguments require '--viz rerun' (or a list containing rerun).")

    effective_web_port = web_port if web_port is not None else _DEFAULT_RERUN_WEB_PORT
    effective_grpc_port = grpc_port if grpc_port is not None else _DEFAULT_RERUN_GRPC_PORT
    if effective_web_port == effective_grpc_port:
        parser.error("Rerun Web Viewer and gRPC server ports must be different.")


def apply_rerun_port_args(env_cfg: ManagerBasedRLEnvCfg, args_cli: argparse.Namespace) -> None:
    """Apply CLI port overrides to the Rerun visualizer configuration."""
    web_port = args_cli.rerun_web_port
    grpc_port = args_cli.rerun_grpc_port
    if web_port is None and grpc_port is None:
        return

    visualizer_cfgs = env_cfg.sim.visualizer_cfgs
    if visualizer_cfgs is None:
        visualizer_cfgs = []
    elif not isinstance(visualizer_cfgs, list):
        visualizer_cfgs = [visualizer_cfgs]

    rerun_cfg = next(
        (cfg for cfg in visualizer_cfgs if getattr(cfg, "visualizer_type", None) == "rerun"),
        None,
    )
    if rerun_cfg is None:
        # Keep the optional visualizer import out of play.py and defer it until Isaac Sim is running.
        from isaaclab_visualizers.rerun import RerunVisualizerCfg

        rerun_cfg = RerunVisualizerCfg()
        visualizer_cfgs.append(rerun_cfg)

    if web_port is not None:
        rerun_cfg.web_port = web_port
    if grpc_port is not None:
        rerun_cfg.grpc_port = grpc_port

    if rerun_cfg.web_port == rerun_cfg.grpc_port:
        raise ValueError("Rerun Web Viewer and gRPC server ports must be different.")

    env_cfg.sim.visualizer_cfgs = visualizer_cfgs
