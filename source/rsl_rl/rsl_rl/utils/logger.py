# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import git
import os
import pathlib
import statistics
import torch
from collections import deque

import rsl_rl
from rsl_rl.utils.metric_averager import MetricAverager
from rsl_rl.utils.tensorboard_utils import TensorboardSummaryWriter

# External logging services that receive the run configuration and git diffs.
_EXTERNAL_LOGGERS = ("wandb", "swanlab")


def _format_duration(seconds: float) -> str:
    """Format a duration as HH:MM:SS without wrapping at 24 hours."""
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class Logger:
    """Logger to save the learning metrics to different logging services."""

    def __init__(
        self,
        log_dir: str | None,
        cfg: dict,
        env_cfg: dict | object,
        num_envs: int,
        is_distributed: bool,
        gpu_world_size: int,
        gpu_global_rank: int,
        device: str,
        env_groups: dict[str, torch.Tensor] | None = None,
    ) -> None:
        self.log_dir = log_dir
        self.cfg = cfg
        self.num_envs = num_envs
        self.gpu_world_size = gpu_world_size
        self.device = device
        self.git_status_repos = [rsl_rl.__file__]
        self.tot_timesteps = 0
        self.tot_time = 0
        self.log_interval = int(self.cfg.get("log_interval", 1))
        if self.log_interval < 1:
            raise ValueError(f"log_interval must be at least 1, got {self.log_interval}.")
        self.metric_averager = MetricAverager()
        self._extra_metric_labels: dict[str, str] = {}
        self._closed = False

        # Create buffers
        self.ep_extras = []
        self.rewbuffer = deque(maxlen=100)
        self.lenbuffer = deque(maxlen=100)
        self.cur_reward_sum = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.cur_episode_length = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.env_groups: dict[str, torch.Tensor] = {}
        self.group_rewbuffers: dict[str, deque] = {}
        self.group_lenbuffers: dict[str, deque] = {}
        for name, indices in (env_groups or {}).items():
            normalized_name = str(name).strip().lower()
            if not normalized_name:
                raise ValueError("Logger environment-group names cannot be empty.")
            group_indices = torch.as_tensor(indices, dtype=torch.long, device=self.device).reshape(-1)
            if group_indices.numel() == 0:
                raise ValueError(f"Logger environment group {normalized_name!r} cannot be empty.")
            if group_indices.min() < 0 or group_indices.max() >= self.num_envs:
                raise ValueError(
                    f"Logger environment group {normalized_name!r} contains indices outside [0, {self.num_envs})."
                )
            if group_indices.unique().numel() != group_indices.numel():
                raise ValueError(f"Logger environment group {normalized_name!r} contains duplicate indices.")
            self.env_groups[normalized_name] = group_indices
            self.group_rewbuffers[normalized_name] = deque(maxlen=100)
            self.group_lenbuffers[normalized_name] = deque(maxlen=100)

        # Decide whether to disable logging
        # Note: We only log from the process with rank 0 (main process)
        self.disable_logs = is_distributed and gpu_global_rank != 0

        # Initialize the writer
        self._prepare_logging_writer()

        # Log code state
        self._store_code_state()

        # Log configuration
        if self.writer and not self.disable_logs and self.logger_type in _EXTERNAL_LOGGERS:
            self.writer.store_config(env_cfg, self.cfg)

    def process_env_step(self, rewards: torch.Tensor, dones: torch.Tensor, extras: dict) -> None:
        """Add metrics from the environment step to the buffers."""
        if self.log_dir is not None and not self.disable_logs:
            if "episode" in extras:
                self.ep_extras.append(extras["episode"])
            elif "log" in extras:
                self.ep_extras.append(extras["log"])

            # Update rewards and episode length
            self.cur_reward_sum += rewards
            self.cur_episode_length += 1

            # Clear data for completed episodes
            done_flags = (dones > 0).reshape(self.num_envs, -1).any(dim=1)
            new_ids = done_flags.nonzero(as_tuple=False).squeeze(-1)
            self.rewbuffer.extend(self.cur_reward_sum[new_ids].cpu().numpy().reshape(-1).tolist())
            self.lenbuffer.extend(self.cur_episode_length[new_ids].cpu().numpy().reshape(-1).tolist())
            for name, group_indices in self.env_groups.items():
                group_done_ids = group_indices[done_flags[group_indices]]
                self.group_rewbuffers[name].extend(
                    self.cur_reward_sum[group_done_ids].cpu().numpy().reshape(-1).tolist()
                )
                self.group_lenbuffers[name].extend(
                    self.cur_episode_length[group_done_ids].cpu().numpy().reshape(-1).tolist()
                )
            self.cur_reward_sum[new_ids] = 0
            self.cur_episode_length[new_ids] = 0

    def log(
        self,
        it: int,
        start_it: int,
        total_it: int,
        collect_time: float,
        learn_time: float,
        loss_dict: dict,
        learning_rate: float,
        action_std: torch.Tensor | None,
        print_minimal: bool = False,
        width: int = 80,
        pad: int = 40,
    ) -> None:
        """Log the training metrics to the logging service and print them to the console."""
        if self.log_dir is None or self.disable_logs:
            return

        collection_size = self.cfg["num_steps_per_env"] * self.num_envs * self.gpu_world_size
        iteration_time = collect_time + learn_time
        self.tot_timesteps += collection_size
        self.tot_time += iteration_time

        metrics: dict[str, float | int | torch.Tensor] = {}

        # Reduce environment metrics immediately so full per-environment tensors
        # are not retained across the multi-iteration averaging window.
        if self.ep_extras:
            # Some environment metrics are emitted only when an episode has
            # reached the corresponding state. Process the union of keys so a
            # sparse metric is not lost when the first reset has no value.
            keys = dict.fromkeys(key for ep_info in self.ep_extras for key in ep_info)
            for key in keys:
                info_values = []
                for ep_info in self.ep_extras:
                    if key not in ep_info:
                        continue
                    value = ep_info[key]
                    if isinstance(value, torch.Tensor):
                        value = value.to(self.device)
                    else:
                        value = torch.as_tensor(value, device=self.device)
                    info_values.append(value.reshape(-1))
                if not info_values:
                    continue
                if key.startswith("Curriculum/terrain_levels/"):
                    tag = "Terrain_Levels/" + key.removeprefix("Curriculum/terrain_levels/")
                    label = f"{tag}:"
                elif key.startswith("Curriculum/reward_metrics/"):
                    tag = key.removeprefix("Curriculum/reward_metrics/")
                    label = f"{tag}:"
                elif "/" in key:
                    tag = key
                    label = f"{key}:"
                else:
                    tag = "Episode/" + key
                    label = f"Mean episode {key}:"
                metrics[tag] = torch.cat(info_values).mean()
                self._extra_metric_labels[tag] = label
        self.ep_extras.clear()

        for key, value in loss_dict.items():
            tag = key if "/" in key else f"Loss/{key}"
            metrics[tag] = value
        metrics["Loss/learning_rate"] = learning_rate
        if action_std is not None:
            metrics["Policy/mean_noise_std"] = action_std.mean()
        metrics["Perf/total_fps"] = collection_size / iteration_time
        metrics["Perf/collection_time"] = collect_time
        metrics["Perf/learning_time"] = learn_time
        metrics["Perf/total_iter_time"] = iteration_time

        if len(self.rewbuffer) > 0:
            metrics["Train/mean_reward"] = statistics.mean(self.rewbuffer)
            metrics["Train/mean_episode_length"] = statistics.mean(self.lenbuffer)
        for name in self.env_groups:
            if self.group_rewbuffers[name]:
                metrics[f"Train/mean_{name}_reward"] = statistics.mean(self.group_rewbuffers[name])
                metrics[f"Train/mean_{name}_episode_length"] = statistics.mean(self.group_lenbuffers[name])

        reductions = {tag: "max" for tag in metrics if tag.endswith("/max_level_reached")}
        self.metric_averager.update(metrics, reductions=reductions)

        # Keep the conventional zero-based first point without consuming the
        # first averaging window. The write at step ``log_interval`` below
        # still averages iterations 0 through ``log_interval - 1``.
        if start_it == 0 and it == 1:
            for tag, value in metrics.items():
                self.writer.add_scalar(tag, value, 0)

        done_it = it - start_it
        should_write = done_it % self.log_interval == 0 or it >= total_it
        if should_write:
            averaged_metrics = self.metric_averager.mean_and_reset()
            for tag, value in averaged_metrics.items():
                self.writer.add_scalar(tag, value, it)

            if self.logger_type != "wandb":
                if "Train/mean_reward" in averaged_metrics:
                    self.writer.add_scalar(
                        "Train/mean_reward/time", averaged_metrics["Train/mean_reward"], int(self.tot_time)
                    )
                if "Train/mean_episode_length" in averaged_metrics:
                    self.writer.add_scalar(
                        "Train/mean_episode_length/time",
                        averaged_metrics["Train/mean_episode_length"],
                        int(self.tot_time),
                    )
                for name in self.env_groups:
                    reward_tag = f"Train/mean_{name}_reward"
                    length_tag = f"Train/mean_{name}_episode_length"
                    if reward_tag in averaged_metrics:
                        self.writer.add_scalar(f"{reward_tag}/time", averaged_metrics[reward_tag], int(self.tot_time))
                    if length_tag in averaged_metrics:
                        self.writer.add_scalar(f"{length_tag}/time", averaged_metrics[length_tag], int(self.tot_time))

        extras_string = ""
        for tag, label in self._extra_metric_labels.items():
            if tag in metrics:
                extras_string += f"""{label:>{pad}} {torch.as_tensor(metrics[tag]).item():.4f}\n"""
        self._extra_metric_labels.clear()

        current_fps = float(metrics["Perf/total_fps"])
        current_collect_time = float(metrics["Perf/collection_time"])
        current_learn_time = float(metrics["Perf/learning_time"])

        # Console output reports every iteration; log_interval only controls
        # persistent logging (TensorBoard, W&B, or SwanLab).
        log_string = f"""{"#" * width}\n"""
        log_string += f"""\033[1m{f" Learning iteration {it}/{total_it} ".center(width)}\033[0m \n\n"""
        run_name = self.cfg.get("run_name")
        log_string += f"""{"Run name:":>{pad}} {run_name}\n""" if run_name else ""
        log_string += (
            f"""{"Total steps:":>{pad}} {self.tot_timesteps} \n"""
            f"""{"Steps per second:":>{pad}} {current_fps:.0f} \n"""
            f"""{"Collection time:":>{pad}} {current_collect_time:.3f}s \n"""
            f"""{"Learning time:":>{pad}} {current_learn_time:.3f}s \n"""
        )

        for key in loss_dict:
            tag = key if "/" in key else f"Loss/{key}"
            if tag in metrics:
                label = f"{key}:" if "/" in key else f"Mean {key} loss:"
                log_string += f"""{label:>{pad}} {torch.as_tensor(metrics[tag]).item():.4f}\n"""

        if "Train/mean_reward" in metrics:
            log_string += f"""{"Mean reward:":>{pad}} {float(metrics["Train/mean_reward"]):.2f}\n"""
        if "Train/mean_episode_length" in metrics:
            log_string += f"""{"Mean episode length:":>{pad}} {float(metrics["Train/mean_episode_length"]):.2f}\n"""
        for name in self.env_groups:
            reward_tag = f"Train/mean_{name}_reward"
            length_tag = f"Train/mean_{name}_episode_length"
            if reward_tag in metrics:
                label = f"Mean {name} reward:"
                log_string += f"""{label:>{pad}} {float(metrics[reward_tag]):.2f}\n"""
            if length_tag in metrics:
                label = f"Mean {name} episode length:"
                log_string += f"""{label:>{pad}} {float(metrics[length_tag]):.2f}\n"""

        if "Policy/mean_noise_std" in metrics:
            log_string += (
                f"""{"Mean action noise std:":>{pad}} """
                f"""{torch.as_tensor(metrics["Policy/mean_noise_std"]).item():.2f}\n"""
            )
        if not print_minimal:
            log_string += extras_string

        remaining_it = total_it - start_it - done_it
        eta = self.tot_time / done_it * remaining_it
        log_string += (
            f"""{"-" * width}\n"""
            f"""{"Iteration time:":>{pad}} {iteration_time:.2f}s\n"""
            f"""{"Time elapsed:":>{pad}} {_format_duration(self.tot_time)}\n"""
            f"""{"ETA:":>{pad}} {_format_duration(eta)}\n"""
        )
        print(log_string)

    def add_git_repo(self, repo_file_path: str) -> None:
        """Register a repository and immediately store its current Git state."""
        if repo_file_path not in self.git_status_repos:
            self.git_status_repos.append(repo_file_path)
        self._store_code_state()

    def close(self) -> None:
        """Stop the external logger and close its local summary writer once."""
        if self._closed:
            return
        self._closed = True

        if self.writer is None:
            return

        try:
            if self.logger_type in _EXTERNAL_LOGGERS:
                self.writer.stop()
        finally:
            self.writer.close()

    def _prepare_logging_writer(self) -> None:
        """Prepare the logging writer, which can be Tensorboard, W&B or SwanLab."""
        if self.log_dir is not None and not self.disable_logs:
            self.logger_type = self.cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()

            if self.logger_type == "wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter

                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
            elif self.logger_type == "swanlab":
                from rsl_rl.utils.swanlab_utils import SwanlabSummaryWriter

                self.writer = SwanlabSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
            elif self.logger_type == "tensorboard":
                self.writer = TensorboardSummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise ValueError("Logger type not found. Please choose 'wandb', 'swanlab', or 'tensorboard'.")
        else:
            self.writer = None

    def _store_code_state(self) -> None:
        """Store the current git diff of the code repositories involved in the experiment."""
        if self.log_dir is not None and not self.disable_logs:
            git_log_dir = os.path.join(self.log_dir, "git")
            os.makedirs(git_log_dir, exist_ok=True)
            file_paths = []
            # Iterate over all repositories to log
            for repository_file_path in self.git_status_repos:
                try:
                    repo = git.Repo(repository_file_path, search_parent_directories=True)
                    commit = repo.head.commit
                    commit_tree = commit.tree
                except Exception:
                    print(f"Could not find git repository in {repository_file_path}. Skipping.")
                    continue
                # Get the name of the repository
                repo_name = pathlib.Path(repo.working_dir).name
                diff_file_name = os.path.join(git_log_dir, f"{repo_name}.diff")
                # Check if the diff file already exists
                if os.path.isfile(diff_file_name):
                    continue
                if repo.head.is_detached:
                    branch_name = "(detached HEAD)"
                    upstream_name = "(none)"
                else:
                    branch = repo.active_branch
                    branch_name = branch.name
                    tracking_branch = branch.tracking_branch()
                    upstream_name = str(tracking_branch) if tracking_branch is not None else "(none)"
                # Write the diff file
                print(f"Storing git diff for '{repo_name}' in: {diff_file_name}")
                with open(diff_file_name, "x", encoding="utf-8") as f:
                    content = (
                        "--- git metadata ---\n"
                        f"repository: {pathlib.Path(repo.working_dir).resolve()}\n"
                        f"branch: {branch_name}\n"
                        f"upstream: {upstream_name}\n"
                        f"commit: {commit.hexsha}\n"
                        f"short_commit: {commit.hexsha[:12]}\n"
                        f"detached: {str(repo.head.is_detached).lower()}\n"
                        f"dirty: {str(repo.is_dirty(untracked_files=True)).lower()}\n"
                        f"author: {commit.author.name} <{commit.author.email}>\n"
                        f"author_date: {commit.authored_datetime.isoformat()}\n"
                        f"committer: {commit.committer.name} <{commit.committer.email}>\n"
                        f"commit_date: {commit.committed_datetime.isoformat()}\n"
                        f"message:\n{commit.message.rstrip()}\n\n\n"
                        f"--- git status ---\n{repo.git.status()}\n\n\n"
                        f"--- git diff (HEAD vs working tree) ---\n{repo.git.diff(commit_tree)}"
                    )
                    f.write(content)
                # Add the file path to the list of files to be uploaded
                file_paths.append(diff_file_name)

            # Upload diff files to external logging services
            if self.writer and self.logger_type in _EXTERNAL_LOGGERS and file_paths:
                for path in file_paths:
                    self.writer.save_file(path)
