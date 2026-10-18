"""Run an exported Go2 velocity policy in MuJoCo.

python deploy/deploy_mujoco/deploy_go2.py \
    --policy logs/rsl_rl/go2_mixed/<run>/exported/policy.pt \
    --terrain flat --command 1.0 0.0 0.0

The policy consumes one 45-dimensional proprioceptive frame and maintains its
10-frame observation history internally. Commands are body-frame linear
velocities and yaw rate [vx, vy, yaw_rate] in [m/s, m/s, rad/s]. The gamepad's
left stick controls forward/lateral velocity and its right stick controls yaw.
Use --no-joystick for scripted commands and --headless --no-real-time for batch
runs. Set MUJOCO_GL=egl when recording an offscreen video.
CPU inference uses one thread by default; tune it with --inference-threads.
The interactive viewer refreshes at --viewer-fps (default 60) independently of physics.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import torch

from utils.mujoco import (
    DEPLOY_DIR,
    clip_torque_speed,
    configure_mujoco,
    extract_policy_action,
    get_gravity_orientation,
    get_joint_indices,
    get_velocity_joystick_command,
    initialize_joystick,
    load_config,
    pd_control,
    quat_rotate_inverse,
    resolve_model_path,
    resolve_policy_path,
)
from utils.render_utils import MujocoRenderUtils


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the velocity deployment command-line arguments."""
    parser = argparse.ArgumentParser(description="Run an exported Go2 velocity policy in MuJoCo.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--policy", type=str, default=None, help="Override config policy_path.")
    parser.add_argument("--terrain", type=str, default=None, help="Terrain name from the config.")
    parser.add_argument("--xml", type=str, default=None, help="Use a custom MuJoCo scene XML.")
    parser.add_argument(
        "--command",
        type=float,
        nargs=3,
        metavar=("VX", "VY", "YAW_RATE"),
        help="Body-frame velocity [vx, vy, yaw_rate] in [m/s, m/s, rad/s].",
    )
    parser.add_argument("--duration", type=float, default=None, help="Maximum wall-clock duration in seconds.")
    parser.add_argument("--no-joystick", action="store_true", help="Ignore the joystick and use the config command.")
    parser.add_argument("--no-real-time", action="store_true")
    parser.add_argument("--inference-threads", type=int, default=1, help="CPU threads for policy inference.")
    parser.add_argument("--viewer-fps", type=float, default=60.0, help="Maximum interactive viewer refresh rate.")
    parser.add_argument("--headless", action="store_true", help="Run without the interactive MuJoCo viewer.")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--video-path", type=Path, default=None)
    return parser


class VelocityDeployment:
    """Run an exported proprioceptive velocity policy with the training motor model."""

    def __init__(self, args: argparse.Namespace, config_path: Path) -> None:
        """Retain the deployment arguments and configuration path."""
        self.args = args
        self.config_path = config_path

    def run(self) -> None:
        """Run physics and policy control with independently paced visualization."""
        args = self.args
        if args.inference_threads < 1:
            raise ValueError("inference_threads must be at least one")
        if not math.isfinite(args.viewer_fps) or args.viewer_fps <= 0.0:
            raise ValueError("viewer_fps must be finite and positive")
        torch.set_num_threads(args.inference_threads)
        config = load_config(self.config_path)
        command_kind = str(config.get("command_kind", ""))
        if command_kind != "command/body/velocity":
            raise ValueError(f"Expected command_kind=command/body/velocity, got {command_kind!r}.")
        configured_command = np.asarray(
            args.command if args.command is not None else config["command"], dtype=np.float32
        )
        maximum_velocity_command = np.asarray(config["max_velocity_command"], dtype=np.float32)
        for name, value in (
            ("command", configured_command),
            ("max_velocity_command", maximum_velocity_command),
        ):
            if value.shape != (3,) or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must contain three finite values")
        if np.any(maximum_velocity_command < 0.0):
            raise ValueError("max_velocity_command must contain nonnegative velocity limits")

        raw_policy_path = args.policy or config["policy_path"]
        if not raw_policy_path:
            raise ValueError("Set policy_path in the config or pass --policy /path/to/exported/policy.pt")
        policy_path = resolve_policy_path(raw_policy_path, self.config_path)
        if not policy_path.is_file():
            raise FileNotFoundError(f"Policy does not exist: {policy_path}")

        terrain = args.terrain or config["terrain"]
        xml_path = resolve_model_path(config, terrain, args.xml)
        if not xml_path.is_file():
            raise FileNotFoundError(f"MuJoCo model does not exist: {xml_path}")

        model = mujoco.MjModel.from_xml_path(str(xml_path))
        data = mujoco.MjData(model)
        configure_mujoco(model, config)

        num_actions = int(config["num_actions"])
        num_observations = int(config["num_obs"])
        if num_observations != 9 + 3 * num_actions:
            raise ValueError("Expected num_obs = 9 + 3 * num_actions for the current policy observation layout")

        mujoco_joint_names = list(config["mujoco_joint_names"])
        policy_joint_names = list(config["policy_joint_names"])
        if set(mujoco_joint_names) != set(policy_joint_names) or len(mujoco_joint_names) != num_actions:
            raise ValueError("MuJoCo and policy joint-name lists must contain the same num_actions joints")

        policy_from_mujoco = np.array([mujoco_joint_names.index(name) for name in policy_joint_names])
        mujoco_from_policy = np.array([policy_joint_names.index(name) for name in mujoco_joint_names])
        qpos_indices, qvel_indices, actuator_indices = get_joint_indices(model, mujoco_joint_names)

        stiffness_policy = np.asarray(config["kps"], dtype=np.float32)
        damping_policy = np.asarray(config["kds"], dtype=np.float32)
        default_position_policy = np.asarray(config["default_angles"], dtype=np.float32)
        for name, value in (
            ("kps", stiffness_policy),
            ("kds", damping_policy),
            ("default_angles", default_position_policy),
        ):
            if value.shape != (num_actions,):
                raise ValueError(f"{name} must contain {num_actions} values")

        stiffness_mujoco = stiffness_policy[mujoco_from_policy]
        damping_mujoco = damping_policy[mujoco_from_policy]
        default_position_mujoco = default_position_policy[mujoco_from_policy]
        data.qpos[qpos_indices] = default_position_mujoco
        mujoco.mj_forward(model, data)

        policy = torch.jit.load(str(policy_path), map_location="cpu")
        policy.eval()
        if hasattr(policy, "reset"):
            policy.reset()

        command_scale = np.asarray(config["command_scale"], dtype=np.float32)
        if command_scale.shape != (3,) or not np.all(np.isfinite(command_scale)):
            raise ValueError("command_scale must contain three finite values.")

        action = np.zeros(num_actions, dtype=np.float32)
        target_position_mujoco = default_position_mujoco.copy()
        observation = np.zeros(num_observations, dtype=np.float32)
        observation_tensor = torch.from_numpy(observation).unsqueeze(0)
        # Compile/warm the stateful export before starting the simulation clock.
        # Restore history so warmup observations cannot affect the first action.
        if hasattr(policy, "reset"):
            with torch.inference_mode():
                for _ in range(10):
                    policy(observation_tensor)
                policy.reset()

        control_decimation = int(config["control_decimation"])
        if control_decimation < 1:
            raise ValueError("control_decimation must be at least one")
        control_dt = model.opt.timestep * control_decimation
        duration = float(args.duration if args.duration is not None else config["simulation_duration"])
        real_time = bool(config["real_time"]) and not args.no_real_time
        pygame, joystick = initialize_joystick(bool(config["joystick"]) and not args.no_joystick)
        command_source = "joystick" if joystick is not None else "config"

        def update_command() -> np.ndarray:
            """Sample body-frame velocity from the joystick or scripted command."""
            if joystick is None:
                return configured_command
            return get_velocity_joystick_command(pygame, joystick, maximum_velocity_command)

        command = update_command()
        render_utils = MujocoRenderUtils()

        video_renderer = None
        video_writer = None
        video_frame_interval = 1
        if args.save_video:
            import imageio.v2 as imageio

            video_path = args.video_path or DEPLOY_DIR / "videos" / f"{policy_path.stem}_{terrain}.mp4"
            video_path = video_path.expanduser().resolve()
            video_path.parent.mkdir(parents=True, exist_ok=True)
            video_renderer = mujoco.Renderer(model, height=360, width=640)
            video_fps = 50
            video_frame_interval = max(1, round(1.0 / (model.opt.timestep * video_fps)))
            video_writer = imageio.get_writer(video_path, fps=video_fps)
            print(f"[INFO] Recording video to: {video_path}")

        print(f"[INFO] Policy: {policy_path}")
        print(f"[INFO] Terrain: {terrain} ({xml_path})")
        print(f"[INFO] Command kind: {command_kind}")
        print(f"[INFO] Command source: {command_source}")
        print(f"[INFO] Physics dt: {model.opt.timestep:.4f} s, control dt: {control_dt:.4f} s")
        print(f"[INFO] PD: Kp={stiffness_policy[0]:g}, Kd={damping_policy[0]:g}")
        print(f"[INFO] CPU inference threads: {torch.get_num_threads()}, viewer limit: {args.viewer_fps:g} Hz")

        viewer = None
        if not args.headless:
            viewer = mujoco.viewer.launch_passive(model, data)
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
            viewer.cam.distance = 2.0
            viewer.cam.elevation = -20.0
            viewer.cam.azimuth = 60.0

        def _reset_deployment_episode() -> None:
            """Reset simulation, policy history and cached command inputs together."""
            mujoco.mj_resetData(model, data)
            data.qpos[qpos_indices] = default_position_mujoco
            data.ctrl.fill(0.0)
            mujoco.mj_forward(model, data)

            action.fill(0.0)
            target_position_mujoco[:] = default_position_mujoco
            observation.fill(0.0)
            with torch.inference_mode():
                if hasattr(policy, "reset"):
                    policy.reset()

        counter = 0
        previous_sim_time = float(data.time)
        start_time = time.perf_counter()
        next_step_time = start_time
        next_viewer_time = start_time
        next_status_time = start_time
        viewer_interval = 1.0 / args.viewer_fps
        maximum_catchup = max(0.1, control_dt)
        try:
            while (viewer is None or viewer.is_running()) and time.perf_counter() - start_time < duration:
                joint_position = data.qpos[qpos_indices]
                joint_velocity = data.qvel[qvel_indices]
                effort = pd_control(
                    target_position_mujoco,
                    joint_position,
                    stiffness_mujoco,
                    joint_velocity,
                    damping_mujoco,
                )
                effort = clip_torque_speed(effort, joint_velocity, config["motor"])
                data.ctrl[actuator_indices] = effort
                mujoco.mj_step(model, data)
                post_step_time = float(data.time)

                render_utils.update(command, data)
                if viewer is not None and time.perf_counter() >= next_viewer_time:
                    with viewer.lock():
                        render_utils.update_external_rendering(viewer, target="viewer")
                    viewer.sync()
                    next_viewer_time += viewer_interval
                    if next_viewer_time < time.perf_counter():
                        next_viewer_time = time.perf_counter() + viewer_interval

                # The native viewer's Reset button rewinds data.time. Detect a
                # reset whether it occurred before mj_step or during viewer.sync().
                reset_detected = (
                    post_step_time + 1.0e-9 < previous_sim_time
                    or float(data.time) + 1.0e-9 < post_step_time
                )
                if reset_detected:
                    _reset_deployment_episode()
                    command = update_command()
                    counter = 0
                    previous_sim_time = float(data.time)
                    next_step_time = time.perf_counter()
                    next_viewer_time = next_step_time
                    print("\n[INFO] Viewer reset: simulation, policy state and command source restored.")
                    continue

                if video_writer is not None and video_renderer is not None and counter % video_frame_interval == 0:
                    camera = viewer.cam if viewer is not None else -1
                    video_renderer.update_scene(data, camera=camera)
                    render_utils.update_external_rendering(video_renderer, target="renderer")
                    video_writer.append_data(video_renderer.render())

                counter += 1
                if counter % control_decimation == 0:
                    joint_position_policy = data.qpos[qpos_indices][policy_from_mujoco]
                    joint_velocity_policy = data.qvel[qvel_indices][policy_from_mujoco]
                    observation[0:3] = data.qvel[3:6] * float(config["ang_vel_scale"])
                    observation[3:6] = get_gravity_orientation(data.qpos[3:7])
                    observation[6:9] = command * command_scale
                    observation[9 : 9 + num_actions] = (
                        joint_position_policy - default_position_policy
                    ) * float(config["dof_pos_scale"])
                    observation[9 + num_actions : 9 + 2 * num_actions] = (
                        joint_velocity_policy * float(config["dof_vel_scale"])
                    )
                    observation[9 + 2 * num_actions : 9 + 3 * num_actions] = action

                    with torch.inference_mode():
                        policy_output = policy(observation_tensor)
                    action = extract_policy_action(policy_output).detach().cpu().numpy().reshape(-1)
                    if action.shape != (num_actions,):
                        raise ValueError(f"Policy returned action shape {action.shape}, expected {(num_actions,)}")
                    action = np.clip(action, -float(config["action_clip"]), float(config["action_clip"]))
                    target_position_policy = default_position_policy + action * float(config["action_scale"])
                    target_position_mujoco = target_position_policy[mujoco_from_policy]

                    command = update_command()
                if time.perf_counter() >= next_status_time:
                    next_status_time = time.perf_counter() + 0.2
                    quaternion = data.qpos[3:7]
                    local_velocity = quat_rotate_inverse(quaternion, data.qvel[:3])
                    qw, qx, qy, qz = quaternion
                    heading = math.atan2(
                        2.0 * (qw * qz + qx * qy),
                        1.0 - 2.0 * (qy * qy + qz * qz),
                    )
                    print(
                        f"\r{command_source} command "
                        f"[{command[0]:+.2f}, {command[1]:+.2f}, {command[2]:+.2f}]  "
                        f"Velocity [{local_velocity[0]:+.2f}, {local_velocity[1]:+.2f}]  "
                        f"Heading {math.degrees(heading):+6.1f} deg",
                        end="",
                    )

                previous_sim_time = float(data.time)

                if real_time:
                    next_step_time += model.opt.timestep
                    now = time.perf_counter()
                    # Recover from ordinary overruns without accumulating drift;
                    # discard excessive backlog after pauses or long GUI stalls.
                    if now - next_step_time > maximum_catchup:
                        next_step_time = now
                    remaining = next_step_time - now
                    if remaining > 0.0:
                        time.sleep(remaining)
        finally:
            print()
            if viewer is not None:
                viewer.close()
            if video_writer is not None:
                video_writer.close()
            if video_renderer is not None:
                video_renderer.close()
            if pygame is not None:
                pygame.quit()


def main() -> None:
    """Parse CLI arguments and launch velocity deployment."""
    args = build_arg_parser().parse_args()
    VelocityDeployment(args, args.config.expanduser().resolve()).run()


if __name__ == "__main__":
    main()
