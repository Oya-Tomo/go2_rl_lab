"""Run an exported go2_rl_lab velocity policy through unitree_sdk2 low-level DDS.

The controller exchanges ``rt/lowstate`` and ``rt/lowcmd`` with unitree_mujoco's
simulate_python exactly as it would with the robot. The policy runs at the exported
control rate (50 Hz), while LowCmd repeats the latest joint targets at 500 Hz because
the simulator recomputes PD torque only when a command arrives.

Keys (focus the "Go2 SDK2 policy" window):
    1            passive -> stand up to the policy's default posture
    2            stand -> run the policy
    Space        passive damping from any state
    W/S A/D Q/E  forward, lateral and yaw velocity; hold Shift for full speed
    Esc          passive damping and quit
Unitree remote L2+B also enters passive damping from any state.

Simulator: start unitree_mujoco simulate_python first (DOMAIN_ID = 1, INTERFACE = "lo"), then
    ./go2_rl_lab.sh --deploy-sdk2 --policy /path/to/exported/policy.pt

Real robot: pass the interface connected to the robot (domain 0 by default). After a typed
confirmation the script lies the robot down and releases the active motion-control mode
(sport service), so the official controller no longer fights LowCmd. It does not restore that
mode on exit; reselect it with the app or reboot the robot.
    ./go2_rl_lab.sh --deploy-sdk2 --interface enp3s0 --policy /path/to/exported/policy.pt
"""

from __future__ import annotations

import argparse
import math
import threading
import time
from pathlib import Path

import numpy as np
import pygame
import torch
import yaml
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.go2.sport.sport_client import SportClient
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_, SportModeState_
from unitree_sdk2py.utils.crc import CRC

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
# Policy observation, action and gain settings are shared with the MuJoCo deployment.
DEFAULT_DEPLOY_CONFIG = REPOSITORY_ROOT / "deploy/deploy_mujoco/config.yaml"

# unitree_sdk2 motor order of LowState/LowCmd.
SDK_JOINT_NAMES = [f"{leg}_{part}_joint" for leg in ("FR", "FL", "RR", "RL") for part in ("hip", "thigh", "calf")]
NUM_SDK_MOTORS = 20

LOWCMD_DT = 0.002
STAND_DURATION = 2.0
STAND_KP = 50.0
STAND_KD = 3.5
PASSIVE_KD = 2.0
# Leave the policy when projected gravity z rises above this value (base tilt beyond ~60 deg).
MAX_TILT_GRAVITY_Z = -0.5
LOWSTATE_TIMEOUT = 0.1
SHUTDOWN_PASSIVE_DURATION = 0.5
KEYBOARD_COMMAND_FRACTION = 0.5

SIMULATOR_INTERFACE = "lo"
SIMULATOR_DOMAIN_ID = 1
ROBOT_DOMAIN_ID = 0
# LowState.wireless_remote button bits: byte 2 bit 5 is L2, byte 3 bit 1 is B.
REMOTE_L2 = (2, 5)
REMOTE_B = (3, 1)
SERVICE_TIMEOUT = 5.0
STAND_DOWN_WAIT = 3.0
MODE_RELEASE_ATTEMPTS = 5


def get_gravity_orientation(quaternion: np.ndarray) -> np.ndarray:
    """Return the gravity direction expressed in the robot frame for a wxyz quaternion."""
    qw, qx, qy, qz = quaternion
    return np.array(
        [2.0 * (-qz * qx + qw * qy), -2.0 * (qz * qy + qw * qx), 1.0 - 2.0 * (qw * qw + qz * qz)],
        dtype=np.float32,
    )


def quat_rotate_inverse(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Rotate a world-frame vector into the frame of a wxyz quaternion."""
    w, xyz = quaternion[0], quaternion[1:]
    return vector * (2.0 * w**2 - 1.0) - np.cross(xyz, vector) * w * 2.0 + xyz * np.dot(xyz, vector) * 2.0


def remote_passive_requested(low_state: LowState_) -> bool:
    """Return whether L2+B is held on the Unitree wireless remote."""
    remote = low_state.wireless_remote
    return all((remote[byte] >> bit) & 1 for byte, bit in (REMOTE_L2, REMOTE_B))


def release_motion_control() -> str:
    """Lie the robot down and release its active motion-control mode, returning the released mode name."""
    sport = SportClient()
    sport.SetTimeout(SERVICE_TIMEOUT)
    sport.Init()
    switcher = MotionSwitcherClient()
    switcher.SetTimeout(SERVICE_TIMEOUT)
    switcher.Init()

    code, mode = switcher.CheckMode()
    if code != 0:
        raise SystemExit(f"motion_switcher service did not answer (code {code}); check the interface and robot.")
    released = mode["name"]
    for _ in range(MODE_RELEASE_ATTEMPTS):
        if not mode["name"]:
            return released
        print(f"[INFO] Lying down and releasing motion-control mode '{mode['name']}'")
        sport.StandDown()
        time.sleep(STAND_DOWN_WAIT)
        switcher.ReleaseMode()
        time.sleep(1.0)
        code, mode = switcher.CheckMode()
        if code != 0:
            raise SystemExit(f"motion_switcher service did not answer (code {code}) while releasing.")
    raise SystemExit(f"Motion-control mode '{mode['name']}' is still active; refusing to send LowCmd.")


class StateBuffer:
    """Keep the latest LowState and SportModeState samples delivered by DDS callbacks."""

    def __init__(self):
        self._lock = threading.Lock()
        self._low_state: LowState_ | None = None
        self._low_state_time = -math.inf
        self._sport_state: SportModeState_ | None = None

    def on_low_state(self, message: LowState_) -> None:
        with self._lock:
            self._low_state = message
            self._low_state_time = time.perf_counter()

    def on_sport_state(self, message: SportModeState_) -> None:
        with self._lock:
            self._sport_state = message

    def latest(self) -> tuple[LowState_ | None, float, SportModeState_ | None]:
        """Return the latest LowState, its age in seconds and the latest SportModeState."""
        with self._lock:
            return self._low_state, time.perf_counter() - self._low_state_time, self._sport_state


class KeyboardInput:
    """Read transition keys and held WASD/QE velocity keys from a small pygame window."""

    TRANSITIONS = {pygame.K_1: "stand", pygame.K_2: "policy", pygame.K_SPACE: "passive", pygame.K_ESCAPE: "quit"}

    def __init__(self):
        pygame.init()
        self._screen = pygame.display.set_mode((440, 110))
        pygame.display.set_caption("Go2 SDK2 policy")
        self._font = pygame.font.Font(None, 22)

    def poll(self, maximum_command: np.ndarray) -> tuple[list[str], np.ndarray]:
        """Return requested transitions and the body-frame ``[vx, vy, yaw_rate]`` command."""
        requests = []
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                requests.append("quit")
            elif event.type == pygame.KEYDOWN and event.key in self.TRANSITIONS:
                requests.append(self.TRANSITIONS[event.key])
        keys = pygame.key.get_pressed()
        direction = np.array(
            [keys[pygame.K_w] - keys[pygame.K_s], keys[pygame.K_a] - keys[pygame.K_d], keys[pygame.K_q] - keys[pygame.K_e]],
            dtype=np.float32,
        )
        full_speed = pygame.key.get_mods() & pygame.KMOD_SHIFT
        return requests, direction * maximum_command * (1.0 if full_speed else KEYBOARD_COMMAND_FRACTION)

    def show(self, status: str) -> None:
        """Draw the key help and the controller status."""
        self._screen.fill((30, 30, 30))
        lines = ("1: stand   2: policy   Space: passive   Esc: quit", "W/S A/D Q/E: velocity, Shift: full speed", status)
        for row, text in enumerate(lines):
            self._screen.blit(self._font.render(text, True, (230, 230, 230)), (12, 14 + 30 * row))
        pygame.display.flip()


class PolicyController:
    """Passive/stand/policy state machine producing SDK-ordered joint targets."""

    def __init__(self, config: dict, policy_path: Path):
        policy_joint_names = list(config["policy_joint_names"])
        if sorted(policy_joint_names) != sorted(SDK_JOINT_NAMES):
            raise ValueError("policy_joint_names must contain the twelve Go2 leg joints")
        self.policy_from_sdk = np.array([SDK_JOINT_NAMES.index(name) for name in policy_joint_names])
        self.sdk_from_policy = np.array([policy_joint_names.index(name) for name in SDK_JOINT_NAMES])

        self.control_dt = float(config["simulation_dt"]) * int(config["control_decimation"])
        self.default_policy = np.asarray(config["default_angles"], dtype=np.float32)
        self.policy_kp_sdk = np.asarray(config["kps"], dtype=np.float32)[self.sdk_from_policy]
        self.policy_kd_sdk = np.asarray(config["kds"], dtype=np.float32)[self.sdk_from_policy]
        self.ang_vel_scale = float(config["ang_vel_scale"])
        self.dof_pos_scale = float(config["dof_pos_scale"])
        self.dof_vel_scale = float(config["dof_vel_scale"])
        self.command_scale = np.asarray(config["command_scale"], dtype=np.float32)
        self.action_scale = float(config["action_scale"])
        self.action_clip = float(config["action_clip"])
        num_actions = int(config["num_actions"])
        if int(config["num_obs"]) != 9 + 3 * num_actions:
            raise ValueError("Expected num_obs = 9 + 3 * num_actions for the current policy observation layout")

        self.policy = torch.jit.load(str(policy_path), map_location="cpu")
        self.policy.eval()
        self.action = np.zeros(num_actions, dtype=np.float32)
        self.observation = np.zeros(int(config["num_obs"]), dtype=np.float32)
        self.observation_tensor = torch.from_numpy(self.observation).unsqueeze(0)
        with torch.inference_mode():
            for _ in range(10):
                self.policy(self.observation_tensor)
        self._reset_policy()

        self.state = "passive"
        self.target_q = np.zeros(len(SDK_JOINT_NAMES), dtype=np.float32)
        self.kp = np.zeros(len(SDK_JOINT_NAMES), dtype=np.float32)
        self.kd = np.full(len(SDK_JOINT_NAMES), PASSIVE_KD, dtype=np.float32)
        self.gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self._stand_start_q = self.target_q.copy()
        self._stand_elapsed = 0.0

    @property
    def stand_finished(self) -> bool:
        return self.state == "stand" and self._stand_elapsed >= STAND_DURATION

    def _reset_policy(self) -> None:
        self.action.fill(0.0)
        self.observation.fill(0.0)
        if hasattr(self.policy, "reset"):
            self.policy.reset()

    def request(self, transition: str, low_state: LowState_) -> str | None:
        """Apply one requested transition, returning a message when it is refused."""
        if transition == "passive":
            self.state = "passive"
        elif transition == "stand":
            if self.state != "passive":
                return "stand is only available from passive"
            self._stand_start_q = np.array([m.q for m in low_state.motor_state[:12]], dtype=np.float32)
            self._stand_elapsed = 0.0
            self.state = "stand"
        elif transition == "policy":
            if not self.stand_finished:
                return "policy starts only after standing up"
            self._reset_policy()
            self.state = "policy"
        return None

    def update(self, low_state: LowState_, command: np.ndarray) -> str | None:
        """Advance one control step, returning a message when a safety stop triggers."""
        q_sdk = np.array([m.q for m in low_state.motor_state[:12]], dtype=np.float32)
        dq_sdk = np.array([m.dq for m in low_state.motor_state[:12]], dtype=np.float32)
        quaternion = np.array(low_state.imu_state.quaternion, dtype=np.float32)
        self.gravity = get_gravity_orientation(quaternion)

        if self.state == "policy" and self.gravity[2] > MAX_TILT_GRAVITY_Z:
            self.state = "passive"
            message = "base tilt limit exceeded; switched to passive"
        else:
            message = None

        if self.state == "passive":
            self.target_q[:] = q_sdk
            self.kp.fill(0.0)
            self.kd.fill(PASSIVE_KD)
        elif self.state == "stand":
            self._stand_elapsed += self.control_dt
            alpha = min(self._stand_elapsed / STAND_DURATION, 1.0)
            default_sdk = self.default_policy[self.sdk_from_policy]
            self.target_q[:] = (1.0 - alpha) * self._stand_start_q + alpha * default_sdk
            self.kp.fill(STAND_KP)
            self.kd.fill(STAND_KD)
        else:
            observation = self.observation
            observation[0:3] = np.array(low_state.imu_state.gyroscope, dtype=np.float32) * self.ang_vel_scale
            observation[3:6] = self.gravity
            observation[6:9] = command * self.command_scale
            observation[9:21] = (q_sdk[self.policy_from_sdk] - self.default_policy) * self.dof_pos_scale
            observation[21:33] = dq_sdk[self.policy_from_sdk] * self.dof_vel_scale
            observation[33:45] = self.action
            with torch.inference_mode():
                output = self.policy(self.observation_tensor)
            output = output[0] if isinstance(output, tuple) else output
            self.action[:] = np.clip(output.numpy().reshape(-1), -self.action_clip, self.action_clip)
            target_policy = self.default_policy + self.action * self.action_scale
            self.target_q[:] = target_policy[self.sdk_from_policy]
            self.kp[:] = self.policy_kp_sdk
            self.kd[:] = self.policy_kd_sdk
        return message

    def enter_passive(self) -> None:
        """Switch to damping immediately without needing a fresh LowState."""
        self.state = "passive"
        self.kp.fill(0.0)
        self.kd.fill(PASSIVE_KD)

    def fill_low_cmd(self, low_cmd: LowCmd_) -> None:
        for i in range(len(SDK_JOINT_NAMES)):
            motor = low_cmd.motor_cmd[i]
            motor.q = float(self.target_q[i])
            motor.dq = 0.0
            motor.kp = float(self.kp[i])
            motor.kd = float(self.kd[i])
            motor.tau = 0.0


def create_low_cmd() -> LowCmd_:
    low_cmd = unitree_go_msg_dds__LowCmd_()
    low_cmd.head[0] = 0xFE
    low_cmd.head[1] = 0xEF
    low_cmd.level_flag = 0xFF
    low_cmd.gpio = 0
    for i in range(NUM_SDK_MOTORS):
        low_cmd.motor_cmd[i].mode = 0x01
        low_cmd.motor_cmd[i].q = 0.0
        low_cmd.motor_cmd[i].dq = 0.0
        low_cmd.motor_cmd[i].kp = 0.0
        low_cmd.motor_cmd[i].kd = 0.0
        low_cmd.motor_cmd[i].tau = 0.0
    return low_cmd


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a go2_rl_lab policy over unitree_sdk2 low-level DDS.")
    parser.add_argument("--policy", type=Path, default=None, help="Exported TorchScript policy.pt; defaults to the config policy_path.")
    parser.add_argument("--config", type=Path, default=DEFAULT_DEPLOY_CONFIG, help="Deployment config shared with deploy_mujoco.")
    parser.add_argument("--interface", type=str, default=SIMULATOR_INTERFACE, help="'lo' for the simulator, else the robot NIC.")
    parser.add_argument("--domain-id", type=int, default=None, help="Defaults to 1 for the simulator and 0 for the robot.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    real_robot = args.interface != SIMULATOR_INTERFACE
    domain_id = args.domain_id if args.domain_id is not None else (ROBOT_DOMAIN_ID if real_robot else SIMULATOR_DOMAIN_ID)

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    policy_path = args.policy or REPOSITORY_ROOT / config["policy_path"]
    torch.set_num_threads(1)
    controller = PolicyController(config, policy_path)
    maximum_command = np.asarray(config["max_velocity_command"], dtype=np.float32)
    policy_decimation = round(controller.control_dt / LOWCMD_DT)
    print(f"[INFO] Policy: {policy_path}")
    print(f"[INFO] LowCmd {1.0 / LOWCMD_DT:.0f} Hz, policy {1.0 / controller.control_dt:.0f} Hz")

    print(f"[INFO] {'REAL ROBOT' if real_robot else 'Simulator'}: interface {args.interface}, domain {domain_id}")

    ChannelFactoryInitialize(domain_id, args.interface)
    states = StateBuffer()
    low_state_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
    low_state_subscriber.Init(states.on_low_state, 10)
    sport_state_subscriber = ChannelSubscriber("rt/sportmodestate", SportModeState_)
    sport_state_subscriber.Init(states.on_sport_state, 10)

    # Confirm the DDS link before touching robot services or publishing LowCmd.
    deadline = time.perf_counter() + 5.0
    while states.latest()[0] is None:
        if time.perf_counter() > deadline:
            hint = f"check that {args.interface} is up on 192.168.123.0/24 and the robot is on" if real_robot else (
                "start unitree_mujoco simulate_python first")
            raise SystemExit(f"No rt/lowstate received; {hint}.")
        time.sleep(0.01)

    released_mode = None
    if real_robot:
        answer = input("The robot will lie down and its motion-control mode will be released. "
                       "Clear the area, keep the remote ready (L2+B = damping). Type 'yes' to continue: ")
        if answer.strip() != "yes":
            raise SystemExit("Aborted.")
        released_mode = release_motion_control()
    low_cmd_publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
    low_cmd_publisher.Init()

    keyboard = KeyboardInput()
    low_cmd = create_low_cmd()
    crc = CRC()
    command = np.zeros(3, dtype=np.float32)
    tick = 0
    next_tick = time.perf_counter()

    def write_low_cmd() -> None:
        controller.fill_low_cmd(low_cmd)
        low_cmd.crc = crc.Crc(low_cmd)
        low_cmd_publisher.Write(low_cmd)

    try:
        while True:
            if tick % policy_decimation == 0:
                low_state, low_state_age, sport_state = states.latest()
                requests, command = keyboard.poll(maximum_command)
                if "quit" in requests:
                    break
                messages = [controller.request(request, low_state) for request in requests]
                if controller.state != "passive" and remote_passive_requested(low_state):
                    controller.enter_passive()
                    messages.append("remote L2+B pressed; switched to passive")
                if low_state_age > LOWSTATE_TIMEOUT and controller.state != "passive":
                    controller.enter_passive()
                    messages.append("rt/lowstate timed out; switched to passive")
                messages.append(controller.update(low_state, command))
                for message in filter(None, messages):
                    print(f"\n[WARN] {message}")

                if tick % (policy_decimation * 10) == 0:
                    velocity = np.zeros(3, dtype=np.float32)
                    if sport_state is not None:
                        quaternion = np.array(low_state.imu_state.quaternion, dtype=np.float32)
                        velocity = quat_rotate_inverse(quaternion, np.array(sport_state.velocity, dtype=np.float32))
                    tilt = math.degrees(math.acos(float(np.clip(-controller.gravity[2], -1.0, 1.0))))
                    status = (
                        f"{controller.state.upper():7s} cmd [{command[0]:+.2f} {command[1]:+.2f} {command[2]:+.2f}]  "
                        f"vel [{velocity[0]:+.2f} {velocity[1]:+.2f}]  tilt {tilt:4.1f} deg"
                    )
                    keyboard.show(status)
                    print(f"\r{status}", end="", flush=True)

            write_low_cmd()
            tick += 1
            next_tick += LOWCMD_DT
            remaining = next_tick - time.perf_counter()
            if remaining > 0.0:
                time.sleep(remaining)
            elif remaining < -0.1:
                next_tick = time.perf_counter()
    finally:
        print()
        controller.enter_passive()
        for _ in range(round(SHUTDOWN_PASSIVE_DURATION / LOWCMD_DT)):
            write_low_cmd()
            time.sleep(LOWCMD_DT)
        pygame.quit()
        if released_mode:
            print(f"[INFO] Motion-control mode '{released_mode}' is still released; reselect it in the app or reboot.")


if __name__ == "__main__":
    main()
