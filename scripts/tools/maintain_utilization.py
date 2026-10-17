"""Occupy the free memory of the GPU(s) used by a training process.

Overview:
    Without ``--pid`` the script lists the processes of the target user (default:
    the current user) that currently hold a CUDA device file and lets you choose
    one interactively; every row shows the local pid, its ``CUDA_VISIBLE_DEVICES``
    value, the memory usage and the command line, sorted by ``CUDA_VISIBLE_DEVICES``
    ascending, and helper children spawned by another listed process are hidden
    (``--all`` lists every process of the user instead).
    The chosen process is mapped to its GPU through the ``CUDA_VISIBLE_DEVICES``
    value in ``/proc/<pid>/environ``; a host pid reported by nvidia-smi inside a
    container can also be passed with ``--pid`` and is then tracked through the
    nvidia-smi compute process list.  When no GPU can be resolved, the script asks
    for the GPU id through an interactive prompt (or accepts ``--gpu``).  The free
    memory of every resolved GPU is then held by this process, leaving only the
    configured reserve plus a small slack free.  Newly released memory is grabbed
    on every poll, and the occupation stops as soon as the target process exits.

Quick Start:
    python scripts/tools/maintain_utilization.py

Full Command:
    python scripts/tools/maintain_utilization.py --user wutianyang --gpu 2,3 \
        --reserve-mb 500 --slack-mb 64 --poll-interval 2.0

Options:
    --pid: Training process id (local pid or nvidia-smi host pid); omit to select interactively.
    --user: User whose processes are listed for interactive selection.
    --all: List every process of the user instead of only the GPU-holding processes.
    --gpu: Fallback GPU id list, used when the target GPU cannot be resolved automatically.
    --reserve-mb: Memory kept free on each occupied GPU.
    --slack-mb: Extra headroom kept on top of the reserve to avoid allocation failures.
    --poll-interval: Seconds between free-memory checks and process liveness checks.

Notes:
    Run this script on the same host as the training process, inside an environment
    with CUDA-enabled torch (for example the Isaac Lab container).  The interactive
    list marks a process as GPU-holding when it has an open ``/dev/nvidia*`` file
    descriptor; inside a container the listed pids are container pids, while a host
    pid reported by nvidia-smi is only accepted through ``--pid`` and its liveness
    is polled through nvidia-smi.  Ctrl-C releases the held memory early.  Because
    the occupation also covers the memory the training process may want to grow
    into later, the training run itself can run out of memory beyond the configured
    reserve.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import pwd
import signal
import subprocess
import sys
import threading
import time

DEFAULT_RESERVE_MB = 500
DEFAULT_SLACK_MB = 64
DEFAULT_POLL_INTERVAL = 2.0
DEFAULT_CHUNK_MB = 1024
MIN_CHUNK_MB = 64


class GpuMemoryProbe:
    """Read per-GPU memory counters from nvidia-smi."""

    @staticmethod
    def free_mb(gpu_id: str) -> float:
        """Return the free memory of one GPU in MiB.

        Args:
            gpu_id: GPU index or UUID understood by ``nvidia-smi -i``.

        Returns:
            The reported free memory in MiB.
        """
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits", "-i", gpu_id],
            check=True,
            capture_output=True,
            text=True,
        )
        return float(result.stdout.strip().splitlines()[0])

    @staticmethod
    def overview() -> str:
        """Return a compact per-GPU memory overview used by the interactive prompt."""
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.free", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    @staticmethod
    def gpu_memory_used() -> dict[str, int]:
        """Return the used memory per GPU in MiB, keyed by both index and UUID."""
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
        )
        usage = {}
        for line in result.stdout.strip().splitlines():
            index, uuid, used_mb = [field.strip() for field in line.split(",", 2)]
            usage[index] = int(used_mb)
            usage[uuid] = int(used_mb)
        return usage

    @staticmethod
    def compute_processes() -> list[dict[str, str]]:
        """Return the running compute processes reported by nvidia-smi.

        Returns:
            One dict per process with the keys ``pid``, ``gpu_uuid``, ``used_mb`` and ``name``.
        """
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,gpu_uuid,used_memory,process_name",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        processes = []
        for line in result.stdout.strip().splitlines():
            fields = [field.strip() for field in line.split(",", 3)]
            if len(fields) < 4 or not fields[0].isdigit():
                continue
            processes.append({"pid": fields[0], "gpu_uuid": fields[1], "used_mb": fields[2], "name": fields[3]})
        return processes


class ProcessWatcher:
    """Track the target training process and detect its exit or pid reuse."""

    def __init__(self, pid: int):
        self.pid = pid
        stat_fields = self._stat_fields()
        if stat_fields is None:
            raise SystemExit(missing_pid_message(pid))
        self.start_time = stat_fields[19]

    def _stat_fields(self) -> list[str] | None:
        """Return the whitespace-split fields of /proc/<pid>/stat after the command name."""
        try:
            stat_text = Path(f"/proc/{self.pid}/stat").read_text()
        except OSError:
            return None
        return stat_text[stat_text.rfind(")") + 2 :].split()

    def alive(self) -> bool:
        """Return True while the target process runs and has not been replaced by pid reuse."""
        stat_fields = self._stat_fields()
        if stat_fields is None or stat_fields[0] == "Z":
            return False
        return stat_fields[19] == self.start_time


def missing_pid_message(pid: int) -> str:
    """Explain a missing /proc entry, calling out host pids reported by nvidia-smi in containers.

    Args:
        pid: Target process id that is not part of this PID namespace.

    Returns:
        The error message shown to the user.
    """
    message = f"pid {pid} does not exist"
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return message
    compute_pids = {line.strip() for line in result.stdout.splitlines()}
    if str(pid) in compute_pids:
        return (
            f"pid {pid} is a host pid reported by nvidia-smi but is not visible in this PID namespace; "
            "it is not a running compute process either, rerun and select the training process again"
        )
    return message


@dataclass
class UserProcess:
    """One process owned by the target user, as seen from this /proc view."""

    pid: int
    cmdline: str
    cuda_visible_devices: str | None
    mem_mb: int | None = None
    mem_exact: bool = False


def cuda_sort_key(cuda_visible_devices: str | None) -> tuple[int, int, str]:
    """Build a sort key that orders numeric CUDA_VISIBLE_DEVICES values ascending and ids last.

    Args:
        cuda_visible_devices: Raw CUDA_VISIBLE_DEVICES value of a process.

    Returns:
        The sort key used to order the interactive listing.
    """
    if cuda_visible_devices is None:
        return (2, 0, "")
    token = cuda_visible_devices.split(",")[0].strip()
    if token.isdigit():
        return (0, int(token), cuda_visible_devices)
    return (1, 0, cuda_visible_devices)


class ComputeAppWatcher:
    """Track a GPU process through nvidia-smi when /proc is not shared with it."""

    def __init__(self, pid: int):
        self.pid = pid

    def alive(self) -> bool:
        """Return True while nvidia-smi still reports the pid as a compute process."""
        return any(int(process["pid"]) == self.pid for process in GpuMemoryProbe.compute_processes())


def read_uid(pid: int) -> int | None:
    """Read the real uid of a process.

    Args:
        pid: Target process id.

    Returns:
        The real uid, or None when the process is gone or unreadable.
    """
    try:
        status = Path(f"/proc/{pid}/status").read_text()
    except OSError:
        return None
    uid_line = next((line for line in status.splitlines() if line.startswith("Uid:")), "")
    uid_fields = uid_line.split()
    return int(uid_fields[1]) if len(uid_fields) >= 2 else None


def read_cmdline(pid: int) -> str | None:
    """Read the command line of a process.

    Args:
        pid: Target process id.

    Returns:
        The space-joined command line, or None when the process is gone or unreadable.
    """
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return None
    arguments = [argument.decode(errors="replace") for argument in cmdline.split(b"\0") if argument]
    return " ".join(arguments) if arguments else None


def holds_nvidia_fd(pid: int) -> bool:
    """Check whether a process holds an open CUDA device file.

    Args:
        pid: Target process id.

    Returns:
        True when any file descriptor points to ``/dev/nvidia*``.
    """
    try:
        entries = os.listdir(f"/proc/{pid}/fd")
    except OSError:
        return False
    for entry in entries:
        try:
            target = os.readlink(f"/proc/{pid}/fd/{entry}")
        except OSError:
            continue
        if target.startswith("/dev/nvidia"):
            return True
    return False


def read_ppid(pid: int) -> int | None:
    """Read the parent process id of a process.

    Args:
        pid: Target process id.

    Returns:
        The parent pid, or None when the process is gone or unreadable.
    """
    try:
        stat_text = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    fields = stat_text[stat_text.rfind(")") + 2 :].split()
    return int(fields[1]) if len(fields) > 1 else None


def ancestor_pids(pid: int) -> set[int]:
    """Collect the visible ancestor pids of a process.

    Args:
        pid: Target process id.

    Returns:
        The ancestor pids, excluding the process itself.
    """
    ancestors = set()
    parent = read_ppid(pid)
    while parent is not None and parent > 1:
        ancestors.add(parent)
        parent = read_ppid(parent)
    return ancestors


def read_env_value(pid: int, name: str) -> str | None:
    """Read one environment variable from a process environment.

    Args:
        pid: Target process id.
        name: Environment variable name.

    Returns:
        The variable value, or None when it is unset or the environment is unreadable.
    """
    try:
        environ = Path(f"/proc/{pid}/environ").read_bytes()
    except OSError:
        return None
    prefix = f"{name}=".encode()
    for entry in environ.split(b"\0"):
        if entry.startswith(prefix):
            return entry.split(b"=", 1)[1].decode(errors="replace").strip()
    return None


def read_cuda_visible_devices(pid: int) -> list[str] | None:
    """Read the usable GPU id list from the target process environment.

    Args:
        pid: Target process id.

    Returns:
        The parsed ``CUDA_VISIBLE_DEVICES`` entries, or None when the variable is
        absent or holds a non-GPU value such as ``-1`` or ``all``.
    """
    value = read_env_value(pid, "CUDA_VISIBLE_DEVICES")
    if value is None or value in {"", "-1", "all", "none"}:
        return None
    return [token.strip() for token in value.split(",") if token.strip()]


def resolve_uid(user: str | None) -> tuple[int, str]:
    """Resolve the target user name and uid, defaulting to the current user.

    Args:
        user: User name passed on the command line, or None for the current user.

    Returns:
        The numeric uid and the resolved user name.
    """
    if user is None:
        return os.getuid(), pwd.getpwuid(os.getuid()).pw_name
    try:
        record = pwd.getpwnam(user)
    except KeyError as error:
        raise SystemExit(f"user {user} does not exist") from error
    return record.pw_uid, record.pw_name


def list_user_processes(uid: int) -> list[UserProcess]:
    """List every process owned by the given uid in the current PID namespace.

    Args:
        uid: Numeric user id to match.

    Returns:
        The processes with a readable command line, sorted by pid.
    """
    processes = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        pid = int(entry)
        if read_uid(pid) != uid:
            continue
        command = read_cmdline(pid)
        if command is None:
            continue
        processes.append(UserProcess(pid, command, read_env_value(pid, "CUDA_VISIBLE_DEVICES")))
    return sorted(processes, key=lambda process: process.pid)


def list_gpu_processes(uid: int) -> list[UserProcess]:
    """List the local user processes that hold an open CUDA device file and set CUDA_VISIBLE_DEVICES.

    Processes spawned by another listed process (for example the wandb helper children
    of a training run) are skipped.  The memory value is the exact process usage when
    the pid is resolvable through nvidia-smi, otherwise the total used memory of the
    process GPU.

    Args:
        uid: Numeric user id to match.

    Returns:
        The GPU-holding processes sorted by CUDA_VISIBLE_DEVICES and pid.
    """
    used_by_gpu = GpuMemoryProbe.gpu_memory_used()
    used_by_pid = {int(row["pid"]): int(row["used_mb"]) for row in GpuMemoryProbe.compute_processes()}
    candidates = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        pid = int(entry)
        if read_uid(pid) != uid:
            continue
        cuda_visible_devices = read_env_value(pid, "CUDA_VISIBLE_DEVICES")
        if cuda_visible_devices is None or not holds_nvidia_fd(pid):
            continue
        command = read_cmdline(pid)
        if command is None:
            continue
        mem_mb = used_by_pid.get(pid)
        mem_exact = mem_mb is not None
        if mem_mb is None:
            mem_mb = used_by_gpu.get(cuda_visible_devices.split(",")[0].strip())
        candidates.append(UserProcess(pid, command, cuda_visible_devices, mem_mb, mem_exact))
    candidate_pids = {process.pid for process in candidates}
    processes = [
        process for process in candidates if not candidate_pids.intersection(ancestor_pids(process.pid))
    ]
    return sorted(processes, key=lambda process: (cuda_sort_key(process.cuda_visible_devices), process.pid))


def select_process(uid: int, user: str) -> int:
    """Interactively choose one process of the target user.

    Args:
        uid: Numeric user id to match.
        user: User name used for display and error messages.

    Returns:
        The pid selected by the user.
    """
    if not sys.stdin.isatty():
        raise SystemExit("interactive selection needs a tty; pass --pid instead")
    processes = list_user_processes(uid)
    if not processes:
        raise SystemExit(f"no process found for user {user}")
    print(f"Processes started by user {user}:")
    for index, process in enumerate(processes, start=1):
        cuda = process.cuda_visible_devices or "-"
        print(f"  [{index:>3}] pid={process.pid:<8} cuda={cuda:<12} {process.cmdline[:110]}")
    while True:
        entered = input("Select the process (index or pid, q to quit): ").strip()
        if entered.lower() in {"", "q", "quit"}:
            raise SystemExit("no process selected")
        if entered.isdigit():
            for process in processes:
                if process.pid == int(entered):
                    return process.pid
            if 1 <= int(entered) <= len(processes):
                return processes[int(entered) - 1].pid
        print("invalid selection")


def select_gpu_process(uid: int, user: str) -> UserProcess:
    """Interactively choose one of the user's processes that holds CUDA device files.

    Args:
        uid: Numeric user id to match.
        user: User name used in the listing header.

    Returns:
        The selected GPU-holding process.
    """
    if not sys.stdin.isatty():
        raise SystemExit("interactive selection needs a tty; pass --pid instead")
    processes = list_gpu_processes(uid)
    if not processes:
        raise SystemExit(f"no GPU-holding process found for user {user}; retry with --all")
    print(f"GPU-holding processes of user {user} (sorted by cuda; mem=process usage, ~mem=GPU total):")
    for index, process in enumerate(processes, start=1):
        cuda = process.cuda_visible_devices or "-"
        if process.mem_mb is None:
            mem = "-"
        elif process.mem_exact:
            mem = f"{process.mem_mb / 1024:.1f}G"
        else:
            mem = f"~{process.mem_mb / 1024:.1f}G"
        print(f"  [{index:>3}] pid={process.pid:<8} cuda={cuda:<12} mem={mem:>6}  {process.cmdline[:104]}")
    while True:
        entered = input("Select the process (index or pid, q to quit): ").strip()
        if entered.lower() in {"", "q", "quit"}:
            raise SystemExit("no process selected")
        if entered.isdigit():
            for process in processes:
                if process.pid == int(entered):
                    return process
            if 1 <= int(entered) <= len(processes):
                return processes[int(entered) - 1]
        print("invalid selection")


def prompt_gpu_ids(pid: int) -> list[str]:
    """Ask the user which GPU id(s) to occupy when CUDA_VISIBLE_DEVICES is unavailable.

    Args:
        pid: Target process id, shown for confirmation in the prompt.

    Returns:
        The comma-separated GPU ids entered by the user.
    """
    if not sys.stdin.isatty():
        raise SystemExit(f"pid {pid} has no readable CUDA_VISIBLE_DEVICES and stdin is not a tty; pass --gpu")
    print(f"pid {pid} has no usable CUDA_VISIBLE_DEVICES. Visible GPUs:")
    print(GpuMemoryProbe.overview())
    entered = input("Enter the GPU id(s) to occupy (comma separated): ").strip()
    gpu_ids = [token.strip() for token in entered.split(",") if token.strip()]
    if not gpu_ids:
        raise SystemExit("no GPU id entered")
    return gpu_ids


class GpuMemoryKeeper:
    """Hold the free memory of the resolved GPUs and re-grab memory released later."""

    def __init__(self, gpu_ids: list[str], reserve_mb: int, slack_mb: int, poll_interval: float):
        self.gpu_ids = gpu_ids
        self.devices = [f"cuda:{index}" for index in range(len(gpu_ids))]
        self.reserve_mb = reserve_mb
        self.slack_mb = slack_mb
        self.poll_interval = poll_interval
        self.probe = GpuMemoryProbe()
        self._held: dict[str, list] = {device: [] for device in self.devices}
        self._stop = threading.Event()

    def request_stop(self, *_) -> None:
        """Release the occupation at the next poll after SIGINT or SIGTERM."""
        self._stop.set()

    def run(self, watcher: ProcessWatcher | ComputeAppWatcher) -> None:
        """Occupy the resolved GPUs until the target process exits or a stop signal arrives.

        Args:
            watcher: Liveness tracker of the target process, local or nvidia-smi based.
        """
        try:
            import torch
        except ImportError as error:
            raise SystemExit("CUDA-enabled torch is required to occupy GPU memory") from error
        if torch.cuda.device_count() < len(self.gpu_ids):
            raise SystemExit(
                f"torch sees {torch.cuda.device_count()} CUDA device(s) but {len(self.gpu_ids)} "
                f"GPU id(s) were resolved: {self.gpu_ids}"
            )
        torch.cuda.init()
        self._torch = torch
        self._warmup()
        try:
            while watcher.alive() and not self._stop.is_set():
                for gpu_id, device in zip(self.gpu_ids, self.devices):
                    free_mb = self.probe.free_mb(gpu_id)
                    target_mb = free_mb - self.reserve_mb - self.slack_mb
                    if target_mb >= MIN_CHUNK_MB:
                        grabbed_mb = self._grab(device, target_mb)
                        if grabbed_mb > 0:
                            print(
                                f"[occupy] gpu {gpu_id}: grabbed {grabbed_mb:.0f} MiB (reported free {free_mb:.0f} MiB)",
                                flush=True,
                            )
                if self._stop.is_set() or not watcher.alive():
                    break
                time.sleep(self.poll_interval)
        finally:
            self._release()

    def _warmup(self) -> None:
        """Create the CUDA context on every device so context memory counts as already used."""
        for device in self.devices:
            tensor = self._torch.empty(1, dtype=self._torch.uint8, device=device)
            self._torch.cuda.synchronize(device)
            del tensor

    def _grab(self, device: str, target_mb: float) -> float:
        """Allocate up to target_mb on one device, halving the chunk size on allocation failure.

        Args:
            device: Torch device string such as ``cuda:0``.
            target_mb: Upper bound of the memory to allocate in MiB.

        Returns:
            The memory actually allocated in MiB.
        """
        grabbed_mb = 0.0
        remaining_mb = target_mb
        chunk_mb = min(DEFAULT_CHUNK_MB, remaining_mb)
        while chunk_mb >= MIN_CHUNK_MB and remaining_mb >= MIN_CHUNK_MB:
            chunk_mb = min(chunk_mb, remaining_mb)
            try:
                tensor = self._torch.empty(int(chunk_mb) * 1024 * 1024, dtype=self._torch.uint8, device=device)
            except (self._torch.cuda.OutOfMemoryError, RuntimeError):
                chunk_mb /= 2
                continue
            self._held[device].append(tensor)
            grabbed_mb += chunk_mb
            remaining_mb -= chunk_mb
        return grabbed_mb

    def _release(self) -> None:
        """Drop every held tensor and release the CUDA cache."""
        tensor_count = sum(len(tensors) for tensors in self._held.values())
        for tensors in self._held.values():
            tensors.clear()
        self._torch.cuda.empty_cache()
        print(f"[occupy] released {tensor_count} tensor(s)", flush=True)


def main() -> None:
    """Resolve the target GPU(s) from the training pid and keep their memory occupied."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--pid", type=int, default=None, help="training process id; omit to select interactively")
    parser.add_argument("--user", default=None, help="user whose processes are listed for selection")
    parser.add_argument("--all", action="store_true", help="list every process instead of only the GPU-holding processes")
    parser.add_argument("--gpu", default=None, help="fallback comma-separated GPU id list")
    parser.add_argument("--reserve-mb", type=int, default=DEFAULT_RESERVE_MB, help="memory kept free per GPU")
    parser.add_argument("--slack-mb", type=int, default=DEFAULT_SLACK_MB, help="extra headroom per GPU")
    parser.add_argument("--poll-interval", type=float, default=DEFAULT_POLL_INTERVAL, help="poll interval in seconds")
    args = parser.parse_args()

    pid = args.pid
    if pid is None:
        uid, user = resolve_uid(args.user)
        if args.all:
            pid = select_process(uid, user)
        else:
            pid = select_gpu_process(uid, user).pid
        print(f"[occupy] selected pid {pid}", flush=True)

    gpu_ids = None
    if Path(f"/proc/{pid}/stat").exists():
        watcher = ProcessWatcher(pid)
    else:
        remote = next(
            (row for row in GpuMemoryProbe.compute_processes() if int(row["pid"]) == pid),
            None,
        )
        if remote is None:
            raise SystemExit(missing_pid_message(pid))
        watcher = ComputeAppWatcher(pid)
        gpu_ids = [remote["gpu_uuid"]]
        print(f"[occupy] pid {pid} is a host pid on gpu {remote['gpu_uuid']} (tracked via nvidia-smi)", flush=True)

    if args.gpu:
        gpu_ids = [token.strip() for token in args.gpu.split(",") if token.strip()]
    elif gpu_ids is None:
        gpu_ids = read_cuda_visible_devices(pid)
    if not gpu_ids:
        gpu_ids = prompt_gpu_ids(pid)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
    print(
        f"[occupy] pid {pid} -> gpu {', '.join(gpu_ids)}; reserve {args.reserve_mb} MiB, slack {args.slack_mb} MiB",
        flush=True,
    )

    keeper = GpuMemoryKeeper(gpu_ids, args.reserve_mb, args.slack_mb, args.poll_interval)
    signal.signal(signal.SIGINT, keeper.request_stop)
    signal.signal(signal.SIGTERM, keeper.request_stop)
    keeper.run(watcher)
    print("[occupy] training process ended, occupation stopped", flush=True)


if __name__ == "__main__":
    main()
