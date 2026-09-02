"""One-command operations for the independent Flexiv + Inspire stack."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

import yaml
from flexiv_inspire_control.foot_pedal import resolve_input_event_path

from .system_config import SystemConfigError, load_system_config, render_runtime_configs

_HARDWARE_WRITE_VALUE = "FLEXIV-RDK-WRITES-ENABLED"

_MANAGED_PROCESS_MARKERS = (
    "flexiv-rdk-daemon",
    "flexiv_inspire_control.node",
    "flexiv-inspire-control-bridge",
    "flexiv_inspire_control/control_bridge",
    "flexiv_inspire_control.teleop_input_node",
    "flexiv-inspire-teleop-input",
    "flexiv_inspire_isaac.cameras.ros_node",
    "flexiv-inspire-camera-node",
    "flexiv_inspire_isaac.dftp.ros_node",
    "flexiv-inspire-dftp-node",
    "flexiv_inspire_dftp/dftp_node",
    "flexiv_inspire_isaac.pedal_router",
    "flexiv-inspire-pedal-router",
    "flexiv_inspire_isaac.episode_control",
    "flexiv-inspire-episode-controller",
    "flexiv_inspire_isaac.data_pipeline.episode_manager",
    "isaac-flexiv-episode",
    "flexiv_inspire_isaac.xr_raw_ros_source",
    "flexiv-inspire-xr-raw-source",
    "flexiv_inspire_isaac.oculus_reader_ros_source",
    "flexiv-inspire-oculus-reader-source",
    "flexiv_inspire_control.manus_ergonomics_source",
    "flexiv-inspire-manus-ergonomics-source",
    "flexiv_inspire_isaac.rerun_viz.cli",
    "flexiv-inspire-rerun",
    "flexiv_inspire_xr_bridge.ros_node",
    "flexiv-inspire-xr-bridge",
    "run_manus_plugin.sh",
    "manus_hand_plugin",
    "run_isaac_camera_receiver.sh",
    "flexiv_inspire_isaac.policy_api.ros_adapter",
    "flexiv-inspire-policy-server",
)


def _runtime_dir(config) -> Path:
    return Path(config.document["session"]["runtime_root"]).expanduser() / "launcher"


def _project_root(config) -> Path:
    return config.root


def _write_state(directory: Path, config, processes: list[subprocess.Popen]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    state_path = directory / "processes.json"
    existing_pids: list[int] = []
    if state_path.is_file():
        try:
            existing = json.loads(state_path.read_text(encoding="utf-8"))
            if existing.get("config") == str(config.path):
                existing_pids = [int(pid) for pid in existing.get("pids", [])]
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            existing_pids = []
    state_path.write_text(
        json.dumps(
            {
                "config": str(config.path),
                "config_sha256": config.sha256,
                "started_unix_ns": time.time_ns(),
                "pids": list(
                    dict.fromkeys([*existing_pids, *[item.pid for item in processes]])
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _remove_state_pids(directory: Path, pids: set[int]) -> None:
    """Remove processes that a foreground operation has already reaped."""

    state_path = directory / "processes.json"
    if not state_path.is_file() or not pids:
        return
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state["pids"] = [
            int(pid) for pid in state.get("pids", []) if int(pid) not in pids
        ]
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return
    state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def _start(
    commands: list[Sequence[str]],
    directory: Path,
    *,
    log_prefix: str = "service",
) -> list[subprocess.Popen]:
    directory.mkdir(parents=True, exist_ok=True)
    processes: list[subprocess.Popen] = []
    try:
        for index, command in enumerate(commands):
            selected = list(command)
            if "--allow-hardware-writes" in selected:
                permit = Path(selected[selected.index("--local-permit-file") + 1])
                _ensure_write_permit(permit)
            log = (directory / f"{log_prefix}-{index}.log").open("ab", buffering=0)
            try:
                process = subprocess.Popen(
                    selected,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    env=_command_environment(selected),
                )
            finally:
                # Popen duplicates the descriptor for the child.  Keeping the
                # launcher's copy open leaks one fd per service and delays log
                # file cleanup during repeated record sessions.
                log.close()
            processes.append(process)
    except BaseException:
        # Starting a service group must be atomic from the caller's point of
        # view.  Otherwise a missing executable halfway through startup leaves
        # cameras or the RDK daemon alive and the next run reports "occupied".
        _stop_started_processes(processes)
        raise
    return processes


def _verify_process_startup(
    processes: list[subprocess.Popen],
    logs: list[Path],
    *,
    timeout_s: float = 1.0,
) -> None:
    """Fail the one-command launcher when a child exits immediately."""

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        failed = [
            (index, process.returncode)
            for index, process in enumerate(processes)
            if process.poll() is not None
        ]
        if failed:
            for process in processes:
                if process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
            details: list[str] = []
            for index, returncode in failed:
                log = logs[index]
                tail = (
                    log.read_text(encoding="utf-8", errors="replace")[-3000:]
                    if log.is_file()
                    else ""
                )
                details.append(
                    f"service-{index} 退出码 {returncode}，日志 {log}\n{tail}"
                )
            raise SystemExit("服务启动失败\n" + "\n".join(details))
        time.sleep(0.05)


def _cloudxr_runtime_live() -> bool:
    pid_path = Path.home() / ".cloudxr/run/cloudxr.pid"
    try:
        pid = int(pid_path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return False
    return pid > 0 and Path(f"/proc/{pid}").is_dir()


def _wait_for_cloudxr_runtime(
    processes: Sequence[subprocess.Popen],
    logs: Sequence[Path],
    *,
    timeout_s: float = 60.0,
) -> None:
    """Wait for the XR source's in-process CloudXR launcher to be usable."""

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        failed = [
            (index, process.returncode)
            for index, process in enumerate(processes)
            if process.poll() is not None
        ]
        if failed:
            _stop_started_processes(processes)
            details = []
            for index, returncode in failed:
                tail = (
                    logs[index].read_text(encoding="utf-8", errors="replace")[-3000:]
                    if logs[index].is_file()
                    else ""
                )
                details.append(
                    f"service-{index} 退出码 {returncode}，日志 {logs[index]}\n{tail}"
                )
            raise SystemExit("XR/CloudXR 启动失败\n" + "\n".join(details))
        if _cloudxr_runtime_live():
            print("CloudXR 和 Quest 输入已自动启动", flush=True)
            return
        time.sleep(0.1)
    _stop_started_processes(processes)
    raise SystemExit(
        "CloudXR 在 60 秒内未就绪；请检查 Quest USB 连接和最新 record 日志"
    )


def _report_manus_status(
    commands: Sequence[Sequence[str]],
    processes: Sequence[subprocess.Popen],
    logs: Sequence[Path],
) -> bool:
    """Report current MANUS state without making it a record startup gate.

    Quest tracking and MANUS may complete their handshake in either order.  The
    teleop node already degrades to arm-only commands and automatically adds
    both hand command bits on the first valid MANUS frame, so the launcher only
    needs to report status and must never tear down an otherwise usable run.
    """

    teleop_indices = [
        index
        for index, command in enumerate(commands)
        if "flexiv_inspire_control.teleop_input_node" in " ".join(command)
    ]
    if len(teleop_indices) != 1:
        print(
            "MANUS 状态暂不可用；record 继续运行，机械臂和其余模态不受影响",
            flush=True,
        )
        return False
    teleop_index = teleop_indices[0]
    marker = "MANUS_HANDS_READY"
    log = logs[teleop_index]
    ready = log.is_file() and marker in log.read_text(
        encoding="utf-8", errors="replace"
    )
    if ready:
        print("MANUS 左右手套已连接，灵巧手映射就绪", flush=True)
        return True
    print(
        "MANUS 暂未出有效双手位姿；record 继续运行。当前先遥操双臂，"
        "手套恢复后会自动加入灵巧手控制",
        flush=True,
    )
    return False


def _wait_for_rdk_socket(
    socket_path: Path,
    processes: Sequence[subprocess.Popen],
    logs: Sequence[Path],
    *,
    timeout_s: float = 45.0,
) -> None:
    """Wait until the just-started hardware daemon accepts IPC requests.

    Starting a Flexiv daemon process is not equivalent to it being usable: it
    initializes both robot controllers before binding the local Unix socket.
    Collection preflight needs that socket, so running it immediately after a
    process-only liveness check creates a deterministic startup race.
    """

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if _rdk_socket_live(socket_path):
            return
        failed = [
            (index, process.returncode)
            for index, process in enumerate(processes)
            if process.poll() is not None
        ]
        if failed:
            _stop_started_processes(processes)
            details: list[str] = []
            for index, returncode in failed:
                log = logs[index]
                tail = (
                    log.read_text(encoding="utf-8", errors="replace")[-3000:]
                    if log.is_file()
                    else ""
                )
                details.append(
                    f"service-{index} 退出码 {returncode}，日志 {log}\n{tail}"
                )
            raise SystemExit("RDK daemon 启动失败\n" + "\n".join(details))
        time.sleep(0.05)
    _stop_started_processes(processes)
    raise SystemExit(
        f"RDK daemon 在 {timeout_s:.0f} 秒内未就绪；查看日志："
        f"{socket_path.parent / 'launcher' / 'rdk-daemon.log'}"
    )


def _wait_for_camera_streams(
    commands: Sequence[Sequence[str]],
    processes: Sequence[subprocess.Popen],
    logs: Sequence[Path],
    camera_names: Sequence[str],
    *,
    timeout_s: float = 20.0,
) -> None:
    """Require every configured RealSense stream before collection starts."""

    camera_index = next(
        (
            index
            for index, command in enumerate(commands)
            if "flexiv-inspire-camera-node" in " ".join(command)
        ),
        None,
    )
    if camera_index is None:
        print("复用已运行的相机服务", flush=True)
        return
    log_path = logs[camera_index]
    process = processes[camera_index]
    expected = tuple(str(name) for name in camera_names)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        text = (
            log_path.read_text(encoding="utf-8", errors="replace")
            if log_path.is_file()
            else ""
        )
        if all(
            any(
                f"camera {name} (" in line and ": streaming" in line
                for line in text.splitlines()
            )
            for name in expected
        ):
            print(
                "相机已就绪: " + ", ".join(expected),
                flush=True,
            )
            return
        if process.poll() is not None or "Device or resource busy" in text:
            _stop_started_processes(processes)
            raise SystemExit(
                "相机启动失败；可能仍被其他采集程序占用。"
                f"日志：{log_path}\n{text[-3000:]}"
            )
        time.sleep(0.05)
    _stop_started_processes(processes)
    raise SystemExit(f"相机在 {timeout_s:.0f} 秒内未全部出帧；日志：{log_path}")


def _wait_for_xr_receiver_ready(
    process: subprocess.Popen,
    log_path: Path,
    *,
    expected_streams: int,
    timeout_s: float = 1.0,
) -> bool:
    """Wait briefly for the Quest display without blocking data collection.

    The Isaac Teleop receiver keeps retrying OpenXR while the Quest browser is
    waking up or completing the CloudXR handshake.  A slow headset connection
    must therefore not tear down cameras, robot control, and an otherwise
    usable foreground recording session.
    """

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        text = (
            log_path.read_text(encoding="utf-8", errors="replace")
            if log_path.is_file()
            else ""
        )
        if (
            "OpenXR session is ready" in text
            and text.count("Session Initialization Time:") >= expected_streams
        ):
            print(
                f"Quest 视频已就绪: {expected_streams} 路单目视频",
                flush=True,
            )
            return True
        if process.poll() is not None:
            print(
                "Quest 视频接收端暂时退出；record 继续运行，启动脚本会自动"
                f"重试。日志：{log_path}\n{text[-1000:]}",
                flush=True,
            )
            return False
        time.sleep(0.1)
    print(
        "Quest/OpenXR 暂未完成视频握手；record 继续运行，接收端会每 2 秒"
        f"自动重连。请保持 Quest 页面打开；日志：{log_path}",
        flush=True,
    )
    return False


def _wait_for_xr_bridge_streams(
    commands: Sequence[Sequence[str]],
    processes: Sequence[subprocess.Popen],
    logs: Sequence[Path],
    stream_names: Sequence[str],
    *,
    timeout_s: float = 3.0,
) -> bool:
    """Check RTP video briefly without making it a collection prerequisite."""

    expected = tuple(str(name) for name in stream_names)
    if not expected:
        return True
    bridge_index = next(
        (
            index
            for index, command in enumerate(commands)
            if "flexiv-inspire-xr-bridge" in " ".join(command)
        ),
        None,
    )
    if bridge_index is None:
        print(
            "Quest 视频暂不可用：本次启动没有 RTP bridge；record 继续运行",
            flush=True,
        )
        return False
    process = processes[bridge_index]
    log_path = logs[bridge_index]
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        output = (
            log_path.read_text(encoding="utf-8", errors="replace")
            if log_path.is_file()
            else ""
        )
        if all(f"XR_VIDEO_STREAM_READY: {name} " in output for name in expected):
            print("Quest RTP 已出帧: " + ", ".join(expected), flush=True)
            return True
        if process.poll() is not None:
            print(
                "Quest RTP bridge 暂时退出；视频不可用，但 record 和遥操继续。"
                f"日志：{log_path}\n{output[-1000:]}",
                flush=True,
            )
            return False
        time.sleep(0.05)
    print(
        f"Quest RTP bridge 在 {timeout_s:.0f} 秒内暂未编码出帧；"
        f"视频后台继续尝试，record 和遥操继续。日志：{log_path}",
        flush=True,
    )
    return False


def _report_optional_xr_video_status(
    commands: Sequence[Sequence[str]],
    processes: Sequence[subprocess.Popen],
    logs: Sequence[Path],
    stream_names: Sequence[str],
    *,
    bridge_timeout_s: float = 3.0,
    receiver_timeout_s: float = 1.0,
) -> bool:
    """Report video readiness without making display a control dependency.

    Quest controller tracking already has its own freshness/validity gate in
    teleop_input.  Camera display is useful operator feedback, but coupling it
    to episode startup makes an optional OpenXR renderer failure tear down
    otherwise valid recording, MANUS, and robot services.
    """

    expected = tuple(str(name) for name in stream_names)
    if not expected:
        return False
    if len(processes) != len(commands) or len(logs) != len(commands):
        print(
            "Quest 视频进程未完整启动；record 和遥操继续，机械臂仍由 "
            "Quest 位姿有效性和中踏板控制",
            flush=True,
        )
        return False
    bridge_ready = _wait_for_xr_bridge_streams(
        commands,
        processes,
        logs,
        expected,
        timeout_s=bridge_timeout_s,
    )
    receiver_index = next(
        (
            index
            for index, command in enumerate(commands)
            if "run_isaac_camera_receiver.sh" in " ".join(command)
        ),
        None,
    )
    if receiver_index is None:
        print(
            "Quest 视频接收器未配置；record 和遥操继续",
            flush=True,
        )
        return False
    receiver_ready = _wait_for_xr_receiver_ready(
        processes[receiver_index],
        logs[receiver_index],
        expected_streams=len(expected),
        timeout_s=receiver_timeout_s,
    )
    if bridge_ready and receiver_ready:
        print("Quest 视频已实时显示: " + ", ".join(expected), flush=True)
        return True
    print(
        "Quest 视频暂不可用；record 和遥操继续。机械臂只有在 Quest 腕部"
        "位姿有效且中踏板踩下时才会运动",
        flush=True,
    )
    return False


def _camera_device_conflicts(config) -> list[str]:
    """Return non-managed processes that currently hold V4L2 devices."""

    project_root = str(_project_root(config).resolve())
    conflicts: dict[int, tuple[str, set[str]]] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            command = (
                (entry / "cmdline")
                .read_bytes()
                .replace(b"\0", b" ")
                .decode("utf-8", errors="replace")
                .strip()
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if not command:
            continue
        if project_root in command and any(
            marker in command
            for marker in (
                "flexiv-inspire-camera-node",
                "flexiv_inspire_isaac.cameras.ros_node",
            )
        ):
            continue
        devices: set[str] = set()
        try:
            descriptors = (entry / "fd").iterdir()
            for descriptor in descriptors:
                try:
                    target = os.readlink(descriptor)
                except (FileNotFoundError, PermissionError, OSError):
                    continue
                if target.startswith("/dev/video"):
                    devices.add(target)
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if devices:
            conflicts[int(entry.name)] = (command, devices)
    return [
        f"PID {pid} ({', '.join(sorted(devices))}): {command}"
        for pid, (command, devices) in sorted(conflicts.items())
    ]


def _require_camera_devices_available(config) -> None:
    conflicts = _camera_device_conflicts(config)
    if not conflicts:
        return
    raise SystemExit(
        "RealSense 视频设备正被其他程序占用，请先在对应终端 Ctrl-C：\n"
        + "\n".join(conflicts)
    )


def _stop_started_processes(
    processes: Sequence[subprocess.Popen], *, timeout_s: float = 5.0
) -> None:
    """Stop only services started by the active foreground invocation."""

    live = [process for process in processes if process.poll() is None]
    for process in live:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + timeout_s
    try:
        while live and time.monotonic() < deadline:
            live = [process for process in live if process.poll() is None]
            if live:
                time.sleep(0.05)
    except KeyboardInterrupt:
        # A second Ctrl-C means "finish now". Continue with the bounded
        # force-stop below instead of leaking a launcher traceback.
        live = [process for process in live if process.poll() is None]
    for process in live:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    for process in live:
        try:
            process.wait(timeout=1.0)
        except (subprocess.TimeoutExpired, ChildProcessError):
            pass


def _is_collection_process_command(command: Sequence[str]) -> bool:
    joined = " ".join(str(item) for item in command)
    return any(
        marker in joined
        for marker in (
            "flexiv_inspire_isaac.episode_control",
            "flexiv-inspire-episode-controller",
            "flexiv_inspire_isaac.pedal_router",
            "flexiv-inspire-pedal-router",
        )
    )


def _ensure_write_permit(path: Path) -> None:
    try:
        path.parent.mkdir(mode=0o700, parents=True)
    except FileExistsError:
        if not path.parent.is_dir():
            raise RuntimeError(
                f"local RDK permit parent is not a directory: {path.parent}"
            )
    else:
        path.parent.chmod(0o700)
    try:
        current = path.lstat()
    except FileNotFoundError:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags, 0o600)
        try:
            os.write(descriptor, (_HARDWARE_WRITE_VALUE + "\n").encode("ascii"))
        finally:
            os.close(descriptor)
        return
    if not path.is_file() or current.st_uid != os.getuid():
        raise RuntimeError(f"invalid local RDK permit: {path}")
    if path.read_text(encoding="utf-8").strip() != _HARDWARE_WRITE_VALUE:
        raise RuntimeError(f"invalid local RDK permit content: {path}")
    path.chmod(0o600)


def _command_environment(command: Sequence[str]) -> dict[str, str] | None:
    if "--allow-hardware-writes" not in command:
        if "--ros-args" not in command:
            return None
        # ``(ros-py312)`` only identifies the virtualenv; it does not prove
        # that the caller sourced /opt/ros/jazzy/setup.bash.  Background
        # bridge/hand/pedal processes must nevertheless always get rclpy and
        # the workspace overlays.  Derive their environment explicitly so a
        # bare activated venv cannot make Reset fail with
        # ``ModuleNotFoundError: rclpy``.
        environment = os.environ.copy()
        project_root = Path(__file__).resolve().parents[4]
        setup_script = project_root / "ros2_ws" / "install" / "setup.bash"
        shell = "source /opt/ros/jazzy/setup.bash"
        if setup_script.is_file():
            shell += f" && source {setup_script}"
        shell += " && env -0"
        try:
            raw = subprocess.check_output(
                ["/bin/bash", "-c", shell], env=environment
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(f"无法构造 ROS 子进程环境: {exc}") from exc
        return {
            key.decode("utf-8", errors="surrogateescape"): value.decode(
                "utf-8", errors="surrogateescape"
            )
            for item in raw.split(b"\0")
            if item
            for key, _, value in (item.partition(b"="),)
        }
    environment = os.environ.copy()
    environment["ISAAC_TELEOP_ALLOW_HARDWARE_WRITES"] = _HARDWARE_WRITE_VALUE
    executable = Path(command[0]).resolve()
    environment["PATH"] = (
        f"{executable.parent}:/usr/local/sbin:/usr/local/bin:"
        "/usr/sbin:/usr/bin:/sbin:/bin"
    )
    environment["VIRTUAL_ENV"] = str(executable.parent.parent)
    environment["PYTHONNOUSERSITE"] = "1"
    for key in (
        "PYTHONPATH",
        "LD_LIBRARY_PATH",
        "CMAKE_PREFIX_PATH",
        "AMENT_PREFIX_PATH",
        "COLCON_PREFIX_PATH",
        "ROS_DISTRO",
        "ROS_VERSION",
        "ROS_PYTHON_VERSION",
        "RMW_IMPLEMENTATION",
        "CYCLONEDDS_URI",
    ):
        environment.pop(key, None)
    return environment


def _rdk_socket_live(path: Path) -> bool:
    if not path.is_socket():
        return False
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    probe.settimeout(0.25)
    try:
        probe.connect(str(path))
    except OSError:
        return False
    finally:
        probe.close()
    return True


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rdk_runtime_fingerprint(config, config_path: Path) -> str:
    """Fingerprint daemon configuration, IPC schema, and executable source."""

    digest = hashlib.sha256()
    digest.update(b"flexiv-rdk-runtime-v1\0")
    digest.update(str(config.sha256).encode("ascii"))
    paths = (
        config_path,
        config.root / "apps/flexiv_daemon/src/flexiv_rdk_daemon/main.py",
        config.root / "apps/flexiv_daemon/src/flexiv_rdk_daemon/configuration.py",
        config.root / "apps/flexiv_daemon/src/flexiv_rdk_daemon/server.py",
        config.root / "apps/flexiv_daemon/src/flexiv_rdk_daemon/backend.py",
        config.root / "apps/flexiv_daemon/src/flexiv_rdk_daemon/ft_zero.py",
        config.root / "apps/flexiv_daemon/src/flexiv_rdk_daemon/typed_ipc.py",
        config.root
        / "apps/flexiv_daemon/src/flexiv_rdk_daemon/generated/rdk_ipc_pb2.py",
    )
    for path in paths:
        digest.update(b"\0")
        digest.update(str(path).encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def _reset_ros_runtime_fingerprint(config) -> str:
    """Restart persistent Reset services after their Python source changes."""

    digest = hashlib.sha256()
    digest.update(b"flexiv-reset-ros-runtime-v1\0")
    digest.update(str(config.sha256).encode("ascii"))
    for relative in (
        "ros2_ws/src/flexiv_inspire_control/flexiv_inspire_control/node.py",
        "libs/control_core/src/isaac_teleop_core/control.py",
        "ros2_ws/src/flexiv_inspire_dftp/ros_node.py",
    ):
        path = config.root / relative
        digest.update(b"\0")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def _restart_managed_rdk_if_config_changed(
    state_path: Path,
    *,
    rdk_socket: Path,
    config_sha256: str,
) -> bool:
    """Stop our daemon when its on-disk configuration has changed."""

    if not state_path.is_file():
        return False
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        pid = int(state["pid"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False
    if state.get("config_sha256") == config_sha256:
        return False
    try:
        command = (
            Path(f"/proc/{pid}/cmdline")
            .read_bytes()
            .replace(b"\0", b" ")
            .decode("utf-8", errors="replace")
        )
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return False
    if "flexiv-rdk-daemon" not in command or str(rdk_socket) not in command:
        return False
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return False
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if not _rdk_socket_live(rdk_socket):
            print("RDK 配置或代码已更新，daemon 已自动重启", flush=True)
            return True
        time.sleep(0.05)
    # A thread blocked inside the vendor RDK may prevent graceful Python
    # shutdown. This daemon is locally managed and will be recreated below,
    # so do not strand Reset behind that stale process/socket.
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    kill_deadline = time.monotonic() + 2.0
    while time.monotonic() < kill_deadline:
        if not _rdk_socket_live(rdk_socket):
            print("旧 RDK daemon 无响应，已强制关闭并自动重启", flush=True)
            return True
        time.sleep(0.05)
    raise RuntimeError("旧 RDK daemon 在 SIGKILL 后仍占用 socket")


def _process_running(*markers: str) -> bool:
    return bool(_matching_process_ids(*markers))


def _matching_process_ids(*markers: str) -> list[int]:
    matches: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (
                (entry / "cmdline")
                .read_bytes()
                .replace(b"\0", b" ")
                .decode("utf-8", errors="replace")
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if any(marker in command for marker in markers):
            matches.append(int(entry.name))
    return matches


def _matching_managed_process_ids(config) -> list[int]:
    """Find orphaned stack services whose PID was lost from launcher state."""
    project_root = str(_project_root(config).resolve())
    matches: list[int] = []
    for pid in _matching_process_ids(*_MANAGED_PROCESS_MARKERS):
        if pid == os.getpid():
            continue
        try:
            command = (
                Path(f"/proc/{pid}/cmdline")
                .read_bytes()
                .replace(b"\0", b" ")
                .decode("utf-8", errors="replace")
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if project_root in command:
            matches.append(pid)
    return matches


def _pid_is_running(pid: int) -> bool:
    """Return false for absent and zombie processes.

    A zombie still has a ``/proc`` directory but cannot react to SIGKILL.  It
    must not make foreground record cleanup report a false failure.
    """

    try:
        stat_line = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
        return False
    closing_parenthesis = stat_line.rfind(")")
    if closing_parenthesis < 0 or closing_parenthesis + 2 >= len(stat_line):
        return False
    return stat_line[closing_parenthesis + 2] not in {"Z", "X"}


def _stop_managed_services(config, *, require_existing: bool) -> int:
    """Stop every background service belonging to this checkout.

    ``robot record`` may reuse processes created by its automatic Reset.  They
    are not children of the record launcher itself, so stopping only the local
    ``Popen`` objects leaves the RDK daemon/control bridge alive between runs.
    Resolve both launcher-owned PIDs and precisely matched orphan processes so
    the foreground record command has a complete lifecycle.
    """

    runtime = _runtime_dir(config)
    state_paths = (
        runtime / "processes.json",
        runtime / "rdk-daemon.json",
        runtime / "reset-services.json",
    )
    owned_pids: list[int] = []
    for state_path in state_paths:
        if not state_path.is_file():
            continue
        try:
            values = json.loads(state_path.read_text(encoding="utf-8"))
            owned_pids.extend(int(pid) for pid in values.get("pids", []))
            if values.get("pid") is not None:
                owned_pids.append(int(values["pid"]))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            # Process discovery below remains authoritative if a prior crash
            # left a partially written launcher state file.
            continue

    discovered_pids = set(_matching_managed_process_ids(config))
    owned_pid_set = set(owned_pids)
    missing_owned_pids = owned_pid_set.difference(discovered_pids)
    if missing_owned_pids:
        # Deployment worktrees may intentionally share an environment whose
        # executable path points at another checkout. State-owned PIDs are
        # still safe to accept when their live command matches this stack.
        marked_pids = set(_matching_process_ids(*_MANAGED_PROCESS_MARKERS))
        discovered_pids.update(missing_owned_pids.intersection(marked_pids))
    confirmed_owned_pids = owned_pid_set.intersection(discovered_pids)
    if not discovered_pids:
        if require_existing:
            raise SystemExit("no launcher process state exists")
        for state_path in state_paths:
            state_path.unlink(missing_ok=True)
        return 0

    # Launcher-owned services were started in their own sessions, so stopping
    # their process groups also reaps wrappers and children such as CloudXR.
    for pid in confirmed_owned_pids:
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    # A discovered process without trusted state may share a caller's process
    # group. Signal only that exact PID rather than risking the user's shell.
    for pid in discovered_pids.difference(confirmed_owned_pids):
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    deadline = time.monotonic() + 5.0
    live_pids = set(discovered_pids)
    while live_pids and time.monotonic() < deadline:
        live_pids = {pid for pid in live_pids if _pid_is_running(pid)}
        if live_pids:
            time.sleep(0.05)
    if live_pids:
        # MANUS Core and CloudXR can take longer than ordinary ROS nodes to
        # unwind.  They are still processes precisely identified as belonging
        # to this checkout, so escalate after the graceful deadline instead of
        # leaking them into the next ``robot record`` invocation.
        for pid in live_pids.intersection(confirmed_owned_pids):
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        for pid in live_pids.difference(confirmed_owned_pids):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        kill_deadline = time.monotonic() + 2.0
        while live_pids and time.monotonic() < kill_deadline:
            live_pids = {pid for pid in live_pids if _pid_is_running(pid)}
            if live_pids:
                time.sleep(0.05)
    if live_pids:
        raise RuntimeError(
            "managed services survived SIGKILL: "
            + ",".join(str(pid) for pid in sorted(live_pids))
        )
    for state_path in state_paths:
        state_path.unlink(missing_ok=True)
    return len(discovered_pids)


def _restart_dftp_processes() -> None:
    markers = (
        "flexiv_inspire_isaac.dftp.ros_node",
        "flexiv-inspire-dftp-node",
        "flexiv_inspire_dftp/dftp_node",
    )
    pids = _matching_process_ids(*markers)
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 2.0
    while pids and time.monotonic() < deadline:
        pids = [pid for pid in pids if Path(f"/proc/{pid}").exists()]
        if pids:
            time.sleep(0.05)
    if pids:
        raise RuntimeError(
            "旧 DFTP 手部进程无法停止，请先运行 robot stop 后重试: "
            + ", ".join(str(pid) for pid in pids)
        )


def _restart_reset_ros_processes() -> None:
    markers = (
        "flexiv_inspire_control.node",
        "flexiv-inspire-control-bridge",
        "flexiv_inspire_control/control_bridge",
        "flexiv_inspire_isaac.dftp.ros_node",
        "flexiv-inspire-dftp-node",
        "flexiv_inspire_dftp/dftp_node",
    )
    pids = _matching_process_ids(*markers)
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 3.0
    while pids and time.monotonic() < deadline:
        pids = [pid for pid in pids if _pid_is_running(pid)]
        if pids:
            time.sleep(0.05)
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    kill_deadline = time.monotonic() + 2.0
    while pids and time.monotonic() < kill_deadline:
        pids = [pid for pid in pids if _pid_is_running(pid)]
        if pids:
            time.sleep(0.05)
    if pids:
        raise RuntimeError(
            "旧 ROS Reset 服务在 SIGKILL 后仍未停止: "
            + ", ".join(str(pid) for pid in pids)
        )


def _xr_receiver_command(config, rendered: dict[str, Path]) -> list[str]:
    return [
        str(_project_root(config) / "orchestration/run_isaac_camera_receiver.sh"),
        str(rendered["isaac_camera_receiver.yaml"]),
        str(config.document["xr_video"]["display"]["mode"]),
    ]


def _xr_raw_source_command(config) -> list[str]:
    xr = config.document["xr_video"]
    client = xr["cloudxr_client"]
    command = [
        str(_project_root(config) / "orchestration/run_xr_raw_source.sh"),
        "--transport",
        str(xr["transport"]),
        "--client-per-eye-width",
        str(client["per_eye_width"]),
        "--client-per-eye-height",
        str(client["per_eye_height"]),
        "--client-frame-rate",
        str(client["frame_rate"]),
        "--client-max-bitrate-mbps",
        str(client["max_bitrate_mbps"]),
        "--client-codec",
        str(client["codec"]),
        "--client-enable-tex-sub-image-2d",
        str(bool(client["enable_tex_sub_image_2d"])).lower(),
    ]
    wifi_connection = str(xr.get("wifi_connection", "")).strip()
    if wifi_connection:
        command.extend(["--wifi-connection", wifi_connection])
    return command


def _quest_input_config(config) -> dict:
    """Return the provider config while preserving legacy site files."""

    configured = config.document["teleop"].get("quest_input")
    if configured is None:
        return {
            "provider": "isaac_openxr",
            "publish_rate_hz": 60.0,
            "stale_timeout_ms": 120.0,
            "oculus_reader": {
                "adb_serial": "",
                "package_name": "com.rail.oculus.teleop",
                "auto_install_apk": True,
            },
        }
    return configured


def _oculus_reader_source_command(config) -> list[str]:
    quest = _quest_input_config(config)
    oculus = quest["oculus_reader"]
    command = [
        sys.executable,
        "-m",
        "flexiv_inspire_isaac.oculus_reader_ros_source",
        "--ros-args",
        "-p",
        f"rate_hz:={float(quest['publish_rate_hz'])}",
        "-p",
        f"stale_timeout_s:={float(quest['stale_timeout_ms']) / 1000.0}",
        "-p",
        f"package_name:={oculus['package_name']}",
        "-p",
        "auto_install_apk:=" + str(bool(oculus["auto_install_apk"])).lower(),
    ]
    adb_serial = str(oculus["adb_serial"]).strip()
    if adb_serial:
        command.extend(["-p", f"adb_serial:={adb_serial}"])
    return command


def _quest_input_source_command(config) -> list[str]:
    provider = str(_quest_input_config(config)["provider"])
    if provider == "oculus_reader":
        return _oculus_reader_source_command(config)
    if provider == "isaac_openxr":
        return _xr_raw_source_command(config)
    raise ValueError(f"unsupported Quest input provider: {provider}")


def _manus_plugin_command(config) -> list[str]:
    ergonomics = config.document["teleop"]["manus_ergonomics"]
    return [
        str(_project_root(config) / "orchestration/run_manus_plugin.sh"),
        "--ergonomics-udp",
        f"{ergonomics['udp_host']}:{int(ergonomics['udp_port'])}",
    ]


def _manus_ergonomics_source_command(config) -> list[str]:
    ergonomics = config.document["teleop"]["manus_ergonomics"]
    return [
        sys.executable,
        "-m",
        "flexiv_inspire_control.manus_ergonomics_source",
        "--ros-args",
        "-p",
        f"udp_host:={ergonomics['udp_host']}",
        "-p",
        f"udp_port:={int(ergonomics['udp_port'])}",
        "-p",
        f"left_topic:={ergonomics['left_topic']}",
        "-p",
        f"right_topic:={ergonomics['right_topic']}",
    ]


def _home_motion_timeout_s(root: dict) -> float:
    """Return the worst-case lift plus joint-Home motion budget."""

    home = root["flexiv"]["home"]
    total = float(home["timeout_s"])
    lift = home.get("lift", {})
    if bool(lift.get("enabled", False)):
        lift_count = 1 if bool(lift.get("parallel", False)) else 2
        total += lift_count * float(lift["timeout_s"])
    return total


def _episode_command(config, rendered: dict[str, Path]) -> list[str]:
    root = config.document
    session = root["session"]
    recording = root["recording"]
    runtime_root = Path(session["runtime_root"]).expanduser()
    camera_extrinsics = {
        name: (
            str(config.resolve(str(stream.get("extrinsics", ""))))
            if str(stream.get("extrinsics", "")).strip()
            else ""
        )
        for name, stream in root["cameras"]["streams"].items()
    }
    command = [
        sys.executable,
        "-m",
        "flexiv_inspire_isaac.episode_control",
        "--ros-args",
        "-p",
        f"sessions_root:={config.resolve(recording['output_root'])}",
        "-p",
        f"dataset_name:={recording['dataset_name']}",
        "-p",
        f"task_name:={recording['task_name']}",
        "-p",
        f"episode_count:={int(recording['episode_count'])}",
        "-p",
        "task_description:="
        + json.dumps(str(recording["task_description"]), ensure_ascii=False),
        "-p",
        f"session_id:={session['id']}",
        "-p",
        f"tool_config:={config.resolve(root['flexiv']['tool_payload_config'])}",
        "-p",
        f"ft_zero_record:={runtime_root / 'ft_zero_events.jsonl'}",
        "-p",
        f"camera_config:={rendered['camera.yaml']}",
        "-p",
        f"deviceio_socket:={runtime_root / 'deviceio.sock'}",
        "-p",
        f"runtime_dir:={runtime_root}",
        "-p",
        f"rdk_socket:={runtime_root / 'rdk.sock'}",
        "-p",
        f"camera_recording_mode:={recording['camera_recording_mode']}",
        "-p",
        f"deviceio_mode:={recording['deviceio_mode']}",
        "-p",
        f"ros_mcap_enabled:={str(bool(recording['ros_mcap_enabled'])).lower()}",
        "-p",
        f"deviceio_profile:={recording['deviceio_profile']}",
        "-p",
        f"controlled_side:={root['teleop'].get('controlled_side', 'both')}",
        "-p",
        "record_only_while_pedal_pressed:="
        + str(bool(recording["record_only_while_pedal_pressed"])).lower(),
        "-p",
        f"camera_hz:={float(root['sampling']['camera_hz'])}",
        "-p",
        f"action_hz:={float(root['sampling']['teleop_command_hz'])}",
        "-p",
        f"auto_start:={str(bool(recording.get('auto_start', True))).lower()}",
        "-p",
        f"auto_authorize_home:={str(bool(recording.get('auto_authorize_home', True))).lower()}",
        "-p",
        f"auto_authorize_control:={str(bool(recording.get('auto_authorize_control', True))).lower()}",
        "-p",
        f"home_result_timeout_s:={_home_motion_timeout_s(root) + 10.0}",
    ]
    # ROS 2 rejects an empty override such as ``-p name:=``.  These files are
    # optional: the controller has empty defaults and only records calibration
    # provenance when the user has actually configured one.
    optional_paths = {
        "camera_head_extrinsics": camera_extrinsics["head"],
        "camera_left_wrist_extrinsics": camera_extrinsics["left_wrist"],
        "camera_right_wrist_extrinsics": camera_extrinsics["right_wrist"],
        "manus_calibration": (
            str(config.resolve(root["teleop"]["manus_calibration"]))
            if str(root["teleop"]["manus_calibration"]).strip()
            else ""
        ),
    }
    for name, path in optional_paths.items():
        if path:
            command.extend(("-p", f"{name}:={path}"))
    return command


def _control_command(config, rendered: dict[str, Path]) -> list[str]:
    root = config.document
    session = root["session"]
    runtime_root = Path(session["runtime_root"]).expanduser()
    return [
        sys.executable,
        "-m",
        "flexiv_inspire_control.node",
        "--ros-args",
        "--params-file",
        str(rendered["control_bridge.yaml"]),
        "-p",
        f"session_id:={session['id']}",
        "-p",
        f"rdk_socket:={runtime_root / 'rdk.sock'}",
        "-p",
        f"foot_pedal:={root['pedal']['device']}",
    ]


def _teleop_input_command(config, rendered: dict[str, Path]) -> list[str]:
    root = config.document
    session = root["session"]
    return [
        sys.executable,
        "-m",
        "flexiv_inspire_control.teleop_input_node",
        "--ros-args",
        "--params-file",
        str(rendered["teleop.yaml"]),
        "-p",
        f"session_id:={session['id']}",
        "-p",
        "command_enabled:=" + str(bool(root["teleop"]["control_enabled"])).lower(),
    ]


def _live_rerun_command(config) -> list[str] | None:
    live_rerun = config.document["recording"]["live_rerun"]
    if not bool(live_rerun["enabled"]):
        return None
    return [
        sys.executable,
        "-m",
        "flexiv_inspire_isaac.rerun_viz.cli",
        "--spawn",
        "--viewer-port",
        str(int(live_rerun["viewer_port"])),
        "--telemetry-hz",
        str(float(live_rerun["telemetry_hz"])),
        "--tactile-hz",
        str(float(live_rerun["tactile_hz"])),
        "--image-hz",
        str(float(live_rerun["image_hz"])),
        "--pointcloud-hz",
        str(float(live_rerun["pointcloud_hz"])),
    ]


def _load_policy_server_config(config, raw_path: str | Path) -> dict:
    """Load the standalone hardware-owner RPC server configuration."""

    path = config.resolve(str(raw_path))
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise SystemConfigError(f"policy server config cannot be read: {path}: {exc}") from exc
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise SystemConfigError("policy server config must declare schema_version: 1")
    server = document.get("server")
    if not isinstance(server, dict):
        raise SystemConfigError("policy server config requires a server mapping")
    bind = str(server.get("bind", "127.0.0.1")).strip()
    if not bind:
        raise SystemConfigError("policy server bind cannot be empty")
    try:
        port = int(server.get("port", 50051))
    except (TypeError, ValueError) as exc:
        raise SystemConfigError("policy server port must be an integer") from exc
    if not 1 <= port <= 65535:
        raise SystemConfigError("policy server port must be in [1,65535]")
    server_cert = config.resolve(str(server.get("server_cert", "")))
    server_key = config.resolve(str(server.get("server_key", "")))
    client_ca_raw = str(server.get("client_ca", "")).strip()
    client_ca = config.resolve(client_ca_raw) if client_ca_raw else None
    for label, selected in (("server_cert", server_cert), ("server_key", server_key)):
        if not selected.is_file():
            raise SystemConfigError(f"policy server {label} is not a file: {selected}")
    if client_ca is not None and not client_ca.is_file():
        raise SystemConfigError(f"policy server client_ca is not a file: {client_ca}")
    if bind not in {"127.0.0.1", "::1", "localhost"} and client_ca is None:
        raise SystemConfigError(
            "non-loopback policy server bind requires client_ca for mutual TLS"
        )
    rates = document.get("rates", {})
    if not isinstance(rates, dict):
        raise SystemConfigError("policy server rates must be a mapping")
    normalized_rates = {}
    for name, default in (
        ("arm_hz", 200.0),
        ("hand_hz", 200.0),
        ("tactile_hz", 15.0),
        ("camera_hz", 15.0),
        ("action_hz", 30.0),
    ):
        try:
            value = float(rates.get(name, default))
        except (TypeError, ValueError) as exc:
            raise SystemConfigError(f"policy server {name} must be numeric") from exc
        if not math.isfinite(value) or value <= 0.0:
            raise SystemConfigError(f"policy server {name} must be positive and finite")
        normalized_rates[name] = value
    startup = document.get("startup", {})
    if not isinstance(startup, dict):
        raise SystemConfigError("policy server startup must be a mapping")
    return {
        "path": path,
        "bind": bind,
        "port": port,
        "server_cert": server_cert,
        "server_key": server_key,
        "client_ca": client_ca,
        "rates": normalized_rates,
        "auto_reset_before_serve": bool(
            startup.get("auto_reset_before_serve", True)
        ),
        "auto_authorize_policy": bool(startup.get("auto_authorize_policy", True)),
    }


def _policy_server_command(settings: dict) -> list[str]:
    rates = settings["rates"]
    command = [
        sys.executable,
        "-m",
        "flexiv_inspire_isaac.policy_api.ros_adapter",
        "--bind",
        str(settings["bind"]),
        "--port",
        str(int(settings["port"])),
        "--server-cert",
        str(settings["server_cert"]),
        "--server-key",
        str(settings["server_key"]),
        "--arm-rate-hz",
        str(float(rates["arm_hz"])),
        "--hand-rate-hz",
        str(float(rates["hand_hz"])),
        "--tactile-rate-hz",
        str(float(rates["tactile_hz"])),
        "--camera-rate-hz",
        str(float(rates["camera_hz"])),
        "--action-rate-hz",
        str(float(rates["action_hz"])),
    ]
    if settings["client_ca"] is not None:
        command.extend(("--client-ca", str(settings["client_ca"])))
    return command


def _policy_authorization_command(
    config, rdk_socket: Path, *, clear_hold_latched: bool
) -> list[str]:
    command = [
        "flexiv-inspire-authorize-control",
        "--session-id",
        str(config.document["session"]["id"]),
        "--source",
        "policy",
        "--confirm",
        "FLEXIV-CONTROL-ARM",
        "--socket",
        str(rdk_socket),
    ]
    if clear_hold_latched:
        command.insert(-2, "--clear-hold-latched")
    return command


class _PolicyAuthorizationSupervisor:
    """Keep the locally launched policy source armed for RPC inference."""

    _RETRY_INTERVAL_S = 2.0

    def __init__(
        self, config, rdk_socket: Path, *, require_pedal: bool
    ) -> None:
        if not sys.stdin.isatty():
            raise RuntimeError(
                "automatic policy authorization requires robot policy-serve "
                "to run in a local interactive terminal"
            )
        import rclpy
        from flexiv_inspire_interfaces.msg import ControlState
        from rclpy.executors import ExternalShutdownException
        from std_msgs.msg import Bool

        self._rclpy = rclpy
        self._external_shutdown = ExternalShutdownException
        self._config = config
        self._rdk_socket = rdk_socket
        self._require_pedal = bool(require_pedal)
        self._pedal_pressed = not self._require_pedal
        self._control_state = ""
        self._authorization_pending = not self._require_pedal
        self._last_attempt = 0.0
        self._last_reported_error = ""
        rclpy.init()
        self._node = rclpy.create_node(
            f"flexiv_policy_auto_authorizer_{os.getpid()}"
        )
        self._node.create_subscription(
            Bool, "/teleop/deadman", self._on_pedal, 10
        )
        self._node.create_subscription(
            ControlState, "/control/state", self._on_control_state, 10
        )
        self._node.create_timer(0.25, self._refresh_if_needed)

    def _on_pedal(self, message) -> None:
        if not self._require_pedal:
            return
        pressed = bool(message.data)
        if pressed and not self._pedal_pressed:
            # Exactly one local action by the operator starts a new policy
            # control attempt. Retry only until the bridge confirms it armed.
            self._authorization_pending = True
            self._last_attempt = 0.0
        elif not pressed:
            self._authorization_pending = False
        self._pedal_pressed = pressed

    def _on_control_state(self, message) -> None:
        previous_state = self._control_state
        self._control_state = str(message.state_name).upper()
        if self._control_state in {"POLICY_ARMED", "ACTIVE"}:
            self._authorization_pending = False
        elif (
            not self._require_pedal
            and (
                (
                    self._control_state == "HOLD_LATCHED"
                    and previous_state != "HOLD_LATCHED"
                )
                or (
                    self._control_state == "READY"
                    and previous_state
                    in {"", "DISABLED", "MAINTENANCE", "FAULT"}
                )
            )
        ):
            self._authorization_pending = True

    def _refresh_if_needed(self) -> None:
        if (
            (self._require_pedal and not self._pedal_pressed)
            or not self._authorization_pending
        ):
            return
        if self._control_state not in {"READY", "HOLD_LATCHED"}:
            return
        now = time.monotonic()
        if now - self._last_attempt < self._RETRY_INTERVAL_S:
            return
        self._last_attempt = now
        command = _policy_authorization_command(
            self._config,
            self._rdk_socket,
            clear_hold_latched=self._control_state == "HOLD_LATCHED",
        )
        try:
            result = subprocess.run(
                command,
                stdin=sys.stdin,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=15.0,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            detail = f"{type(exc).__name__}: {exc}"
        else:
            detail = (result.stdout or "").strip()
            if result.returncode == 0:
                # The bridge keeps the accepted token pending until its arm,
                # hand and observation gates are ready. Do not mint another
                # token merely because the READY state publication races this
                # short subprocess.
                self._authorization_pending = False
                self._last_reported_error = ""
                print(
                    (
                        "中踏板：policy 控制已自动授权；策略动作可以申请 lease"
                        if self._require_pedal
                        else "Policy 已自动授权；RPC 策略动作可直接下发"
                    ),
                    flush=True,
                )
                return
            detail = detail or f"exit code {result.returncode}"
        if detail != self._last_reported_error:
            print(
                (
                    f"中踏板：policy 自动授权暂未完成：{detail}"
                    if self._require_pedal
                    else f"Policy 自动授权暂未完成：{detail}"
                ),
                file=sys.stderr,
                flush=True,
            )
            self._last_reported_error = detail

    def spin_once(self) -> None:
        try:
            self._rclpy.spin_once(self._node, timeout_sec=0.1)
        except self._external_shutdown as exc:
            raise KeyboardInterrupt from exc

    def close(self) -> None:
        self._node.destroy_node()
        self._rclpy.try_shutdown()


def _policy_serve_commands(
    config, rendered: dict[str, Path], settings: dict
) -> list[list[str]]:
    """Return the hardware owner stack required by an external policy client."""

    commands = [
        _rdk_command(config),
        _control_command(config, rendered),
        [
            "flexiv-inspire-camera-node",
            "--ros-args",
            "-p",
            f"config:={rendered['camera.yaml']}",
        ],
        [
            "flexiv-inspire-dftp-node",
            "--ros-args",
            "--params-file",
            str(rendered["dftp.yaml"]),
        ],
    ]
    if bool(config.document["flexiv"]["policy_control"]["require_pedal"]):
        commands.append(
            [
                "flexiv-inspire-pedal-router",
                "--ros-args",
                "--params-file",
                str(rendered["pedal.yaml"]),
            ]
        )
    rerun_command = _live_rerun_command(config)
    if rerun_command is not None:
        commands.append(rerun_command)
    commands.append(_policy_server_command(settings))
    return commands


def _wait_for_policy_services(
    processes: Sequence[subprocess.Popen],
    logs: Sequence[Path],
    *,
    spin_once=None,
) -> None:
    """Keep the policy owner foregrounded and fail if an owned service exits."""

    while True:
        for index, process in enumerate(processes):
            status = process.poll()
            if status is None:
                continue
            tail = (
                logs[index].read_text(encoding="utf-8", errors="replace")[-4000:]
                if logs[index].is_file()
                else ""
            )
            raise RuntimeError(
                f"policy service-{index} exited with status {status}; "
                f"log={logs[index]}\n{tail}"
            )
        if spin_once is None:
            time.sleep(0.1)
        else:
            spin_once()


def _commands(
    config, rendered: dict[str, Path], *, include_xr_receiver: bool
) -> list[list[str]]:
    root = config.document
    rdk_command = _rdk_command(config)
    commands = [rdk_command]
    commands += [
        _control_command(config, rendered),
        _teleop_input_command(config, rendered),
        [
            "flexiv-inspire-camera-node",
            "--ros-args",
            "-p",
            f"config:={rendered['camera.yaml']}",
        ],
        [
            "flexiv-inspire-dftp-node",
            "--ros-args",
            "--params-file",
            str(rendered["dftp.yaml"]),
        ],
        [
            "flexiv-inspire-pedal-router",
            "--ros-args",
            "--params-file",
            str(rendered["pedal.yaml"]),
        ],
        _episode_command(config, rendered),
    ]
    rerun_command = _live_rerun_command(config)
    if rerun_command is not None:
        commands.append(rerun_command)
    # Quest controllers and MANUS are command inputs, not part of the optional
    # camera display branch. Exactly one provider owns the stable XR topics.
    commands.append(_quest_input_source_command(config))
    if bool(root["teleop"]["manus_ergonomics"]["enabled"]):
        commands.append(_manus_ergonomics_source_command(config))
    commands.append(_manus_plugin_command(config))
    # RTP encoding and the Holoscan receiver are an optional visual branch.
    # Start both when requested, but never couple their health to collection.
    if bool(root["xr_video"]["enabled"]) and include_xr_receiver:
        commands.append(
            [
                "flexiv-inspire-xr-bridge",
                "--ros-args",
                "--params-file",
                str(rendered["xr_bridge.yaml"]),
            ]
        )
        commands.append(_xr_receiver_command(config, rendered))
    return commands


def _is_optional_xr_video_command(command: Sequence[str]) -> bool:
    joined = " ".join(command)
    return (
        "flexiv-inspire-xr-bridge" in joined
        or "flexiv_inspire_xr_bridge.ros_node" in joined
        or "run_isaac_camera_receiver.sh" in joined
    )


def _rdk_command(config) -> list[str]:
    root = config.document
    runtime_root = Path(root["session"]["runtime_root"]).expanduser()
    configured = list(root["commands"]["rdk_daemon"])
    if not configured:
        raise SystemConfigError("commands.rdk_daemon cannot be empty")
    command = [
        *configured,
        "--config",
        str(config.resolve(root["flexiv"]["rdk_config"])),
        "--socket",
        str(runtime_root / "rdk.sock"),
        "--events",
        str(runtime_root / "ft_zero_events.jsonl"),
    ]
    if "--mock" not in configured:
        command.extend(
            [
                "--hardware",
                "--allow-hardware-writes",
                "--local-permit-file",
                str(runtime_root / f"{root['session']['id']}.write-permit"),
            ]
        )
    return command


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="robot", description="Config-driven independent Flexiv/Inspire operations"
    )
    parser.add_argument(
        "--config", "-c", default="config/site.yaml", help="single site YAML"
    )
    sub = parser.add_subparsers(dest="operation", required=True)
    sub.add_parser("validate", help="validate the site config without devices")
    sub.add_parser("render", help="write immutable child configs and print their paths")
    reset = sub.add_parser(
        "reset",
        help="F/T-zero, arm Home, then cycle both Inspire hands",
    )
    reset.add_argument("--preview-seconds", type=float, default=2.0)
    record = sub.add_parser(
        "record",
        help="run acquisition in the foreground; Ctrl-C finalizes and stops it",
    )
    record.add_argument(
        "--dry-run", action="store_true", help="render and print commands only"
    )
    record.add_argument(
        "--xr",
        action="store_true",
        help="start the optional IsaacTeleop Quest video display",
    )
    policy_serve = sub.add_parser(
        "policy-serve",
        help="run the hardware owner and TLS policy RPC server in the foreground",
    )
    policy_serve.add_argument(
        "--policy-config",
        default="config/policy_server.yaml",
        help="standalone policy server YAML",
    )
    policy_serve.add_argument(
        "--dry-run", action="store_true", help="render and print commands only"
    )
    policy_serve.add_argument(
        "--no-reset",
        action="store_true",
        help="start without F/T zero or Home; guarded actions remain unavailable until locally prepared",
    )
    sub.add_parser("stop", help="recover and stop leftover managed services")
    sub.add_parser(
        "collect", help="run the configured multi-episode collection in the foreground"
    )
    sub.add_parser(
        "visualize", help="visualize the dataset selected by config/playback.yaml"
    )
    sub.add_parser("replay", help="replay the dataset selected by config/playback.yaml")
    convert = sub.add_parser(
        "convert",
        help="export episodes selected by a conversion config to a training dataset",
    )
    convert.add_argument(
        "--action-view",
        choices=(
            "sent_command",
            "absolute_joint_position",
            "absolute_cartesian_pose",
        ),
        default=None,
    )
    convert.add_argument(
        "--manifest",
        default="",
        help="optional manifest; defaults to the latest completed episode",
    )
    convert.add_argument("--output-root", default="")
    convert.add_argument("--repo-id", default="")
    convert.add_argument(
        "--conversion-config",
        default="config/conversion.yaml",
        help="source selection and output settings (LeRobot or RL-100 Zarr)",
    )
    camera_calibrate = sub.add_parser(
        "camera-calibrate",
        help="collect or solve camera hand-eye calibration",
        add_help=False,
    )
    camera_calibrate.add_argument(
        "-h", "--help", action="store_true", dest="calibration_help"
    )
    camera_calibrate.add_argument("calibration_args", nargs=argparse.REMAINDER)
    xr_view = sub.add_parser(
        "xr-view", help="one-command IsaacTeleop camera display in Quest/monitor"
    )
    xr_view.add_argument("--dry-run", action="store_true")
    sub.add_parser(
        "xr-doctor", help="check the selected Quest input provider prerequisites"
    )
    return parser


def _xr_doctor(config) -> int:
    quest = _quest_input_config(config)
    provider = str(quest["provider"])
    if provider == "oculus_reader":
        adb = shutil.which("adb")
        checks = {
            "adb": adb is not None,
            "oculus_reader_python": importlib.util.find_spec("oculus_reader")
            is not None,
            "quest_usb_device": False,
        }
        devices: list[str] = []
        if adb is not None:
            try:
                result = subprocess.run(
                    [adb, "devices", "-l"],
                    capture_output=True,
                    text=True,
                    timeout=3.0,
                    check=False,
                )
                if result.returncode == 0:
                    devices = [
                        line.split()[0]
                        for line in result.stdout.splitlines()[1:]
                        if len(line.split()) >= 2 and line.split()[1] == "device"
                    ]
            except subprocess.TimeoutExpired:
                pass
        configured = str(quest["oculus_reader"]["adb_serial"]).strip()
        checks["quest_usb_device"] = (
            configured in devices if configured else len(devices) == 1
        )
        ready = all(checks.values())
        if not checks["oculus_reader_python"]:
            next_step = "sync requirements/ros-py312.lock"
        elif not checks["quest_usb_device"]:
            next_step = "unlock Quest, allow USB debugging, then run robot xr-doctor"
        else:
            next_step = "robot record"
        print(
            json.dumps(
                {
                    "ready": ready,
                    "provider": provider,
                    "devices": devices,
                    "checks": checks,
                    "next": next_step,
                },
                indent=2,
            )
        )
        return 0 if ready else 2

    root = _project_root(config)
    ffmpeg = str(config.document["xr_video"]["ffmpeg"])
    transport = str(config.document["xr_video"]["transport"])
    checks = {
        "ffmpeg": Path(ffmpeg).is_file() or shutil.which(ffmpeg) is not None,
        "docker": shutil.which("docker") is not None,
        "isaac_camera_streamer": (
            root / "third_party/IsaacTeleop/examples/camera_streamer/camera_streamer.sh"
        ).is_file(),
        "cloudxr_runtime_json": (
            Path.home() / ".cloudxr/openxr_cloudxr.json"
        ).is_file(),
    }
    image = False
    if checks["docker"]:
        image = (
            subprocess.run(
                ["docker", "image", "inspect", "isaac-teleop-camera:latest"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).returncode
            == 0
        )
    checks["isaac_camera_image"] = image
    if transport == "usb_tcp":
        checks["adb"] = shutil.which("adb") is not None
        checks["coturn"] = shutil.which("turnserver") is not None
        quest_ready = False
        if checks["adb"]:
            try:
                state = subprocess.run(
                    ["adb", "get-state"],
                    capture_output=True,
                    text=True,
                    timeout=3.0,
                )
                quest_ready = state.returncode == 0 and state.stdout.strip() == "device"
            except subprocess.TimeoutExpired:
                pass
        checks["quest_usb_device"] = quest_ready
    ready = all(checks.values())
    if transport == "usb_tcp" and not checks.get("quest_usb_device", True):
        next_step = "unlock Quest and allow USB debugging, then run robot xr-doctor"
    elif not ready:
        next_step = "orchestration/setup_xr_receiver.sh"
    else:
        next_step = "robot record"
    print(
        json.dumps(
            {
                "ready": ready,
                "provider": provider,
                "transport": transport,
                "checks": checks,
                "next": next_step,
            },
            indent=2,
        )
    )
    return 0 if ready else 2


def _main(args, config) -> int:
    runtime = _runtime_dir(config)
    if args.operation == "validate":
        print(
            json.dumps(
                {"valid": True, "config": str(config.path), "sha256": config.sha256},
                indent=2,
            )
        )
        return 0
    if args.operation == "render":
        rendered = render_runtime_configs(config, runtime / config.sha256[:12])
        print(
            json.dumps({key: str(value) for key, value in rendered.items()}, indent=2)
        )
        return 0
    if args.operation == "reset":
        return _run_reset(config, args)
    if args.operation == "camera-calibrate":
        from flexiv_inspire_isaac.cameras.calibration import main as calibration_main

        selected = ["--help"] if args.calibration_help else args.calibration_args
        return calibration_main(selected)
    if args.operation == "xr-doctor":
        return _xr_doctor(config)
    if args.operation == "xr-view":
        rendered = render_runtime_configs(config, runtime / config.sha256[:12])
        command = _xr_receiver_command(config, rendered)
        if args.dry_run:
            print(json.dumps(command, indent=2))
            return 0
        return subprocess.call(command)
    if args.operation == "record":
        rendered = render_runtime_configs(config, runtime / config.sha256[:12])
        xr_video_enabled = bool(config.document["xr_video"]["enabled"])
        include_xr = xr_video_enabled and args.xr
        commands = _commands(config, rendered, include_xr_receiver=include_xr)
        if args.dry_run:
            print(json.dumps(commands, indent=2))
            return 0
        if _process_running(
            "flexiv_inspire_isaac.episode_control",
            "flexiv-inspire-episode-controller",
            "flexiv_inspire_isaac.pedal_router",
            "flexiv-inspire-pedal-router",
        ):
            raise SystemExit(
                "已有旧的后台录制/脚踏进程；请先执行一次 robot stop，"
                "之后 robot record 将以前台方式运行"
            )
        _require_camera_devices_available(config)
        if bool(config.document["recording"].get("auto_reset_before_record", False)):
            print("开始自动准备真机（Reset/F/T/Home/双手张开）", flush=True)
            reset_result = _run_reset(config, argparse.Namespace(preview_seconds=2.0))
            if reset_result != 0:
                raise SystemExit(f"自动真机准备失败，Reset 退出码 {reset_result}")
            print("真机准备完成，控制状态将由 READY 进入遥操", flush=True)
        # Pedal routing and the episode controller are started by
        # _run_collection() so the latter can remain attached to this TTY.
        commands = [
            command
            for command in commands
            if not _is_collection_process_command(command)
        ]
        rdk_socket = (
            Path(config.document["session"]["runtime_root"]).expanduser() / "rdk.sock"
        )
        if _rdk_socket_live(rdk_socket):
            commands = commands[1:]
            print("复用已运行的 RDK daemon", flush=True)
        if _process_running(
            "flexiv_inspire_control.node",
            "flexiv_inspire_control/control_bridge",
        ):
            commands = [
                command
                for command in commands
                if "flexiv_inspire_control.node" not in command
            ]
        if _process_running(
            "flexiv_inspire_control.teleop_input_node",
            "flexiv_inspire_control/teleop_input",
        ):
            commands = [
                command
                for command in commands
                if "flexiv_inspire_control.teleop_input_node" not in command
            ]
        # Reset intentionally leaves DFTP running with the control bridge.  All
        # camera/XR/MANUS/Rerun services belong to this foreground record run
        # and must start fresh, otherwise a live-looking stale process can hide
        # a dead video or glove stream.
        reusable_services = (
            (
                (
                    "flexiv_inspire_isaac.dftp.ros_node",
                    "flexiv-inspire-dftp-node",
                    "flexiv_inspire_dftp/dftp_node",
                ),
                ("flexiv-inspire-dftp-node",),
            ),
        )
        for process_markers, command_markers in reusable_services:
            if _process_running(*process_markers):
                commands = [
                    command
                    for command in commands
                    if not any(
                        marker in " ".join(command) for marker in command_markers
                    )
                ]
        optional_video_commands = [
            command for command in commands if _is_optional_xr_video_command(command)
        ]
        commands = [
            command
            for command in commands
            if not _is_optional_xr_video_command(command)
        ]
        log_prefix = f"record-{time.time_ns()}"
        logs = [runtime / f"{log_prefix}-{index}.log" for index in range(len(commands))]
        processes = _start(commands, runtime, log_prefix=log_prefix)
        _write_state(runtime, config, processes)
        _verify_process_startup(processes, logs)
        _wait_for_rdk_socket(rdk_socket, processes, logs)
        _wait_for_camera_streams(
            commands,
            processes,
            logs,
            tuple(
                name
                for name, stream in config.document["cameras"]["streams"].items()
                if bool(stream.get("enabled", True))
            ),
        )
        if any("run_xr_raw_source.sh" in " ".join(command) for command in commands):
            _wait_for_cloudxr_runtime(processes, logs)
        if str(config.document["teleop"]["manus_calibration"]).strip():
            _report_manus_status(commands, processes, logs)

        # Video is an optional display branch. Quest controller freshness and
        # the physical pedal gate motion independently inside teleop_input.
        video_processes: list[subprocess.Popen] = []
        video_logs: list[Path] = []
        if optional_video_commands:
            video_prefix = f"{log_prefix}-video"
            video_logs = [
                runtime / f"{video_prefix}-{index}.log"
                for index in range(len(optional_video_commands))
            ]
            try:
                video_processes = _start(
                    optional_video_commands,
                    runtime,
                    log_prefix=video_prefix,
                )
            except OSError as exc:
                print(
                    f"Quest 视频启动失败；record 和遥操继续：{exc}",
                    flush=True,
                )
            else:
                processes.extend(video_processes)
                logs.extend(video_logs)
                _write_state(runtime, config, video_processes)
                print(
                    "Quest 视频已在后台启动；显示失败不会停止 record 或遥操",
                    flush=True,
                )
        print(
            f"started {len(processes)} supporting services; "
            f"Quest tracking={_quest_input_config(config)['provider']}; "
            f"XR video={'on' if include_xr else 'off'}",
            flush=True,
        )
        try:
            if include_xr:
                enabled_video_streams = tuple(
                    name
                    for name, settings in config.document["xr_video"]["streams"].items()
                    if bool(settings["enabled"])
                )
                _report_optional_xr_video_status(
                    optional_video_commands,
                    video_processes,
                    video_logs,
                    enabled_video_streams,
                )
            try:
                return _run_collection(config, rendered)
            except KeyboardInterrupt:
                return 130
        finally:
            _stop_started_processes(processes)
            _remove_state_pids(runtime, {process.pid for process in processes})
    if args.operation == "policy-serve":
        rendered = render_runtime_configs(config, runtime / config.sha256[:12])
        settings = _load_policy_server_config(config, args.policy_config)
        commands = _policy_serve_commands(config, rendered, settings)
        if args.dry_run:
            print(json.dumps(commands, indent=2))
            return 0
        _require_camera_devices_available(config)
        reset_requested = settings["auto_reset_before_serve"] and not args.no_reset
        if reset_requested:
            print(
                "Policy server: 自动准备真机（F/T 清零 -> 安全抬升 -> Home -> 双手张开）",
                flush=True,
            )
            reset_result = _run_reset(
                config, argparse.Namespace(preview_seconds=2.0)
            )
            if reset_result != 0:
                raise SystemExit(f"Policy server 真机准备失败，Reset 退出码 {reset_result}")
        rdk_socket = (
            Path(config.document["session"]["runtime_root"]).expanduser()
            / "rdk.sock"
        )
        if _rdk_socket_live(rdk_socket):
            commands = [
                command
                for command in commands
                if "flexiv-rdk-daemon" not in " ".join(command)
            ]
            print("Policy server 复用已运行的 RDK daemon", flush=True)
        reusable_services = (
            (
                (
                    "flexiv_inspire_control.node",
                    "flexiv-inspire-control-bridge",
                    "flexiv_inspire_control/control_bridge",
                ),
                ("flexiv_inspire_control.node", "flexiv-inspire-control-bridge"),
            ),
            (
                (
                    "flexiv_inspire_isaac.dftp.ros_node",
                    "flexiv-inspire-dftp-node",
                    "flexiv_inspire_dftp/dftp_node",
                ),
                ("flexiv-inspire-dftp-node",),
            ),
        )
        for process_markers, command_markers in reusable_services:
            if _process_running(*process_markers):
                commands = [
                    command
                    for command in commands
                    if not any(
                        marker in " ".join(command) for marker in command_markers
                    )
                ]
        log_prefix = f"policy-serve-{time.time_ns()}"
        logs = [runtime / f"{log_prefix}-{index}.log" for index in range(len(commands))]
        processes = _start(commands, runtime, log_prefix=log_prefix)
        _write_state(runtime, config, processes)
        authorizer = None
        try:
            _verify_process_startup(processes, logs)
            _wait_for_rdk_socket(rdk_socket, processes, logs)
            _wait_for_camera_streams(
                commands,
                processes,
                logs,
                tuple(
                    name
                    for name, stream in config.document["cameras"]["streams"].items()
                    if bool(stream.get("enabled", True))
                ),
            )
            if settings["auto_authorize_policy"] and reset_requested:
                policy_pedal_required = bool(
                    config.document["flexiv"]["policy_control"][
                        "require_pedal"
                    ]
                )
                if policy_pedal_required:
                    authorizer = _PolicyAuthorizationSupervisor(
                        config,
                        rdk_socket,
                        require_pedal=True,
                    )
                    authorization_status = (
                        "中踏板踩下时自动授权 policy；松开停止，再踩自动恢复"
                    )
                else:
                    # Direct policy mode is action-driven inside the control
                    # bridge.  Running the legacy periodic authorizer here
                    # races that path and can invalidate its one-shot token.
                    authorization_status = (
                        "Policy 直连模式：无需中踏板，收到客户端动作即执行"
                    )
            else:
                authorization_status = (
                    "未启用自动授权；--no-reset 仅用于 shadow/只读诊断"
                )
            print(
                f"Policy RPC ready: {settings['bind']}:{settings['port']}\n"
                "hardware owner=isaac_teleop_flexiv_inspire; "
                "training/inference client=dual_arm_teleop\n"
                f"{authorization_status}\n"
                "Ctrl-C stops server",
                flush=True,
            )
            try:
                _wait_for_policy_services(
                    processes,
                    logs,
                    spin_once=(authorizer.spin_once if authorizer is not None else None),
                )
            except KeyboardInterrupt:
                return 130
        finally:
            if authorizer is not None:
                authorizer.close()
            _stop_started_processes(processes)
            _remove_state_pids(runtime, {process.pid for process in processes})
    if args.operation == "collect":
        rendered = render_runtime_configs(config, runtime / config.sha256[:12])
        return _run_collection(config, rendered)
    if args.operation == "stop":
        _stop_managed_services(config, require_existing=True)
        print("stop requested; captured episodes are retained")
        return 0
    if args.operation == "convert":
        return _run_convert(config, args)
    if args.operation == "visualize":
        from flexiv_inspire_isaac.rerun_viz.offline import main as visualize_main

        return visualize_main([])
    if args.operation == "replay":
        return _run_replay(config)
    from .replay import main as replay_main

    return replay_main([])


def _start_replay_pedal_router(config) -> list[subprocess.Popen]:
    """Start only the physical pedal input needed by standalone replay."""

    if _process_running(
        "flexiv_inspire_isaac.pedal_router",
        "flexiv-inspire-pedal-router",
    ):
        print("Replay 复用已运行的踏板路由", flush=True)
        return []
    runtime = _runtime_dir(config)
    rendered = render_runtime_configs(config, runtime / config.sha256[:12])
    prefix = f"replay-{time.time_ns()}"
    command = [
        sys.executable,
        "-m",
        "flexiv_inspire_isaac.pedal_router",
        "--ros-args",
        "--params-file",
        str(rendered["pedal.yaml"]),
    ]
    processes = _start([command], runtime, log_prefix=prefix)
    logs = [runtime / f"{prefix}-0.log"]
    _verify_process_startup(processes, logs, timeout_s=0.25)
    print("Replay 踏板路由已自动启动", flush=True)
    return processes


def _publish_replay_deadman_release() -> None:
    """Deliver a clutch-release edge before standalone replay cleanup."""

    try:
        import rclpy
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        from std_msgs.msg import Bool

        rclpy.init()
        node = rclpy.create_node("flexiv_inspire_replay_cleanup")
        publisher = node.create_publisher(
            Bool, "/teleop/deadman", QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        )
        try:
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline and publisher.get_subscription_count() < 1:
                rclpy.spin_once(node, timeout_sec=0.05)
            message = Bool()
            message.data = False
            for _ in range(3):
                publisher.publish(message)
                rclpy.spin_once(node, timeout_sec=0.05)
        finally:
            node.destroy_node()
            rclpy.shutdown()
    except Exception as exc:
        # Reset below remains the recovery authority if ROS was already torn
        # down unexpectedly. Keep this best-effort release non-fatal.
        print(f"Replay 清理：踏板释放通知未确认（将继续 Reset）：{exc}", flush=True)


def _run_replay(config) -> int:
    """Prepare the local stack and run one foreground hardware replay."""

    # Replay is a hardware operation just like record: a bare command must be
    # enough on a freshly booted workstation. Reset owns the RDK daemon and
    # ROS bridge lifecycle, establishes this session's F/T zero, and moves to
    # Home before replay's own trajectory checks.
    print("Replay: 自动启动 RDK/ROS 并准备真机（F/T 清零 -> Home）", flush=True)
    reset_result = _run_reset(config, argparse.Namespace(preview_seconds=0.0))
    if reset_result != 0:
        raise SystemExit(f"Replay 自动准备失败，Reset 退出码 {reset_result}")
    pedal_processes = _start_replay_pedal_router(config)
    replay_started = False
    try:
        print("Replay: 真机准备完成，开始校验并回放选中的 episode", flush=True)
        from .replay import main as replay_main

        replay_started = True
        return replay_main([])
    finally:
        if pedal_processes:
            _stop_started_processes(pedal_processes)
        _publish_replay_deadman_release()
        if replay_started:
            print("Replay 已结束，正在自动 Reset（F/T -> Home -> 双手张开）", flush=True)
            try:
                recovery = _run_reset(config, argparse.Namespace(preview_seconds=0.0))
                if recovery != 0:
                    print(f"Replay 收尾 Reset 失败，退出码 {recovery}", flush=True)
            except BaseException as exc:
                print(f"Replay 收尾 Reset 失败：{exc}", flush=True)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = load_system_config(args.config)
    except (OSError, SystemConfigError) as exc:
        raise SystemExit(f"invalid system config: {exc}")

    foreground_operations = {"record", "policy-serve"}
    if args.operation not in foreground_operations or args.dry_run:
        return _main(args, config)

    runtime_root = Path(config.document["session"]["runtime_root"]).expanduser()
    runtime_root.mkdir(parents=True, exist_ok=True)
    lock_name = "record.lock" if args.operation == "record" else "policy-serve.lock"
    operation_lock = (runtime_root / lock_name).open("a+", encoding="utf-8")
    try:
        fcntl.flock(operation_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        operation_lock.close()
        raise SystemExit(f"another robot {args.operation} process is already active")

    failure: BaseException | None = None
    try:
        stale_count = _stop_managed_services(config, require_existing=False)
        if stale_count:
            print(
                f"已自动关闭上次遗留的 {stale_count} 个后台服务",
                flush=True,
            )
        return _main(args, config)
    except BaseException as exc:
        failure = exc
        raise
    finally:
        try:
            _stop_managed_services(config, require_existing=False)
            print(
                f"{args.operation} 已退出；本次启动的全部后台服务已关闭",
                flush=True,
            )
        except Exception as cleanup_error:
            print(
                f"record 退出清理失败: {cleanup_error}",
                file=sys.stderr,
                flush=True,
            )
            if failure is None:
                raise
        finally:
            fcntl.flock(operation_lock.fileno(), fcntl.LOCK_UN)
            operation_lock.close()


def _episode_manifest_paths(dataset_root: Path) -> list[Path]:
    """Find current ``raw/`` episodes and the legacy flat layout."""

    paths = set(dataset_root.glob("raw/*/manifest.json"))
    paths.update(dataset_root.glob("*/manifest.json"))
    return sorted(paths)


def _recording_dataset_root(config) -> Path:
    recording = config.document["recording"]
    return config.resolve(str(recording.get("output_root", "sessions"))) / str(
        recording["dataset_name"]
    )


def _conversion_manifests(config, source: dict) -> list[Path]:
    explicit = str(source.get("manifest", "")).strip()
    if explicit:
        path = config.resolve(explicit)
        if not path.is_file():
            raise FileNotFoundError(f"manifest 不存在: {path}")
        return [path]
    dataset_root_raw = str(source.get("dataset_root", "")).strip()
    dataset_root = (
        config.resolve(dataset_root_raw)
        if dataset_root_raw
        else _recording_dataset_root(config)
    )
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"数据集目录不存在: {dataset_root}")
    selector = str(source.get("episode", "latest")).strip() or "latest"
    candidates: list[tuple[tuple[int, int, int], Path, dict]] = []
    for path in _episode_manifest_paths(dataset_root):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(document, dict) or not bool(document.get("completed", False)):
            continue
        candidates.append(
            (
                (
                    int(document.get("episode_index", 0)),
                    int(document.get("attempt", 0)),
                    path.stat().st_mtime_ns,
                ),
                path,
                document,
            )
        )
    if selector in {"latest", "all"}:
        matching = candidates
    elif selector.isdigit():
        matching = [
            item
            for item in candidates
            if int(item[2].get("episode_index", -1)) == int(selector)
        ]
    else:
        matching = [item for item in candidates if item[1].parent.name == selector]
    if not matching:
        raise FileNotFoundError(
            f"没有匹配的已完成 episode: {dataset_root} / {selector}"
        )
    ordered = sorted(matching, key=lambda item: item[0])
    if selector == "all":
        return [item[1] for item in ordered]
    return [ordered[-1][1]]


def _conversion_manifest(config, source: dict) -> Path:
    """Compatibility helper for callers that need one selected episode."""

    return _conversion_manifests(config, source)[-1]


def _run_convert(config, args) -> int:
    root = _project_root(config)
    conversion_path = config.resolve(args.conversion_config)
    try:
        conversion = yaml.safe_load(conversion_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SystemExit(f"转换配置无法读取: {conversion_path}: {exc}") from exc
    if (
        not isinstance(conversion, dict)
        or int(conversion.get("schema_version", 0)) != 1
    ):
        raise SystemExit("conversion config 必须是 schema_version: 1")
    source = conversion.get("source", {})
    output = conversion.get("output", {})
    if not isinstance(source, dict) or not isinstance(output, dict):
        raise SystemExit("conversion source/output 必须是 mapping")
    manifests = (
        [config.resolve(args.manifest)]
        if str(args.manifest).strip()
        else _conversion_manifests(config, source)
    )
    for manifest in manifests:
        if not manifest.is_file():
            raise SystemExit(f"manifest 不存在: {manifest}")
    output_format = str(output.get("format", "lerobot")).strip() or "lerobot"
    if output_format not in {"lerobot", "rl100_zarr"}:
        raise SystemExit("conversion output.format 必须是 lerobot 或 rl100_zarr")
    export = conversion.get(
        "rl100_zarr_export" if output_format == "rl100_zarr" else "lerobot_export",
        {},
    )
    profile = (
        str(export.get("profile", "")).strip()
        if isinstance(export, dict)
        else ""
    )
    configured_view = (
        export.get("action", {}).get("view")
        if isinstance(export, dict) and isinstance(export.get("action", {}), dict)
        else None
    ) or config.document["lerobot_export"]["action"]["view"]
    action_view = str(args.action_view or configured_view)
    dataset_name = str(config.document["recording"]["dataset_name"])
    episode_subdirectory = bool(output.get("episode_subdirectory", True))
    configured_root = str(output.get("root", "")).strip()
    explicit_output_root = str(args.output_root).strip()
    dataset_root = _recording_dataset_root(config)
    default_output_root = (
        dataset_root / "rl100" / f"{profile or 'dataset'}.zarr"
        if output_format == "rl100_zarr"
        else dataset_root / "lerobot"
    )
    base = (
        config.resolve(explicit_output_root)
        if explicit_output_root
        else (
            config.resolve(configured_root)
            if configured_root
            else default_output_root
        )
    )
    if len(manifests) > 1 and not episode_subdirectory and not profile:
        raise SystemExit("批量转换要求 output.episode_subdirectory: true")
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", dataset_name).strip("-") or "dataset"
    repo_id = (
        str(args.repo_id).strip()
        or str(output.get("repo_id", "")).strip()
        or f"local/{slug}-{profile or action_view}"
    )
    executable_name = (
        "flexiv-inspire-rl100-zarr-export"
        if output_format == "rl100_zarr"
        else (
            "flexiv-inspire-policy-profile-export"
            if profile
            else "flexiv-inspire-lerobot-export"
        )
    )
    executable = root / "envs/data-py312/bin" / executable_name
    if not executable.is_file():
        raise SystemExit("data 环境不存在；先运行 scripts/env/create_envs.sh")
    mcap_files = source.get("mcap_files", [])
    if not isinstance(mcap_files, list):
        raise SystemExit("conversion source.mcap_files 必须是列表")
    mcaps = [config.resolve(str(raw_mcap)) for raw_mcap in mcap_files]
    for mcap in mcaps:
        if not mcap.is_file():
            raise SystemExit(f"MCAP 不存在: {mcap}")

    if output_format == "rl100_zarr":
        if profile not in {
            "joint_proprio_cartesian_v1",
            "right_joint_proprio_cartesian_v1",
        }:
            raise SystemExit(
                "RL-100 Zarr 不支持该 profile"
            )
        if action_view != "sent_command":
            raise SystemExit("RL-100 Zarr 仅支持 action.view: sent_command")
        if episode_subdirectory:
            raise SystemExit(
                "RL-100 Zarr 合并导出要求 output.episode_subdirectory: false"
            )
        if mcaps:
            raise SystemExit(
                "RL-100 Zarr 从每个 manifest 解析 MCAP；请删除 source.mcap_files"
            )
        if base.exists() and (not base.is_dir() or any(base.iterdir())):
            raise SystemExit(f"输出目录已存在且非空: {base}；请指定 --output-root")
        command = [str(executable)]
        for manifest in manifests:
            command.extend(("--manifest", str(manifest)))
        command.extend(
            (
                "--output-root",
                str(base),
                "--export-config",
                str(conversion_path),
            )
        )
        print(
            f"合并转换 {len(manifests)} 条 episode -> RL-100 Zarr ({profile}) -> {base}",
            flush=True,
        )
        status = subprocess.call(command)
        if status:
            print(f"RL-100 Zarr 转换失败（退出码 {status}）", flush=True)
            return 1
        print(f"转换完成：成功合并 {len(manifests)} 条 episode", flush=True)
        return 0

    if profile:
        if action_view != "sent_command":
            raise SystemExit("policy profile 只支持 action.view: sent_command")
        if episode_subdirectory:
            raise SystemExit(
                "policy profile 合并导出要求 output.episode_subdirectory: false"
            )
        if mcaps:
            raise SystemExit(
                "policy profile 批量导出从每个 manifest 解析 MCAP；"
                "请删除 source.mcap_files"
            )
        if base.exists() and (not base.is_dir() or any(base.iterdir())):
            raise SystemExit(f"输出目录已存在且非空: {base}；请指定 --output-root")
        command = [str(executable)]
        for manifest in manifests:
            command.extend(("--manifest", str(manifest)))
        command.extend(
            (
                "--output-root",
                str(base),
                "--repo-id",
                repo_id,
                "--export-config",
                str(conversion_path),
            )
        )
        print(
            f"合并转换 {len(manifests)} 条 episode -> {profile} -> {base}",
            flush=True,
        )
        status = subprocess.call(command)
        if status:
            print(f"合并转换失败（退出码 {status}）", flush=True)
            return 1
        print(f"转换完成：成功合并 {len(manifests)} 条 episode", flush=True)
        return 0

    failed = 0
    converted = 0
    skipped = 0
    for manifest in manifests:
        output_root = (
            base
            if explicit_output_root and len(manifests) == 1
            else (
                base / manifest.parent.name / action_view
                if episode_subdirectory
                else base
            )
        )
        if output_root.exists() and (
            not output_root.is_dir() or any(output_root.iterdir())
        ):
            if len(manifests) == 1:
                raise SystemExit(
                    f"输出目录已存在且非空: {output_root}；请指定 --output-root"
                )
            print(f"跳过已有转换结果: {manifest.parent.name}", flush=True)
            skipped += 1
            continue
        command = [
            str(executable),
            "--manifest",
            str(manifest),
            "--output-root",
            str(output_root),
            "--repo-id",
            repo_id,
            "--export-config",
            str(conversion_path),
            "--action-view",
            action_view,
        ]
        for mcap in mcaps:
            command.extend(("--mcap", str(mcap)))
        print(
            f"转换 {manifest.parent.name} -> {action_view} -> {output_root}",
            flush=True,
        )
        status = subprocess.call(command)
        if status:
            print(f"转换失败: {manifest.parent.name}（退出码 {status}）", flush=True)
            failed += 1
        else:
            converted += 1
    print(
        f"转换完成：成功 {converted}，跳过 {skipped}，失败 {failed}", flush=True
    )
    return 1 if failed else 0


def _run_reset(config, args) -> int:
    """Run the guarded session-start Reset as one local interaction."""

    if not sys.stdin.isatty():
        raise SystemExit("Reset 必须从机器人本机的交互式终端执行")
    root = config.document
    rdk_socket = _ensure_rdk_daemon(config)
    _ensure_reset_ros_services(config)
    tool_config = config.resolve(root["flexiv"]["tool_payload_config"])
    if not tool_config.is_file():
        raise SystemExit(f"工具/负载配置不是文件: {tool_config}")

    print(
        "Reset: 清除双臂控制器故障 -> 双臂 F/T 清零 -> "
        "TCP 抬升并对齐 Home XY -> 配置的双臂 Home -> "
        "双手张开/闭合/张开",
        flush=True,
    )

    from flexiv_inspire_control.zero_ft_local import main as zero_ft_main

    zero_args = [
        "--rdk-socket",
        str(rdk_socket),
        "--tool-payload-config",
        str(tool_config),
        "--preview-seconds",
        str(float(args.preview_seconds)),
        "--hand-warmup-seconds",
        "1.0",
        "--max-hand-delta",
        "150",
        "--skip-preview-if-ft-zeroed",
        "--confirm-ft-unloaded",
        "FLEXIV-FT-UNLOADED",
        "--home-after-zero",
        "--cycle-hands-after-home",
        "--home-timeout",
        str(_home_motion_timeout_s(root)),
        # Zeroing intentionally latches the daemon in maintenance. This
        # authorization clears only that freshly validated hold before Home.
        "--clear-home-hold-latched",
    ]
    try:
        return zero_ft_main(zero_args)
    except (RuntimeError, TimeoutError) as exc:
        message = str(exc)
        # A matching process name only proves that an old bridge/DFTP process
        # still exists.  It does not prove that this ROS domain is receiving
        # /control/state and both hand-state streams.  Recover that stale
        # service reuse locally, before any F/T or Home request is issued.
        if "Reset 等待状态超时" in message:
            print(
                "Reset: 未收到控制或双手状态，正在重启 ROS 控制桥和手部服务后重试一次",
                flush=True,
            )
            _restart_reset_ros_processes()
            _ensure_reset_ros_services(config)
            return zero_ft_main(zero_args)

        # The bridge intentionally latches a hardware command failure as
        # FAULT, but the F/T action can only run from MAINTENANCE. Restarting
        # the ROS bridge resets that software latch while keeping the same RDK
        # daemon; the retried F/T transaction then invokes RDK ClearFault on
        # both real controllers before Enable/ZeroFT/Home.
        if "got FAULT" in message:
            print(
                "Reset: 检测到控制 FAULT，正在重启 ROS 控制桥并调用 RDK ClearFault",
                flush=True,
            )
            _restart_reset_ros_processes()
            _ensure_reset_ros_services(config)
            return zero_ft_main(zero_args)

        # A failed RDK mode transition can leave the controller call inside
        # the daemon blocked even though the Unix socket still accepts peers.
        # Reusing that process merely turns the original Home error into an
        # opaque authorize_home timeout. Reset is a one-shot local operation,
        # so discard the stuck stack and retry once from fresh connections.
        retryable = isinstance(exc, TimeoutError) or any(
            marker in message
            for marker in (
                "timed out",
                "Resource temporarily unavailable",
                "SwitchMode",
                "SendJointPosition",
                "feasible trajectory",
                "closed IPC connection",
                # A failed replay can leave its source latched while the
                # middle pedal is still held. Restarting the locally managed
                # bridge is the practical reset path: it clears that stale
                # source ownership before F/T/Home runs again.
                "source deadman must be released before rearming",
            )
        )
        if not retryable:
            raise
        print(
            "Reset: RDK Home 通道异常，正在自动重启 daemon 和 ROS 服务后重试一次",
            flush=True,
        )
        _stop_managed_services(config, require_existing=False)
        rdk_socket = _ensure_rdk_daemon(config)
        _ensure_reset_ros_services(config)
        zero_args[zero_args.index("--rdk-socket") + 1] = str(rdk_socket)
        return zero_ft_main(zero_args)


def _ensure_reset_ros_services(config) -> None:
    runtime = _runtime_dir(config)
    state_path = runtime / "reset-services.json"
    runtime_sha256 = _reset_ros_runtime_fingerprint(config)
    previous_runtime_sha256 = ""
    if state_path.is_file():
        try:
            previous_runtime_sha256 = str(
                json.loads(state_path.read_text(encoding="utf-8")).get(
                    "runtime_sha256", ""
                )
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            previous_runtime_sha256 = ""
    control_running = _process_running(
        "flexiv_inspire_control.node",
        "flexiv-inspire-control-bridge",
        "flexiv_inspire_control/control_bridge",
    )
    dftp_running = _process_running(
        "flexiv_inspire_isaac.dftp.ros_node",
        "flexiv-inspire-dftp-node",
        "flexiv_inspire_dftp/dftp_node",
    )
    if (control_running or dftp_running) and previous_runtime_sha256 != runtime_sha256:
        _restart_reset_ros_processes()
        control_running = False
        dftp_running = False
        print("Reset 配置已更新，ROS 控制与手部服务已自动重启", flush=True)

    rendered = render_runtime_configs(config, runtime / config.sha256[:12])
    commands: list[list[str]] = []
    if not control_running:
        commands.append(_control_command(config, rendered))
    if not dftp_running:
        commands.append(
            [
                sys.executable,
                "-m",
                "flexiv_inspire_isaac.dftp.ros_node",
                "--ros-args",
                "--params-file",
                str(rendered["dftp.yaml"]),
            ]
        )
    log_prefix = f"reset-{time.time_ns()}"
    processes = _start(commands, runtime, log_prefix=log_prefix)
    log_paths = [
        runtime / f"{log_prefix}-{index}.log" for index in range(len(commands))
    ]
    state_path.write_text(
        json.dumps(
            {
                "pids": [process.pid for process in processes],
                "commands": commands,
                "logs": [str(path) for path in log_paths],
                "started_unix_ns": time.time_ns(),
                "config_sha256": config.sha256,
                "runtime_sha256": runtime_sha256,
                "reused": {
                    "control_bridge": control_running,
                    "dftp": dftp_running,
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if commands:
        print("Reset 所需的 ROS 控制和手部服务已自动启动", flush=True)
        # Catch an immediate import/configuration failure without paying the old
        # unconditional two-second sleep.  zero_ft_local then waits for actual
        # fresh control and hand messages, so ROS readiness—not elapsed time—
        # decides when Reset may continue.
        _verify_process_startup(processes, log_paths, timeout_s=0.25)
    else:
        print("Reset 复用已就绪的 ROS 控制和手部服务", flush=True)


def _ensure_rdk_daemon(config, timeout_s: float = 45.0) -> Path:
    runtime_root = Path(config.document["session"]["runtime_root"]).expanduser()
    rdk_socket = runtime_root / "rdk.sock"
    command = _rdk_command(config)
    if "--mock" in command:
        raise SystemExit(
            "当前 config 把 RDK daemon 配置为 mock；真机 Reset 需要 hardware 配置"
        )
    config_path = Path(command[command.index("--config") + 1])
    config_sha256 = _rdk_runtime_fingerprint(config, config_path)
    launcher = _runtime_dir(config)
    state_path = launcher / "rdk-daemon.json"
    if _rdk_socket_live(rdk_socket):
        restarted = _restart_managed_rdk_if_config_changed(
            state_path,
            rdk_socket=rdk_socket,
            config_sha256=config_sha256,
        )
        if not restarted:
            return rdk_socket

    permit = Path(command[command.index("--local-permit-file") + 1])
    _ensure_write_permit(permit)
    launcher.mkdir(parents=True, exist_ok=True)
    log_path = launcher / "rdk-daemon.log"
    log = log_path.open("ab", buffering=0)
    process = subprocess.Popen(
        command,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=_command_environment(command),
    )
    state_path.write_text(
        json.dumps(
            {
                "pid": process.pid,
                "command": command,
                "config_sha256": config_sha256,
                "started_unix_ns": time.time_ns(),
                "log": str(log_path),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"RDK daemon 未运行，已自动启动（PID {process.pid}）", flush=True)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if _rdk_socket_live(rdk_socket):
            return rdk_socket
        status = process.poll()
        if status is not None:
            detail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
            raise SystemExit(f"RDK daemon 启动失败，退出码 {status}\n{detail}")
        time.sleep(0.1)
    raise SystemExit(f"RDK daemon 启动超时；查看日志：{log_path}")


def _wait_for_collection_processes(
    controller: subprocess.Popen,
    pedal: subprocess.Popen,
    *,
    poll_s: float = 0.1,
) -> int:
    """Wait for collection while treating the pedal router as essential."""

    while True:
        controller_status = controller.poll()
        if controller_status is not None:
            return int(controller_status)
        pedal_status = pedal.poll()
        if pedal_status is not None:
            raise RuntimeError(
                f"脚踏输入进程意外退出，已停止本次录制，退出码 {pedal_status}"
            )
        time.sleep(poll_s)


def _run_collection(config, rendered: dict[str, Path]) -> int:
    _collection_preflight(config, rendered)
    runtime_root = Path(config.document["session"]["runtime_root"]).expanduser()
    lock = (runtime_root / "collection.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise SystemExit("another collection process is already active")
    recording = config.document["recording"]
    output = (
        config.resolve(recording["output_root"])
        / recording["dataset_name"]
        / "raw"
    )
    print(
        f"collection starting: {recording['episode_count']} episodes -> {output}\n"
        f"task name: {recording['task_name']}\n"
        f"task: {recording['task_description']}\n"
        + (
            "capture: middle pedal pressed only; action=sent_command\n"
            if recording.get("record_only_while_pedal_pressed", False)
            and recording.get("deviceio_profile", "full")
            in {"training", "right_training"}
            else "capture: whole episode\n"
        )
        + "right=commit/Home/next, left=discard/Home/retry, Quest A=pause/Home/resume, Ctrl-C=save/stop",
        flush=True,
    )
    pedal = None
    controller = None
    collection_pids: set[int] = set()
    try:
        pedal = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "flexiv_inspire_isaac.pedal_router",
                "--ros-args",
                "--params-file",
                str(rendered["pedal.yaml"]),
            ],
            start_new_session=True,
        )
        controller = subprocess.Popen(
            _episode_command(config, rendered), start_new_session=True
        )
        collection_pids = {pedal.pid, controller.pid}
        _write_state(_runtime_dir(config), config, [pedal, controller])
        try:
            return _wait_for_collection_processes(controller, pedal)
        except KeyboardInterrupt:
            return 130
    finally:
        # SIGINT lets EpisodeController finalize the current attempt and flush
        # both ROS MCAP and native DeviceIO before supporting services stop.
        if controller is not None and controller.poll() is None:
            controller.send_signal(signal.SIGINT)
            try:
                # EpisodeController may legitimately spend 30 seconds flushing
                # rosbag/MCAP. Leave headroom before escalating.
                controller.wait(timeout=45.0)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                # The recorder and rosbag inherit the controller's process
                # group. Stop the whole group so a recorder cannot be orphaned.
                try:
                    os.killpg(controller.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    controller.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(controller.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    controller.wait(timeout=5.0)
        if pedal is not None and pedal.poll() is None:
            pedal.terminate()
            try:
                pedal.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                pedal.kill()
                pedal.wait(timeout=5.0)
        _remove_state_pids(_runtime_dir(config), collection_pids)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


def _collection_preflight(config, rendered: dict[str, Path]) -> None:
    root = config.document
    runtime_root = Path(root["session"]["runtime_root"]).expanduser()
    required_commands = ("ros2",)
    missing_commands = [
        command for command in required_commands if shutil.which(command) is None
    ]
    if missing_commands:
        raise SystemExit(
            "collection preflight failed; missing commands: "
            + ", ".join(missing_commands)
        )
    if (
        bool(root["recording"].get("auto_authorize_home", True))
        or bool(root["recording"].get("auto_authorize_control", True))
    ) and not sys.stdin.isatty():
        raise SystemExit(
            "collection preflight failed: automatic local authorization requires an interactive local TTY"
        )
    try:
        pedal_path = resolve_input_event_path(root["pedal"]["device"])
    except FileNotFoundError as exc:
        raise SystemExit(f"collection preflight failed: {exc}") from exc
    paths = {
        "foot pedal": pedal_path,
        "RDK socket": runtime_root / "rdk.sock",
        "F/T-zero event record": runtime_root / "ft_zero_events.jsonl",
        "tool config": config.resolve(root["flexiv"]["tool_payload_config"]),
        "rendered camera config": rendered["camera.yaml"],
    }
    manus = str(root["teleop"]["manus_calibration"]).strip()
    if manus:
        paths["MANUS calibration"] = config.resolve(manus)
    for camera, stream in root["cameras"]["streams"].items():
        extrinsics = str(stream.get("extrinsics", "")).strip()
        if extrinsics:
            paths[f"{camera} camera extrinsics"] = config.resolve(extrinsics)
    missing = [label for label, path in paths.items() if not path.exists()]
    if missing:
        details = ", ".join(f"{label}={paths[label]}" for label in missing)
        raise SystemExit(f"collection preflight failed; missing: {details}")
    if not os.access(paths["foot pedal"], os.R_OK):
        raise SystemExit(
            f"collection preflight failed: foot pedal is not readable: {paths['foot pedal']}"
        )
    if not paths["RDK socket"].is_socket():
        raise SystemExit(
            f"collection preflight failed: RDK endpoint is not a Unix socket: {paths['RDK socket']}"
        )


def collect_main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if argv:
        raise SystemExit(
            "flexiv-inspire-collect takes no arguments; edit config/recording.yaml"
        )
    project_root = Path(__file__).resolve().parents[4]
    config = load_system_config(project_root / "config" / "site.yaml")
    runtime = _runtime_dir(config)
    rendered = render_runtime_configs(config, runtime / config.sha256[:12])
    return _run_collection(config, rendered)


def reset_main(argv: list[str] | None = None) -> int:
    selected = list(sys.argv[1:] if argv is None else argv)
    project_root = Path(__file__).resolve().parents[4]
    return main(
        ["--config", str(project_root / "config" / "site.yaml"), "reset", *selected]
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
