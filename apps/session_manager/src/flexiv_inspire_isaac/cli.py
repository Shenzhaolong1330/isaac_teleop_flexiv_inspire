"""One-command operations for the independent Flexiv + Inspire stack."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from typing import Sequence

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
    "flexiv_inspire_xr_bridge.ros_node",
    "flexiv-inspire-xr-bridge",
    "run_manus_plugin.sh",
    "manus_hand_plugin",
    "run_isaac_camera_receiver.sh",
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
    state_path.write_text(json.dumps({
        "config": str(config.path), "config_sha256": config.sha256,
        "started_unix_ns": time.time_ns(),
        "pids": list(dict.fromkeys([*existing_pids, *[item.pid for item in processes]])),
    }, indent=2) + "\n", encoding="utf-8")


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
    processes = []
    for index, command in enumerate(commands):
        selected = list(command)
        if "--allow-hardware-writes" in selected:
            permit = Path(selected[selected.index("--local-permit-file") + 1])
            _ensure_write_permit(permit)
        log = (directory / f"{log_prefix}-{index}.log").open("ab", buffering=0)
        processes.append(
            subprocess.Popen(
                selected,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=_command_environment(selected),
            )
        )
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
            print("CloudXR、Quest 输入和 MANUS 已自动启动", flush=True)
            return
        time.sleep(0.1)
    _stop_started_processes(processes)
    raise SystemExit(
        "CloudXR 在 60 秒内未就绪；请检查 Quest USB 连接和最新 record 日志"
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
    while live and time.monotonic() < deadline:
        live = [process for process in live if process.poll() is None]
        if live:
            time.sleep(0.05)
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
        return None
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
        command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(
            "utf-8", errors="replace"
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
            print("RDK 配置已更新，daemon 已自动重启", flush=True)
            return True
        time.sleep(0.05)
    raise RuntimeError("RDK daemon 配置已更新，但旧进程无法停止")


def _process_running(*markers: str) -> bool:
    return bool(_matching_process_ids(*markers))


def _matching_process_ids(*markers: str) -> list[int]:
    matches: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", errors="replace"
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
            command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(
                b"\0", b" "
            ).decode("utf-8", errors="replace")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if project_root in command:
            matches.append(pid)
    return matches


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


def _xr_receiver_command(config, rendered: dict[str, Path]) -> list[str]:
    return [
        str(_project_root(config) / "orchestration/run_isaac_camera_receiver.sh"),
        str(rendered["isaac_camera_receiver.yaml"]),
        str(config.document["xr_video"]["display"]["mode"]),
    ]


def _xr_raw_source_command(config) -> list[str]:
    xr = config.document["xr_video"]
    command = [
        str(_project_root(config) / "orchestration/run_xr_raw_source.sh"),
        "--transport",
        str(xr["transport"]),
    ]
    wifi_connection = str(xr.get("wifi_connection", "")).strip()
    if wifi_connection:
        command.extend(["--wifi-connection", wifi_connection])
    return command


def _manus_plugin_command(config) -> list[str]:
    return [
        str(_project_root(config) / "orchestration/run_manus_plugin.sh")
    ]


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
        "-p", f"sessions_root:={config.resolve(recording['output_root'])}",
        "-p", f"dataset_name:={recording['dataset_name']}",
        "-p", f"episode_count:={int(recording['episode_count'])}",
        "-p", "task_description:=" + json.dumps(
            str(recording["task_description"]), ensure_ascii=False
        ),
        "-p", f"session_id:={session['id']}",
        "-p", f"tool_config:={config.resolve(root['flexiv']['tool_payload_config'])}",
        "-p", f"ft_zero_record:={runtime_root / 'ft_zero_events.jsonl'}",
        "-p", f"camera_config:={rendered['camera.yaml']}",
        "-p", f"deviceio_socket:={runtime_root / 'deviceio.sock'}",
        "-p", f"runtime_dir:={runtime_root}",
        "-p", f"rdk_socket:={runtime_root / 'rdk.sock'}",
        "-p", f"camera_recording_mode:={recording['camera_recording_mode']}",
        "-p", f"deviceio_mode:={recording['deviceio_mode']}",
        "-p", f"auto_start:={str(bool(recording.get('auto_start', True))).lower()}",
        "-p", f"auto_authorize_home:={str(bool(recording.get('auto_authorize_home', True))).lower()}",
        "-p", f"auto_authorize_control:={str(bool(recording.get('auto_authorize_control', True))).lower()}",
        "-p", f"home_result_timeout_s:={float(root['flexiv']['home']['timeout_s']) + 10.0}",
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
        "command_enabled:="
        + str(bool(root["teleop"]["control_enabled"])).lower(),
    ]


def _commands(config, rendered: dict[str, Path], *, include_xr_receiver: bool) -> list[list[str]]:
    root = config.document
    rdk_command = _rdk_command(config)
    commands = [rdk_command]
    commands += [
        _control_command(config, rendered),
        _teleop_input_command(config, rendered),
        ["flexiv-inspire-camera-node", "--ros-args", "-p", f"config:={rendered['camera.yaml']}"],
        ["flexiv-inspire-dftp-node", "--ros-args", "--params-file", str(rendered["dftp.yaml"])],
        ["flexiv-inspire-pedal-router", "--ros-args", "--params-file", str(rendered["pedal.yaml"])],
        _episode_command(config, rendered),
    ]
    if bool(root["xr_video"]["enabled"]):
        commands.append(_xr_raw_source_command(config))
        commands.append(_manus_plugin_command(config))
        commands.append(["flexiv-inspire-xr-bridge", "--ros-args", "--params-file", str(rendered["xr_bridge.yaml"])])
        if include_xr_receiver:
            commands.append(_xr_receiver_command(config, rendered))
    return commands


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
                str(
                    runtime_root
                    / f"{root['session']['id']}.write-permit"
                ),
            ]
        )
    return command


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="robot", description="Config-driven independent Flexiv/Inspire operations")
    parser.add_argument("--config", "-c", default="config/site.yaml", help="single site YAML")
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
    record.add_argument("--dry-run", action="store_true", help="render and print commands only")
    xr_group = record.add_mutually_exclusive_group()
    xr_group.add_argument("--with-xr", action="store_true", help="start the IsaacTeleop Quest video receiver")
    xr_group.add_argument("--no-xr", action="store_true", help="keep RTP bridge available but do not start the Quest receiver")
    sub.add_parser("stop", help="recover and stop leftover managed services")
    sub.add_parser("collect", help="run the configured multi-episode collection in the foreground")
    sub.add_parser("visualize", help="play the configured dataset in read-only Rerun")
    sub.add_parser("replay", help="guarded hardware replay from config/playback.yaml")
    convert = sub.add_parser(
        "convert",
        help="export the latest completed episode to LeRobot",
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
    camera_calibrate = sub.add_parser(
        "camera-calibrate",
        help="collect or solve camera hand-eye calibration",
        add_help=False,
    )
    camera_calibrate.add_argument(
        "-h", "--help", action="store_true", dest="calibration_help"
    )
    camera_calibrate.add_argument("calibration_args", nargs=argparse.REMAINDER)
    xr_view = sub.add_parser("xr-view", help="one-command IsaacTeleop camera display in Quest/monitor")
    xr_view.add_argument("--dry-run", action="store_true")
    sub.add_parser("xr-doctor", help="check ffmpeg, Docker, IsaacTeleop and CloudXR prerequisites")
    return parser


def _xr_doctor(config) -> int:
    root = _project_root(config)
    ffmpeg = str(config.document["xr_video"]["ffmpeg"])
    checks = {
        "ffmpeg": Path(ffmpeg).is_file() or shutil.which(ffmpeg) is not None,
        "docker": shutil.which("docker") is not None,
        "isaac_camera_streamer": (root / "third_party/IsaacTeleop/examples/camera_streamer/camera_streamer.sh").is_file(),
        "cloudxr_runtime_json": (Path.home() / ".cloudxr/openxr_cloudxr.json").is_file(),
    }
    image = False
    if checks["docker"]:
        image = subprocess.run(["docker", "image", "inspect", "isaac-teleop-camera:latest"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    checks["isaac_camera_image"] = image
    print(json.dumps({"ready": all(checks.values()), "checks": checks,
                      "next": "orchestration/setup_xr_receiver.sh" if not all(checks.values()) else "flexiv-inspire xr-view"}, indent=2))
    return 0 if all(checks.values()) else 2


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = load_system_config(args.config)
    except (OSError, SystemConfigError) as exc:
        raise SystemExit(f"invalid system config: {exc}")
    runtime = _runtime_dir(config)
    if args.operation == "validate":
        print(json.dumps({"valid": True, "config": str(config.path), "sha256": config.sha256}, indent=2)); return 0
    if args.operation == "render":
        rendered = render_runtime_configs(config, runtime / config.sha256[:12])
        print(json.dumps({key: str(value) for key, value in rendered.items()}, indent=2)); return 0
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
            print(json.dumps(command, indent=2)); return 0
        return subprocess.call(command)
    if args.operation == "record":
        rendered = render_runtime_configs(config, runtime / config.sha256[:12])
        default_xr = bool(config.document["xr_video"]["auto_start_with_record"])
        include_xr = args.with_xr or (default_xr and not args.no_xr)
        commands = _commands(config, rendered, include_xr_receiver=include_xr)
        if args.dry_run:
            print(json.dumps(commands, indent=2)); return 0
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
        # Pedal routing and the episode controller are started by
        # _run_collection() so the latter can remain attached to this TTY.
        commands = [
            command
            for command in commands
            if not _is_collection_process_command(command)
        ]
        rdk_socket = Path(
            config.document["session"]["runtime_root"]
        ).expanduser() / "rdk.sock"
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
        reusable_services = (
            (
                (
                    "flexiv_inspire_isaac.dftp.ros_node",
                    "flexiv-inspire-dftp-node",
                    "flexiv_inspire_dftp/dftp_node",
                ),
                ("flexiv-inspire-dftp-node",),
            ),
            (
                ("flexiv-inspire-camera-node", "flexiv_inspire_isaac.cameras.ros_node"),
                ("flexiv-inspire-camera-node",),
            ),
            (
                ("flexiv-inspire-xr-raw-source", "flexiv_inspire_isaac.xr_raw_ros_source"),
                ("run_xr_raw_source.sh",),
            ),
            (
                ("manus_hand_plugin",),
                ("run_manus_plugin.sh",),
            ),
            (
                ("flexiv-inspire-xr-bridge", "flexiv_inspire_xr_bridge.ros_node"),
                ("flexiv-inspire-xr-bridge",),
            ),
            (
                ("run_isaac_camera_receiver.sh",),
                ("run_isaac_camera_receiver.sh",),
            ),
        )
        for process_markers, command_markers in reusable_services:
            if _process_running(*process_markers):
                commands = [
                    command
                    for command in commands
                    if not any(
                        marker in " ".join(command)
                        for marker in command_markers
                    )
                ]
        log_prefix = f"record-{time.time_ns()}"
        logs = [
            runtime / f"{log_prefix}-{index}.log"
            for index in range(len(commands))
        ]
        processes = _start(commands, runtime, log_prefix=log_prefix)
        _write_state(runtime, config, processes)
        _verify_process_startup(processes, logs)
        if bool(config.document["xr_video"]["enabled"]):
            _wait_for_cloudxr_runtime(processes, logs)
        print(
            f"started {len(processes)} supporting services; "
            f"XR receiver={'on' if include_xr else 'off'}",
            flush=True,
        )
        try:
            try:
                return _run_collection(config, rendered)
            except KeyboardInterrupt:
                return 130
        finally:
            _stop_started_processes(processes)
            _remove_state_pids(runtime, {process.pid for process in processes})
            print("record stopped; current episode was finalized", flush=True)
    if args.operation == "collect":
        rendered = render_runtime_configs(config, runtime / config.sha256[:12])
        return _run_collection(config, rendered)
    if args.operation == "stop":
        owned_pids: list[int] = []
        state = runtime / "processes.json"
        daemon_state = runtime / "rdk-daemon.json"
        reset_state = runtime / "reset-services.json"
        if state.is_file():
            values = json.loads(state.read_text(encoding="utf-8"))
            owned_pids.extend(int(pid) for pid in values.get("pids", []))
        if daemon_state.is_file():
            values = json.loads(daemon_state.read_text(encoding="utf-8"))
            if values.get("pid") is not None:
                owned_pids.append(int(values["pid"]))
        if reset_state.is_file():
            values = json.loads(reset_state.read_text(encoding="utf-8"))
            owned_pids.extend(int(pid) for pid in values.get("pids", []))
        discovered_pids = set(_matching_managed_process_ids(config))
        # Launcher state can survive a crash for long enough that the kernel
        # reuses a PID. Never signal a state-file PID unless its current
        # command line still identifies a managed process in this checkout.
        confirmed_owned_pids = set(owned_pids).intersection(discovered_pids)
        if not confirmed_owned_pids and not discovered_pids:
            raise SystemExit("no launcher process state exists")
        for pid in confirmed_owned_pids:
            try:
                os.killpg(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        # Reset state records can be overwritten when a later reset reuses an
        # already-running bridge. Stop those precisely matched orphan services
        # by PID without assuming ownership of their process group.
        for pid in discovered_pids.difference(confirmed_owned_pids):
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        requested_pids = discovered_pids
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            live_pids = {
                pid for pid in requested_pids if Path(f"/proc/{pid}").exists()
            }
            if not live_pids:
                break
            time.sleep(0.05)
        else:
            raise RuntimeError(
                "managed services did not stop within 5 seconds: "
                + ",".join(str(pid) for pid in sorted(live_pids))
            )
        print("stop requested; captured episodes are retained"); return 0
    if args.operation == "convert":
        return _run_convert(config, args)
    if args.operation == "visualize":
        from flexiv_inspire_isaac.rerun_viz.offline import main as visualize_main

        return visualize_main([])
    from .replay import main as replay_main

    return replay_main([])


def _latest_completed_manifest(config) -> Path:
    recording = config.document["recording"]
    dataset_root = (
        config.resolve(recording["output_root"])
        / str(recording["dataset_name"])
    )
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"数据集目录不存在: {dataset_root}")
    candidates: list[tuple[tuple[int, int, int], Path]] = []
    for path in dataset_root.glob("*/manifest.json"):
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
            )
        )
    if not candidates:
        raise FileNotFoundError(f"没有已完成 episode: {dataset_root}")
    return max(candidates, key=lambda item: item[0])[1]


def _run_convert(config, args) -> int:
    root = _project_root(config)
    manifest = (
        config.resolve(args.manifest)
        if str(args.manifest).strip()
        else _latest_completed_manifest(config)
    )
    if not manifest.is_file():
        raise SystemExit(f"manifest 不存在: {manifest}")
    configured_view = config.document["lerobot_export"]["action"]["view"]
    action_view = str(args.action_view or configured_view)
    dataset_name = str(config.document["recording"]["dataset_name"])
    output_root = (
        config.resolve(args.output_root)
        if str(args.output_root).strip()
        else root
        / "artifacts"
        / "lerobot"
        / dataset_name
        / manifest.parent.name
        / action_view
    )
    if output_root.exists() and (
        not output_root.is_dir() or any(output_root.iterdir())
    ):
        raise SystemExit(
            f"输出目录已存在且非空: {output_root}；请指定 --output-root"
        )
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", dataset_name).strip("-") or "dataset"
    repo_id = str(args.repo_id).strip() or f"local/{slug}-{action_view}"
    executable = root / "envs/data-py312/bin/flexiv-inspire-lerobot-export"
    if not executable.is_file():
        raise SystemExit("data 环境不存在；先运行 scripts/env/create_envs.sh")
    command = [
        str(executable),
        "--manifest",
        str(manifest),
        "--output-root",
        str(output_root),
        "--repo-id",
        repo_id,
        "--export-config",
        str(config.path),
        "--action-view",
        action_view,
    ]
    print(
        f"转换 {manifest.parent.name} -> {action_view} -> {output_root}",
        flush=True,
    )
    return subprocess.call(command)


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
        "Reset: 双臂 F/T 清零 -> 配置的双臂 Home -> 双手张开/闭合/张开",
        flush=True,
    )

    from flexiv_inspire_control.zero_ft_local import main as zero_ft_main

    return zero_ft_main(
        [
            "--rdk-socket",
            str(rdk_socket),
            "--tool-payload-config",
            str(tool_config),
            "--preview-seconds",
            str(float(args.preview_seconds)),
            "--skip-preview-if-ft-zeroed",
            "--confirm-ft-unloaded",
            "FLEXIV-FT-UNLOADED",
            "--home-after-zero",
            "--cycle-hands-after-home",
            "--home-timeout",
            str(float(root["flexiv"]["home"]["timeout_s"])),
            # Zeroing intentionally latches the daemon in maintenance. This
            # authorization clears only that freshly validated hold before Home.
            "--clear-home-hold-latched",
        ]
    )


def _ensure_reset_ros_services(config) -> None:
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

    runtime = _runtime_dir(config)
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
    (runtime / "reset-services.json").write_text(
        json.dumps(
            {
                "pids": [process.pid for process in processes],
                "commands": commands,
                "logs": [str(path) for path in log_paths],
                "started_unix_ns": time.time_ns(),
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
    runtime_root = Path(
        config.document["session"]["runtime_root"]
    ).expanduser()
    rdk_socket = runtime_root / "rdk.sock"
    command = _rdk_command(config)
    if "--mock" in command:
        raise SystemExit(
            "当前 config 把 RDK daemon 配置为 mock；真机 Reset 需要 hardware 配置"
        )
    config_path = Path(command[command.index("--config") + 1])
    config_sha256 = _sha256_file(config_path)
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
            raise SystemExit(
                f"RDK daemon 启动失败，退出码 {status}\n{detail}"
            )
        time.sleep(0.1)
    raise SystemExit(
        f"RDK daemon 启动超时；查看日志：{log_path}"
    )


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
    output = config.resolve(recording["output_root"]) / recording["dataset_name"]
    print(
        f"collection starting: {recording['episode_count']} episodes -> {output}\n"
        f"task: {recording['task_description']}\n"
        "right=commit/Home/next, left=discard/Home/retry, Quest A=pause/Home/resume, Ctrl-C=save/stop",
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
            return int(controller.wait())
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
            except subprocess.TimeoutExpired:
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
        raise SystemExit("flexiv-inspire-collect takes no arguments; edit config/recording.yaml")
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
