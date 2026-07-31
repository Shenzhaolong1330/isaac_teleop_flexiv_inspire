"""One-command operations for the independent Flexiv + Inspire stack."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from typing import Sequence

from .system_config import SystemConfigError, load_system_config, render_runtime_configs


def _runtime_dir(config) -> Path:
    return Path(config.document["session"]["runtime_root"]).expanduser() / "launcher"


def _project_root(config) -> Path:
    return config.root


def _write_state(directory: Path, config, processes: list[subprocess.Popen]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "processes.json").write_text(json.dumps({
        "config": str(config.path), "config_sha256": config.sha256,
        "started_unix_ns": time.time_ns(), "pids": [item.pid for item in processes],
    }, indent=2) + "\n", encoding="utf-8")


def _start(commands: list[Sequence[str]], directory: Path) -> list[subprocess.Popen]:
    directory.mkdir(parents=True, exist_ok=True)
    processes = []
    for index, command in enumerate(commands):
        log = (directory / f"service-{index}.log").open("ab", buffering=0)
        processes.append(subprocess.Popen(list(command), stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
    return processes


def _xr_receiver_command(config, rendered: dict[str, Path]) -> list[str]:
    return [
        str(_project_root(config) / "orchestration/run_isaac_camera_receiver.sh"),
        str(rendered["isaac_camera_receiver.yaml"]),
        str(config.document["xr_video"]["display"]["mode"]),
    ]


def _commands(config, rendered: dict[str, Path], *, include_xr_receiver: bool) -> list[list[str]]:
    root = config.document
    session = root["session"]
    runtime_root = Path(session["runtime_root"]).expanduser()
    rdk_socket = runtime_root / "rdk.sock"
    ft_zero_record = runtime_root / "ft_zero_events.jsonl"
    rdk_command = [
        *root["commands"]["rdk_daemon"],
        "--config",
        str(config.resolve(root["flexiv"]["rdk_config"])),
        "--socket",
        str(rdk_socket),
        "--events",
        str(ft_zero_record),
    ]
    commands = [rdk_command]
    commands += [
        [*root["commands"]["teleop_launch"], f"session_id:={session['id']}",
         f"rdk_socket:={rdk_socket}", f"foot_pedal:={root['pedal']['device']}",
         f"control_config:={rendered['control_bridge.yaml']}", f"teleop_config:={rendered['teleop.yaml']}"],
        ["flexiv-inspire-camera-node", "--ros-args", "-p", f"config:={rendered['camera.yaml']}"],
        ["flexiv-inspire-dftp-node", "--ros-args", "--params-file", str(rendered["dftp.yaml"])],
        ["flexiv-inspire-pedal-router", "--ros-args", "--params-file", str(rendered["pedal.yaml"])],
        ["flexiv-inspire-episode-controller", "--ros-args", "-p", f"sessions_root:={config.resolve(root['session']['sessions_root'])}",
         "-p", f"session_id:={session['id']}", "-p", f"tool_config:={config.resolve(root['flexiv']['tool_payload_config'])}",
         "-p", f"ft_zero_record:={ft_zero_record}",
         "-p", f"camera_config:={rendered['camera.yaml']}",
         "-p", f"manus_calibration:={config.resolve(root['teleop']['manus_calibration']) if str(root['teleop']['manus_calibration']).strip() else ''}",
         "-p", f"deviceio_socket:={runtime_root / 'deviceio.sock'}",
         "-p", f"camera_recording_mode:={root['recording']['camera_recording_mode']}",
         "-p", f"home_result_timeout_s:={float(root['flexiv']['home']['timeout_s']) + 10.0}"],
    ]
    if bool(root["xr_video"]["enabled"]):
        commands.append(["flexiv-inspire-xr-bridge", "--ros-args", "--params-file", str(rendered["xr_bridge.yaml"])])
        if include_xr_receiver:
            commands.append(_xr_receiver_command(config, rendered))
    return commands


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="flexiv-inspire", description="Config-driven independent Flexiv/Inspire operations")
    parser.add_argument("--config", "-c", default="config/site.yaml", help="single site YAML")
    sub = parser.add_subparsers(dest="operation", required=True)
    sub.add_parser("validate", help="validate the site config without devices")
    sub.add_parser("render", help="write immutable child configs and print their paths")
    record = sub.add_parser("record", help="launch the config-selected acquisition and teleoperation stack")
    record.add_argument("--dry-run", action="store_true", help="render and print commands only")
    xr_group = record.add_mutually_exclusive_group()
    xr_group.add_argument("--with-xr", action="store_true", help="start the IsaacTeleop Quest video receiver")
    xr_group.add_argument("--no-xr", action="store_true", help="keep RTP bridge available but do not start the Quest receiver")
    sub.add_parser("stop", help="stop services started by record")
    visualize = sub.add_parser("visualize", help="start read-only live Rerun")
    visualize.add_argument("--save", default="")
    replay = sub.add_parser("replay", help="offline replay inspection only; never moves hardware")
    replay.add_argument("--episode", required=True, type=Path)
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
        processes = _start(commands, runtime)
        _write_state(runtime, config, processes)
        print(f"started {len(processes)} services; XR receiver={'on' if include_xr else 'off'}; robot writes remain locally gated")
        return 0
    if args.operation == "stop":
        state = runtime / "processes.json"
        if not state.is_file():
            raise SystemExit("no launcher process state exists")
        values = json.loads(state.read_text(encoding="utf-8"))
        for pid in values.get("pids", []):
            try:
                os.killpg(int(pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
        print("stop requested; captured episodes are retained"); return 0
    if args.operation == "visualize":
        command = ["flexiv-inspire-rerun"] + (["--save", args.save] if args.save else ["--spawn"])
        return subprocess.call(command)
    episode = args.episode.expanduser().resolve(strict=True)
    manifest = episode / "manifest.json"
    if not manifest.is_file():
        raise SystemExit("--episode must be an episode directory containing manifest.json")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    print(json.dumps({"mode": "offline-shadow-replay", "hardware_writes": False, "episode": str(episode),
                      "completed": bool(data.get("completed")), "reason": data.get("completion_reason", "")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
