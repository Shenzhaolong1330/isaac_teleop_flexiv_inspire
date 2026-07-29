"""One-command operations for the independent Flexiv + Inspire stack."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Sequence

from .system_config import SystemConfigError, load_system_config, render_runtime_configs


def _runtime_dir(config) -> Path:
    return Path(config.document["session"]["runtime_root"]).expanduser() / "launcher"


def _write_state(directory: Path, config, processes: list[subprocess.Popen]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "processes.json").write_text(json.dumps({
        "config": str(config.path), "config_sha256": config.sha256,
        "started_unix_ns": time.time_ns(), "pids": [item.pid for item in processes],
    }, indent=2) + "\n", encoding="utf-8")


def _start(commands: list[Sequence[str]], directory: Path) -> list[subprocess.Popen]:
    processes = []
    for index, command in enumerate(commands):
        log = (directory / f"service-{index}.log").open("ab", buffering=0)
        processes.append(subprocess.Popen(list(command), stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
    return processes


def _commands(config, rendered: dict[str, Path]) -> list[list[str]]:
    root = config.document
    session = root["session"]
    commands = [list(root["commands"]["rdk_daemon"])]
    commands += [
        [*root["commands"]["teleop_launch"], f"session_id:={session['id']}", f"rdk_socket:={Path(session['runtime_root']) / 'rdk.sock'}", f"foot_pedal:={root['pedal']['device']}", f"control_config:={rendered['control_bridge.yaml']}", f"teleop_config:={rendered['teleop.yaml']}"],
        ["flexiv-inspire-camera-node", "--ros-args", "--params-file", str(rendered["camera.yaml"])],
        ["flexiv-inspire-dftp-node", "--ros-args", "--params-file", str(rendered["dftp.yaml"])],
        ["flexiv-inspire-pedal-router", "--ros-args", "--params-file", str(rendered["pedal.yaml"])],
        ["flexiv-inspire-episode-controller", "--ros-args", "-p", f"sessions_root:={root['session']['sessions_root']}", "-p", f"session_id:={session['id']}", "-p", f"tool_config:={root['flexiv']['tool_payload_config']}", "-p", f"camera_config:={rendered['camera.yaml']}", "-p", f"manus_calibration:={root['teleop']['manus_calibration']}", "-p", f"deviceio_socket:={Path(session['runtime_root']) / 'deviceio.sock'}", "-p", f"camera_recording_mode:={root['recording']['camera_recording_mode']}"],
    ]
    return commands


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="flexiv-inspire", description="Config-driven independent Flexiv/Inspire operations")
    parser.add_argument("--config", "-c", default="config/site.yaml", help="single site YAML")
    sub = parser.add_subparsers(dest="operation", required=True)
    sub.add_parser("validate", help="validate the site config without devices")
    sub.add_parser("render", help="write immutable child configs and print their paths")
    record = sub.add_parser("record", help="launch shadow-safe stack; right pedal starts/stops episodes")
    record.add_argument("--dry-run", action="store_true", help="render and print commands only")
    sub.add_parser("stop", help="stop services started by record")
    visualize = sub.add_parser("visualize", help="start read-only live Rerun")
    visualize.add_argument("--save", default="")
    replay = sub.add_parser("replay", help="offline replay inspection only; never moves hardware")
    replay.add_argument("--episode", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = load_system_config(args.config)
    except (OSError, SystemConfigError) as exc:
        raise SystemExit(f"invalid system config: {exc}")
    runtime = _runtime_dir(config)
    if args.operation == "validate":
        print(json.dumps({"valid": True, "config": str(config.path), "sha256": config.sha256}, indent=2))
        return 0
    if args.operation == "render":
        rendered = render_runtime_configs(config, runtime / config.sha256[:12])
        print(json.dumps({key: str(value) for key, value in rendered.items()}, indent=2))
        return 0
    if args.operation == "record":
        rendered = render_runtime_configs(config, runtime / config.sha256[:12])
        commands = _commands(config, rendered)
        if args.dry_run:
            print(json.dumps(commands, indent=2))
            return 0
        processes = _start(commands, runtime)
        _write_state(runtime, config, processes)
        print(f"started {len(processes)} motionless/shadow-safe services; right pedal toggles episodes")
        return 0
    if args.operation == "stop":
        state = runtime / "processes.json"
        if not state.is_file():
            raise SystemExit("no launcher process state exists")
        values = json.loads(state.read_text(encoding="utf-8"))
        for pid in values.get("pids", []):
            try:
                # Process groups make stop include each ROS launch child.
                import os
                os.killpg(int(pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
        print("stop requested; captured episodes are retained")
        return 0
    if args.operation == "visualize":
        command = ["flexiv-inspire-rerun"] + (["--save", args.save] if args.save else ["--spawn"])
        return subprocess.call(command)
    episode = args.episode.expanduser().resolve(strict=True)
    manifest = episode / "manifest.json"
    if not manifest.is_file():
        raise SystemExit("--episode must be an episode directory containing manifest.json")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    print(json.dumps({"mode": "offline-shadow-replay", "hardware_writes": False, "episode": str(episode), "completed": bool(data.get("completed")), "reason": data.get("completion_reason", "")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
