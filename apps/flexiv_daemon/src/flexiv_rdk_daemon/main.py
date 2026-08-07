"""CLI for the independent Flexiv RDK daemon."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import threading

import numpy as np

from .backend import FlexivRDKBackend, RobotSpec
from .configuration import (
    load_daemon_configuration,
    read_tool_payload_identity,
)
from .ft_zero import FTZeroConfig, FTZeroManager, JsonlEventRecorder
from .guard import HardwareWriteGuard
from .ipc import SeqpacketServer, StructEnvelopeCodec
from .mock_backend import MockBackend
from .server import DaemonInterlock, HandObservationCache, RDKRequestDispatcher
from isaac_teleop_core.deviceio import AsyncDeviceIOEmitter


def _default_config() -> Path:
    return Path(__file__).resolve().parents[2] / "config" / "robots.yaml"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    mode = result.add_mutually_exclusive_group()
    mode.add_argument(
        "--mock", action="store_true", help="deterministic no-hardware backend (default)"
    )
    mode.add_argument(
        "--hardware",
        action="store_true",
        help="connect to configured RDK robots, observation only by default",
    )
    result.add_argument("--config", type=Path, default=_default_config())
    result.add_argument("--socket", type=Path, default=None)
    result.add_argument("--events", type=Path, default=None)
    result.add_argument("--allow-hardware-writes", action="store_true")
    result.add_argument("--local-permit-file", type=Path, default=None)
    result.add_argument(
        "--verify-compatibility",
        action="store_true",
        help="read RobotInfo, validate both robots, print JSON and exit",
    )
    result.add_argument(
        "--print-tool-payload-hash",
        action="store_true",
        help="validate local tool/payload record, print canonical SHA-256 and exit",
    )
    result.add_argument(
        "--dev-struct-ipc",
        action="store_true",
        help="mock/development only; typed protobuf remains mandatory for hardware",
    )
    return result


def _ft_config(raw: dict[str, object]) -> FTZeroConfig:
    window = float(raw["sample_window_s"])
    rate = float(raw["sample_rate_hz"])
    return FTZeroConfig(
        external_contact_check_enabled=bool(
            raw["external_contact_check_enabled"]
        ),
        sample_window_s=window,
        sample_rate_hz=rate,
        min_samples=max(1, int(math.ceil(window * rate * 0.8))),
        operational_timeout_s=float(raw["operational_timeout_s"]),
        primitive_timeout_s=float(raw["primitive_timeout_s"]),
        enable_settle_timeout_s=float(raw["enable_settle_timeout_s"]),
        enable_settle_window_s=float(raw["enable_settle_window_s"]),
        max_joint_velocity_norm=float(raw["max_joint_velocity_norm"]),
        max_tcp_velocity_norm=float(raw["max_tcp_velocity_norm"]),
        max_wrench_std_force_n=float(raw["max_wrench_std_force_n"]),
        max_wrench_std_torque_nm=float(raw["max_wrench_std_torque_nm"]),
        max_residual_force_n=float(raw["max_residual_force_n"]),
        max_residual_torque_nm=float(raw["max_residual_torque_nm"]),
        max_pre_external_mean_force_n=float(raw["max_pre_external_mean_force_n"]),
        max_pre_external_mean_torque_nm=float(raw["max_pre_external_mean_torque_nm"]),
        max_pre_external_peak_force_n=float(raw["max_pre_external_peak_force_n"]),
        max_pre_external_peak_torque_nm=float(raw["max_pre_external_peak_torque_nm"]),
        max_hand_delta=float(raw["max_hand_delta"]),
    )


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    config = load_daemon_configuration(args.config)
    if args.print_tool_payload_hash:
        identity = read_tool_payload_identity(config.tool_payload_path)
        print(
            json.dumps(
                {
                    "source_path": str(identity.source_path),
                    "canonical_sha256": identity.sha256,
                    "mtime_ns": identity.mtime_ns,
                    "size": identity.size,
                    "locally_verified": True,
                },
                sort_keys=True,
            )
        )
        return 0
    if args.verify_compatibility and not args.hardware:
        raise SystemExit("--verify-compatibility requires --hardware")

    runtime = (
        Path(os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}"))
        / "isaac_teleop"
    )
    socket_path = args.socket or runtime / config.socket_name
    event_path = args.events or runtime / "ft_zero_events.jsonl"
    if args.allow_hardware_writes and not args.hardware:
        raise SystemExit("--allow-hardware-writes is meaningful only with --hardware")
    if args.hardware and args.dev_struct_ipc:
        raise SystemExit("generic Struct IPC is forbidden with --hardware")

    guard = HardwareWriteGuard(
        cli_enabled=args.allow_hardware_writes,
        permit_file=args.local_permit_file,
        test_backend=not args.hardware,
    )
    if args.hardware:
        specs = tuple(
            RobotSpec(
                item.side,
                item.serial,
                expected_model=item.expected_model,
                expected_active_tool=item.expected_active_tool,
                require_ft_sensor=item.require_ft_sensor,
                expected_software_prefix=item.expected_software_prefix,
                required_license=item.required_license,
            )
            for item in config.robots
        )
        backend = FlexivRDKBackend(specs, write_guard=guard)
        backend.connect()  # read-only connection + RobotInfo compatibility audit
        print(
            json.dumps(
                {
                    "event": "rdk_compatibility_preflight",
                    "config": str(config.source_path),
                    "rdk_required": "1.9.x",
                    "robots": backend.compatibility_metadata,
                    "controller_upgrade_performed": False,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if args.verify_compatibility:
            backend.disconnect()
            return 0
    else:
        backend = MockBackend()

    hands = HandObservationCache()
    if not args.hardware:
        hands.update(np.zeros(6), np.zeros(6))
    interlock = DaemonInterlock()
    events = JsonlEventRecorder(event_path)

    def verified_tool_payload_identity():
        identity = read_tool_payload_identity(config.tool_payload_path)
        if args.hardware:
            backend.verify_active_tool_payload(identity.canonical_json)
        return identity

    ft_zero = FTZeroManager(
        backend,
        write_guard=guard,
        interlock=interlock,
        read_hand_positions=hands.read,
        event_sink=events,
        config=_ft_config(config.ft_zero),
        hand_monitor_snapshot=hands.monitor_snapshot,
        tool_payload_identity=(
            verified_tool_payload_identity
            if args.hardware
            else None
        ),
    )
    deviceio = AsyncDeviceIOEmitter("rdk")
    dispatcher = RDKRequestDispatcher(
        backend,
        ft_zero,
        hands,
        interlock,
        deviceio_emitter=deviceio,
        cartesian_limits=config.cartesian_limits,
        world_frame=config.world_frame,
        world_from_base=config.world_from_base,
    )
    dispatcher.start_watchdog()
    codec = StructEnvelopeCodec() if args.dev_struct_ipc else None
    server = SeqpacketServer(
        socket_path,
        dispatcher,
        codec=codec,
        max_packet_bytes=config.max_packet_bytes,
    )
    stop = threading.Event()

    def request_stop(signum: int, frame: object) -> None:
        stop.set()
        server.close()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    try:
        server.serve_forever()
    finally:
        server.close()
        dispatcher.close()
        deviceio.close()
        if args.hardware:
            backend.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
