"""Local-TTY control arm client; remote policy processes cannot mint this token."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import rclpy
from std_msgs.msg import String

from .ipc_client import RDKIPCClient


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--source", choices=("teleop", "policy", "replay"), required=True)
    parser.add_argument("--confirm", required=True, help="must equal FLEXIV-CONTROL-ARM")
    parser.add_argument("--clear-hold-latched", action="store_true")
    parser.add_argument("--socket", type=Path, default=None)
    args = parser.parse_args(argv)
    if not sys.stdin.isatty():
        raise SystemExit("control authorization must run from a local interactive TTY")
    runtime = Path(os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}"))
    client = RDKIPCClient(args.socket or runtime / "isaac_teleop" / "rdk.sock")
    kind, payload = client.request(
        "authorize_control",
        {
            "session_id": args.session_id,
            "source": args.source,
            "operator_confirmation": args.confirm,
            "clear_hold_latched": args.clear_hold_latched,
        },
        # Clearing a measured hardware hold can switch both controllers back
        # into Cartesian mode.  The old 50 ms IPC default was shorter than a
        # normal real-robot mode transition and produced false timeouts.
        timeout_s=10.0 if args.clear_hold_latched else 2.0,
    )
    client.close()
    if kind != "authorize_control_result" or not payload.get("authorized", False):
        raise SystemExit(f"authorization rejected: {payload.get('reason', kind)}")
    outgoing = String()
    outgoing.data = json.dumps(
        {
            "session_id": args.session_id,
            "source": args.source,
            "one_time_token": payload["one_time_token"],
            "expires_monotonic_ns": payload["expires_monotonic_ns"],
        },
        separators=(",", ":"),
    )
    rclpy.init()
    node = rclpy.create_node("local_control_authorizer")
    publisher = node.create_publisher(String, "/control/local_arm_authorization", 1)
    deadline = time.monotonic() + 1.0
    while publisher.get_subscription_count() == 0 and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
    for _ in range(3):
        publisher.publish(outgoing)
        rclpy.spin_once(node, timeout_sec=0.05)
    node.destroy_node()
    rclpy.shutdown()
    return 0
