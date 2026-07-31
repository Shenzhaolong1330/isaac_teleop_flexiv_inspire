"""Mint and deliver a short-lived local authorization for configured Home."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import rclpy
from std_msgs.msg import Empty, String

from .ipc_client import RDKIPCClient


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-id", required=True)
    parser.add_argument(
        "--confirm",
        required=True,
        help="must equal FLEXIV-HOME-MOVE",
    )
    parser.add_argument("--clear-hold-latched", action="store_true")
    parser.add_argument(
        "--request",
        action="store_true",
        help="publish /control/home_request after delivering the token",
    )
    parser.add_argument("--socket", type=Path, default=None)
    args = parser.parse_args(argv)
    if not sys.stdin.isatty():
        raise SystemExit("Home authorization must run from a local interactive TTY")
    runtime = Path(
        os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
    )
    client = RDKIPCClient(
        args.socket or runtime / "isaac_teleop" / "rdk.sock"
    )
    kind, payload = client.request(
        "authorize_home",
        {
            "session_id": args.session_id,
            "operator_confirmation": args.confirm,
            "clear_hold_latched": args.clear_hold_latched,
        },
    )
    client.close()
    if kind != "authorize_home_result" or not payload.get("authorized", False):
        raise SystemExit(
            f"Home authorization rejected: {payload.get('reason', kind)}"
        )
    outgoing = String()
    outgoing.data = json.dumps(
        {
            "session_id": args.session_id,
            "one_time_token": payload["one_time_token"],
            "expires_monotonic_ns": payload["expires_monotonic_ns"],
        },
        separators=(",", ":"),
    )
    rclpy.init()
    node = rclpy.create_node("local_home_authorizer")
    publisher = node.create_publisher(
        String, "/control/local_home_authorization", 1
    )
    deadline = time.monotonic() + 1.0
    while publisher.get_subscription_count() == 0 and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
    for _ in range(3):
        publisher.publish(outgoing)
        rclpy.spin_once(node, timeout_sec=0.05)
    if args.request:
        request_publisher = node.create_publisher(
            Empty, "/control/home_request", 1
        )
        deadline = time.monotonic() + 1.0
        while (
            request_publisher.get_subscription_count() == 0
            and time.monotonic() < deadline
        ):
            rclpy.spin_once(node, timeout_sec=0.05)
        request_publisher.publish(Empty())
        rclpy.spin_once(node, timeout_sec=0.10)
    node.destroy_node()
    rclpy.shutdown()
    return 0


def request_main(argv: list[str] | None = None) -> int:
    selected = list(sys.argv[1:] if argv is None else argv)
    if "--request" not in selected:
        selected.append("--request")
    return main(selected)
