"""Mint a short-lived local authorization for the ROS ZeroFTSensors action."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import socket
import sys

from .typed_ipc import TypedEnvelopeCodec


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Local现场 confirmation; this command does not itself move or zero a robot."
    )
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--tool-payload-config-hash", required=True)
    parser.add_argument("--confirm-ft-unloaded", required=True)
    parser.add_argument("--socket", type=Path, default=None)
    args = parser.parse_args(argv)
    if not sys.stdin.isatty():
        raise SystemExit("authorization must be run interactively from a local TTY")
    runtime = Path(os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")) / "isaac_teleop"
    socket_path = args.socket or runtime / "rdk.sock"
    codec = TypedEnvelopeCodec.load()
    request = codec.encode(
        "authorize_zero_ft",
        1,
        {
            "session_id": args.session_id,
            "operator_confirmation": args.confirm_ft_unloaded,
            "tool_payload_config_hash": args.tool_payload_config_hash,
        },
    )
    with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as client:
        client.connect(str(socket_path))
        client.sendall(request)
        response = client.recv(65536)
    kind, _, payload = codec.decode(response)
    if kind != "authorize_zero_ft_result" or not payload.get("authorized", False):
        raise SystemExit(f"authorization rejected: {payload.get('reason', kind)}")
    # This token is intentionally short-lived and single-use. The local ROS
    # Action client passes it in ZeroFTSensors.Goal.local_authorization_token.
    print(payload["one_time_token"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
