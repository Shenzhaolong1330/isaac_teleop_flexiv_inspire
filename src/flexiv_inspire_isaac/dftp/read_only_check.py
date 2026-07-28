"""Read-only DFTP hardware validation.

This module deliberately imports only :class:`ReadOnlyModbusTcpClient`; there
is no code path to Modbus function 0x06 or 0x10.
"""

from __future__ import annotations

import argparse
import json
import time

from .modbus import ReadOnlyModbusTcpClient
from .protocol import Register, TACTILE_LAYOUT, decode_modbus_i16, decode_tactile


def check_hand(host: str, port: int, include_tactile: bool) -> dict:
    result: dict = {"host": host, "port": port, "read_only": True}
    started = time.monotonic_ns()
    with ReadOnlyModbusTcpClient(host, port) as client:
        result["angles"] = list(
            decode_modbus_i16(client.read_raw(Register.ANGLE_ACTUAL, 6))
        )
        result["forces_g"] = list(
            decode_modbus_i16(client.read_raw(Register.FORCE_ACTUAL, 6))
        )
        if include_tactile:
            count = 0
            for surface in TACTILE_LAYOUT:
                count += len(
                    decode_tactile(
                        client.read_raw(surface.start_address, surface.taxels), surface
                    )
                )
            result["tactile_taxels"] = count
    result["duration_ms"] = (time.monotonic_ns() - started) / 1e6
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", default="192.168.5.11")
    parser.add_argument("--right", default="192.168.5.12")
    parser.add_argument("--port", type=int, default=6000)
    parser.add_argument("--include-tactile", action="store_true")
    args = parser.parse_args()
    output = [
        check_hand(args.left, args.port, args.include_tactile),
        check_hand(args.right, args.port, args.include_tactile),
    ]
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
