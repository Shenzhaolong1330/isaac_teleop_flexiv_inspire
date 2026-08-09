"""Read-only DFTP hardware validation.

This module deliberately imports only :class:`ReadOnlyModbusTcpClient`; there
is no code path to Modbus function 0x06 or 0x10.
"""

from __future__ import annotations

import argparse
import json
import time

from .modbus import ReadOnlyModbusTcpClient
from .profiles import hand_profile
from .protocol import (
    HAND_STATE_BLOCK_START,
    HAND_STATE_BLOCK_WORDS,
    Register,
    decode_hand_state_block,
    decode_tactile,
)


def _decode_byte_registers(payload: bytes) -> list[int]:
    """Restore ascending byte-address order from Modbus word byte order."""

    if len(payload) % 2:
        raise ValueError("packed byte registers require complete Modbus words")
    return [
        value
        for index in range(0, len(payload), 2)
        for value in payload[index : index + 2][::-1]
    ]


def check_hand(
    host: str, port: int, include_tactile: bool, model: str, side: str
) -> dict:
    profile = hand_profile(model, side=side)
    result: dict = {
        "host": host,
        "port": port,
        "model": profile.name,
        "product": profile.product_name,
        "read_only": True,
    }
    started = time.monotonic_ns()
    with ReadOnlyModbusTcpClient(host, port) as client:
        result["hand_id"] = int(client.read_raw(1000, 1)[-1])
        result["configured_ip"] = ".".join(
            str(value)
            for value in _decode_byte_registers(client.read_raw(Register.IP_ADDRESS, 2))
        )
        state = decode_hand_state_block(
            client.read_raw(HAND_STATE_BLOCK_START, HAND_STATE_BLOCK_WORDS)
        )
        result["angles"] = list(state["angle"])
        result["forces_g"] = list(state["force"])
        result["currents_ma"] = list(state["current"])
        result["errors"] = list(state["error"])
        result["temperatures_c"] = list(state["temperature"])
        if include_tactile:
            count = 0
            minimum = 65535
            maximum = 0
            for surface in profile.tactile_layout:
                values = decode_tactile(
                    client.read_raw(surface.start_address, surface.taxels), surface
                )
                count += len(
                    values
                )
                minimum = min(minimum, *values)
                maximum = max(maximum, *values)
            result["tactile_taxels"] = count
            result["tactile_raw_range"] = [minimum, maximum]
    result["duration_ms"] = (time.monotonic_ns() - started) / 1e6
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", default="192.168.5.11")
    parser.add_argument("--right", default="192.168.5.12")
    parser.add_argument("--port", type=int, default=6000)
    parser.add_argument("--left-model", default="rh56dftp_2")
    parser.add_argument("--right-model", default="rh56dftp_2")
    parser.add_argument("--side", choices=("left", "right", "both"), default="both")
    parser.add_argument("--include-tactile", action="store_true")
    args = parser.parse_args()
    output = []
    if args.side in {"left", "both"}:
        output.append(
            check_hand(
                args.left, args.port, args.include_tactile, args.left_model, "left"
            )
        )
    if args.side in {"right", "both"}:
        output.append(
            check_hand(
                args.right,
                args.port,
                args.include_tactile,
                args.right_model,
                "right",
            )
        )
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
