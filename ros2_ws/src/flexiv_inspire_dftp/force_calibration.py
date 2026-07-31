"""One-shot, locally authorized RH56DFTP force-sensor calibration."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from typing import Any

from .modbus import CommandCapableModbusTcpClient, LocalWritePermit
from .protocol import (
    Register,
    decode_modbus_i16,
    decode_packed_u8,
)


CONFIRMATION = "INSPIRE-FORCE-CALIBRATION"
MIN_OPEN_ANGLE = 850
MAX_IDLE_CURRENT_MA = 100


def _read_state(client: CommandCapableModbusTcpClient) -> dict[str, list[int]]:
    return {
        "angles": list(decode_modbus_i16(client.read_raw(Register.ANGLE_ACTUAL, 6))),
        "forces_g": list(
            decode_modbus_i16(client.read_raw(Register.FORCE_ACTUAL, 6))
        ),
        "currents_ma": list(
            decode_modbus_i16(client.read_raw(Register.CURRENT, 6))
        ),
        "errors": list(
            decode_packed_u8(client.read_raw(Register.ERROR, 3))
        ),
    }


def _assert_safe_to_calibrate(state: dict[str, list[int]]) -> None:
    if any(angle < MIN_OPEN_ANGLE for angle in state["angles"]):
        raise RuntimeError(
            f"hand is not fully open: every angle must be >= {MIN_OPEN_ANGLE}"
        )
    if any(abs(current) > MAX_IDLE_CURRENT_MA for current in state["currents_ma"]):
        raise RuntimeError(
            "hand is not idle: actuator current exceeds "
            f"{MAX_IDLE_CURRENT_MA} mA"
        )
    if any(state["errors"]):
        raise RuntimeError(f"hand reports errors: {state['errors']}")


def calibrate_hand(
    host: str,
    port: int,
    *,
    side: str,
    wait: Callable[[float], None] = time.sleep,
    client_factory: Callable[..., CommandCapableModbusTcpClient] = (
        CommandCapableModbusTcpClient
    ),
) -> dict[str, Any]:
    permit = LocalWritePermit.issue_after_local_authorization(
        f"force-calibration-{side}", "DFTP-LOCAL-CONTROL-AUTHORIZED"
    )
    with client_factory(host, port, timeout_s=1.0, permit=permit) as client:
        before = _read_state(client)
        _assert_safe_to_calibrate(before)
        client.write_single_u16(Register.FORCE_CALIBRATION, 1)
        # The vendor routine completes asynchronously. Six seconds is used by
        # RH56-series documentation; one extra second avoids an early readback.
        wait(7.0)
        after = _read_state(client)
    return {
        "side": side,
        "host": host,
        "port": port,
        "register": Register.FORCE_CALIBRATION,
        "written_value": 1,
        "before": before,
        "after": after,
        "force_delta_g": [
            after_value - before_value
            for before_value, after_value in zip(
                before["forces_g"], after["forces_g"], strict=True
            )
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate exactly one Inspire RH56DFTP force sensor set. "
            "Keep the selected hand open and completely unloaded."
        )
    )
    parser.add_argument("--side", choices=("left", "right"), required=True)
    parser.add_argument("--left", default="192.168.5.11")
    parser.add_argument("--right", default="192.168.5.12")
    parser.add_argument("--port", type=int, default=6000)
    parser.add_argument(
        "--confirm",
        required=True,
        help=f"must be exactly {CONFIRMATION}",
    )
    args = parser.parse_args()
    if args.confirm != CONFIRMATION:
        parser.error(f"--confirm must be exactly {CONFIRMATION}")
    host = args.left if args.side == "left" else args.right
    result = calibrate_hand(host, args.port, side=args.side)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
