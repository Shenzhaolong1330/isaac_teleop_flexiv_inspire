"""Explicit local utility for assigning an Inspire hand's Modbus TCP IP."""

from __future__ import annotations

import argparse
from ipaddress import IPv4Address
import json
from typing import Any

from .modbus import CommandCapableModbusTcpClient, LocalWritePermit
from .profiles import hand_profile
from .protocol import (
    HAND_STATE_BLOCK_START,
    HAND_STATE_BLOCK_WORDS,
    Register,
    decode_hand_state_block,
)


CONFIRMATION = "INSPIRE-NETWORK-CONFIG"


def decode_ipv4_registers(payload: bytes) -> IPv4Address:
    if len(payload) != 4:
        raise ValueError("IP register payload must contain four bytes")
    return IPv4Address(bytes((payload[1], payload[0], payload[3], payload[2])))


def encode_ipv4_registers(address: IPv4Address | str) -> tuple[int, int]:
    octets = IPv4Address(address).packed
    return (
        int.from_bytes((octets[1], octets[0]), "big"),
        int.from_bytes((octets[3], octets[2]), "big"),
    )


def configure_hand_ip(
    host: str,
    new_ip: str,
    *,
    port: int,
    side: str,
    model: str,
    expected_current_ip: str,
    confirmation: str,
) -> dict[str, Any]:
    if confirmation != CONFIRMATION:
        raise PermissionError(f"confirmation must be exactly {CONFIRMATION}")
    profile = hand_profile(model, side=side)
    expected = IPv4Address(expected_current_ip)
    target = IPv4Address(new_ip)
    if target.is_multicast or target.is_unspecified or target.is_loopback:
        raise ValueError(f"new IP is not a usable device address: {target}")
    permit = LocalWritePermit.issue_after_local_authorization(
        f"network-config-{side}", "DFTP-LOCAL-CONTROL-AUTHORIZED"
    )
    with CommandCapableModbusTcpClient(
        host, port, timeout_s=1.0, permit=permit
    ) as client:
        current = decode_ipv4_registers(client.read_raw(Register.IP_ADDRESS, 2))
        if current != expected or IPv4Address(host) != expected:
            raise RuntimeError(
                f"refusing IP write: connected={host}, register={current}, "
                f"expected={expected}"
            )
        state = decode_hand_state_block(
            client.read_raw(HAND_STATE_BLOCK_START, HAND_STATE_BLOCK_WORDS)
        )
        if any(state["error"]):
            raise RuntimeError(f"hand reports actuator errors: {state['error']}")
        client.write_i16(Register.IP_ADDRESS, encode_ipv4_registers(target))
        client.write_single_u16(Register.SAVE_PARAMETERS, 1)
        stored = decode_ipv4_registers(client.read_raw(Register.IP_ADDRESS, 2))
        if stored != target:
            raise RuntimeError(f"IP readback mismatch: wrote {target}, read {stored}")
    return {
        "side": side,
        "model": profile.name,
        "product": profile.product_name,
        "connected_ip": str(expected),
        "stored_ip": str(stored),
        "angles": list(state["angle"]),
        "errors": list(state["error"]),
        "saved_to_flash": True,
        "requires_power_cycle": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Assign one Inspire RH56 Modbus TCP address without moving it."
    )
    parser.add_argument("--host", required=True)
    parser.add_argument("--new-ip", required=True)
    parser.add_argument("--expected-current-ip", required=True)
    parser.add_argument("--port", type=int, default=6000)
    parser.add_argument("--side", choices=("left", "right"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--confirm", required=True)
    args = parser.parse_args()
    result = configure_hand_ip(
        args.host,
        args.new_ip,
        port=args.port,
        side=args.side,
        model=args.model,
        expected_current_ip=args.expected_current_ip,
        confirmation=args.confirm,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
