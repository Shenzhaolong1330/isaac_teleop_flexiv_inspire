"""Provider-independent ROS contract for Quest controller input."""

from __future__ import annotations

import time
from typing import Any

import msgpack
from isaac_teleop_core.octet_sequence import encode_octet_sequence


def controller_payload(
    *,
    left_squeeze_value: float,
    right_squeeze_value: float,
    left_primary_click: bool,
    right_primary_click: bool,
    left_is_active: bool,
    right_is_active: bool,
    timestamp_ns: int | None = None,
) -> dict[str, Any]:
    """Build the stable controller payload consumed by ``teleop_input``."""

    timestamp = time.time_ns() if timestamp_ns is None else int(timestamp_ns)
    if timestamp < 0:
        raise ValueError("controller timestamp must be non-negative")
    squeezes = (float(left_squeeze_value), float(right_squeeze_value))
    if any(value < 0.0 or value > 1.0 for value in squeezes):
        raise ValueError("controller squeeze values must be in [0,1]")
    return {
        "timestamp": timestamp,
        "left_squeeze_value": squeezes[0],
        "right_squeeze_value": squeezes[1],
        "left_primary_click": bool(left_primary_click),
        "right_primary_click": bool(right_primary_click),
        "left_is_active": bool(left_is_active),
        "right_is_active": bool(right_is_active),
    }


def encode_controller_payload(payload: dict[str, Any]) -> list[int]:
    """Encode the contract exactly as ``std_msgs/ByteMultiArray.data``."""

    return encode_octet_sequence(msgpack.packb(payload, use_bin_type=True))
