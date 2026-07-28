"""Versioned localhost IPC protocol shared by ROS and hardware processes."""

from __future__ import annotations

import json
import math
from typing import Any

from .mapping import AXES, SIDES, action_vectors

PROTOCOL_VERSION = 1
ARM_KEYS = tuple(
    f"{side}_delta_ee_pose.{axis}" for side in SIDES for axis in AXES
)


def encode_packet(packet: dict[str, Any]) -> bytes:
    payload = dict(packet)
    payload["protocol_version"] = PROTOCOL_VERSION
    return json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def decode_packet(data: bytes, *, max_packet_bytes: int = 65536) -> dict[str, Any]:
    if len(data) > int(max_packet_bytes):
        raise ValueError("IPC packet exceeds configured maximum")
    decoded = json.loads(data.decode("utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError("IPC packet must decode to a mapping")
    if int(decoded.get("protocol_version", -1)) != PROTOCOL_VERSION:
        raise ValueError("unsupported IPC protocol version")
    return decoded


def validate_action(
    action: Any,
    *,
    max_translation_step_m: float,
    max_rotation_step_rad: float,
) -> dict[str, float | bool]:
    if not isinstance(action, dict):
        raise ValueError("action must be a mapping")
    missing = [key for key in ARM_KEYS if key not in action]
    if missing:
        raise ValueError(f"action is missing arm fields: {missing}")
    unexpected_delta = [
        key
        for key in action
        if "_delta_ee_pose." in str(key) and key not in ARM_KEYS
    ]
    if unexpected_delta:
        raise ValueError(f"action has unexpected delta fields: {unexpected_delta}")
    clean: dict[str, float | bool] = {}
    for key in ARM_KEYS:
        value = float(action[key])
        if not math.isfinite(value):
            raise ValueError(f"action field {key} must be finite")
        clean[key] = value
    clean["teleop_enable_pressed"] = bool(
        action.get("teleop_enable_pressed", False)
    )
    left_dp, left_dr, right_dp, right_dr = action_vectors(clean)
    for side, translation, rotation in (
        ("left", left_dp, left_dr),
        ("right", right_dp, right_dr),
    ):
        translation_norm = float((translation @ translation) ** 0.5)
        rotation_norm = float((rotation @ rotation) ** 0.5)
        if translation_norm > max_translation_step_m + 1e-9:
            raise ValueError(
                f"{side} translation step {translation_norm:.6f} exceeds "
                f"{max_translation_step_m:.6f}"
            )
        if rotation_norm > max_rotation_step_rad + 1e-9:
            raise ValueError(
                f"{side} rotation step {rotation_norm:.6f} exceeds "
                f"{max_rotation_step_rad:.6f}"
            )
    if not clean["teleop_enable_pressed"]:
        raise ValueError("action packet is not deadman-enabled")
    return clean

