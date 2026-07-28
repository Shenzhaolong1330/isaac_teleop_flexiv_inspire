"""Pure validation for the ROS `/control/sent_command` DFTP boundary."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping


class SafeCommandRejected(ValueError):
    pass


@dataclass(frozen=True)
class GateSnapshot:
    configured_session: str
    control_session: str
    control_active: bool
    fault_latched: bool
    control_state_received_ns: int
    control_state_timeout_ns: int
    last_hand_state_ns: Mapping[str, int]
    hand_state_timeout_ns: int
    last_sequence: int


@dataclass(frozen=True)
class ValidatedHandTarget:
    side: str
    sequence: int
    angles: tuple[int, ...]
    remaining_ttl_ns: int
    source: str


def message_time_ns(message: Any) -> int:
    return int(message.sec) * 1_000_000_000 + int(message.nanosec)


def validate_safe_command_message(
    message: Any,
    snapshot: GateSnapshot,
    *,
    now_monotonic_ns: int,
    now_ros_ns: int,
) -> tuple[ValidatedHandTarget, ...]:
    if snapshot.fault_latched or not snapshot.control_active:
        raise SafeCommandRejected("control is not ACTIVE")
    if (
        now_monotonic_ns - snapshot.control_state_received_ns
        > snapshot.control_state_timeout_ns
    ):
        raise SafeCommandRejected("control state is stale")
    if (
        int(message.schema_version) != 1
        or str(message.session_id) != snapshot.configured_session
        or str(message.session_id) != snapshot.control_session
        or not bool(message.deadman)
    ):
        raise SafeCommandRejected("schema/session/deadman gate failed")
    if int(message.sequence) <= snapshot.last_sequence:
        raise SafeCommandRejected("sequence is not monotonic")
    ttl_ns = message_time_ns(message.ttl)
    stamp_ns = message_time_ns(message.header.stamp)
    age_ns = now_ros_ns - stamp_ns
    if ttl_ns <= 0 or ttl_ns > 1_000_000_000:
        raise SafeCommandRejected("TTL must be in (0, 1s]")
    if age_ns < -50_000_000 or age_ns >= ttl_ns:
        raise SafeCommandRejected("command is future-dated or expired")
    if len(message.trajectory) != 1:
        raise SafeCommandRejected("sent_command must contain exactly one safe point")
    point = message.trajectory[0]
    if message_time_ns(point.execute_after) >= ttl_ns:
        raise SafeCommandRejected("execute_after exceeds TTL")
    remaining = ttl_ns - max(0, age_ns)
    output = []
    for side, constant_name, fallback_bit in (
        ("left", "LEFT_HAND_VALID", 4),
        ("right", "RIGHT_HAND_VALID", 8),
    ):
        bit = int(getattr(type(message), constant_name, fallback_bit))
        if not int(message.valid_mask) & bit:
            continue
        if (
            now_monotonic_ns - int(snapshot.last_hand_state_ns[side])
            > snapshot.hand_state_timeout_ns
        ):
            raise SafeCommandRejected(f"{side} hand state is stale")
        raw = tuple(float(value) for value in getattr(point, f"{side}_hand_targets"))
        if len(raw) != 6 or not all(math.isfinite(value) for value in raw):
            raise SafeCommandRejected(f"{side} hand target must be six finite values")
        angles = tuple(int(round(value)) for value in raw)
        if any(value < 0 or value > 1000 for value in angles):
            raise SafeCommandRejected(f"{side} hand target is outside 0..1000")
        output.append(
            ValidatedHandTarget(
                side=side,
                sequence=int(message.sequence),
                angles=angles,
                remaining_ttl_ns=remaining,
                source=f"safe:{message.source}",
            )
        )
    return tuple(output)
